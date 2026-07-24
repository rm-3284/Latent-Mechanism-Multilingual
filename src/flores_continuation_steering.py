"""
Steer held-out FLORES+ sentence prefixes with per-language latents and continue.

For each held-out sentence (prepared by prepare_flores_heldout.py), each prompt
language, each steering language, and each feature-selection method, applies
feature interventions and autoregressively continues the truncated prefix.

Cross-lingual steering ablates prompt-language features and amplifies
steering-language features (same pattern as evaluate_selected_features.py).
Same-language steering uses amplification only.

Example:
  python flores_continuation_steering.py \\
    --model gemma-2-2b \\
    --features-file ../data/additional_experiments/gemma-2-2b/selected_features/AnnSel/selection_threshold_0.1.json \\
    --output-dir ../data/flores_continuation/gemma-2-2b/default_features

  python flores_continuation_steering.py --model gemma-2-2b --features-file ... --sentence-idx 0 --prompt-lang fr --steer-lang de
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ablation_amplification_intervention import ablation_and_amplification, activation_dict
from circuit_tracer_import import ReplacementModel
from device_setup import device
from evaluate_selected_features import (
    build_cross_lang_intervention,
    build_intervention_maps,
    load_feature_payload,
)
from hyperparameter_sensitivity import repo_data_dir
from intervention import ablation, amplification
from models import hf_model_names, hf_transcoder_names
from multiple_words_intervention import get_logprob_with_intervention
from template import lang_to_flores_key


def resolve_output_dir(path: str | None, model_name: str, feature_stem: str) -> str:
    if path is not None:
        return path if os.path.isabs(path) else os.path.join(
            os.path.dirname(repo_data_dir()), path
        )
    return os.path.join(
        repo_data_dir(),
        "flores_continuation",
        model_name,
        feature_stem,
    )


def build_steering_intervention(
    ablations: dict[str, list[tuple[int, int, int, float]]],
    amplifications: dict[str, list[tuple[int, int, int, float]]],
    prompt_lang: str,
    steer_lang: str,
) -> list[tuple[int, int, int, float]]:
    if prompt_lang == steer_lang:
        return amplifications[steer_lang]
    return build_cross_lang_intervention(ablations, amplifications, prompt_lang, steer_lang)


def generate_with_intervention(
    prompt: str,
    model: ReplacementModel,
    interventions: list[tuple[int, int, int, float]],
    max_new_tokens: int,
    top_p: float = 0.9,
    greedy: bool = False,
) -> str:
    generated = prompt
    eos_id = model.tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        new_logits, _ = model.feature_intervention(generated, interventions)
        next_token_logits = new_logits[0, -1, :]

        if greedy:
            next_token_id = next_token_logits.argmax().item()
        else:
            probs = F.softmax(next_token_logits.float(), dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            indices_to_remove = cumulative_probs > top_p
            indices_to_remove[..., 1:] = indices_to_remove[..., :-1].clone()
            indices_to_remove[..., 0] = False
            sorted_probs[indices_to_remove] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum()
            next_token_index = torch.multinomial(sorted_probs, num_samples=1)
            next_token_id = sorted_indices[next_token_index].item()

        if eos_id is not None and next_token_id == eos_id:
            break

        token = model.tokenizer.decode([next_token_id], skip_special_tokens=False)
        generated += token

        if token.strip() and token.strip()[-1] in ".?!":
            break

    return generated[len(prompt) :]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FLORES held-out continuation steering")
    parser.add_argument("--model", type=str, default="gemma-2-2b", choices=hf_model_names.keys())
    parser.add_argument("--features-file", type=str, required=True)
    parser.add_argument("--heldout-file", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--sentence-idx", type=int, default=None)
    parser.add_argument("--prompt-lang", type=str, default=None, choices=lang_to_flores_key.keys())
    parser.add_argument("--steer-lang", type=str, default=None, choices=lang_to_flores_key.keys())
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--score-gold", action="store_true", default=True)
    parser.add_argument("--no-score-gold", action="store_false", dest="score_gold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    langs = list(lang_to_flores_key.keys())

    heldout_path = args.heldout_file or os.path.join(
        repo_data_dir(), "flores_heldout", args.model, "heldout_sentences.json"
    )
    with open(heldout_path, "r", encoding="utf-8") as f:
        heldout = json.load(f)

    payload = load_feature_payload(args.features_file)
    features_by_lang = payload["selected_features"]
    method = payload.get("method", "unknown")
    feature_stem = os.path.splitext(os.path.basename(args.features_file))[0]

    amp_dir = os.path.join(repo_data_dir(), "amplification_values", args.model)
    feature_values = activation_dict(features_by_lang, amp_dir, langs)
    ablations = {lang: ablation(feature_values, lang) for lang in langs}
    amplifications = {lang: amplification(feature_values, lang) for lang in langs}

    model_name = args.model
    model = ReplacementModel.from_pretrained(
        hf_model_names[model_name],
        hf_transcoder_names[model_name],
        device=device,
        dtype=torch.bfloat16,
    )

    sentence_indices = (
        [args.sentence_idx]
        if args.sentence_idx is not None
        else list(range(len(heldout["sentences"])))
    )
    prompt_langs = [args.prompt_lang] if args.prompt_lang else langs
    steer_langs = [args.steer_lang] if args.steer_lang else langs

    results: list[dict[str, Any]] = []
    total = len(sentence_indices) * len(prompt_langs) * len(steer_langs)
    bar = tqdm(total=total, desc=f"{method} continuation")

    for sent_idx in sentence_indices:
        sentence_entry = heldout["sentences"][sent_idx]
        flores_index = sentence_entry["flores_index"]

        for prompt_lang in prompt_langs:
            lang_data = sentence_entry["languages"][prompt_lang]
            prefix = lang_data["prefix"]
            gold_suffix = lang_data["suffix"]
            max_tokens = max(args.max_new_tokens, len(model.tokenizer.encode(gold_suffix)) + 8)

            for steer_lang in steer_langs:
                interventions = build_steering_intervention(
                    ablations, amplifications, prompt_lang, steer_lang
                )
                continuation = generate_with_intervention(
                    prefix,
                    model,
                    interventions,
                    max_new_tokens=max_tokens,
                    top_p=args.top_p,
                    greedy=args.greedy,
                )

                record: dict[str, Any] = {
                    "method": method,
                    "flores_index": flores_index,
                    "sentence_idx": sent_idx,
                    "prompt_lang": prompt_lang,
                    "steer_lang": steer_lang,
                    "prefix": prefix,
                    "gold_suffix": gold_suffix,
                    "generated_continuation": continuation,
                    "generated_full": prefix + continuation,
                }

                if args.score_gold:
                    record["gold_logprob_baseline"] = _safe_logprob(
                        prefix, gold_suffix, model, None
                    )
                    record["gold_logprob_steered"] = _safe_logprob(
                        prefix, gold_suffix, model, interventions
                    )

                results.append(record)
                bar.update(1)
                torch.cuda.empty_cache()
                gc.collect()

    bar.close()

    out_dir = resolve_output_dir(args.output_dir, args.model, feature_stem)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{method}__continuations.json")
    output = {
        "model": args.model,
        "method": method,
        "features_file": os.path.abspath(args.features_file),
        "heldout_file": os.path.abspath(heldout_path),
        "top_p": args.top_p,
        "greedy": args.greedy,
        "max_new_tokens": args.max_new_tokens,
        "num_results": len(results),
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(results)} continuation records to {out_path}")


def _safe_logprob(
    prefix: str,
    suffix: str,
    model: ReplacementModel,
    interventions: list[tuple[int, int, int, float]] | None,
) -> float | None:
    try:
        if interventions is None:
            from ablation_amplification_intervention import model_run

            _, logits = model_run(prefix + suffix, model)
            prompt_ids = model.tokenizer.encode(prefix, add_special_tokens=False)
            target_ids = model.tokenizer.encode(suffix, add_special_tokens=False)
            log_probs = F.log_softmax(
                logits[0, len(prompt_ids) - 1 : len(prompt_ids) - 1 + len(target_ids)],
                dim=-1,
            )
            total = 0.0
            for i, tid in enumerate(target_ids):
                total += log_probs[i, tid].item()
            return total
        return get_logprob_with_intervention(prefix, suffix, model, interventions)
    except Exception as exc:
        return None


if __name__ == "__main__":
    main()
