#!/usr/bin/env python3
"""Diagnose why feature-intervention logprobs are NaN for qwen3-4b multiple_words data.

Run on a GPU node from the repo root:
  PYTHONPATH=src python scripts/maintenance/debug_nan_logprobs.py \\
    --model qwen3-4b --prompt-lang en --list-lang en

Writes a text report to stdout and optional --report path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import torch
import torch.nn.functional as F

from lib.ablation_amplification_intervention import (
    activation_dict,
    combine_except_one,
    description_based_features,
)
from lib.circuit_tracer_import import ReplacementModel
from lib.device_setup import device
from lib.intervention import ablation, amplification, normalize_intervention_list
from lib.models import hf_model_names, hf_transcoder_names, layer_num
from pipeline.multiple_words_intervention import (
    base_prompts,
    categories,
    get_logprob_of_string,
    get_logprob_with_intervention,
    number_of_choices_to_display,
    options,
    transform_intervention,
)
from lib.template import lang_to_flores_key


def parse_args():
    p = argparse.ArgumentParser(description="Debug NaN logprobs in multiple_words interventions")
    p.add_argument("--model", default="qwen3-4b", choices=hf_model_names.keys())
    p.add_argument("--prompt-lang", default="en")
    p.add_argument("--list-lang", default="en")
    p.add_argument("--category", default="months", help="Category key in multiple_words_intervention.py")
    p.add_argument("--report", default=None, help="Optional path to write report")
    return p.parse_args()


def build_prompt_and_ans(prompt_lang: str, list_lang: str, category_key: str, langs: list[str]):
    category_text = categories[category_key][prompt_lang]
    n_display = number_of_choices_to_display[category_key]
    display = options[category_key][list_lang][:n_display]
    if list_lang in ("ja", "zh"):
        fill_in = ",".join(display) + ","
    else:
        fill_in = ", ".join(display) + ","
    prompt = base_prompts[prompt_lang].format(category=category_text, choices=fill_in)
    ans = {lang: options[category_key][lang][n_display:] for lang in langs}
    for lang in langs:
        if lang in ("ja", "zh"):
            ans[lang] = ",".join(ans[lang])
        else:
            ans[lang] = " " + ", ".join(ans[lang])
    return prompt, ans


def diagnose_logprob_step(
    label: str,
    prompt: str,
    target: str,
    model: ReplacementModel,
    intervention: list,
) -> dict:
    prompt_ids = model.tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = model.tokenizer.encode(target, add_special_tokens=False)
    input_ids = prompt_ids + target_ids
    full_input = model.tokenizer.decode(input_ids)
    start_pos = len(prompt_ids)
    last_pos = len(input_ids) - 1
    transformed = transform_intervention(intervention, start_pos, last_pos)

    out = {
        "label": label,
        "prompt_len": len(prompt_ids),
        "target_len": len(target_ids),
        "total_len": len(input_ids),
        "intervention_len": len(intervention),
        "transformed_len": len(transformed),
        "positions_spanned": last_pos - start_pos + 1,
    }

    try:
        logits, _ = model.feature_intervention(
            full_input, transformed, return_activations=False
        )
        slice_logits = logits[0, len(prompt_ids) - 1 : len(input_ids) - 1, :]
        out["logits_shape"] = tuple(slice_logits.shape)
        out["logits_has_nan"] = bool(torch.isnan(slice_logits).any().item())
        out["logits_has_inf"] = bool(torch.isinf(slice_logits).any().item())
        log_probs = F.log_softmax(slice_logits, dim=-1)
        out["log_probs_has_nan"] = bool(torch.isnan(log_probs).any().item())
        total = 0.0
        ok = True
        for i, tid in enumerate(target_ids):
            lp = log_probs[i, tid].item()
            if not math.isfinite(lp):
                ok = False
            total += lp
        out["total_logprob"] = total
        out["finite"] = ok and math.isfinite(total)
    except Exception as exc:
        out["error"] = repr(exc)
        out["finite"] = False
        out["total_logprob"] = float("nan")

    return out


def main():
    args = parse_args()
    langs = list(lang_to_flores_key.keys())
    lines: list[str] = []

    def log(msg: str = ""):
        lines.append(msg)
        print(msg)

    repo = _REPO_ROOT
    data_dir = os.path.join(repo, "data")
    model_name = args.model

    log(f"device={device}")
    log(f"model={model_name} prompt_lang={args.prompt_lang} list_lang={args.list_lang}")

    prompt, ans = build_prompt_and_ans(
        args.prompt_lang, args.list_lang, args.category, langs
    )
    target_en = ans["en"]
    log(f"prompt: {prompt!r}")
    log(f"target (en): {target_en!r}")

    # Load features
    flores = os.path.join(data_dir, "flores_features", model_name)
    amp_dir = os.path.join(data_dir, "amplification_values", model_name)
    desc_features = description_based_features(flores, langs, 0.1)
    desc_interventions = activation_dict(desc_features, amp_dir, langs)

    log("\n=== Feature counts (description, threshold 0.1) ===")
    for lang in langs:
        abl = ablation(desc_interventions, lang)
        amp = amplification(desc_interventions, lang)
        log(
            f"  {lang}: {len(desc_features[lang])} features, "
            f"ablation={len(abl)} tuples, amp={len(amp)} tuples, "
            f"abl[0] type={type(abl[0][3]).__name__ if abl else 'n/a'}"
        )

    log("\n=== Loading ReplacementModel ===")
    model = ReplacementModel.from_pretrained(
        hf_model_names[model_name],
        hf_transcoder_names[model_name],
        device=device,
        dtype=torch.bfloat16,
    )

    json_path = os.path.join(
        data_dir,
        "interventions_multiple_words",
        model_name,
        args.prompt_lang,
        args.list_lang,
        "interventions_and_results_description.json",
    )
    if os.path.isfile(json_path):
        with open(json_path, encoding="utf-8") as f:
            stored = json.load(f)
        if prompt in stored.get("distractor ablation", {}):
            block = stored["distractor ablation"][prompt]["en"]["en"]
            stored_val = block.get(target_en)
            log(f"\nStored JSON value for distractor ablation en->en: {stored_val}")

    cases = [
        ("baseline (no intervention)", []),
    ]
    for ilang in ("en", "fr", "de"):
        raw = ablation(desc_interventions, ilang)
        cases.append((f"ablation desc features ({ilang}), raw tensors", raw))
        cases.append(
            (f"ablation desc features ({ilang}), float normalized", normalize_intervention_list(raw))
        )
        if raw:
            cases.append((f"ablation desc features ({ilang}), first feature only", raw[:1]))

    log("\n=== Logprob diagnostics ===")
    for label, intervention in cases:
        d = diagnose_logprob_step(label, prompt, target_en, model, intervention)
        log(f"\n{label}:")
        for k, v in d.items():
            if k != "label":
                log(f"  {k}: {v}")

    # amplification / combined
    log("\n=== Amplification (en) ===")
    amp_en = amplification(desc_interventions, "en")
    d = diagnose_logprob_step("amplification en", prompt, target_en, model, amp_en)
    for k, v in d.items():
        if k != "label":
            log(f"  {k}: {v}")

    log("\n=== feature_everything ablation (excludes en) ===")
    feat = {lang: ablation(desc_interventions, lang) for lang in langs}
    everything = combine_except_one(feat, "en")
    d = diagnose_logprob_step("feature_everything (not en)", prompt, target_en, model, everything)
    for k, v in d.items():
        if k != "label":
            log(f"  {k}: {v}")

    log("\n=== Compare get_logprob_of_string vs intervention ===")
    try:
        base_lp = get_logprob_of_string(prompt, target_en, model)
        log(f"  get_logprob_of_string: {base_lp}")
    except Exception as exc:
        log(f"  get_logprob_of_string ERROR: {exc}")

    for ilang in ("en", "fr"):
        try:
            interv = normalize_intervention_list(ablation(desc_interventions, ilang))
            lp = get_logprob_with_intervention(prompt, target_en, model, interv)
            log(f"  get_logprob_with_intervention ({ilang}): {lp}")
        except Exception as exc:
            log(f"  get_logprob_with_intervention ({ilang}) ERROR: {exc}")

    log("\n=== Summary ===")
    log(
        "If only 'en' ablation produces NaN: English description features break "
        "feature_intervention for this prompt (likely too many / wrong-layer hooks), "
        "not a JSON/recompute key bug. Check transformed_len and logits_has_nan above."
    )

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nWrote report to {args.report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
