import argparse
import csv
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ablation_amplification_intervention import (
    activation_dict,
    amplification,
    description_based_features,
    freq_based_features,
    mean_value_based_features,
    model_intervention,
)
from circuit_tracer_import import ReplacementModel
from direction_ablation import project_orthogonally
from intervention import get_best_base, get_best_rank, logit_diff_single
from models import hf_model_names, hf_transcoder_names
from pipeline_data.adjectives import big_data
from device_setup import device
from template import base_strings, lang_to_flores_key


METHODS = ["desc", "val", "freq"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute target amplification, remove a second method's projection component, "
            "and measure the isolated residual effect."
        )
    )
    parser.add_argument("--model", type=str, default="gemma-2-2b", choices=hf_model_names.keys())
    parser.add_argument("--prompt-lang", type=str, default=None)
    parser.add_argument("--adj-lang", type=str, default=None)
    parser.add_argument("--intervention-lang", type=str, default=None)
    parser.add_argument("--projection-lang", type=str, default=None)
    parser.add_argument("--amplification-method", type=str, default=None, choices=METHODS)
    parser.add_argument("--projection-method", type=str, default=None, choices=METHODS)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--output-file", type=str, default=None)
    return parser.parse_args()


def default_output_file(model_name: str, prompt_lang: str | None, adj_lang: str | None) -> str:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(repo_root, "data", "projected_amplification_isolation", model_name)
    os.makedirs(out_dir, exist_ok=True)
    prompt_part = prompt_lang or "all"
    adj_part = adj_lang or "all"
    return os.path.join(out_dir, f"prompt_{prompt_part}__adj_{adj_part}.csv")


def load_method_interventions(model_name: str) -> dict[str, dict[str, list[tuple[int, int, int, torch.Tensor]]]]:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = os.path.join(repo_root, "data")
    flores_directory = os.path.join(data_root, "flores_features", model_name)
    lang_specific_directory = os.path.join(data_root, "language_specific_features", model_name)
    multilingual_features_directory = os.path.join(data_root, "multilingual_llm_features", model_name)
    amplification_values_directory = os.path.join(data_root, "amplification_values", model_name)

    langs = list(lang_to_flores_key.keys())

    desc_features = description_based_features(flores_directory, langs, 0.1)
    val_features = mean_value_based_features(multilingual_features_directory, langs, 50)
    freq_features = freq_based_features(lang_specific_directory, langs)

    desc_interventions = activation_dict(desc_features, amplification_values_directory, langs)
    val_interventions = activation_dict(val_features, amplification_values_directory, langs)
    freq_interventions = activation_dict(freq_features, amplification_values_directory, langs)

    method_interventions: dict[str, dict[str, list[tuple[int, int, int, torch.Tensor]]]] = {
        "desc": {},
        "val": {},
        "freq": {},
    }
    for lang in langs:
        method_interventions["desc"][lang] = amplification(desc_interventions, lang)
        method_interventions["val"][lang] = amplification(val_interventions, lang)
        method_interventions["freq"][lang] = amplification(freq_interventions, lang)
    return method_interventions


def group_interventions_by_layer(interventions: list[tuple[int, int, int, torch.Tensor]]) -> dict[int, list[tuple[int, int, int, torch.Tensor]]]:
    grouped: dict[int, list[tuple[int, int, int, torch.Tensor]]] = defaultdict(list)
    for layer, pos, feature_idx, val in interventions:
        grouped[int(layer)].append((int(layer), int(pos), int(feature_idx), val))
    return grouped


def tensorize(values: list[float], reference: torch.Tensor) -> torch.Tensor:
    return torch.tensor(values, device=reference.device, dtype=torch.float32)


def build_isolated_interventions(
    model: ReplacementModel,
    original_activations: torch.Tensor,
    amp_interventions: list[tuple[int, int, int, torch.Tensor]],
    proj_interventions: list[tuple[int, int, int, torch.Tensor]],
) -> list[tuple[int, int, int, torch.Tensor]]:
    amp_by_layer = group_interventions_by_layer(amp_interventions)
    proj_by_layer = group_interventions_by_layer(proj_interventions)
    final_interventions: list[tuple[int, int, int, torch.Tensor]] = []

    for layer, amp_items in amp_by_layer.items():
        amp_features = [feature_idx for _, _, feature_idx, _ in amp_items]
        if not amp_features:
            continue

        transcoder = model.transcoders[layer]
        W_dec = transcoder.W_dec

        orig_vals = tensorize(
            [original_activations[layer, -1, feature_idx].item() for feature_idx in amp_features],
            W_dec,
        )
        amp_vals = tensorize(
            [value.item() if isinstance(value, torch.Tensor) else float(value) for _, _, _, value in amp_items],
            W_dec,
        )
        amp_delta = amp_vals - orig_vals

        proj_items = proj_by_layer.get(layer, [])
        if proj_items:
            proj_features = [feature_idx for _, _, feature_idx, _ in proj_items]
            proj_basis = F.normalize(W_dec[proj_features].to(torch.float32), p=2, dim=1)

            amp_basis = W_dec[amp_features].to(torch.float32)
            hidden_delta = amp_delta @ amp_basis
            residual_hidden = project_orthogonally(hidden_delta.view(1, 1, -1), proj_basis).view(-1)

            amp_basis_columns = W_dec.T[:, amp_features].to(torch.float32)
            residual_coefficients = residual_hidden @ torch.linalg.pinv(amp_basis_columns).T
            final_vals = orig_vals + residual_coefficients
        else:
            final_vals = amp_vals

        for feature_idx, value in zip(amp_features, final_vals):
            final_interventions.append((layer, -1, feature_idx, value.to(dtype=W_dec.dtype)))

    return final_interventions


def main() -> None:
    args = parse_args()
    model_name = args.model
    langs = list(lang_to_flores_key.keys())

    if args.prompt_lang is not None and args.prompt_lang not in langs:
        raise KeyError(f"Unsupported prompt language: {args.prompt_lang}")
    if args.adj_lang is not None and args.adj_lang not in langs:
        raise KeyError(f"Unsupported adjective language: {args.adj_lang}")
    if args.intervention_lang is not None and args.intervention_lang not in langs:
        raise KeyError(f"Unsupported intervention language: {args.intervention_lang}")
    if args.projection_lang is not None and args.projection_lang not in langs:
        raise KeyError(f"Unsupported projection language: {args.projection_lang}")

    # When both are explicitly provided, they must agree.
    if (
        args.intervention_lang is not None
        and args.projection_lang is not None
        and args.intervention_lang != args.projection_lang
    ):
        raise ValueError("projection_lang must be the same as intervention_lang for this experiment.")

    prompt_langs = [args.prompt_lang] if args.prompt_lang is not None else langs
    adj_langs = [args.adj_lang] if args.adj_lang is not None else langs
    intervention_langs = [args.intervention_lang] if args.intervention_lang is not None else langs
    amplification_methods = [args.amplification_method] if args.amplification_method is not None else METHODS
    projection_methods = [args.projection_method] if args.projection_method is not None else METHODS

    transcoder_name = hf_transcoder_names.get(model_name, "gemma")
    model = ReplacementModel.from_pretrained(hf_model_names[model_name], transcoder_name, device=device, dtype=torch.bfloat16)

    method_interventions = load_method_interventions(model_name)
    dataset = big_data if args.max_items is None else big_data[: args.max_items]

    output_file = args.output_file or default_output_file(model_name, args.prompt_lang, args.adj_lang)
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    rows: list[dict[str, object]] = []
    for prompt_idx, (adj, ans) in enumerate(tqdm(dataset, desc="Adjectives")):
        for pl in prompt_langs:
            if pl not in ans:
                continue

            prompt = base_strings[pl].format(adj=adj[pl])
            original_logits, original_activations = model.get_activations(prompt)

            for al in adj_langs:
                if al not in ans:
                    continue

                # Match all_langs_intervention_logit_change: base token is chosen from the
                # adjective language answer pool (not the prompt language pool).
                base = get_best_base(original_logits, ans[al], model)

                for il in intervention_langs:
                    # Enforce projection_lang == intervention_lang.
                    projection_lang = il
                    if args.projection_lang is not None and args.projection_lang != il:
                        continue

                    for amplification_method in amplification_methods:
                        amp_interventions = method_interventions[amplification_method][il]
                        amplified_logits, _ = model.feature_intervention(prompt, amp_interventions)

                        for projection_method in projection_methods:
                            proj_interventions = method_interventions[projection_method][projection_lang]
                            isolated_interventions = build_isolated_interventions(
                                model,
                                original_activations,
                                amp_interventions,
                                proj_interventions,
                            )
                            isolated_logits, _ = model.feature_intervention(prompt, isolated_interventions)

                            for measure_lang in langs:
                                if measure_lang not in ans:
                                    continue
                                target = get_best_base(original_logits, ans[measure_lang], model)
                                original_diff, amplified_diff, _ = logit_diff_single(
                                    original_logits,
                                    amplified_logits,
                                    target,
                                    base,
                                    model,
                                )
                                _, isolated_diff, _ = logit_diff_single(
                                    original_logits,
                                    isolated_logits,
                                    target,
                                    base,
                                    model,
                                )
                                rows.append(
                                    {
                                        "model": model_name,
                                        "prompt_idx": prompt_idx,
                                        "prompt_lang": pl,
                                        "adj_lang": al,
                                        "intervention_lang": il,
                                        "projection_lang": projection_lang,
                                        "measure_lang": measure_lang,
                                        "amplification_method": amplification_method,
                                        "projection_method": projection_method,
                                        "original_target_minus_base": original_diff,
                                        "amplified_target_minus_base": amplified_diff,
                                        "isolated_target_minus_base": isolated_diff,
                                        "amplified_minus_original": amplified_diff - original_diff,
                                        "isolated_minus_original": isolated_diff - original_diff,
                                        "original_rank": get_best_rank(original_logits, ans[measure_lang], model),
                                        "amplified_rank": get_best_rank(amplified_logits, ans[measure_lang], model),
                                        "isolated_rank": get_best_rank(isolated_logits, ans[measure_lang], model),
                                    }
                                )

    fieldnames = [
        "model",
        "prompt_idx",
        "prompt_lang",
        "adj_lang",
        "intervention_lang",
        "projection_lang",
        "measure_lang",
        "amplification_method",
        "projection_method",
        "original_target_minus_base",
        "amplified_target_minus_base",
        "isolated_target_minus_base",
        "amplified_minus_original",
        "isolated_minus_original",
        "original_rank",
        "amplified_rank",
        "isolated_rank",
    ]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_file}")


if __name__ == "__main__":
    main()
