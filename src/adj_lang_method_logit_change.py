import argparse
import csv
import os
from typing import Any

import torch

from ablation_amplification_intervention import (
    activation_dict,
    description_based_features,
    freq_based_features,
    mean_value_based_features,
    model_intervention,
    model_run,
)
from circuit_tracer_import import ReplacementModel
from device_setup import device
from intervention import ablation, amplification, get_best_base, get_best_rank
from models import hf_model_names, hf_transcoder_names
from pipeline_data.adjectives import big_data
from template import base_strings, lang_to_flores_key


AVAILABLE_METHODS = ["desc", "val", "freq"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Iterate through prompt_lang, adj_lang, and method (desc/val/freq); "
            "apply zero ablation and target amplification on adj_lang features; "
            "record target-logit changes."
        )
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="gemma-2-2b",
        choices=hf_model_names.keys(),
        help="Model to run interventions with.",
    )
    parser.add_argument(
        "--prompt-lang",
        type=str,
        default=None,
        help="Optional prompt language filter (e.g., en).",
    )
    parser.add_argument(
        "--adj-lang",
        type=str,
        default=None,
        help="Optional adjective language filter (e.g., ja).",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional cap on number of adjective pairs from big_data.",
    )
    parser.add_argument(
        "--output-file",
        "-o",
        type=str,
        default=None,
        help=(
            "Optional output CSV path. Defaults to "
            "data/adj_lang_method_logit_change/<model>/prompt_<prompt_lang-or-all>__adj_<adj_lang-or-all>/adj_lang_method_logit_change.csv"
        ),
    )
    parser.add_argument(
        "--ablation-methods",
        nargs="+",
        default=AVAILABLE_METHODS,
        choices=AVAILABLE_METHODS,
        help=(
            "Methods to use for zero ablation. "
            "Default: desc val freq"
        ),
    )
    parser.add_argument(
        "--amplification-methods",
        nargs="+",
        default=AVAILABLE_METHODS,
        choices=AVAILABLE_METHODS,
        help=(
            "Methods to use for target amplification. "
            "Default: desc val freq"
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        choices=AVAILABLE_METHODS,
        help=(
            "Backward-compatible alias: when provided, uses the same methods for both "
            "--ablation-methods and --amplification-methods."
        ),
    )
    return parser.parse_args()


def token_id_for_word(model: ReplacementModel, word: str) -> int:
    ids = model.tokenizer.encode(word)
    if not ids:
        raise ValueError(f"Tokenizer produced empty tokenization for word: {word}")

    special_ids = set(getattr(model.tokenizer, "all_special_ids", []))
    for token_id in ids:
        if token_id not in special_ids:
            return token_id

    plain_ids = model.tokenizer.encode(word, add_special_tokens=False)
    if not plain_ids:
        raise ValueError(f"Tokenizer produced only special tokens for word: {word}")
    return plain_ids[0]


def token_logit(logits: torch.Tensor, token_id: int) -> float:
    last_logits = logits.squeeze(0)[-1]
    return float(last_logits[token_id].item())


def validate_lang_filter(value: str | None, langs: list[str], arg_name: str) -> None:
    if value is not None and value not in langs:
        raise ValueError(f"{arg_name}={value} is invalid. Choose from: {langs}")


def build_method_interventions(
    methods: list[str],
    langs: list[str],
    flores_directory: str,
    multilingual_features_directory: str,
    lang_specific_directory: str,
    amplification_values_directory: str,
) -> dict[str, dict[str, list[tuple[int, int, int, torch.Tensor]]]]:
    desc_features = description_based_features(flores_directory, langs, 0.1)
    val_features = mean_value_based_features(multilingual_features_directory, langs, 50)
    freq_features = freq_based_features(lang_specific_directory, langs)

    desc_supernodes = activation_dict(desc_features, amplification_values_directory, langs)
    val_supernodes = activation_dict(val_features, amplification_values_directory, langs)
    freq_supernodes = activation_dict(freq_features, amplification_values_directory, langs)

    all_method_supernodes = {
        "desc": desc_supernodes,
        "val": val_supernodes,
        "freq": freq_supernodes,
    }
    method_supernodes = {method: all_method_supernodes[method] for method in methods}

    method_interventions: dict[str, dict[str, list[tuple[int, int, int, torch.Tensor]]]] = {}
    for method, supernodes in method_supernodes.items():
        method_interventions[method] = {}
        for lang in langs:
            method_interventions[method][f"{lang}:ablation"] = ablation(supernodes, lang, alpha=0)
            method_interventions[method][f"{lang}:amplification"] = amplification(supernodes, lang)

    return method_interventions


def combine_interventions_with_ablation_priority(
    ablation_interventions: list[tuple[int, int, int, torch.Tensor]],
    amplification_interventions: list[tuple[int, int, int, torch.Tensor]],
) -> list[tuple[int, int, int, torch.Tensor]]:
    # If a feature is present in both lists, keep zero-ablation and drop amplification.
    ablation_feature_keys = {(layer, pos, feature_idx) for layer, pos, feature_idx, _ in ablation_interventions}
    combined = list(ablation_interventions)
    for layer, pos, feature_idx, value in amplification_interventions:
        if (layer, pos, feature_idx) in ablation_feature_keys:
            continue
        combined.append((layer, pos, feature_idx, value))
    return combined


def run_experiment(args: argparse.Namespace) -> str:
    model_name = args.model
    transcoder_name = hf_transcoder_names.get(model_name)
    model = ReplacementModel.from_pretrained(
        hf_model_names[model_name],
        transcoder_name,
        device=device,
        dtype=torch.bfloat16,
    )

    current_directory = os.path.abspath(os.path.dirname(__file__))
    data_directory = os.path.join(os.path.dirname(current_directory), "data")

    flores_directory = os.path.join(data_directory, "flores_features", model_name)
    lang_specific_directory = os.path.join(data_directory, "language_specific_features", model_name)
    multilingual_features_directory = os.path.join(data_directory, "multilingual_llm_features", model_name)
    amplification_values_directory = os.path.join(data_directory, "amplification_values", model_name)

    langs = sorted(lang_to_flores_key.keys())
    validate_lang_filter(args.prompt_lang, langs, "--prompt-lang")
    validate_lang_filter(args.adj_lang, langs, "--adj-lang")

    ablation_methods = args.ablation_methods
    amplification_methods = args.amplification_methods
    if args.methods is not None:
        ablation_methods = args.methods
        amplification_methods = args.methods

    selected_methods = sorted(set(ablation_methods + amplification_methods))

    method_interventions = build_method_interventions(
        methods=selected_methods,
        langs=langs,
        flores_directory=flores_directory,
        multilingual_features_directory=multilingual_features_directory,
        lang_specific_directory=lang_specific_directory,
        amplification_values_directory=amplification_values_directory,
    )

    output_file = args.output_file
    if output_file is None:
        prompt_lang_label = args.prompt_lang if args.prompt_lang is not None else "all"
        adj_lang_label = args.adj_lang if args.adj_lang is not None else "all"
        lang_pair_dir = f"prompt_{prompt_lang_label}__adj_{adj_lang_label}"

        output_dir = os.path.join(
            data_directory,
            "adj_lang_method_logit_change",
            model_name,
            lang_pair_dir,
        )
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "adj_lang_method_logit_change.csv")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    rows: list[dict[str, Any]] = []
    dataset = big_data if args.max_items is None else big_data[: args.max_items]

    for prompt_lang in langs:
        if args.prompt_lang is not None and prompt_lang != args.prompt_lang:
            continue

        prompt_template = base_strings[prompt_lang]

        for adj_lang in langs:
            if args.adj_lang is not None and adj_lang != args.adj_lang:
                continue

            for sample_idx, (adj, ans) in enumerate(dataset):
                if adj_lang not in adj or adj_lang not in ans:
                    continue

                adjective = adj[adj_lang]
                prompt = prompt_template.format(adj=adjective)

                _, baseline_logits = model_run(prompt, model)
                baseline_best_target = get_best_base(baseline_logits, ans[adj_lang], model)
                baseline_target_id = token_id_for_word(model, baseline_best_target)
                baseline_target_logit = token_logit(baseline_logits, baseline_target_id)
                baseline_adj_rank = get_best_rank(baseline_logits, ans[adj_lang], model)

                for amplification_method in amplification_methods:
                    amplification_key = f"{adj_lang}:amplification"
                    amplification_interventions = method_interventions[amplification_method][
                        amplification_key
                    ]

                    # Apples-to-apples normal amplification (no ablation).
                    _, normal_amp_logits = model_intervention(
                        prompt,
                        model,
                        amplification_interventions,
                    )
                    normal_amp_target_logit = token_logit(normal_amp_logits, baseline_target_id)
                    normal_amp_row = {
                        "model": model_name,
                        "sample_idx": sample_idx,
                        "intervention_type": "normal_amplification",
                        "ablation_method": "none",
                        "amplification_method": amplification_method,
                        "prompt_lang": prompt_lang,
                        "adj_lang": adj_lang,
                        "adjective": adjective,
                        "prompt": prompt,
                        "tracked_target": baseline_best_target,
                        "baseline_adj_rank": baseline_adj_rank,
                        "combined_adj_rank": get_best_rank(normal_amp_logits, ans[adj_lang], model),
                        "baseline_target_logit": baseline_target_logit,
                        "combined_target_logit": normal_amp_target_logit,
                        "delta_combined_logit": normal_amp_target_logit - baseline_target_logit,
                    }
                    rows.append(normal_amp_row)

                    for ablation_method in ablation_methods:
                        ablation_key = f"{adj_lang}:ablation"
                        ablation_interventions = method_interventions[ablation_method][ablation_key]

                        combined_interventions = combine_interventions_with_ablation_priority(
                            ablation_interventions,
                            amplification_interventions,
                        )

                        _, combined_logits = model_intervention(
                            prompt,
                            model,
                            combined_interventions,
                        )

                        combined_target_logit = token_logit(combined_logits, baseline_target_id)

                        row = {
                            "model": model_name,
                            "sample_idx": sample_idx,
                            "intervention_type": "combined",
                            "ablation_method": ablation_method,
                            "amplification_method": amplification_method,
                            "prompt_lang": prompt_lang,
                            "adj_lang": adj_lang,
                            "adjective": adjective,
                            "prompt": prompt,
                            "tracked_target": baseline_best_target,
                            "baseline_adj_rank": baseline_adj_rank,
                            "combined_adj_rank": get_best_rank(combined_logits, ans[adj_lang], model),
                            "baseline_target_logit": baseline_target_logit,
                            "combined_target_logit": combined_target_logit,
                            "delta_combined_logit": combined_target_logit - baseline_target_logit,
                        }
                        rows.append(row)

    fieldnames = [
        "model",
        "sample_idx",
        "intervention_type",
        "ablation_method",
        "amplification_method",
        "prompt_lang",
        "adj_lang",
        "adjective",
        "prompt",
        "tracked_target",
        "baseline_adj_rank",
        "combined_adj_rank",
        "baseline_target_logit",
        "combined_target_logit",
        "delta_combined_logit",
    ]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_file}")
    return output_file


if __name__ == "__main__":
    run_experiment(parse_args())
