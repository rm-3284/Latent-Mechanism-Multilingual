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
from intervention import ablation, amplification, get_best_rank
from models import hf_model_names, hf_transcoder_names
from pipeline_data.adjectives import big_data
from template import base_strings, lang_to_flores_key


AVAILABLE_METHODS = ["desc", "val", "freq"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run intervention-style experiments over all intervention languages. "
            "For each prompt/adjective sample, apply zero-ablation + target-amplification "
            "features from intervention_lang and measure per-target-word logit/rank changes "
            "for all measure languages."
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
        "--intervention-lang",
        type=str,
        default=None,
        help="Optional intervention language filter (e.g., de).",
    )
    parser.add_argument(
        "--measure-lang",
        type=str,
        default=None,
        help="Optional measure language filter (e.g., zh).",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional cap on number of adjective pairs from big_data.",
    )
    parser.add_argument(
        "--ablation-methods",
        nargs="+",
        default=AVAILABLE_METHODS,
        choices=AVAILABLE_METHODS,
        help="Methods to use for zero ablation. Default: desc val freq",
    )
    parser.add_argument(
        "--amplification-methods",
        nargs="+",
        default=AVAILABLE_METHODS,
        choices=AVAILABLE_METHODS,
        help="Methods to use for target amplification. Default: desc val freq",
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
    parser.add_argument(
        "--output-file",
        "-o",
        type=str,
        default=None,
        help=(
            "Optional output CSV path. Defaults to "
            "data/all_langs_intervention_logit_change/<model>/"
            "prompt_<prompt_lang-or-all>__adj_<adj_lang-or-all>__"
            "intervention_<intervention_lang-or-all>__measure_<measure_lang-or-all>/"
            "all_langs_intervention_logit_change.csv"
        ),
    )
    return parser.parse_args()


def validate_lang_filter(value: str | None, langs: list[str], arg_name: str) -> None:
    if value is not None and value not in langs:
        raise ValueError(f"{arg_name}={value} is invalid. Choose from: {langs}")


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


def token_rank(logits: torch.Tensor, token_id: int) -> int:
    last_logits = logits.squeeze(0)[-1]
    _, indices = torch.sort(last_logits, dim=-1, descending=True)
    mask = (indices == token_id)
    rank = torch.argmax(mask.int(), dim=-1)
    return int(rank.item() if isinstance(rank, torch.Tensor) else rank)


def get_best_adj_target(ans: dict[str, list[str]], adj_lang: str, logits: torch.Tensor, model: ReplacementModel) -> str:
    best_target = None
    best_rank = None
    for candidate in ans[adj_lang]:
        token_id = token_id_for_word(model, candidate)
        rank = token_rank(logits, token_id)
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_target = candidate
    if best_target is None:
        raise ValueError(f"No targets found for adj_lang={adj_lang}")
    return best_target


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
    validate_lang_filter(args.intervention_lang, langs, "--intervention-lang")
    validate_lang_filter(args.measure_lang, langs, "--measure-lang")

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

    intervention_lang_label = args.intervention_lang if args.intervention_lang is not None else "all"
    measure_lang_label = args.measure_lang if args.measure_lang is not None else "all"

    output_file = args.output_file
    if output_file is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    rows: list[dict[str, Any]] = []
    rows_by_lang_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
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
                baseline_adj_rank = get_best_rank(baseline_logits, ans[adj_lang], model)

                base_word = get_best_adj_target(ans, adj_lang, baseline_logits, model)
                base_token_id = token_id_for_word(model, base_word)
                baseline_base_logit = token_logit(baseline_logits, base_token_id)

                for intervention_lang in langs:
                    if args.intervention_lang is not None and intervention_lang != args.intervention_lang:
                        continue

                    for ablation_method in ablation_methods:
                        ablation_key = f"{intervention_lang}:ablation"
                        ablation_interventions = method_interventions[ablation_method][ablation_key]

                        for amplification_method in amplification_methods:
                            amplification_key = f"{intervention_lang}:amplification"
                            amplification_interventions = method_interventions[amplification_method][
                                amplification_key
                            ]

                            combined_interventions = combine_interventions_with_ablation_priority(
                                ablation_interventions,
                                amplification_interventions,
                            )

                            _, intervened_logits = model_intervention(
                                prompt,
                                model,
                                combined_interventions,
                            )

                            intervened_adj_rank = get_best_rank(intervened_logits, ans[adj_lang], model)
                            intervened_base_logit = token_logit(intervened_logits, base_token_id)

                            for measure_lang in langs:
                                if args.measure_lang is not None and measure_lang != args.measure_lang:
                                    continue
                                if measure_lang not in ans:
                                    continue

                                for target_word in ans[measure_lang]:
                                    target_token_id = token_id_for_word(model, target_word)

                                    baseline_target_logit = token_logit(baseline_logits, target_token_id)
                                    intervened_target_logit = token_logit(intervened_logits, target_token_id)

                                    baseline_target_rank = token_rank(baseline_logits, target_token_id)
                                    intervened_target_rank = token_rank(intervened_logits, target_token_id)

                                    baseline_target_minus_base = baseline_target_logit - baseline_base_logit
                                    intervened_target_minus_base = intervened_target_logit - intervened_base_logit

                                    row = {
                                        "model": model_name,
                                        "sample_idx": sample_idx,
                                        "ablation_method": ablation_method,
                                        "amplification_method": amplification_method,
                                        "prompt_lang": prompt_lang,
                                        "adj_lang": adj_lang,
                                        "intervention_lang": intervention_lang,
                                        "measure_lang": measure_lang,
                                        "target_word": target_word,
                                        "adjective": adjective,
                                        "prompt": prompt,
                                        "base_word": base_word,
                                        "baseline_adj_rank": baseline_adj_rank,
                                        "intervened_adj_rank": intervened_adj_rank,
                                        "baseline_target_rank": baseline_target_rank,
                                        "intervened_target_rank": intervened_target_rank,
                                        "baseline_target_logit": baseline_target_logit,
                                        "intervened_target_logit": intervened_target_logit,
                                        "delta_target_logit": intervened_target_logit - baseline_target_logit,
                                        "baseline_target_minus_base": baseline_target_minus_base,
                                        "intervened_target_minus_base": intervened_target_minus_base,
                                        "delta_target_minus_base": intervened_target_minus_base - baseline_target_minus_base,
                                    }
                                    rows.append(row)
                                    pair_key = (prompt_lang, adj_lang)
                                    if pair_key not in rows_by_lang_pair:
                                        rows_by_lang_pair[pair_key] = []
                                    rows_by_lang_pair[pair_key].append(row)

    fieldnames = [
        "model",
        "sample_idx",
        "ablation_method",
        "amplification_method",
        "prompt_lang",
        "adj_lang",
        "intervention_lang",
        "measure_lang",
        "target_word",
        "adjective",
        "prompt",
        "base_word",
        "baseline_adj_rank",
        "intervened_adj_rank",
        "baseline_target_rank",
        "intervened_target_rank",
        "baseline_target_logit",
        "intervened_target_logit",
        "delta_target_logit",
        "baseline_target_minus_base",
        "intervened_target_minus_base",
        "delta_target_minus_base",
    ]

    if output_file is not None:
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"Wrote {len(rows)} rows to {output_file}")
        return output_file

    base_output_dir = os.path.join(
        data_directory,
        "all_langs_intervention_logit_change",
        model_name,
    )
    written_files: list[str] = []
    for (prompt_lang, adj_lang), pair_rows in sorted(rows_by_lang_pair.items()):
        pair_dir = os.path.join(base_output_dir, prompt_lang, adj_lang)
        os.makedirs(pair_dir, exist_ok=True)

        pair_file = os.path.join(
            pair_dir,
            (
                "all_langs_intervention_logit_change"
                f"__intervention_{intervention_lang_label}"
                f"__measure_{measure_lang_label}.csv"
            ),
        )

        with open(pair_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(pair_rows)
        written_files.append(pair_file)

    print(f"Wrote {len(rows)} total rows across {len(written_files)} files")
    for path in written_files:
        print(f"- {path}")

    return base_output_dir


if __name__ == "__main__":
    run_experiment(parse_args())
