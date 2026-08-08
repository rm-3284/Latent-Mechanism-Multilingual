"""
Per-language metrics for antonym and enumeration benchmarks.

Antonyms: mean delta in top-1 logit margin (target_logit - base_logit), where base is
the highest-ranked antonym candidate at baseline (get_best_base).

Enumerations: mean delta in normalized target-sequence logprob (total logprob / #tokens).

Aggregates on-language pairs where prompt_lang = adj/list_lang = intervention context
and measure_lang equals the amplified language (for feature-intervention) or
intervention_lang (for ablation/amplification).

Example:
  python analyze_per_lang_metrics.py --feature-set-file ../data/.../min_default_counts....json
  python analyze_per_lang_metrics.py --defaults-only --intervention-types "distractor ablation"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Any


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def repo_data_dir() -> str:
    from lib.paths import data_dir_str
    return data_dir_str()




def _gpu_imports():
    import torch

    from lib.ablation_amplification_intervention import model_intervention, model_run
    from lib.circuit_tracer_import import ReplacementModel
    from lib.device_setup import device
    from sensitivity.evaluate_selected_features import (
        build_cross_lang_intervention,
        build_enumeration_prompt,
        build_intervention_maps,
        discover_feature_files,
        load_feature_payload,
    )
    from sensitivity.hyperparameter_sensitivity import repo_data_dir as _repo_data_dir
    from lib.intervention import get_best_base
    from pipeline.interventions_to_json import get_logits_and_ranks
    from lib.models import hf_model_names, hf_transcoder_names
    from pipeline.multiple_words_intervention import (
        categories,
        get_logprob_with_intervention,
        get_logprobs_for_candidates,
    )
    from lib.pipeline_data.adjectives import big_data
    from lib.template import base_strings, lang_to_flores_key

    return {
        "torch": torch,
        "model_intervention": model_intervention,
        "model_run": model_run,
        "ReplacementModel": ReplacementModel,
        "device": device,
        "build_cross_lang_intervention": build_cross_lang_intervention,
        "build_enumeration_prompt": build_enumeration_prompt,
        "build_intervention_maps": build_intervention_maps,
        "discover_feature_files": discover_feature_files,
        "load_feature_payload": load_feature_payload,
        "repo_data_dir": _repo_data_dir,
        "get_best_base": get_best_base,
        "get_logits_and_ranks": get_logits_and_ranks,
        "hf_model_names": hf_model_names,
        "hf_transcoder_names": hf_transcoder_names,
        "categories": categories,
        "get_logprob_with_intervention": get_logprob_with_intervention,
        "get_logprobs_for_candidates": get_logprobs_for_candidates,
        "big_data": big_data,
        "base_strings": base_strings,
        "lang_to_flores_key": lang_to_flores_key,
    }


def token_count(model, text: str) -> int:
    token_ids = model.tokenizer.encode(text, add_special_tokens=False)
    return max(len(token_ids), 1)


def antonym_margin_delta(
    model,
    model_name: str,
    prompt: str,
    ans: dict[str, list[str]],
    adj_lang: str,
    measure_lang: str,
    interventions: list[tuple[int, int, int, float]],
    imports: dict,
) -> float | None:
    model_run = imports["model_run"]
    model_intervention = imports["model_intervention"]
    get_best_base = imports["get_best_base"]
    get_logits_and_ranks = imports["get_logits_and_ranks"]
    if measure_lang not in ans:
        return None
    _, baseline_logits = model_run(prompt, model)
    _, intervened_logits = model_intervention(prompt, model, interventions)

    base_word = get_best_base(baseline_logits, ans[adj_lang], model)
    target_word = get_best_base(baseline_logits, ans[measure_lang], model)

    baseline = get_logits_and_ranks(baseline_logits, ans, model.tokenizer, model_name)
    intervened = get_logits_and_ranks(intervened_logits, ans, model.tokenizer, model_name)

    baseline_margin = (
        baseline[measure_lang][target_word][0] - baseline[adj_lang][base_word][0]
    )
    intervened_margin = (
        intervened[measure_lang][target_word][0] - intervened[adj_lang][base_word][0]
    )
    return intervened_margin - baseline_margin


def enumeration_norm_logprob_delta(
    model,
    prompt: str,
    ans: dict[str, str],
    measure_lang: str,
    interventions: list[tuple[int, int, int, float]],
    imports: dict,
) -> float | None:
    get_logprobs_for_candidates = imports["get_logprobs_for_candidates"]
    get_logprob_with_intervention = imports["get_logprob_with_intervention"]
    if measure_lang not in ans:
        return None
    candidate = ans[measure_lang]
    baseline = get_logprobs_for_candidates(prompt, ans, model)[measure_lang][candidate]
    intervened = get_logprob_with_intervention(prompt, candidate, model, interventions)
    norm = token_count(model, candidate)
    return (intervened / norm) - (baseline / norm)


def interventions_for_type(
    intervention_type: str,
    ablations: dict[str, list[tuple[int, int, int, float]]],
    amplifications: dict[str, list[tuple[int, int, int, float]]],
    prompt_lang: str,
    target_lang: str,
    imports: dict,
) -> list[tuple[int, int, int, float]]:
    build_cross_lang_intervention = imports["build_cross_lang_intervention"]
    if intervention_type == "distractor ablation":
        return ablations[target_lang]
    if intervention_type == "amplification":
        return amplifications[target_lang]
    if intervention_type == "feature-intervention":
        return build_cross_lang_intervention(
            ablations, amplifications, prompt_lang, target_lang
        )
    raise ValueError(f"Unknown intervention type: {intervention_type}")


def evaluate_feature_set(
    model,
    model_name: str,
    features_by_lang: dict[str, list[str]],
    langs: list[str],
    intervention_types: list[str],
    max_items: int | None,
    imports: dict,
) -> dict[str, Any]:
    build_intervention_maps = imports["build_intervention_maps"]
    base_strings = imports["base_strings"]
    big_data = imports["big_data"]
    build_enumeration_prompt = imports["build_enumeration_prompt"]
    categories = imports["categories"]

    amplification_values_directory = os.path.join(
        repo_data_dir(), "amplification_values", model_name
    )
    ablations, amplifications = build_intervention_maps(
        features_by_lang, amplification_values_directory, langs
    )

    antonym_rows: list[dict[str, Any]] = []
    enum_rows: list[dict[str, Any]] = []
    dataset = big_data if max_items is None else big_data[:max_items]

    for pl in langs:
        base = base_strings[pl]
        for al in langs:
            for _sample_idx, (adj, ans) in enumerate(dataset):
                if al not in adj:
                    continue
                prompt = base.format(adj=adj[al])
                for intervention_type in intervention_types:
                    if intervention_type == "feature-intervention" and pl == al:
                        continue
                    for measure_lang in langs:
                        interventions = interventions_for_type(
                            intervention_type,
                            ablations,
                            amplifications,
                            pl,
                            al if intervention_type == "feature-intervention" else measure_lang,
                            imports,
                        )
                        margin_delta = antonym_margin_delta(
                            model,
                            model_name,
                            prompt,
                            ans,
                            al,
                            measure_lang,
                            interventions,
                            imports,
                        )
                        if margin_delta is not None:
                            antonym_rows.append(
                                {
                                    "prompt_lang": pl,
                                    "adj_lang": al,
                                    "measure_lang": measure_lang,
                                    "intervention_type": intervention_type,
                                    "delta_top1_margin": margin_delta,
                                    "on_language_pair": pl == al == measure_lang,
                                }
                            )

    for pl in langs:
        for ll in langs:
            for category_key in categories:
                prompt, ans = build_enumeration_prompt(pl, ll, category_key)
                for intervention_type in intervention_types:
                    if intervention_type == "feature-intervention" and pl == ll:
                        continue
                    for measure_lang in langs:
                        interventions = interventions_for_type(
                            intervention_type,
                            ablations,
                            amplifications,
                            pl,
                            ll if intervention_type == "feature-intervention" else measure_lang,
                            imports,
                        )
                        lp_delta = enumeration_norm_logprob_delta(
                            model, prompt, ans, measure_lang, interventions, imports
                        )
                        if lp_delta is not None:
                            enum_rows.append(
                                {
                                    "prompt_lang": pl,
                                    "list_lang": ll,
                                    "measure_lang": measure_lang,
                                    "category": category_key,
                                    "intervention_type": intervention_type,
                                    "delta_norm_logprob": lp_delta,
                                    "on_language_pair": pl == ll == measure_lang,
                                }
                            )

    return {"antonyms": antonym_rows, "enumerations": enum_rows}


def summarize_per_lang(
    rows: list[dict[str, Any]],
    value_key: str,
    on_language_only: bool,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if on_language_only and not row.get("on_language_pair"):
            continue
        grouped[row["measure_lang"]].append(row[value_key])
    return {lang: mean(vals) for lang, vals in sorted(grouped.items())}


def load_original_antonym_margins(
    model_name: str,
    intervention_type: str,
    langs: list[str],
) -> dict[str, dict[str, float]]:
    """From all_langs CSV: diagonal on-lang delta_target_minus_base."""
    base = os.path.join(repo_data_dir(), "all_langs_intervention_logit_change", model_name)
    method_map = {"desc": "AnnSel", "val": "ValSel", "freq": "FreqSel"}
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)

    for pl in langs:
        csv_path = os.path.join(
            base,
            pl,
            pl,
            "all_langs_intervention_logit_change__intervention_all__measure_all.csv",
        )
        if not os.path.exists(csv_path):
            continue
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["prompt_lang"] != pl or row["adj_lang"] != pl:
                    continue
                if row["intervention_lang"] != row["measure_lang"]:
                    continue
                ab = row["ablation_method"]
                amp = row["amplification_method"]
                ml = row["measure_lang"]
                if intervention_type == "distractor ablation":
                    if ab != amp:
                        continue
                elif intervention_type == "amplification":
                    if ab != amp:
                        continue
                elif intervention_type == "feature-intervention":
                    if ab == amp:
                        continue
                else:
                    continue
                method = method_map[ab]
                grouped[(method, ml)].append(float(row["delta_target_minus_base"]))

    return {
        method: {lang: mean(grouped[(method, lang)]) for lang in langs if grouped[(method, lang)]}
        for method in ["AnnSel", "ValSel", "FreqSel"]
    }


def load_original_enumeration_norm_logprob(
    model_name: str,
    intervention_type: str,
    langs: list[str],
) -> dict[str, dict[str, float]]:
    """From interventions_multiple_words normalized JSON."""
    base = os.path.join(repo_data_dir(), "interventions_multiple_words", model_name)
    method_files = {
        "AnnSel": "interventions_and_results_description_normalized.json",
        "ValSel": "interventions_and_results_value_normalized.json",
        "FreqSel": "interventions_and_results_frequency_normalized.json",
    }
    exp_key = intervention_type
    if intervention_type == "amplification":
        exp_key = "non-distractor amplification"

    results: dict[str, dict[str, float]] = {}
    for method, fname in method_files.items():
        per_lang: dict[str, list[float]] = defaultdict(list)
        for pl in langs:
            for ll in langs:
                path = os.path.join(base, pl, ll, fname)
                if not os.path.exists(path):
                    continue
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                if exp_key not in data or "original" not in data:
                    continue
                for prompt in data["original"]:
                    orig = data["original"][prompt]["logprobs"]
                    post = data[exp_key][prompt]
                    if intervention_type == "feature-intervention":
                        # ablation_lang -> amplification_lang -> measure_lang
                        if pl not in post or ll not in post[pl]:
                            continue
                        post_block = post[pl][ll]
                    else:
                        if ll not in post:
                            continue
                        post_block = post[ll]
                    for ml in langs:
                        if ml not in orig or ml not in post_block:
                            continue
                        if pl != ll or ml != ll:
                            continue
                        cand = next(iter(orig[ml].keys()))
                        if cand not in post_block[ml]:
                            continue
                        delta = post_block[ml][cand] - orig[ml][cand]
                        per_lang[ml].append(delta)
        results[method] = {lang: mean(vals) for lang, vals in sorted(per_lang.items())}
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-language benchmark metrics.")
    parser.add_argument("--model", "-m", default="gemma-2-2b")
    parser.add_argument("--features-dir", default=None)
    parser.add_argument("--features-file", default=None)
    parser.add_argument("--defaults-only", action="store_true")
    parser.add_argument(
        "--budget-label",
        default=None,
        help="Label for output (e.g. matched_budget, original_default)",
    )
    parser.add_argument(
        "--intervention-types",
        nargs="+",
        default=["distractor ablation", "amplification", "feature-intervention"],
    )
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument(
        "--use-existing-original-data",
        action="store_true",
        help="Load original defaults from precomputed intervention artifacts instead of re-running.",
    )
    parser.add_argument("--output-csv", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    imports = _gpu_imports() if not args.use_existing_original_data else None
    langs = list(
        imports["lang_to_flores_key"].keys()
        if imports
        else ["en", "fr", "de", "es", "zh", "ja", "ko"]
    )
    rows_out: list[dict[str, Any]] = []

    feature_files: list[tuple[str, str, str]] = []
    if args.use_existing_original_data:
        feature_files = [("original_default", "defaults", "")]
    else:
        assert imports is not None
        discover_feature_files = imports["discover_feature_files"]
        load_feature_payload = imports["load_feature_payload"]
        features_dir = args.features_dir or os.path.join(
            repo_data_dir(), "additional_experiments", args.model, "selected_features"
        )
        for path in discover_feature_files(features_dir, args.features_file, args.defaults_only):
            payload = load_feature_payload(path)
            label = args.budget_label or (
                "matched_budget"
                if "min_default_counts" in path
                else "original_default"
            )
            feature_files.append((label, payload.get("method", "?"), path))

    for budget_label, method_hint, feature_path in feature_files:
        for intervention_type in args.intervention_types:
            if args.use_existing_original_data or (
                budget_label == "original_default" and feature_path == ""
            ):
                ant = load_original_antonym_margins(args.model, intervention_type, langs)
                enum = load_original_enumeration_norm_logprob(
                    args.model, intervention_type, langs
                )
                for method in ["AnnSel", "ValSel", "FreqSel"]:
                    for lang in langs:
                        rows_out.append(
                            {
                                "budget": budget_label,
                                "method": method,
                                "intervention_type": intervention_type,
                                "language": lang,
                                "antonym_delta_top1_margin": ant.get(method, {}).get(lang),
                                "enumeration_delta_norm_logprob": enum.get(method, {}).get(lang),
                            }
                        )
                continue

            assert imports is not None
            load_feature_payload = imports["load_feature_payload"]
            ReplacementModel = imports["ReplacementModel"]
            hf_model_names = imports["hf_model_names"]
            hf_transcoder_names = imports["hf_transcoder_names"]
            device = imports["device"]
            torch = imports["torch"]

            payload = load_feature_payload(feature_path)
            method = payload.get("method", method_hint)

            print(f"Evaluating {method} {budget_label} {intervention_type} ...")
            model = ReplacementModel.from_pretrained(
                hf_model_names[args.model],
                hf_transcoder_names[args.model],
                device=device,
                dtype=torch.bfloat16,
            )
            result = evaluate_feature_set(
                model,
                args.model,
                payload["selected_features"],
                langs,
                [intervention_type],
                args.max_items,
                imports,
            )
            ant_summary = summarize_per_lang(
                [r for r in result["antonyms"] if r["intervention_type"] == intervention_type],
                "delta_top1_margin",
                on_language_only=True,
            )
            enum_summary = summarize_per_lang(
                [r for r in result["enumerations"] if r["intervention_type"] == intervention_type],
                "delta_norm_logprob",
                on_language_only=True,
            )
            for lang in langs:
                rows_out.append(
                    {
                        "budget": budget_label,
                        "method": method,
                        "intervention_type": intervention_type,
                        "language": lang,
                        "antonym_delta_top1_margin": ant_summary.get(lang),
                        "enumeration_delta_norm_logprob": enum_summary.get(lang),
                    }
                )
            del model
            torch.cuda.empty_cache()

    output_csv = args.output_csv or os.path.join(
        repo_data_dir(),
        "additional_experiments",
        args.model,
        "per_lang_metrics_summary.csv",
    )
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    fieldnames = [
        "budget",
        "method",
        "intervention_type",
        "language",
        "antonym_delta_top1_margin",
        "enumeration_delta_norm_logprob",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"Wrote {len(rows_out)} rows to {output_csv}")


if __name__ == "__main__":
    main()
