"""
Generate budget-matched feature sets for AnnSel, ValSel, and FreqSel.

Default mode (--budget-mode min_default_counts): for each language L, all methods
receive K_L latents, where K_L is the minimum default selection size across
methods for that language (e.g. ko=7 from AnnSel, de=50 from ValSel).

Features are ranked by each method's native score and truncated to the top K_L
per language.

Example:
  python budget_matched_features.py --report-only
  python budget_matched_features.py
  python budget_matched_features.py --budget-mode per_lang_min
  python budget_matched_features.py --budget-mode uniform --uniform-budgets 7,10
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any

from hyperparameter_sensitivity import (
    DEFAULT_ANNSEL_THRESHOLD,
    DEFAULT_FREQSEL_CROSS_LINGUAL_THRES,
    DEFAULT_FREQSEL_EXAMPLE_THRES,
    DEFAULT_FREQSEL_TOKEN_THRES,
    DEFAULT_VALSEL_TOPK,
    choose_language_specific_features,
    description_based_features,
    freqsel_from_params,
    load_sparse_stats,
    mean_value_based_features,
    repo_data_dir,
    save_selected_features,
    summarize_feature_set,
    try_load_cached_freqsel,
)
from models import hf_model_names
from template import lang_to_flores_key

DEFAULT_FREQSEL_CANDIDATE_PARAMS = {
    "example_thres": 0.5,
    "cross_lingual_thres": 0.5,
    "token_thres": 0.01,
}
PER_LANG_MIN_POOL_SWEEP = "per_lang_min_pool"
MIN_DEFAULT_COUNTS_SWEEP = "min_default_counts"


def annsel_ranked_candidates(
    model_name: str, langs: list[str]
) -> dict[str, list[tuple[str, float]]]:
    """Rank AnnSel candidates by annotation match count (descending)."""
    flores_dir = os.path.join(repo_data_dir(), "flores_features", model_name)
    ranked: dict[str, list[tuple[str, float]]] = {}
    for lang in langs:
        path = os.path.join(flores_dir, f"{lang}_features.json")
        with open(path, "r") as f:
            features_freq: dict[str, float] = json.load(f)
        ranked[lang] = sorted(
            features_freq.items(), key=lambda item: item[1], reverse=True
        )
    return ranked


def valsel_ranked_candidates(
    model_name: str, langs: list[str]
) -> dict[str, list[tuple[str, float]]]:
    """Rank ValSel candidates by per-language v-score (descending)."""
    val_dir = os.path.join(repo_data_dir(), "multilingual_llm_features", model_name)
    ranked: dict[str, list[tuple[str, float]]] = {}
    for lang in langs:
        with open(os.path.join(val_dir, f"{lang}.json"), "r") as f:
            feature_val: dict[str, float] = json.load(f)
        ranked[lang] = sorted(
            feature_val.items(), key=lambda item: item[1], reverse=True
        )
    return ranked


def freqsel_ranked_candidates(
    langs: list[str],
    activation_per_pos: dict[str, dict[str, float]],
    active_example_ratio: dict[str, dict[str, float]],
    example_thres: float,
    cross_lingual_thres: float,
    token_thres: float,
) -> dict[str, list[tuple[str, float]]]:
    """Rank FreqSel candidates by activation_per_pos within assigned language."""
    selected = choose_language_specific_features(
        langs,
        activation_per_pos,
        active_example_ratio,
        cross_lingual_thres=cross_lingual_thres,
        example_thres=example_thres,
        token_thres=token_thres,
    )
    ranked: dict[str, list[tuple[str, float]]] = {lang: [] for lang in langs}
    for lang, features in selected.items():
        scores = []
        for layer, feature_idx in features:
            feature_key = f"{layer}.{feature_idx}"
            score = activation_per_pos[lang].get(feature_key, 0.0)
            scores.append((feature_key, score))
        ranked[lang] = sorted(scores, key=lambda item: item[1], reverse=True)
    return ranked


def truncate_ranked_candidates_uniform(
    ranked: dict[str, list[tuple[str, float]]],
    budget_k: int,
) -> tuple[dict[str, list[str]], dict[str, int], dict[str, int]]:
    """Take top-K features per language from ranked candidate lists."""
    selected: dict[str, list[str]] = {}
    requested = {lang: budget_k for lang in ranked}
    actual: dict[str, int] = {}
    for lang, candidates in ranked.items():
        top = candidates[:budget_k]
        selected[lang] = [feature for feature, _ in top]
        actual[lang] = len(top)
    return selected, requested, actual


def truncate_ranked_candidates_per_lang(
    ranked: dict[str, list[tuple[str, float]]],
    per_lang_budgets: dict[str, int],
) -> tuple[dict[str, list[str]], dict[str, int], dict[str, int]]:
    """Take top-K_L features per language using language-specific budgets."""
    selected: dict[str, list[str]] = {}
    actual: dict[str, int] = {}
    for lang, candidates in ranked.items():
        budget_k = per_lang_budgets[lang]
        top = candidates[:budget_k]
        selected[lang] = [feature for feature, _ in top]
        actual[lang] = len(top)
    return selected, per_lang_budgets, actual


def pool_size_summary(
    ranked: dict[str, list[tuple[str, float]]],
) -> dict[str, Any]:
    per_lang = {lang: len(candidates) for lang, candidates in ranked.items()}
    return {
        "per_lang_pool_size": per_lang,
        "total_pool_size": sum(per_lang.values()),
        "min_per_lang_pool_size": min(per_lang.values()) if per_lang else 0,
    }


def per_lang_min_across_methods(
    per_method_per_lang: dict[str, dict[str, int]],
    langs: list[str],
) -> dict[str, int]:
    """K_L = min value across methods for each language L."""
    return {
        lang: min(per_method_per_lang[method][lang] for method in per_method_per_lang)
        for lang in langs
    }


def load_default_selection_counts(
    model_name: str,
    langs: list[str],
) -> dict[str, dict[str, int]]:
    """Default selection sizes per language for AnnSel, ValSel, and FreqSel."""
    flores_dir = os.path.join(repo_data_dir(), "flores_features", model_name)
    val_dir = os.path.join(repo_data_dir(), "multilingual_llm_features", model_name)
    activation_per_pos, active_example_ratio = load_sparse_stats(model_name, langs)

    annsel = description_based_features(
        flores_dir, langs, threshold=DEFAULT_ANNSEL_THRESHOLD
    )
    valsel = mean_value_based_features(val_dir, langs, topk=DEFAULT_VALSEL_TOPK)
    freqsel = try_load_cached_freqsel(
        model_name,
        example_thres=DEFAULT_FREQSEL_EXAMPLE_THRES,
        cross_lingual_thres=DEFAULT_FREQSEL_CROSS_LINGUAL_THRES,
        token_thres=DEFAULT_FREQSEL_TOKEN_THRES,
        langs=langs,
    )
    if freqsel is None:
        freqsel = freqsel_from_params(
            langs,
            activation_per_pos,
            active_example_ratio,
            example_thres=DEFAULT_FREQSEL_EXAMPLE_THRES,
            cross_lingual_thres=DEFAULT_FREQSEL_CROSS_LINGUAL_THRES,
            token_thres=DEFAULT_FREQSEL_TOKEN_THRES,
        )
    return {
        "AnnSel": {lang: len(features) for lang, features in annsel.items()},
        "ValSel": {lang: len(features) for lang, features in valsel.items()},
        "FreqSel": {lang: len(features) for lang, features in freqsel.items()},
    }


def per_lang_min_achievable_budgets(
    method_pools: dict[str, dict[str, list[tuple[str, float]]]],
    langs: list[str],
) -> dict[str, int]:
    """K_L = min candidate-pool size across methods for each language L."""
    return per_lang_min_across_methods(
        {
            method: {lang: len(ranked[lang]) for lang in langs}
            for method, ranked in method_pools.items()
        },
        langs,
    )


def per_lang_pool_sizes_by_method(
    method_pools: dict[str, dict[str, list[tuple[str, float]]]],
    langs: list[str],
) -> dict[str, dict[str, int]]:
    return {
        method: {lang: len(ranked[lang]) for lang in langs}
        for method, ranked in method_pools.items()
    }


def binding_method_per_lang(
    per_method_per_lang: dict[str, dict[str, int]],
    per_lang_budgets: dict[str, int],
    langs: list[str],
) -> dict[str, str]:
    """Which method's value sets K_L for each language."""
    binding: dict[str, str] = {}
    for lang in langs:
        k_l = per_lang_budgets[lang]
        tied = [
            method
            for method, values in per_method_per_lang.items()
            if values[lang] == k_l
        ]
        binding[lang] = tied[0] if len(tied) == 1 else ",".join(tied)
    return binding


def max_uniform_budget(
    method_pools: dict[str, dict[str, list[tuple[str, float]]]],
) -> int:
    """Largest K such that every method has at least K candidates in every language."""
    mins = []
    for ranked in method_pools.values():
        mins.append(min(len(candidates) for candidates in ranked.values()))
    return min(mins) if mins else 0


def parse_uniform_budgets(budgets_arg: str) -> list[int]:
    budgets = [int(item.strip()) for item in budgets_arg.split(",") if item.strip()]
    if not budgets:
        raise ValueError("At least one uniform budget value is required.")
    if any(k <= 0 for k in budgets):
        raise ValueError("Budget values must be positive integers.")
    return sorted(set(budgets))


def method_extra(
    method: str,
    freqsel_candidate_params: dict[str, float] | None = None,
) -> dict[str, Any]:
    extra: dict[str, Any] = {
        "budget_matched": True,
        "source_defaults": {
            "AnnSel": {"threshold": DEFAULT_ANNSEL_THRESHOLD},
            "ValSel": {"topk": DEFAULT_VALSEL_TOPK},
            "FreqSel": {
                "example_thres": DEFAULT_FREQSEL_EXAMPLE_THRES,
                "cross_lingual_thres": DEFAULT_FREQSEL_CROSS_LINGUAL_THRES,
                "token_thres": DEFAULT_FREQSEL_TOKEN_THRES,
            },
        }[method],
    }
    if method == "FreqSel" and freqsel_candidate_params is not None:
        extra["freqsel_candidate_hyperparameters"] = freqsel_candidate_params
    return extra


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate budget-matched feature sets across selection methods."
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="gemma-2-2b",
        choices=hf_model_names.keys(),
    )
    parser.add_argument(
        "--budget-mode",
        choices=["min_default_counts", "per_lang_min", "uniform"],
        default="min_default_counts",
        help=(
            "min_default_counts: K_L = min default selection size across methods "
            "(default); per_lang_min: K_L = min candidate-pool size; "
            "uniform: same K for every language"
        ),
    )
    parser.add_argument(
        "--uniform-budgets",
        type=str,
        default="5,7,10,11",
        help="Comma-separated K values for --budget-mode uniform",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="annsel,valsel,freqsel",
        help="Comma-separated methods: annsel,valsel,freqsel",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Output root (default: data/additional_experiments/<model>/budget_matched_features)",
    )
    parser.add_argument(
        "--freqsel-example-thres",
        type=float,
        default=DEFAULT_FREQSEL_CANDIDATE_PARAMS["example_thres"],
        help="FreqSel candidate-pool example_thres (looser than default selection)",
    )
    parser.add_argument(
        "--freqsel-cross-lingual-thres",
        type=float,
        default=DEFAULT_FREQSEL_CANDIDATE_PARAMS["cross_lingual_thres"],
        help="FreqSel candidate-pool cross_lingual_thres",
    )
    parser.add_argument(
        "--freqsel-token-thres",
        type=float,
        default=DEFAULT_FREQSEL_CANDIDATE_PARAMS["token_thres"],
        help="FreqSel candidate-pool token_thres",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Print candidate pool sizes and exit without writing feature sets",
    )
    parser.add_argument(
        "--skip-overbudget",
        action="store_true",
        help="Uniform mode only: skip K when any method has fewer than K candidates",
    )
    return parser.parse_args()


def load_method_pools(
    args: argparse.Namespace,
    model_name: str,
    langs: list[str],
    selected_methods: set[str],
) -> tuple[dict[str, dict[str, list[tuple[str, float]]]], dict[str, Any]]:
    method_pools: dict[str, dict[str, list[tuple[str, float]]]] = {}
    pool_report: dict[str, Any] = {"model": model_name, "languages": langs, "methods": {}}

    if "annsel" in selected_methods:
        ranked = annsel_ranked_candidates(model_name, langs)
        method_pools["AnnSel"] = ranked
        pool_report["methods"]["AnnSel"] = pool_size_summary(ranked)

    if "valsel" in selected_methods:
        ranked = valsel_ranked_candidates(model_name, langs)
        method_pools["ValSel"] = ranked
        pool_report["methods"]["ValSel"] = pool_size_summary(ranked)

    if "freqsel" in selected_methods:
        activation_per_pos, active_example_ratio = load_sparse_stats(model_name, langs)
        ranked = freqsel_ranked_candidates(
            langs,
            activation_per_pos,
            active_example_ratio,
            example_thres=args.freqsel_example_thres,
            cross_lingual_thres=args.freqsel_cross_lingual_thres,
            token_thres=args.freqsel_token_thres,
        )
        method_pools["FreqSel"] = ranked
        pool_report["methods"]["FreqSel"] = {
            **pool_size_summary(ranked),
            "candidate_hyperparameters": {
                "example_thres": args.freqsel_example_thres,
                "cross_lingual_thres": args.freqsel_cross_lingual_thres,
                "token_thres": args.freqsel_token_thres,
            },
        }

    per_lang_pool_sizes = per_lang_pool_sizes_by_method(method_pools, langs)
    default_selection_counts = load_default_selection_counts(model_name, langs)
    per_lang_min_pool_budgets = per_lang_min_achievable_budgets(method_pools, langs)
    per_lang_min_default_budgets = per_lang_min_across_methods(
        default_selection_counts, langs
    )

    pool_report["per_lang_pool_sizes_by_method"] = per_lang_pool_sizes
    pool_report["default_selection_counts_by_method"] = default_selection_counts
    pool_report["per_lang_min_achievable_budgets"] = per_lang_min_pool_budgets
    pool_report["per_lang_min_default_budgets"] = per_lang_min_default_budgets
    pool_report["per_lang_binding_method_pool"] = binding_method_per_lang(
        per_lang_pool_sizes, per_lang_min_pool_budgets, langs
    )
    pool_report["per_lang_binding_method_default"] = binding_method_per_lang(
        default_selection_counts, per_lang_min_default_budgets, langs
    )
    pool_report["total_min_achievable_budget"] = sum(per_lang_min_pool_budgets.values())
    pool_report["total_min_default_budget"] = sum(per_lang_min_default_budgets.values())
    pool_report["max_uniform_budget"] = max_uniform_budget(method_pools)
    return method_pools, pool_report


def generate_per_lang_budget_sets(
    method_pools: dict[str, dict[str, list[tuple[str, float]]]],
    per_lang_budgets: dict[str, int],
    budget_mode: str,
    sweep_name: str,
    binding_method: dict[str, str],
    features_dir: str,
    model_name: str,
    freqsel_candidate_params: dict[str, float] | None,
    default_selection_counts: dict[str, dict[str, int]] | None = None,
) -> list[dict[str, Any]]:
    generated: list[dict[str, Any]] = []

    for method, ranked in method_pools.items():
        selected, requested, actual = truncate_ranked_candidates_per_lang(
            ranked, per_lang_budgets
        )
        hyperparameters = {
            "budget_mode": budget_mode,
            "total_features": sum(per_lang_budgets.values()),
        }
        extra = method_extra(method, freqsel_candidate_params)
        extra.update(
            {
                "per_lang_requested": requested,
                "per_lang_actual": actual,
                "per_lang_binding_method": binding_method,
                "fully_matched": actual == requested,
            }
        )
        if default_selection_counts is not None:
            extra["default_selection_counts_by_method"] = default_selection_counts
        out_path = save_selected_features(
            features_dir,
            method,
            model_name,
            sweep_name,
            hyperparameters,
            is_default=False,
            selected_features=selected,
            extra=extra,
        )
        summary = summarize_feature_set(
            method,
            selected,
            descriptions={},
            baseline_by_lang=None,
            annotation_selected=(method == "AnnSel"),
        )
        entry = {
            "method": method,
            "budget_mode": budget_mode,
            "per_lang_budgets": per_lang_budgets,
            "selected_features_file": out_path,
            "fully_matched": extra["fully_matched"],
            "per_lang_actual": actual,
            **summary,
        }
        generated.append(entry)
        print(
            f"[OK] {method} {budget_mode}: per_lang={actual}, "
            f"total={sum(actual.values())}, file={out_path}"
        )
    return generated


def generate_uniform_sets(
    method_pools: dict[str, dict[str, list[tuple[str, float]]]],
    features_dir: str,
    model_name: str,
    langs: list[str],
    budgets: list[int],
    skip_overbudget: bool,
    freqsel_candidate_params: dict[str, float] | None,
) -> list[dict[str, Any]]:
    generated: list[dict[str, Any]] = []
    for budget_k in budgets:
        for method, ranked in method_pools.items():
            min_pool = min(len(candidates) for candidates in ranked.values())
            if skip_overbudget and budget_k > min_pool:
                print(f"Skipping {method} K={budget_k}: min pool size is {min_pool}")
                continue

            selected, requested, actual = truncate_ranked_candidates_uniform(
                ranked, budget_k
            )
            hyperparameters = {"budget_mode": "uniform", "budget_k": budget_k}
            extra = method_extra(method, freqsel_candidate_params)
            extra.update(
                {
                    "per_lang_requested": requested,
                    "per_lang_actual": actual,
                    "fully_matched": all(count == budget_k for count in actual.values()),
                }
            )
            out_path = save_selected_features(
                features_dir,
                method,
                model_name,
                "budget_k",
                hyperparameters,
                is_default=False,
                selected_features=selected,
                extra=extra,
            )
            summary = summarize_feature_set(
                method,
                selected,
                descriptions={},
                baseline_by_lang=None,
                annotation_selected=(method == "AnnSel"),
            )
            entry = {
                "method": method,
                "budget_mode": "uniform",
                "budget_k": budget_k,
                "selected_features_file": out_path,
                "fully_matched": extra["fully_matched"],
                "per_lang_actual": actual,
                **summary,
            }
            generated.append(entry)
            status = "OK" if extra["fully_matched"] else "PARTIAL"
            print(
                f"[{status}] {method} K={budget_k}: per_lang={actual}, file={out_path}"
            )
    return generated


def write_summary_csv(
    csv_path: str,
    generated: list[dict[str, Any]],
    langs: list[str],
) -> None:
    if not generated:
        return
    fieldnames = [
        "method",
        "budget_mode",
        "fully_matched",
        "total_features",
        "mean_features_per_lang",
        "selected_features_file",
    ]
    if any("budget_k" in entry for entry in generated):
        fieldnames.insert(2, "budget_k")
    if any("per_lang_budgets" in entry for entry in generated):
        fieldnames.insert(2, "per_lang_budgets")
    for lang in langs:
        fieldnames.append(f"count_{lang}")
    rows = []
    for entry in generated:
        row = {
            "method": entry["method"],
            "budget_mode": entry["budget_mode"],
            "fully_matched": entry["fully_matched"],
            "total_features": entry["total_features"],
            "mean_features_per_lang": round(entry["mean_features_per_lang"], 2),
            "selected_features_file": entry["selected_features_file"],
        }
        if "budget_k" in entry:
            row["budget_k"] = entry["budget_k"]
        if "per_lang_budgets" in entry:
            row["per_lang_budgets"] = json.dumps(entry["per_lang_budgets"])
        for lang in langs:
            row[f"count_{lang}"] = entry["per_lang_actual"][lang]
        rows.append(row)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    model_name = args.model
    langs = list(lang_to_flores_key.keys())
    selected_methods = {m.strip().lower() for m in args.methods.split(",")}

    output_dir = args.output_dir or os.path.join(
        repo_data_dir(), "additional_experiments", model_name, "budget_matched_features"
    )
    features_dir = output_dir
    os.makedirs(features_dir, exist_ok=True)

    method_pools, pool_report = load_method_pools(
        args, model_name, langs, selected_methods
    )
    pool_report["budget_mode"] = args.budget_mode

    report_path = os.path.join(output_dir, "candidate_pool_report.json")
    with open(report_path, "w") as f:
        json.dump(pool_report, f, indent=2)

    print(f"Wrote candidate pool report to {report_path}")
    for method, summary in pool_report["methods"].items():
        print(
            f"{method}: min_per_lang_pool={summary['min_per_lang_pool_size']}, "
            f"per_lang={summary['per_lang_pool_size']}"
        )
    print("Default selection counts by method:")
    for method, counts in pool_report["default_selection_counts_by_method"].items():
        print(f"  {method}: {counts} (total={sum(counts.values())})")
    print(
        "Per-language min-default budgets: "
        f"{pool_report['per_lang_min_default_budgets']}"
    )
    print(
        f"Binding method (default): {pool_report['per_lang_binding_method_default']}"
    )
    print(f"Total min-default budget: {pool_report['total_min_default_budget']}")
    print(
        "Per-language min-pool budgets: "
        f"{pool_report['per_lang_min_achievable_budgets']}"
    )
    print(f"Total min-pool budget: {pool_report['total_min_achievable_budget']}")

    if args.report_only:
        return

    freqsel_candidate_params = None
    if "FreqSel" in method_pools:
        freqsel_candidate_params = {
            "example_thres": args.freqsel_example_thres,
            "cross_lingual_thres": args.freqsel_cross_lingual_thres,
            "token_thres": args.freqsel_token_thres,
        }

    if args.budget_mode == "min_default_counts":
        per_lang_budgets = pool_report["per_lang_min_default_budgets"]
        generated = generate_per_lang_budget_sets(
            method_pools,
            per_lang_budgets,
            budget_mode="min_default_counts",
            sweep_name=MIN_DEFAULT_COUNTS_SWEEP,
            binding_method=pool_report["per_lang_binding_method_default"],
            features_dir=features_dir,
            model_name=model_name,
            freqsel_candidate_params=freqsel_candidate_params,
            default_selection_counts=pool_report["default_selection_counts_by_method"],
        )
        manifest_budgets = per_lang_budgets
    elif args.budget_mode == "per_lang_min":
        per_lang_budgets = pool_report["per_lang_min_achievable_budgets"]
        generated = generate_per_lang_budget_sets(
            method_pools,
            per_lang_budgets,
            budget_mode="per_lang_min_achievable",
            sweep_name=PER_LANG_MIN_POOL_SWEEP,
            binding_method=pool_report["per_lang_binding_method_pool"],
            features_dir=features_dir,
            model_name=model_name,
            freqsel_candidate_params=freqsel_candidate_params,
        )
        manifest_budgets = per_lang_budgets
    else:
        budgets = parse_uniform_budgets(args.uniform_budgets)
        pool_report["requested_uniform_budgets"] = budgets
        generated = generate_uniform_sets(
            method_pools,
            features_dir,
            model_name,
            langs,
            budgets,
            args.skip_overbudget,
            freqsel_candidate_params,
        )
        manifest_budgets = budgets

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "model": model_name,
                "budget_mode": args.budget_mode,
                "budgets": manifest_budgets,
                "methods": sorted(method_pools.keys()),
                "candidate_pool_report": pool_report,
                "generated_sets": generated,
            },
            f,
            indent=2,
        )

    csv_path = os.path.join(output_dir, "summary.csv")
    write_summary_csv(csv_path, generated, langs)
    print(f"Wrote manifest to {manifest_path}")
    print(f"Wrote summary to {csv_path}")
    print(f"Generated {len(generated)} budget-matched feature set(s) under {features_dir}")


if __name__ == "__main__":
    main()
