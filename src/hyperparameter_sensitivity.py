"""
Hyperparameter sensitivity analysis for AnnSel, ValSel, and FreqSel.

Uses precomputed feature-extraction artifacts (no GPU required).

AnnSel has two hyperparameter stages:
  1. Tracing (GPU; see annsel_tracing_sensitivity.py):
     throughput_threshold, node_threshold, edge_threshold,
     threshold_first, threshold_last, max_iterations
  2. Selection (this script; offline):
     threshold on annotation match counts

ValSel and FreqSel sweeps are fully offline here.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from itertools import combinations
from typing import Any

from models import hf_model_names
from template import identifiers, lang_to_flores_key


def description_based_features(
    dir_name: str, languages: list[str], threshold: float
) -> dict[str, list[str]]:
    lang_features: dict[str, list[str]] = {}
    for lang in languages:
        with open(os.path.join(dir_name, f"{lang}_features.json"), "r") as f:
            features_freq = json.load(f)
        max_val = max(features_freq.values())
        lang_features[lang] = [
            key for key, val in features_freq.items() if val >= max_val * threshold
        ]
    return lang_features


def mean_value_based_features(
    dir_name: str, languages: list[str], topk: int
) -> dict[str, list[str]]:
    lang_features: dict[str, list[str]] = {}
    for lang in languages:
        with open(os.path.join(dir_name, f"{lang}.json"), "r") as f:
            feature_val = json.load(f)
        sorted_features = sorted(
            feature_val.items(), key=lambda item: item[1], reverse=True
        )[:topk]
        lang_features[lang] = [feature for feature, _ in sorted_features]
    return lang_features


def choose_language_specific_features(
    langs: list[str],
    active_tokens: dict[str, dict[str, float]],
    active_examples: dict[str, dict[str, float]],
    cross_lingual_thres: float,
    example_thres: float,
    token_thres: float = 0.1,
) -> dict[str, list[tuple[int, int]]]:
    all_features: set[str] = set()
    for lang in langs:
        all_features.update(active_tokens[lang].keys())

    language_specific_features = {lang: [] for lang in langs}
    for feature_key in all_features:
        passes_threshold = False
        for lang in langs:
            token_val = active_tokens[lang].get(feature_key, 0)
            example_val = active_examples[lang].get(feature_key, 0)
            if token_val > token_thres and example_val > example_thres:
                passes_threshold = True
                break
        if not passes_threshold:
            continue

        vals = [active_tokens[lang].get(feature_key, 0) for lang in langs]
        max_val = max(vals)
        if max_val == 0:
            continue

        active_langs = [
            i for i, val in enumerate(vals) if val >= max_val * cross_lingual_thres
        ]
        if len(active_langs) == 1:
            lang = langs[active_langs[0]]
            layer, feature_idx = feature_key.split(".")
            language_specific_features[lang].append((int(layer), int(feature_idx)))
    return language_specific_features

DEFAULT_ANNSEL_THRESHOLD = 0.1
DEFAULT_ANNSEL_TRACING = {
    "throughput_threshold": 0.1,
    "node_threshold": 0.8,
    "edge_threshold": 0.98,
    "threshold_first": 0.5,
    "threshold_last": 0.25,
    "max_iterations": 75,
}
DEFAULT_VALSEL_TOPK = 50
DEFAULT_FREQSEL_EXAMPLE_THRES = 0.98
DEFAULT_FREQSEL_CROSS_LINGUAL_THRES = 0.8
DEFAULT_FREQSEL_TOKEN_THRES = 0.1

ANNSEL_THRESHOLDS = [0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
VALSEL_TOPKS = [10, 20, 30, 40, 50, 75, 100, 150, 200]
FREQSEL_EXAMPLE_THRES = [0.8, 0.85, 0.9, 0.95, 0.98]
FREQSEL_CROSS_LINGUAL_THRES = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
FREQSEL_TOKEN_THRES = [0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25]


def repo_data_dir() -> str:
    current_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(current_directory), "data")


def load_description_dicts(model_name: str, langs: list[str]) -> dict[str, dict[str, str]]:
    """Merge per-language Neuronpedia description files for ValSel and FreqSel."""
    descriptions: dict[str, dict[str, str]] = {
        "val": {},
        "freq": {},
    }
    val_dir = os.path.join(repo_data_dir(), "multilingual_llm_features", model_name)
    freq_dir = os.path.join(repo_data_dir(), "language_specific_features", model_name)
    for lang in langs:
        val_path = os.path.join(val_dir, f"{lang}_description.json")
        if os.path.exists(val_path):
            with open(val_path, "r") as f:
                descriptions["val"].update(json.load(f))
        freq_path = os.path.join(freq_dir, f"{lang}_description.json")
        if os.path.exists(freq_path):
            with open(freq_path, "r") as f:
                descriptions["freq"].update(json.load(f))
    return descriptions


def load_annsel_descriptions(model_name: str, langs: list[str]) -> dict[str, str]:
    """AnnSel descriptions from the shared Neuronpedia cache."""
    from feature_extraction import (
        load_neuronpedia_description_cache,
        neuronpedia_description_cache_path,
    )

    cache_path = neuronpedia_description_cache_path(model_name)
    return load_neuronpedia_description_cache(cache_path)


def load_sparse_stats(model_name: str, langs: list[str]) -> tuple[
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
]:
    """Load per-language activation stats needed for FreqSel selection."""
    lang_specific_dir = os.path.join(
        repo_data_dir(), "language_specific_features", model_name
    )
    activation_per_pos: dict[str, dict[str, float]] = {}
    active_example_ratio: dict[str, dict[str, float]] = {}
    for lang in langs:
        for suffix in ("_sparse_long.json", "_sparse.json"):
            path = os.path.join(lang_specific_dir, f"{lang}{suffix}")
            if os.path.exists(path):
                with open(path, "r") as f:
                    data = json.load(f)
                activation_per_pos[lang] = data["activation_per_pos"]
                active_example_ratio[lang] = data["active_example_ratio"]
                break
        if lang not in activation_per_pos:
            raise FileNotFoundError(
                f"Missing sparse stats for {lang}. "
                f"Run language_specific_features.py --model {model_name} first."
            )
    return activation_per_pos, active_example_ratio


def tuples_to_strings(
    feature_dict: dict[str, list[tuple[int, int]]],
) -> dict[str, list[str]]:
    return {
        lang: [f"{layer}.{feature_idx}" for layer, feature_idx in features]
        for lang, features in feature_dict.items()
    }


def freqsel_from_params(
    langs: list[str],
    activation_per_pos: dict[str, dict[str, float]],
    active_example_ratio: dict[str, dict[str, float]],
    example_thres: float,
    cross_lingual_thres: float,
    token_thres: float,
) -> dict[str, list[str]]:
    selected = choose_language_specific_features(
        langs,
        activation_per_pos,
        active_example_ratio,
        cross_lingual_thres=cross_lingual_thres,
        example_thres=example_thres,
        token_thres=token_thres,
    )
    return tuples_to_strings(selected)


def try_load_cached_freqsel(
    model_name: str,
    example_thres: float,
    cross_lingual_thres: float,
    token_thres: float,
    langs: list[str],
) -> dict[str, list[str]] | None:
    """Use precomputed features_{example_thres}.json when defaults match."""
    if (
        cross_lingual_thres != DEFAULT_FREQSEL_CROSS_LINGUAL_THRES
        or token_thres != DEFAULT_FREQSEL_TOKEN_THRES
    ):
        return None
    path = os.path.join(
        repo_data_dir(),
        "language_specific_features",
        model_name,
        f"features_{example_thres}.json",
    )
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        feature_dict = json.load(f)
    return {
        lang: [f"{layer}.{feature_idx}" for layer, feature_idx in feature_dict[lang]]
        for lang in langs
    }


def count_annsel_lang_name_matches(
    features_by_lang: dict[str, list[str]],
) -> dict[str, tuple[int, int]]:
    """AnnSel features are selected from identifier-matched counts by construction."""
    return {lang: (len(features), len(features)) for lang, features in features_by_lang.items()}


def count_lang_name_matches(
    features_by_lang: dict[str, list[str]],
    descriptions: dict[str, str],
) -> dict[str, tuple[int, int]]:
    counts: dict[str, tuple[int, int]] = {}
    for lang, features in features_by_lang.items():
        identifier = identifiers[lang]
        matched = 0
        for feature in features:
            description = descriptions.get(feature, "")
            if any(item in description for item in identifier):
                matched += 1
        counts[lang] = (len(features), matched)
    return counts

def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def stability_vs_baseline(
    features_by_lang: dict[str, list[str]],
    baseline_by_lang: dict[str, list[str]],
) -> dict[str, float]:
    return {
        lang: jaccard(set(features_by_lang[lang]), set(baseline_by_lang[lang]))
        for lang in features_by_lang
    }


def mean_cross_lang_jaccard(features_by_lang: dict[str, list[str]]) -> float:
    langs = sorted(features_by_lang.keys())
    if len(langs) < 2:
        return 1.0
    scores = []
    for lang_a, lang_b in combinations(langs, 2):
        scores.append(
            jaccard(
                set(features_by_lang[lang_a]),
                set(features_by_lang[lang_b]),
            )
        )
    return sum(scores) / len(scores)


def format_hyperparameter_tag(hyperparameters: dict[str, Any]) -> str:
    parts = []
    for key, value in sorted(hyperparameters.items()):
        if isinstance(value, float):
            parts.append(f"{key}_{value:g}")
        else:
            parts.append(f"{key}_{value}")
    return "__".join(parts)


def format_setting_filename(sweep_name: str, hyperparameters: dict[str, Any]) -> str:
    if len(hyperparameters) == 1:
        _, value = next(iter(hyperparameters.items()))
        if isinstance(value, float):
            return f"{sweep_name}_{value:g}"
        return f"{sweep_name}_{value}"
    return f"{sweep_name}__{format_hyperparameter_tag(hyperparameters)}"


def save_selected_features(
    features_dir: str,
    method: str,
    model_name: str,
    sweep_name: str,
    hyperparameters: dict[str, Any],
    is_default: bool,
    selected_features: dict[str, list[str]],
    extra: dict[str, Any] | None = None,
    filename_suffix: str | None = None,
) -> str:
    """Write per-setting selected feature lists for downstream intervention runs."""
    method_dir = os.path.join(features_dir, method)
    os.makedirs(method_dir, exist_ok=True)
    stem = format_setting_filename(sweep_name, hyperparameters)
    if filename_suffix:
        stem = f"{stem}__{filename_suffix}"
    path = os.path.join(method_dir, f"{stem}.json")
    payload: dict[str, Any] = {
        "method": method,
        "model": model_name,
        "sweep": sweep_name,
        "hyperparameters": hyperparameters,
        "is_default": is_default,
        "selected_features": selected_features,
    }
    if extra:
        payload.update(extra)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def load_selected_features(path: str) -> dict[str, list[str]]:
    """Load selected feature lists written by save_selected_features."""
    with open(path, "r") as f:
        payload = json.load(f)
    return payload["selected_features"]


def summarize_feature_set(
    method: str,
    features_by_lang: dict[str, list[str]],
    descriptions: dict[str, str],
    baseline_by_lang: dict[str, list[str]] | None,
    *,
    annotation_selected: bool = False,
) -> dict[str, Any]:
    per_lang_counts = {lang: len(features) for lang, features in features_by_lang.items()}
    if annotation_selected:
        lang_name_counts = count_annsel_lang_name_matches(features_by_lang)
    else:
        lang_name_counts = count_lang_name_matches(features_by_lang, descriptions)
    lang_name_rates = {
        lang: (matched / total if total else 0.0)
        for lang, (total, matched) in lang_name_counts.items()
    }
    summary: dict[str, Any] = {
        "method": method,
        "per_lang_counts": per_lang_counts,
        "total_features": sum(per_lang_counts.values()),
        "mean_features_per_lang": sum(per_lang_counts.values()) / len(per_lang_counts),
        "lang_name_counts": {
            lang: {"total": total, "matched": matched}
            for lang, (total, matched) in lang_name_counts.items()
        },
        "mean_lang_name_match_rate": sum(lang_name_rates.values()) / len(lang_name_rates),
        "mean_cross_lang_jaccard": mean_cross_lang_jaccard(features_by_lang),
    }
    if baseline_by_lang is not None:
        jaccards = stability_vs_baseline(features_by_lang, baseline_by_lang)
        summary["jaccard_vs_default"] = jaccards
        summary["mean_jaccard_vs_default"] = sum(jaccards.values()) / len(jaccards)
    return summary


def sweep_annsel(
    model_name: str,
    langs: list[str],
    thresholds: list[float],
    descriptions: dict[str, str],
    features_dir: str,
) -> list[dict[str, Any]]:
    flores_dir = os.path.join(repo_data_dir(), "flores_features", model_name)
    baseline = description_based_features(
        flores_dir, langs, threshold=DEFAULT_ANNSEL_THRESHOLD
    )
    results = []
    for threshold in thresholds:
        features = description_based_features(flores_dir, langs, threshold=threshold)
        hyperparameters = {"threshold": threshold}
        is_default = threshold == DEFAULT_ANNSEL_THRESHOLD
        entry = {
            "hyperparameters": hyperparameters,
            "is_default": is_default,
            "selected_features_file": save_selected_features(
                features_dir,
                "AnnSel",
                model_name,
                "selection_threshold",
                hyperparameters,
                is_default,
                features,
            ),
        }
        entry.update(
            summarize_feature_set(
                "AnnSel",
                features,
                descriptions,
                baseline if not is_default else None,
                annotation_selected=True,
            )
        )
        results.append(entry)
    return results


def sweep_valsel(
    model_name: str,
    langs: list[str],
    topks: list[int],
    descriptions: dict[str, str],
    features_dir: str,
) -> list[dict[str, Any]]:
    val_dir = os.path.join(repo_data_dir(), "multilingual_llm_features", model_name)
    baseline = mean_value_based_features(val_dir, langs, topk=DEFAULT_VALSEL_TOPK)
    results = []
    for topk in topks:
        features = mean_value_based_features(val_dir, langs, topk=topk)
        hyperparameters = {"topk": topk}
        is_default = topk == DEFAULT_VALSEL_TOPK
        entry = {
            "hyperparameters": hyperparameters,
            "is_default": is_default,
            "selected_features_file": save_selected_features(
                features_dir,
                "ValSel",
                model_name,
                "topk",
                hyperparameters,
                is_default,
                features,
            ),
        }
        entry.update(
            summarize_feature_set(
                "ValSel",
                features,
                descriptions,
                baseline if not is_default else None,
            )
        )
        results.append(entry)
    return results


def sweep_freqsel_axis(
    model_name: str,
    langs: list[str],
    axis_name: str,
    axis_values: list[float],
    activation_per_pos: dict[str, dict[str, float]],
    active_example_ratio: dict[str, dict[str, float]],
    descriptions: dict[str, str],
    features_dir: str,
) -> list[dict[str, Any]]:
    baseline = freqsel_from_params(
        langs,
        activation_per_pos,
        active_example_ratio,
        example_thres=DEFAULT_FREQSEL_EXAMPLE_THRES,
        cross_lingual_thres=DEFAULT_FREQSEL_CROSS_LINGUAL_THRES,
        token_thres=DEFAULT_FREQSEL_TOKEN_THRES,
    )
    results = []
    for value in axis_values:
        params = {
            "example_thres": DEFAULT_FREQSEL_EXAMPLE_THRES,
            "cross_lingual_thres": DEFAULT_FREQSEL_CROSS_LINGUAL_THRES,
            "token_thres": DEFAULT_FREQSEL_TOKEN_THRES,
            axis_name: value,
        }
        # Always recompute from sparse stats. Legacy features_{example_thres}.json
        # caches are not comparable (different counts / not monotonic in example_thres).
        features = freqsel_from_params(
            langs,
            activation_per_pos,
            active_example_ratio,
            example_thres=params["example_thres"],
            cross_lingual_thres=params["cross_lingual_thres"],
            token_thres=params["token_thres"],
        )
        is_default = (
            params["example_thres"] == DEFAULT_FREQSEL_EXAMPLE_THRES
            and params["cross_lingual_thres"] == DEFAULT_FREQSEL_CROSS_LINGUAL_THRES
            and params["token_thres"] == DEFAULT_FREQSEL_TOKEN_THRES
        )
        entry = {
            "sweep_axis": axis_name,
            "hyperparameters": params,
            "is_default": is_default,
            "selected_features_file": save_selected_features(
                features_dir,
                "FreqSel",
                model_name,
                axis_name,
                params,
                is_default,
                features,
            ),
        }
        entry.update(
            summarize_feature_set(
                "FreqSel",
                features,
                descriptions,
                baseline if not is_default else None,
            )
        )
        results.append(entry)
    return results


def write_csv(path: str, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def flatten_sweep_rows(method: str, sweep_name: str, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for entry in entries:
        hp = entry["hyperparameters"]
        row = {
            "method": method,
            "sweep": sweep_name,
            "is_default": entry["is_default"],
            "total_features": entry["total_features"],
            "mean_features_per_lang": round(entry["mean_features_per_lang"], 2),
            "mean_lang_name_match_rate": round(entry["mean_lang_name_match_rate"], 4),
            "mean_cross_lang_jaccard": round(entry["mean_cross_lang_jaccard"], 4),
            "mean_jaccard_vs_default": (
                round(entry["mean_jaccard_vs_default"], 4)
                if "mean_jaccard_vs_default" in entry
                else 1.0 if entry["is_default"] else ""
            ),
        }
        for key, value in hp.items():
            row[key] = value
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hyperparameter sensitivity analysis for latent selection methods."
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="gemma-2-2b",
        choices=hf_model_names.keys(),
        help="Model key (default: gemma-2-2b)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="all",
        help="Comma-separated methods: annsel,valsel,freqsel (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Output directory (default: data/additional_experiments/<model>)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_name = args.model
    langs = list(lang_to_flores_key.keys())
    output_dir = args.output_dir or os.path.join(
        repo_data_dir(), "additional_experiments", model_name
    )
    features_dir = os.path.join(output_dir, "selected_features")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)

    selected_methods = (
        {"annsel", "valsel", "freqsel"}
        if args.methods == "all"
        else {m.strip().lower() for m in args.methods.split(",")}
    )

    description_dicts = load_description_dicts(model_name, langs)
    annsel_descriptions = load_annsel_descriptions(model_name, langs)

    report: dict[str, Any] = {
        "model": model_name,
        "languages": langs,
        "defaults": {
            "AnnSel_tracing": DEFAULT_ANNSEL_TRACING,
            "AnnSel_selection": {"threshold": DEFAULT_ANNSEL_THRESHOLD},
            "ValSel": {"topk": DEFAULT_VALSEL_TOPK},
            "FreqSel": {
                "example_thres": DEFAULT_FREQSEL_EXAMPLE_THRES,
                "cross_lingual_thres": DEFAULT_FREQSEL_CROSS_LINGUAL_THRES,
                "token_thres": DEFAULT_FREQSEL_TOKEN_THRES,
            },
        },
        "notes": {
            "AnnSel_tracing": (
                "Tracing hyperparameters require GPU re-extraction. "
                "Run annsel_tracing_sensitivity.py."
            ),
            "selected_features": (
                "Per-setting feature lists are saved under selected_features/. "
                "Each file has selected_features[lang] = list of 'layer.feature_idx' strings."
            ),
        },
        "selected_features_dir": features_dir,
        "sweeps": {},
    }
    csv_rows: list[dict[str, Any]] = []

    if "annsel" in selected_methods:
        annsel_results = sweep_annsel(
            model_name,
            langs,
            ANNSEL_THRESHOLDS,
            annsel_descriptions or description_dicts["val"],
            features_dir,
        )
        report["sweeps"]["AnnSel_selection_threshold"] = annsel_results
        csv_rows.extend(flatten_sweep_rows("AnnSel", "selection_threshold", annsel_results))

    if "valsel" in selected_methods:
        valsel_results = sweep_valsel(
            model_name,
            langs,
            VALSEL_TOPKS,
            description_dicts["val"],
            features_dir,
        )
        report["sweeps"]["ValSel_topk"] = valsel_results
        csv_rows.extend(flatten_sweep_rows("ValSel", "topk", valsel_results))

    if "freqsel" in selected_methods:
        activation_per_pos, active_example_ratio = load_sparse_stats(model_name, langs)
        freqsel_sweeps = {
            "FreqSel_example_thres": sweep_freqsel_axis(
                model_name,
                langs,
                "example_thres",
                FREQSEL_EXAMPLE_THRES,
                activation_per_pos,
                active_example_ratio,
                description_dicts["freq"],
                features_dir,
            ),
            "FreqSel_cross_lingual_thres": sweep_freqsel_axis(
                model_name,
                langs,
                "cross_lingual_thres",
                FREQSEL_CROSS_LINGUAL_THRES,
                activation_per_pos,
                active_example_ratio,
                description_dicts["freq"],
                features_dir,
            ),
            "FreqSel_token_thres": sweep_freqsel_axis(
                model_name,
                langs,
                "token_thres",
                FREQSEL_TOKEN_THRES,
                activation_per_pos,
                active_example_ratio,
                description_dicts["freq"],
                features_dir,
            ),
        }
        report["sweeps"].update(freqsel_sweeps)
        for sweep_name, entries in freqsel_sweeps.items():
            axis = sweep_name.replace("FreqSel_", "")
            csv_rows.extend(flatten_sweep_rows("FreqSel", axis, entries))

    json_path = os.path.join(output_dir, f"hyperparameter_sensitivity_{model_name}.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    csv_path = os.path.join(output_dir, f"hyperparameter_sensitivity_{model_name}.csv")
    csv_fieldnames = [
        "method",
        "sweep",
        "is_default",
        "threshold",
        "topk",
        "example_thres",
        "cross_lingual_thres",
        "token_thres",
        "total_features",
        "mean_features_per_lang",
        "mean_lang_name_match_rate",
        "mean_cross_lang_jaccard",
        "mean_jaccard_vs_default",
    ]
    write_csv(csv_path, csv_rows, csv_fieldnames)

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Selected features saved under {features_dir}")
    for row in csv_rows:
        if row["is_default"]:
            print(
                f"[default] {row['method']} {row['sweep']}: "
                f"total={row['total_features']}, "
                f"lang_name_rate={row['mean_lang_name_match_rate']}, "
                f"cross_lang_jaccard={row['mean_cross_lang_jaccard']}"
            )


if __name__ == "__main__":
    main()
