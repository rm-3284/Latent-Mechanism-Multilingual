"""
GPU-based hyperparameter sensitivity for AnnSel circuit-tracing stage.

AnnSel has two stages:
  1. Tracing (this script): widest-path extraction in attribution graphs
     - throughput_threshold, node_threshold, edge_threshold
     - threshold_first, threshold_last, max_iterations
  2. Selection (hyperparameter_sensitivity.py): filter annotation counts
     - threshold

This script re-runs path pruning + annotation filtering for each tracing
hyperparameter value (one-at-a-time sweeps). Attribution graphs are built
once per sentence and cached; only path pruning is repeated per setting.

Example (single language smoke test):
  python annsel_tracing_sensitivity.py --model gemma-2-2b --lang en --num-sentences 5

Full sweep:
  python annsel_tracing_sensitivity.py --model gemma-2-2b
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import gc
import json
import os
import time
from typing import Any

from sensitivity.hyperparameter_sensitivity import (
    DEFAULT_ANNSEL_THRESHOLD,
    flatten_sweep_rows,
    format_setting_filename,
    repo_data_dir,
    save_selected_features,
    summarize_feature_set,
    write_csv,
)
from lib.models import hf_model_names
from lib.template import identifiers, lang_to_flores_key

DEFAULT_TRACING = {
    "throughput_threshold": 0.1,
    "node_threshold": 0.8,
    "edge_threshold": 0.98,
    "threshold_first": 0.5,
    "threshold_last": 0.25,
    "max_iterations": 75,
}

TRACING_SWEEPS = {
    "throughput_threshold": [0.05, 0.075, 0.1, 0.15, 0.2, 0.25],
    "node_threshold": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    "edge_threshold": [0.9, 0.92, 0.95, 0.98, 0.99],
    "threshold_first": [0.3, 0.4, 0.5, 0.6, 0.7],
    "threshold_last": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
}


def graph_cache_complete(
    graph_cache_root: str,
    langs: list[str],
    num_sentences: int,
) -> bool:
    from pipeline.flores_feature_extraction import graph_cache_manifest_path

    for lang in langs:
        lang_dir = os.path.join(graph_cache_root, lang)
        manifest_path = graph_cache_manifest_path(lang_dir)
        if not os.path.exists(manifest_path):
            return False
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        if manifest.get("num_sentences", len(manifest.get("sentences", []))) < num_sentences:
            return False
        for idx in range(num_sentences):
            if not os.path.exists(os.path.join(lang_dir, f"{idx:04d}.pt")):
                return False
    return True


def load_sentences_from_manifest(
    langs: list[str],
    graph_cache_root: str,
    num_sentences: int,
) -> dict[str, list[str]]:
    from pipeline.flores_feature_extraction import graph_cache_manifest_path

    sentences_by_lang: dict[str, list[str]] = {}
    for lang in langs:
        lang_dir = os.path.join(graph_cache_root, lang)
        with open(graph_cache_manifest_path(lang_dir), "r") as f:
            manifest = json.load(f)
        sentences = manifest["sentences"]
        if len(sentences) < num_sentences:
            raise ValueError(
                f"Graph cache for {lang} has {len(sentences)} sentences, "
                f"need {num_sentences}"
            )
        sentences_by_lang[lang] = sentences[:num_sentences]
        print(f"Loaded {num_sentences} cached sentences for {lang} from graph manifest")
    return sentences_by_lang


def load_sentences(
    langs: list[str],
    model: Any,
    num_sentences: int,
) -> dict[str, list[str]]:
    from datasets import load_dataset
    from lib.pipeline_data.generic_sentences import alphabet_char, filter_sentences

    sentences_by_lang: dict[str, list[str]] = {}
    for lang in langs:
        ds_key = lang_to_flores_key[lang]
        print(f"Loading FLORES dev for {lang} ({ds_key})")
        last_error: OSError | None = None
        ds = None
        for attempt in range(6):
            try:
                ds = load_dataset("openlanguagedata/flores_plus", ds_key, split="dev")
                break
            except OSError as exc:
                last_error = exc
                if exc.errno == 116 and attempt + 1 < 6:
                    delay = 5 * (attempt + 1)
                    print(
                        f"  stale NFS handle loading FLORES for {lang}; "
                        f"retrying in {delay}s ({attempt + 1}/5)"
                    )
                    time.sleep(delay)
                    continue
                raise
        if ds is None:
            assert last_error is not None
            raise last_error
        ds = ds.shuffle(seed=42)
        batch = [example["text"] for i, example in enumerate(ds) if i < 150]
        sentences_by_lang[lang] = filter_sentences(
            batch, alphabet_char[lang], model, num_sentences=num_sentences
        )
        del batch, ds
    return sentences_by_lang


def ensure_graph_caches_for_langs(
    model: Any,
    langs: list[str],
    sentences_by_lang: dict[str, list[str]],
    graph_cache_root: str,
) -> None:
    from pipeline.flores_feature_extraction import ensure_attribution_graph_cache

    os.makedirs(graph_cache_root, exist_ok=True)
    for lang in langs:
        lang_cache_dir = os.path.join(graph_cache_root, lang)
        ensure_attribution_graph_cache(
            model,
            lang,
            sentences_by_lang[lang],
            lang_cache_dir,
        )


def run_tracing_for_params(
    model: Any,
    langs: list[str],
    sentences_by_lang: dict[str, list[str]],
    tracing_params: dict[str, float | int],
    cache_dir: str | None,
    graph_cache_root: str | None = None,
    prune_workers: int = 1,
) -> dict[str, list[tuple[int, int]]]:
    import torch
    from pipeline.flores_feature_extraction import (
        features_from_graph_cache,
        iterate_through_sentences,
    )

    traced_by_lang: dict[str, list[tuple[int, int]]] = {}
    for lang in langs:
        cache_path = None
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, f"{lang}.json")
            if os.path.exists(cache_path):
                with open(cache_path, "r") as f:
                    traced_by_lang[lang] = [tuple(item) for item in json.load(f)]
                print(f"  loaded cached traced features for {lang}")
                continue

        if graph_cache_root is not None:
            lang_graph_dir = os.path.join(graph_cache_root, lang)
            print(
                f"  pruning paths for {lang} "
                f"({len(sentences_by_lang[lang])} cached graphs)"
            )
            features = features_from_graph_cache(
                lang_graph_dir,
                len(sentences_by_lang[lang]),
                tracing_params,
                prune_workers=prune_workers,
            )
        else:
            print(f"  tracing {lang} ({len(sentences_by_lang[lang])} sentences)")
            features = iterate_through_sentences(
                model,
                sentences_by_lang[lang],
                max_feature_nodes=None,
                throughput_threshold=tracing_params["throughput_threshold"],
                node_threshold=tracing_params["node_threshold"],
                edge_threshold=tracing_params["edge_threshold"],
                MAX_ITERATIONS=tracing_params["max_iterations"],
                threshold_first=tracing_params["threshold_first"],
                threshold_last=tracing_params["threshold_last"],
            )
        traced_by_lang[lang] = features
        if cache_path is not None:
            with open(cache_path, "w") as f:
                json.dump(features, f)
        torch.cuda.empty_cache()
        gc.collect()
    return traced_by_lang


def annotation_counts_from_traced(
    model_name: str,
    langs: list[str],
    traced_by_lang: dict[str, list[tuple[int, int]]],
    cache_dir: str | None,
    description_cache_path: str | None = None,
) -> dict[str, dict[str, int]]:
    """Run Neuronpedia identifier matching on traced features."""
    from lib.feature_extraction import choose_language_features, neuronpedia_description_cache_path

    feature_descriptions: dict[str, str] = {}
    counts_by_lang: dict[str, dict[str, int]] = {}
    desc_cache = description_cache_path or neuronpedia_description_cache_path(model_name)
    for lang in langs:
        cache_path = None
        if cache_dir is not None:
            cache_path = os.path.join(cache_dir, f"{lang}_features.json")
            if os.path.exists(cache_path):
                with open(cache_path, "r") as f:
                    counts_by_lang[lang] = json.load(f)
                print(f"  loaded cached annotation counts for {lang}")
                continue

            annotate_lock = f"{cache_path}.annotate.lock"
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            with open(annotate_lock, "w") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                if os.path.exists(cache_path):
                    with open(cache_path, "r") as f:
                        counts_by_lang[lang] = json.load(f)
                    print(f"  loaded cached annotation counts for {lang} (after wait)")
                    continue

                print(f"  annotation filtering for {lang}")
                lang_counts, feature_descriptions = choose_language_features(
                    traced_by_lang[lang],
                    identifiers[lang],
                    feature_descriptions,
                    model=model_name,
                    description_cache_path=desc_cache,
                )
                counts_by_lang[lang] = lang_counts
                with open(cache_path, "w") as f:
                    json.dump(lang_counts, f)
            continue

        print(f"  annotation filtering for {lang}")
        lang_counts, feature_descriptions = choose_language_features(
            traced_by_lang[lang],
            identifiers[lang],
            feature_descriptions,
            model=model_name,
            description_cache_path=desc_cache,
        )
        counts_by_lang[lang] = lang_counts
        if cache_path is not None:
            with open(cache_path, "w") as f:
                json.dump(lang_counts, f)
    return counts_by_lang


def select_from_counts(
    langs: list[str],
    counts_by_lang: dict[str, dict[str, int]],
    threshold: float,
) -> dict[str, list[str]]:
    selected: dict[str, list[str]] = {}
    for lang in langs:
        features_freq = counts_by_lang[lang]
        if not features_freq:
            selected[lang] = []
            continue
        max_val = max(features_freq.values())
        selected[lang] = [
            key for key, val in features_freq.items() if val >= max_val * threshold
        ]
    return selected


def selected_features_path(
    features_dir: str,
    axis_name: str,
    axis_value: float,
    lang: str | None = None,
) -> str:
    """Path to a saved AnnSel tracing feature-set JSON."""
    tracing_params = copy.deepcopy(DEFAULT_TRACING)
    tracing_params[axis_name] = axis_value
    stem = format_setting_filename(f"tracing_{axis_name}", tracing_params)
    if lang is not None:
        stem = f"{stem}__{lang}"
    return os.path.join(features_dir, "AnnSel", f"{stem}.json")


def cache_tag(tracing_params: dict[str, float | int]) -> str:
    parts = [
        f"tp{tracing_params['throughput_threshold']}",
        f"nd{tracing_params['node_threshold']}",
        f"ed{tracing_params['edge_threshold']}",
        f"tf{tracing_params['threshold_first']}",
        f"tl{tracing_params['threshold_last']}",
        f"mi{tracing_params['max_iterations']}",
    ]
    return "__".join(parts)


def sweep_tracing_axis(
    model: Any,
    model_name: str,
    langs: list[str],
    sentences_by_lang: dict[str, list[str]],
    axis_name: str,
    axis_values: list[float],
    sweep_root: str,
    features_dir: str,
    reuse_cache: bool,
    graph_cache_root: str | None = None,
    prune_workers: int = 1,
    description_cache_path: str | None = None,
) -> list[dict[str, Any]]:
    baseline_params = copy.deepcopy(DEFAULT_TRACING)
    baseline_cache_dir = os.path.join(sweep_root, "baseline", cache_tag(baseline_params))
    baseline_counts = annotation_counts_from_traced(
        model_name,
        langs,
        run_tracing_for_params(
            model,
            langs,
            sentences_by_lang,
            baseline_params,
            baseline_cache_dir if reuse_cache else None,
            graph_cache_root=graph_cache_root,
            prune_workers=prune_workers,
        ),
        baseline_cache_dir if reuse_cache else None,
        description_cache_path=description_cache_path,
    )
    baseline_selected = select_from_counts(
        langs, baseline_counts, DEFAULT_ANNSEL_THRESHOLD
    )

    results: list[dict[str, Any]] = []
    for value in axis_values:
        tracing_params = copy.deepcopy(DEFAULT_TRACING)
        tracing_params[axis_name] = value
        is_default = value == DEFAULT_TRACING[axis_name]
        setting_cache_dir = os.path.join(
            sweep_root, f"{axis_name}={value}", cache_tag(tracing_params)
        )

        traced = run_tracing_for_params(
            model,
            langs,
            sentences_by_lang,
            tracing_params,
            setting_cache_dir if reuse_cache else None,
            graph_cache_root=graph_cache_root,
            prune_workers=prune_workers,
        )
        counts = annotation_counts_from_traced(
            model_name,
            langs,
            traced,
            setting_cache_dir if reuse_cache else None,
            description_cache_path=description_cache_path,
        )
        selected = select_from_counts(langs, counts, DEFAULT_ANNSEL_THRESHOLD)

        raw_traced_total = sum(len(features) for features in traced.values())
        annotation_candidate_total = sum(len(counts[lang]) for lang in langs)

        entry: dict[str, Any] = {
            "sweep_axis": axis_name,
            "hyperparameters": tracing_params,
            "selection_threshold": DEFAULT_ANNSEL_THRESHOLD,
            "is_default": is_default,
            "raw_traced_features_total": raw_traced_total,
            "annotation_candidates_total": annotation_candidate_total,
            "selected_features_file": save_selected_features(
                features_dir,
                "AnnSel",
                model_name,
                f"tracing_{axis_name}",
                tracing_params,
                is_default,
                selected,
                extra={"selection_threshold": DEFAULT_ANNSEL_THRESHOLD},
                filename_suffix=langs[0] if len(langs) == 1 else None,
            ),
        }
        entry.update(
            summarize_feature_set(
                "AnnSel",
                selected,
                descriptions={},
                baseline_by_lang=baseline_selected if not is_default else None,
                annotation_selected=True,
            )
        )
        results.append(entry)
        print(
            f"[{axis_name}={value}] traced={raw_traced_total}, "
            f"annotated={annotation_candidate_total}, "
            f"selected={entry['total_features']}"
        )
    return results


def ensure_baseline_artifacts(
    model: Any,
    model_name: str,
    langs: list[str],
    sentences_by_lang: dict[str, list[str]],
    sweep_root: str,
    reuse_cache: bool,
    graph_cache_root: str | None = None,
    prune_workers: int = 1,
    description_cache_path: str | None = None,
) -> None:
    """Build baseline traced features + annotation counts once per language."""
    baseline_params = copy.deepcopy(DEFAULT_TRACING)
    baseline_cache_dir = os.path.join(sweep_root, "baseline", cache_tag(baseline_params))
    print("Ensuring baseline traced features and annotation counts")
    annotation_counts_from_traced(
        model_name,
        langs,
        run_tracing_for_params(
            model,
            langs,
            sentences_by_lang,
            baseline_params,
            baseline_cache_dir if reuse_cache else None,
            graph_cache_root=graph_cache_root,
            prune_workers=prune_workers,
        ),
        baseline_cache_dir if reuse_cache else None,
        description_cache_path=description_cache_path,
    )
    print("Baseline artifacts ready")


def merge_axis_results(
    existing: list[dict[str, Any]],
    new: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge partial axis runs keyed by full hyperparameter dict."""
    merged: dict[tuple[tuple[str, Any], ...], dict[str, Any]] = {}
    for entry in existing + new:
        key = tuple(sorted(entry["hyperparameters"].items()))
        merged[key] = entry
    return list(merged.values())


def default_report_template(
    model_name: str,
    langs: list[str],
    num_sentences: int,
    features_dir: str,
) -> dict[str, Any]:
    return {
        "model": model_name,
        "languages": langs,
        "num_sentences": num_sentences,
        "tracing_defaults": DEFAULT_TRACING,
        "selection_threshold": DEFAULT_ANNSEL_THRESHOLD,
        "selected_features_dir": features_dir,
        "sweeps": {},
    }


def load_report_json(json_path: str, default: dict[str, Any]) -> dict[str, Any]:
    """Load report JSON under an exclusive lock."""
    lock_path = f"{json_path}.lock"
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                report = json.load(f)
            report.setdefault("sweeps", {})
            return report
        return copy.deepcopy(default)


def save_report_json(json_path: str, report: dict[str, Any]) -> None:
    """Atomically save report JSON under an exclusive lock."""
    lock_path = f"{json_path}.lock"
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        tmp_path = f"{json_path}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(report, f, indent=2)
        os.replace(tmp_path, json_path)


def merge_report_sweep_results(
    json_path: str,
    default: dict[str, Any],
    sweep_key: str,
    axis_results: list[dict[str, Any]],
) -> None:
    """Load, merge one sweep axis, and save report JSON atomically."""
    lock_path = f"{json_path}.lock"
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                report = json.load(f)
            report.setdefault("sweeps", {})
        else:
            report = copy.deepcopy(default)
        prior = report["sweeps"].get(sweep_key, [])
        report["sweeps"][sweep_key] = merge_axis_results(prior, axis_results)
        tmp_path = f"{json_path}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(report, f, indent=2)
        os.replace(tmp_path, json_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AnnSel circuit-tracing hyperparameter sensitivity (GPU)."
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="gemma-2-2b",
        choices=hf_model_names.keys(),
    )
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        choices=lang_to_flores_key.keys(),
        help="Optional single language (recommended for smoke tests)",
    )
    parser.add_argument(
        "--num-sentences",
        type=int,
        default=100,
        help="FLORES sentences per language (use fewer for quick tests)",
    )
    parser.add_argument(
        "--axes",
        type=str,
        default="all",
        help="Comma-separated tracing axes to sweep, or 'all'",
    )
    parser.add_argument(
        "--axis-values",
        type=str,
        default=None,
        help="Comma-separated values for a single-axis sweep (requires one --axes entry)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Output directory for sensitivity results",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache traced/annotation artifacts (default: <output-dir>/cache)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching of per-setting traced features",
    )
    parser.add_argument(
        "--no-graph-cache",
        action="store_true",
        help="Disable shared attribution-graph cache (re-run attribute per setting)",
    )
    parser.add_argument(
        "--prune-workers",
        type=int,
        default=1,
        help="CPU workers for parallel path pruning from cached graphs",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Only build baseline traced features + annotation cache, then exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sweep grid and exit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_name = args.model
    langs = [args.lang] if args.lang else list(lang_to_flores_key.keys())
    output_dir = args.output_dir or os.path.join(
        repo_data_dir(), "additional_experiments", model_name, "annsel_tracing_sensitivity"
    )
    cache_root = None if args.no_cache else (args.cache_dir or os.path.join(output_dir, "cache"))
    graph_cache_root = None
    if cache_root is not None and not args.no_graph_cache:
        graph_cache_root = os.path.join(cache_root, "attribution_graphs")
    features_dir = os.path.join(output_dir, "selected_features")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)

    selected_axes = (
        list(TRACING_SWEEPS.keys())
        if args.axes == "all"
        else [axis.strip() for axis in args.axes.split(",")]
    )
    for axis in selected_axes:
        if axis not in TRACING_SWEEPS:
            raise ValueError(f"Unknown axis '{axis}'. Options: {list(TRACING_SWEEPS)}")

    axis_value_overrides: dict[str, list[float]] | None = None
    if args.axis_values is not None:
        if len(selected_axes) != 1:
            raise ValueError("--axis-values requires exactly one axis in --axes")
        axis_value_overrides = {
            selected_axes[0]: [float(v.strip()) for v in args.axis_values.split(",")]
        }

    if args.dry_run:
        print("AnnSel tracing sweep plan:")
        print(f"  model={model_name}, langs={langs}, num_sentences={args.num_sentences}")
        print(f"  defaults={DEFAULT_TRACING}")
        print(f"  selection_threshold={DEFAULT_ANNSEL_THRESHOLD}")
        if args.baseline_only:
            print("  mode=baseline-only")
        for axis in selected_axes:
            values = (
                axis_value_overrides[axis]
                if axis_value_overrides is not None
                else TRACING_SWEEPS[axis]
            )
            print(f"  {axis}: {values}")
        return

    if args.baseline_only and args.axis_values is not None:
        raise ValueError("--baseline-only cannot be combined with --axis-values")

    print(f"Loading model {model_name}")
    import torch
    from lib.circuit_tracer_import import ReplacementModel
    from lib.device_setup import device
    from lib.models import hf_transcoder_names

    model = None
    sentences_by_lang: dict[str, list[str]]
    if (
        graph_cache_root is not None
        and graph_cache_complete(graph_cache_root, langs, args.num_sentences)
    ):
        print(f"Graph cache complete under {graph_cache_root}; skipping GPU model load")
        sentences_by_lang = load_sentences_from_manifest(
            langs, graph_cache_root, args.num_sentences
        )
    else:
        transcoder_name = hf_transcoder_names[model_name]
        model = ReplacementModel.from_pretrained(
            hf_model_names[model_name],
            transcoder_name,
            device=device,
            dtype=torch.bfloat16,
        )
        sentences_by_lang = load_sentences(langs, model, args.num_sentences)

    if graph_cache_root is not None:
        print(f"Ensuring attribution graph cache under {graph_cache_root}")
        if model is None:
            print("  graph cache already complete; nothing to build")
        else:
            ensure_graph_caches_for_langs(model, langs, sentences_by_lang, graph_cache_root)

    lang_tag = langs[0] if len(langs) == 1 else "all"
    json_path = os.path.join(output_dir, f"annsel_tracing_sensitivity_{model_name}_{lang_tag}.json")
    csv_path = os.path.join(output_dir, f"annsel_tracing_sensitivity_{model_name}_{lang_tag}.csv")
    report_defaults = default_report_template(
        model_name, langs, args.num_sentences, features_dir
    )

    from lib.feature_extraction import neuronpedia_description_cache_path

    # Shared across all AnnSel jobs: layer.feature_idx -> Neuronpedia description text.
    description_cache_path = neuronpedia_description_cache_path(model_name)

    if args.baseline_only:
        ensure_baseline_artifacts(
            model=model,
            model_name=model_name,
            langs=langs,
            sentences_by_lang=sentences_by_lang,
            sweep_root=cache_root or os.path.join(output_dir, "tmp_cache"),
            reuse_cache=cache_root is not None,
            graph_cache_root=graph_cache_root,
            prune_workers=max(1, args.prune_workers),
            description_cache_path=description_cache_path,
        )
        print("Baseline-only run complete")
        return

    for axis in selected_axes:
        print(f"\n=== Sweeping {axis} ===")
        axis_values = (
            axis_value_overrides[axis]
            if axis_value_overrides is not None
            else TRACING_SWEEPS[axis]
        )
        axis_results = sweep_tracing_axis(
            model=model,
            model_name=model_name,
            langs=langs,
            sentences_by_lang=sentences_by_lang,
            axis_name=axis,
            axis_values=axis_values,
            sweep_root=cache_root or os.path.join(output_dir, "tmp_cache"),
            features_dir=features_dir,
            reuse_cache=cache_root is not None,
            graph_cache_root=graph_cache_root,
            prune_workers=max(1, args.prune_workers),
            description_cache_path=description_cache_path,
        )
        sweep_key = f"AnnSel_tracing_{axis}"
        merge_report_sweep_results(
            json_path, report_defaults, sweep_key, axis_results
        )

    report = load_report_json(json_path, report_defaults)
    csv_rows = []
    for sweep_key, axis_results in report["sweeps"].items():
        sweep_name = sweep_key.removeprefix("AnnSel_tracing_")
        csv_rows.extend(flatten_sweep_rows("AnnSel", f"tracing_{sweep_name}", axis_results))

    csv_fieldnames = [
        "method",
        "sweep",
        "is_default",
        "throughput_threshold",
        "node_threshold",
        "edge_threshold",
        "threshold_first",
        "threshold_last",
        "max_iterations",
        "total_features",
        "mean_features_per_lang",
        "mean_lang_name_match_rate",
        "mean_cross_lang_jaccard",
        "mean_jaccard_vs_default",
    ]
    write_csv(csv_path, csv_rows, csv_fieldnames)

    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Selected features saved under {features_dir}")


if __name__ == "__main__":
    main()
