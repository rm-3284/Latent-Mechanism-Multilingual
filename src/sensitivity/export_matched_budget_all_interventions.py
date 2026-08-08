"""
Export all_interventions.csv-style summaries for matched-budget feature sets.

Antonyms (all_interventions*.csv):
  mean = post-intervention top-1 logit margin (target_logit - base_logit)

Enumerations (all_enumerations_normalized*.csv):
  mean = post-intervention normalized target-sequence logprob (total logprob / #tokens),
  using the same normalization as data/interventions_multiple_words/gemma-2-2b/average.py

Both use columns: Experiment, Method, Run, mean, stdev
Subtract the `original` row from intervention rows for delta metrics.

Example:
  python export_matched_budget_all_interventions.py --benchmark both \\
    --features-file ../data/.../AnnSel/min_default_counts__...json

  python export_matched_budget_all_interventions.py --merge-methods \\
    --input-dir ../data/.../min_default_counts/en/de

  # Hyperparameter sensitivity (one feature set -> per-lang-pair all_interventions.csv):
  sbatch jobscripts/run_export_sensitivity_all_interventions.sh \\
    --features-file data/additional_experiments/gemma-2-2b/selected_features/FreqSel/example_thres__....json
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
from collections import defaultdict
from typing import Any

import torch
from tqdm import tqdm

from lib.ablation_amplification_intervention import (
    model_intervention,
    model_run,
)
from lib.circuit_tracer_import import ReplacementModel
from lib.device_setup import device
from sensitivity.evaluate_selected_features import (
    build_cross_lang_intervention,
    build_enumeration_prompt,
    build_intervention_maps,
    feature_set_id,
    load_feature_payload,
    resolve_repo_relative_path,
)
from sensitivity.hyperparameter_sensitivity import repo_data_dir
from lib.intervention import _first_content_token_id, get_best_base
from lib.models import hf_model_names, hf_transcoder_names
from pipeline.multiple_words_intervention import (
    categories,
    get_logprob_with_intervention,
    get_logprobs_for_candidates,
)
from lib.pipeline_data.adjectives import big_data
from lib.template import base_strings, lang_to_flores_key


EXPERIMENTS = [
    "original",
    "distractor ablation",
    "amplification",
    "feature-intervention",
]

CSV_PREFIXES = {
    "antonyms": ("all_interventions", "all_interventions.csv"),
    "enumerations": ("all_enumerations_normalized", "all_enumerations_normalized.csv"),
}


def norm_denom(model, candidate: str) -> int:
    tokenized = model.tokenizer(candidate, return_tensors="pt")
    return max(int(tokenized.input_ids.shape[1]) - 1, 1)


def normalized_target_logprob(
    prompt: str,
    ans: dict[str, str],
    measure_lang: str,
    model,
    interventions: list[tuple[int, int, int, float]] | None = None,
) -> float:
    candidate = ans[measure_lang]
    if interventions:
        logprob = get_logprob_with_intervention(prompt, candidate, model, interventions)
    else:
        logprob = get_logprobs_for_candidates(prompt, ans, model)[measure_lang][candidate]
    return logprob / norm_denom(model, candidate)


def post_margin(
    baseline_logits: torch.Tensor,
    logits: torch.Tensor,
    ans: dict[str, list[str]],
    adj_lang: str,
    measure_lang: str,
    model,
) -> float:
    base_word = get_best_base(baseline_logits, ans[adj_lang], model)
    target_word = get_best_base(logits, ans[measure_lang], model)
    n_logits = logits.squeeze(0)[-1]
    base_id = _first_content_token_id(model, base_word)
    target_id = _first_content_token_id(model, target_word)
    margin = n_logits[target_id] - n_logits[base_id]
    return margin.item() if isinstance(margin, torch.Tensor) else float(margin)


def collect_margins_for_pair(
    model,
    ablations: dict[str, list[tuple[int, int, int, float]]],
    amplifications: dict[str, list[tuple[int, int, int, float]]],
    langs: list[str],
    prompt_lang: str,
    adj_lang: str,
    max_items: int | None,
) -> dict[str, dict[str, list[float]]]:
    grouped: dict[str, dict[str, list[float]]] = {
        experiment: {lang: [] for lang in langs} for experiment in EXPERIMENTS
    }
    dataset = big_data if max_items is None else big_data[:max_items]
    base = base_strings[prompt_lang]

    for _sample_idx, (adj, ans) in enumerate(dataset):
        if adj_lang not in adj:
            continue
        prompt = base.format(adj=adj[adj_lang])
        _, baseline_logits = model_run(prompt, model)

        for measure_lang in langs:
            if measure_lang not in ans:
                continue
            grouped["original"][measure_lang].append(
                post_margin(baseline_logits, baseline_logits, ans, adj_lang, measure_lang, model)
            )

        for intervention_type in ("distractor ablation", "amplification"):
            for intervention_lang in langs:
                interventions = (
                    ablations[intervention_lang]
                    if intervention_type == "distractor ablation"
                    else amplifications[intervention_lang]
                )
                _, post_logits = model_intervention(prompt, model, interventions)
                for measure_lang in langs:
                    if measure_lang not in ans:
                        continue
                    if intervention_lang != measure_lang:
                        continue
                    grouped[intervention_type][measure_lang].append(
                        post_margin(
                            baseline_logits, post_logits, ans, adj_lang, measure_lang, model
                        )
                    )

        if prompt_lang != adj_lang:
            interventions = build_cross_lang_intervention(
                ablations, amplifications, prompt_lang, adj_lang
            )
            _, post_logits = model_intervention(prompt, model, interventions)
            for measure_lang in langs:
                if measure_lang not in ans:
                    continue
                grouped["feature-intervention"][measure_lang].append(
                    post_margin(
                        baseline_logits, post_logits, ans, adj_lang, measure_lang, model
                    )
                )

    return grouped


def collect_enumeration_norm_logprobs_for_pair(
    model,
    ablations: dict[str, list[tuple[int, int, int, float]]],
    amplifications: dict[str, list[tuple[int, int, int, float]]],
    langs: list[str],
    prompt_lang: str,
    list_lang: str,
) -> dict[str, dict[str, list[float]]]:
    grouped: dict[str, dict[str, list[float]]] = {
        experiment: {lang: [] for lang in langs} for experiment in EXPERIMENTS
    }

    for category_key in categories:
        prompt, ans = build_enumeration_prompt(prompt_lang, list_lang, category_key)

        for measure_lang in langs:
            if measure_lang not in ans:
                continue
            grouped["original"][measure_lang].append(
                normalized_target_logprob(prompt, ans, measure_lang, model)
            )

        for intervention_type in ("distractor ablation", "amplification"):
            for intervention_lang in langs:
                interventions = (
                    ablations[intervention_lang]
                    if intervention_type == "distractor ablation"
                    else amplifications[intervention_lang]
                )
                for measure_lang in langs:
                    if measure_lang not in ans:
                        continue
                    if intervention_lang != measure_lang:
                        continue
                    grouped[intervention_type][measure_lang].append(
                        normalized_target_logprob(
                            prompt, ans, measure_lang, model, interventions
                        )
                    )

        if prompt_lang != list_lang:
            interventions = build_cross_lang_intervention(
                ablations, amplifications, prompt_lang, list_lang
            )
            for measure_lang in langs:
                if measure_lang not in ans:
                    continue
                grouped["feature-intervention"][measure_lang].append(
                    normalized_target_logprob(
                        prompt, ans, measure_lang, model, interventions
                    )
                )

    return grouped


def aggregate_rows(
    grouped: dict[str, dict[str, list[float]]],
    method: str,
    langs: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for experiment in EXPERIMENTS:
        for lang in langs:
            values = grouped[experiment][lang]
            if not values:
                continue
            rows.append(
                {
                    "Experiment": experiment,
                    "Method": method,
                    "Run": lang,
                    "mean": statistics.fmean(values),
                    "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "n": len(values),
                }
            )
    return rows


def write_summary_csv(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["Experiment", "Method", "Run", "mean", "stdev"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def merge_method_csvs(
    input_dir: str,
    benchmark: str,
) -> str:
    partial_prefix, output_name = CSV_PREFIXES[benchmark]
    partial_paths = sorted(
        os.path.join(input_dir, name)
        for name in os.listdir(input_dir)
        if name.startswith(f"{partial_prefix}_") and name.endswith(".csv")
    )
    if not partial_paths:
        raise FileNotFoundError(
            f"No partial {partial_prefix}_*.csv files in {input_dir}"
        )

    merged: list[dict[str, Any]] = []
    for path in partial_paths:
        with open(path, newline="", encoding="utf-8") as f:
            merged.extend(csv.DictReader(f))

    def sort_key(row: dict[str, str]) -> tuple[int, int, str]:
        exp_order = {name: idx for idx, name in enumerate(EXPERIMENTS)}
        method_order = {"AnnSel": 0, "ValSel": 1, "FreqSel": 2}
        return (
            exp_order.get(row["Experiment"], 99),
            method_order.get(row["Method"], 99),
            row["Run"],
        )

    merged.sort(key=sort_key)
    out_path = os.path.join(input_dir, output_name)
    write_summary_csv(out_path, merged)
    return out_path


def merge_all_benchmarks(input_dir: str) -> list[str]:
    outputs: list[str] = []
    for benchmark in CSV_PREFIXES:
        if not any(
            name.startswith(f"{CSV_PREFIXES[benchmark][0]}_") and name.endswith(".csv")
            for name in os.listdir(input_dir)
        ):
            continue
        outputs.append(merge_method_csvs(input_dir, benchmark))
    if not outputs:
        raise FileNotFoundError(f"No partial export CSVs found in {input_dir}")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export matched-budget all_interventions.csv files.")
    parser.add_argument("--model", "-m", default="gemma-2-2b", choices=hf_model_names.keys())
    parser.add_argument("--features-file", default=None)
    parser.add_argument("--method", default=None, choices=["AnnSel", "ValSel", "FreqSel"])
    parser.add_argument("--prompt-lang", default=None, choices=lang_to_flores_key.keys())
    parser.add_argument("--adj-lang", default=None, choices=lang_to_flores_key.keys())
    parser.add_argument("--list-lang", default=None, choices=lang_to_flores_key.keys())
    parser.add_argument(
        "--benchmark",
        choices=["antonyms", "enumerations", "both"],
        default="both",
    )
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Root output dir (default: data/additional_experiments/<model>/budget_matched_interventions/min_default_counts)",
    )
    parser.add_argument(
        "--merge-methods",
        action="store_true",
        help="Merge partial method CSVs in --input-dir (antonyms and/or enumerations)",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory containing partial method CSVs (for --merge-methods)",
    )
    parser.add_argument(
        "--nest-under-feature-id",
        action="store_true",
        help="Write under <output-dir>/<feature_set_id>/prompt_lang/pair_lang/",
    )
    parser.add_argument(
        "--write-merged-csv",
        action="store_true",
        help="Write all_interventions.csv (not all_interventions_<Method>.csv)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip prompt/pair langs whose output CSVs already exist (resume partial runs)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.merge_methods:
        if args.input_dir is None:
            raise ValueError("--input-dir is required with --merge-methods")
        for out_path in merge_all_benchmarks(args.input_dir):
            print(f"Wrote merged {out_path}")
        return

    if args.features_file is None:
        raise ValueError("--features-file is required unless --merge-methods is set")

    payload = load_feature_payload(resolve_repo_relative_path(args.features_file))
    method = args.method or payload.get("method")
    if method is None:
        raise ValueError("Could not determine method; pass --method")

    langs = list(lang_to_flores_key.keys())
    prompt_langs = [args.prompt_lang] if args.prompt_lang else langs
    pair_langs = (
        [args.adj_lang]
        if args.adj_lang
        else ([args.list_lang] if args.list_lang else langs)
    )
    run_antonyms = args.benchmark in {"antonyms", "both"}
    run_enumerations = args.benchmark in {"enumerations", "both"}

    output_root = args.output_dir or os.path.join(
        repo_data_dir(),
        "additional_experiments",
        args.model,
        "budget_matched_interventions",
        "min_default_counts",
    )
    features_path = resolve_repo_relative_path(args.features_file)
    if args.nest_under_feature_id:
        set_id = feature_set_id(payload, features_path)
        output_root = os.path.join(output_root, set_id)

    antonyms_csv_name = (
        CSV_PREFIXES["antonyms"][1]
        if args.write_merged_csv
        else f"{CSV_PREFIXES['antonyms'][0]}_{method}.csv"
    )
    enumerations_csv_name = (
        CSV_PREFIXES["enumerations"][1]
        if args.write_merged_csv
        else f"{CSV_PREFIXES['enumerations'][0]}_{method}.csv"
    )

    print(f"Loading model {args.model}")
    model = ReplacementModel.from_pretrained(
        hf_model_names[args.model],
        hf_transcoder_names[args.model],
        device=device,
        dtype=torch.bfloat16,
    )

    amplification_values_directory = os.path.join(
        repo_data_dir(), "amplification_values", args.model
    )
    ablations, amplifications = build_intervention_maps(
        payload["selected_features"], amplification_values_directory, langs
    )

    for prompt_lang in tqdm(prompt_langs, desc="prompt_lang"):
        for pair_lang in tqdm(pair_langs, desc="pair_lang", leave=False):
            antonyms_path = os.path.join(
                output_root, prompt_lang, pair_lang, antonyms_csv_name
            )
            enumerations_path = os.path.join(
                output_root, prompt_lang, pair_lang, enumerations_csv_name
            )
            need_antonyms = run_antonyms and not (
                args.skip_existing and os.path.exists(antonyms_path)
            )
            need_enumerations = run_enumerations and not (
                args.skip_existing and os.path.exists(enumerations_path)
            )
            if not need_antonyms and not need_enumerations:
                print(f"Skipping existing {prompt_lang}/{pair_lang}")
                continue

            if need_antonyms:
                grouped = collect_margins_for_pair(
                    model=model,
                    ablations=ablations,
                    amplifications=amplifications,
                    langs=langs,
                    prompt_lang=prompt_lang,
                    adj_lang=pair_lang,
                    max_items=args.max_items,
                )
                rows = aggregate_rows(grouped, method, langs)
                write_summary_csv(antonyms_path, rows)
                print(f"Wrote {antonyms_path} ({len(rows)} rows)")

            if need_enumerations:
                grouped = collect_enumeration_norm_logprobs_for_pair(
                    model=model,
                    ablations=ablations,
                    amplifications=amplifications,
                    langs=langs,
                    prompt_lang=prompt_lang,
                    list_lang=pair_lang,
                )
                rows = aggregate_rows(grouped, method, langs)
                write_summary_csv(enumerations_path, rows)
                print(f"Wrote {enumerations_path} ({len(rows)} rows)")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
