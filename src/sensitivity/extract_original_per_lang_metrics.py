"""Extract per-language metrics from precomputed original (default) intervention artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Any


def repo_data_dir() -> str:
    from lib.paths import data_dir_str
    return data_dir_str()




def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def load_antonym_margins(
    model_name: str,
    intervention_type: str,
    langs: list[str],
) -> dict[str, dict[str, float]]:
    """On-language delta in top-1 logit margin from all_langs CSVs."""
    base = os.path.join(repo_data_dir(), "all_langs_intervention_logit_change", model_name)
    method_map = {"desc": "AnnSel", "val": "ValSel", "freq": "FreqSel"}
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)

    if intervention_type == "feature-intervention":
        for pl in langs:
            for al in langs:
                if pl == al:
                    continue
                csv_path = os.path.join(
                    base,
                    pl,
                    al,
                    "all_langs_intervention_logit_change__intervention_all__measure_all.csv",
                )
                if not os.path.exists(csv_path):
                    continue
                with open(csv_path, newline="", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        if row["prompt_lang"] != pl or row["adj_lang"] != al:
                            continue
                        if row["measure_lang"] != al:
                            continue
                        if row["ablation_method"] == row["amplification_method"]:
                            continue
                        method = method_map[row["ablation_method"]]
                        grouped[(method, al)].append(float(row["delta_target_minus_base"]))
        return {
            method: {
                lang: mean(grouped[(method, lang)])
                for lang in langs
                if grouped[(method, lang)]
            }
            for method in ["AnnSel", "ValSel", "FreqSel"]
        }

    if intervention_type not in {"distractor ablation", "amplification"}:
        return {method: {} for method in ["AnnSel", "ValSel", "FreqSel"]}

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
            for row in csv.DictReader(f):
                if row["prompt_lang"] != pl or row["adj_lang"] != pl:
                    continue
                if row["intervention_lang"] != pl or row["measure_lang"] != pl:
                    continue
                ab = row["ablation_method"]
                amp = row["amplification_method"]
                if intervention_type == "distractor ablation":
                    if ab != amp:
                        continue
                elif intervention_type == "amplification":
                    # Pure amplification is not stored in all_langs CSVs (they always combine
                    # ablation + amplification). Leave empty; compute via GPU re-evaluation.
                    continue
                method = method_map[ab]
                grouped[(method, pl)].append(float(row["delta_target_minus_base"]))

    return {
        method: {lang: mean(grouped[(method, lang)]) for lang in langs if grouped[(method, lang)]}
        for method in ["AnnSel", "ValSel", "FreqSel"]
    }


def load_enumeration_norm_logprob(
    model_name: str,
    intervention_type: str,
    langs: list[str],
) -> dict[str, dict[str, float]]:
    """On-language delta in normalized target-sequence logprob."""
    base = os.path.join(repo_data_dir(), "interventions_multiple_words", model_name)
    method_files = {
        "AnnSel": "interventions_and_results_description_normalized.json",
        "ValSel": "interventions_and_results_value_normalized.json",
        "FreqSel": "interventions_and_results_frequency_normalized.json",
    }
    exp_key = (
        "non-distractor amplification"
        if intervention_type == "amplification"
        else intervention_type
    )

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
                        if pl not in post or ll not in post[pl]:
                            continue
                        post_block = post[pl][ll]
                    else:
                        if ll not in post:
                            continue
                        post_block = post[ll]
                    for ml in langs:
                        if pl != ll or ml != ll:
                            continue
                        if ml not in orig or ml not in post_block:
                            continue
                        cand = next(iter(orig[ml].keys()))
                        if cand not in post_block[ml]:
                            continue
                        per_lang[ml].append(post_block[ml][cand] - orig[ml][cand])
        results[method] = {lang: mean(vals) for lang, vals in sorted(per_lang.items())}
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma-2-2b")
    parser.add_argument(
        "--intervention-types",
        nargs="+",
        default=["distractor ablation", "amplification", "feature-intervention"],
    )
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    langs = ["en", "fr", "de", "es", "zh", "ja", "ko"]
    rows: list[dict[str, Any]] = []

    for intervention_type in args.intervention_types:
        ant = load_antonym_margins(args.model, intervention_type, langs)
        enum = load_enumeration_norm_logprob(args.model, intervention_type, langs)
        for method in ["AnnSel", "ValSel", "FreqSel"]:
            for lang in langs:
                rows.append(
                    {
                        "budget": "original_default",
                        "method": method,
                        "intervention_type": intervention_type,
                        "language": lang,
                        "antonym_delta_top1_margin": ant.get(method, {}).get(lang),
                        "enumeration_delta_norm_logprob": enum.get(method, {}).get(lang),
                    }
                )

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    fieldnames = [
        "budget",
        "method",
        "intervention_type",
        "language",
        "antonym_delta_top1_margin",
        "enumeration_delta_norm_logprob",
    ]
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
