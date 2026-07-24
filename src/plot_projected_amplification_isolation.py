"""Scatter plots for projected amplification isolation CSVs.

Style is aligned with all_langs_intervention_logit_change analysis plots:
- one x-axis of intervention conditions
- colored points per measure language with small x-offsets
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import statistics
from collections import defaultdict
from typing import Optional

import matplotlib.pyplot as plt

METHOD_ORDER = ["desc", "val", "freq"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate projected_amplification_isolation CSVs and write bar plots: "
            "original logit gap, mean amplification delta per method, and isolated "
            "(projected) deltas for each (amplification_method, projection_method) pair."
        )
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Path to repo data directory. Defaults to <repo>/data.",
    )
    parser.add_argument("--model", type=str, default=None, help="Optional model filter.")
    parser.add_argument("--prompt-lang", type=str, default=None)
    parser.add_argument("--adj-lang", type=str, default=None)
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help="Defaults to data/projected_amplification_isolation/analysis/plots",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default=None,
        help="Defaults to data/projected_amplification_isolation/analysis/plot_summary.csv",
    )
    return parser.parse_args()


def repo_root_from_file() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_data_root(arg: Optional[str]) -> str:
    if arg is not None:
        return os.path.abspath(arg)
    return os.path.join(repo_root_from_file(), "data")


def parse_prompt_adj_from_filename(path: str) -> tuple[str, str]:
    base = os.path.basename(path).replace(".csv", "")
    if "__adj_" not in base:
        raise ValueError(f"Unexpected CSV name: {path}")
    left, right = base.split("__adj_", 1)
    if not left.startswith("prompt_"):
        raise ValueError(f"Unexpected CSV name: {path}")
    prompt_lang = left[len("prompt_") :]
    adj_token = right
    return prompt_lang, adj_token


def load_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def filtered_rows(rows: list[dict[str, str]], prompt_lang: str, adj_lang: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        if (row.get("prompt_lang") or "").strip() != prompt_lang:
            continue
        if (row.get("adj_lang") or "").strip() != adj_lang:
            continue
        if (row.get("intervention_lang") or "").strip() != adj_lang:
            continue
        if (row.get("projection_lang") or "").strip() != adj_lang:
            continue
        out.append(row)
    return out


def fmean(xs: list[float]) -> float:
    finite = [x for x in xs if math.isfinite(x)]
    if not finite:
        return float("nan")
    return statistics.fmean(finite)


def aggregate_block_by_measure_lang(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    """Return measure_lang -> {original, amp_*, iso_*} means."""
    vals: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    amp_deltas: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    iso_pairs: dict[str, dict[tuple[str, str], list[float]]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        try:
            o = float(row["original_target_minus_base"])
            a_delta = float(row["amplified_minus_original"])
            i_delta = float(row["isolated_minus_original"])
        except (KeyError, ValueError):
            continue
        measure_lang = (row.get("measure_lang") or "").strip()
        if not measure_lang:
            continue
        amp = (row.get("amplification_method") or "").strip()
        proj = (row.get("projection_method") or "").strip()
        if amp not in METHOD_ORDER or proj not in METHOD_ORDER:
            continue
        vals[measure_lang]["original"].append(o)
        amp_deltas[measure_lang][amp].append(a_delta)
        iso_pairs[measure_lang][(amp, proj)].append(i_delta)

    out: dict[str, dict[str, float]] = {}
    for measure_lang in sorted(vals.keys()):
        per_lang: dict[str, float] = {}
        per_lang["original"] = fmean(vals[measure_lang]["original"])
        for m in METHOD_ORDER:
            per_lang[f"amp_{m}"] = fmean(amp_deltas[measure_lang][m])
        for a in METHOD_ORDER:
            for p in METHOD_ORDER:
                per_lang[f"iso_{a}__proj_{p}"] = fmean(iso_pairs[measure_lang][(a, p)])
        out[measure_lang] = per_lang
    return out


def plot_order_labels() -> tuple[list[str], list[str]]:
    labels: list[str] = ["original"]
    keys: list[str] = ["original"]
    for m in METHOD_ORDER:
        labels.append(f"target_amp:{m}")
        keys.append(f"amp_{m}")
    for a in METHOD_ORDER:
        for p in METHOD_ORDER:
            labels.append(f"projected\namp={a}\nremove={p}")
            keys.append(f"iso_{a}__proj_{p}")
    return keys, labels


def make_figure(
    title: str,
    keys: list[str],
    labels: list[str],
    metrics_by_lang: dict[str, dict[str, float]],
) -> None:
    x = list(range(len(keys)))
    measure_lang_order = ["en", "fr", "de", "es", "zh", "ja", "ko"]
    language_colors = {
        "en": "#1f77b4",
        "fr": "#ff7f0e",
        "de": "#2ca02c",
        "es": "#d62728",
        "zh": "#9467bd",
        "ja": "#8c564b",
        "ko": "#e377c2",
    }
    offsets = {
        "en": -0.24,
        "fr": -0.16,
        "de": -0.08,
        "es": 0.0,
        "zh": 0.08,
        "ja": 0.16,
        "ko": 0.24,
    }

    fig, ax = plt.subplots(figsize=(18, 6))
    present_langs = [lang for lang in measure_lang_order if lang in metrics_by_lang]
    for measure_lang in present_langs:
        values_by_label = metrics_by_lang.get(measure_lang, {})
        y_vals = [values_by_label.get(label, float("nan")) for label in keys]
        x_vals = [idx + offsets[measure_lang] for idx in x]
        ax.scatter(
            x_vals,
            y_vals,
            color=language_colors[measure_lang],
            s=45,
            label=measure_lang,
            zorder=3,
        )

    ax.axhline(0, color="black", linewidth=1)
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.set_ylabel("Target-Base Logit Difference")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Method Pair")
    if present_langs:
        ax.legend(title="measure_lang", ncol=2, fontsize=9)
    fig.suptitle(title, y=0.99)
    fig.tight_layout()


def main() -> None:
    args = parse_args()
    data_root = get_data_root(args.data_root)
    iso_dir = os.path.join(data_root, "projected_amplification_isolation")
    plot_dir = args.plot_dir or os.path.join(iso_dir, "analysis", "plots")
    summary_csv = args.summary_csv or os.path.join(iso_dir, "analysis", "plot_summary.csv")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(summary_csv)), exist_ok=True)

    pattern = os.path.join(iso_dir, "*", "prompt_*__adj_*.csv")
    # Use only explicit prompt/adj pair files; ignore bundled adj_all inputs.
    paths = sorted(p for p in glob.glob(pattern) if "__adj_all" not in p)
    if args.model:
        paths = [p for p in paths if p.split(os.sep)[-2] == args.model]
    if args.prompt_lang:
        paths = [p for p in paths if f"prompt_{args.prompt_lang}__" in os.path.basename(p)]

    keys, labels = plot_order_labels()
    summary_rows: list[dict[str, object]] = []
    plot_count = 0

    for path in paths:
        model = path.split(os.sep)[-2]
        prompt_token, adj_token = parse_prompt_adj_from_filename(path)
        rows = load_rows(path)

        if adj_token == "all":
            continue
        adj_langs: list[str] = [adj_token]

        for adj_lang in adj_langs:
            if args.adj_lang and adj_lang != args.adj_lang:
                continue
            block = filtered_rows(rows, prompt_token, adj_lang)
            if not block:
                continue
            metrics_by_lang = aggregate_block_by_measure_lang(block)
            if not metrics_by_lang:
                continue
            title = f"{model} | prompt={prompt_token} | adj={adj_lang}"
            make_figure(title, keys, labels, metrics_by_lang)
            model_dir = os.path.join(plot_dir, model)
            os.makedirs(model_dir, exist_ok=True)
            out_png = os.path.join(model_dir, f"prompt_{prompt_token}__adj_{adj_lang}.png")
            plt.savefig(out_png, dpi=150)
            plt.close()
            plot_count += 1

            for measure_lang, metrics in metrics_by_lang.items():
                for k in keys:
                    summary_rows.append(
                        {
                            "model": model,
                            "prompt_lang": prompt_token,
                            "adj_lang": adj_lang,
                            "measure_lang": measure_lang,
                            "series": k,
                            "mean_value": metrics.get(k, float("nan")),
                            "source_csv": path,
                        }
                    )

    fieldnames = ["model", "prompt_lang", "adj_lang", "measure_lang", "series", "mean_value", "source_csv"]
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)

    print(f"Data root: {data_root}")
    print(f"Plots written: {plot_count}")
    print(f"Plot directory: {plot_dir}")
    print(f"Summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
