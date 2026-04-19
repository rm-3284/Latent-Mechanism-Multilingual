import argparse
import csv
import glob
import math
import os
import statistics
from collections import defaultdict
from typing import Optional

import matplotlib.pyplot as plt


METHOD_TO_INTERVENTION = {
    "desc": "AnnSel",
    "val": "ValSel",
    "freq": "FreqSel",
}

METHOD_ORDER = ["desc", "val", "freq"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare adj_lang_method_logit_change deltas to aggregated intervention "
            "metrics (default experiment: amplification)."
        )
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Path to repo data directory. Defaults to <repo>/data.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model filter (e.g., qwen3-4b).",
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
        "--experiment",
        type=str,
        default="non-distractor amplification",
        help="Experiment name in all_interventions.csv to compare against.",
    )
    parser.add_argument(
        "--normal-amplification-experiment",
        type=str,
        default="non-distractor amplification",
        choices=["non-distractor amplification", "amplification"],
        help=(
            "Source experiment for normal amplification bars. "
            "Use 'non-distractor amplification' for adj-lang-only amplification "
            "(default, based on source code semantics)."
        ),
    )
    parser.add_argument(
        "--ablation-filter",
        type=str,
        default="all",
        choices=["all", "same-as-amplification", "none"],
        help=(
            "How to include rows from adj_lang_method_logit_change: "
            "'all' keeps all ablation methods (grouped separately), "
            "'same-as-amplification' keeps rows where ablation_method == amplification_method, "
            "'none' keeps rows where ablation_method is empty or 'none'."
        ),
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help=(
            "Output CSV path. Defaults to "
            "data/adj_lang_method_logit_change/analysis/adj_logit_vs_interventions.csv"
        ),
    )
    parser.add_argument(
        "--summary-plot-csv",
        type=str,
        default=None,
        help=(
            "Optional output CSV path for 12-type plot summaries. Defaults to "
            "data/adj_lang_method_logit_change/analysis/adj_logit_plot_summary.csv"
        ),
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help=(
            "Optional output directory for plots. Defaults to "
            "data/adj_lang_method_logit_change/analysis/plots"
        ),
    )
    return parser.parse_args()


def repo_root_from_file() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_data_root(arg_data_root: Optional[str]) -> str:
    if arg_data_root is not None:
        return os.path.abspath(arg_data_root)
    return os.path.join(repo_root_from_file(), "data")


def include_adj_row(row: dict[str, str], mode: str) -> bool:
    if mode == "all":
        return True

    ablation_method = (row.get("ablation_method") or "").strip()
    amplification_method = (row.get("amplification_method") or "").strip()

    if mode == "same-as-amplification":
        return ablation_method == amplification_method

    if mode == "none":
        return ablation_method in {"", "none", "null", "None"}

    return True


def load_adj_aggregates(
    data_root: str,
    model_filter: Optional[str],
    prompt_lang_filter: Optional[str],
    adj_lang_filter: Optional[str],
    ablation_filter: str,
) -> dict[tuple[str, str, str, str, str], dict[str, float]]:
    pattern = os.path.join(
        data_root,
        "adj_lang_method_logit_change",
        "*",
        "prompt_*__adj_*",
        "adj_lang_method_logit_change.csv",
    )

    grouped: dict[tuple[str, str, str, str, str], list[float]] = defaultdict(list)

    for path in glob.glob(pattern):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                intervention_type = (row.get("intervention_type") or "combined").strip()
                if intervention_type != "combined":
                    continue

                model = (row.get("model") or "").strip()
                prompt_lang = (row.get("prompt_lang") or "").strip()
                adj_lang = (row.get("adj_lang") or "").strip()
                ablation_method = (row.get("ablation_method") or "").strip()
                amplification_method = (row.get("amplification_method") or "").strip()

                if model_filter is not None and model != model_filter:
                    continue
                if prompt_lang_filter is not None and prompt_lang != prompt_lang_filter:
                    continue
                if adj_lang_filter is not None and adj_lang != adj_lang_filter:
                    continue
                if amplification_method not in METHOD_TO_INTERVENTION:
                    continue
                if not include_adj_row(row, ablation_filter):
                    continue

                raw_delta = row.get("delta_combined_logit")
                if raw_delta in (None, ""):
                    continue

                try:
                    delta = float(raw_delta)
                except ValueError:
                    continue

                key = (model, prompt_lang, adj_lang, ablation_method, amplification_method)
                grouped[key].append(delta)

    aggregates: dict[tuple[str, str, str, str, str], dict[str, float]] = {}
    for key, values in grouped.items():
        finite_values = [v for v in values if math.isfinite(v)]
        if not finite_values:
            continue
        mean_val = statistics.fmean(finite_values)
        if len(finite_values) > 1:
            variance = sum((v - mean_val) ** 2 for v in finite_values) / (len(finite_values) - 1)
            stdev_val = math.sqrt(variance)
        else:
            stdev_val = 0.0
        aggregates[key] = {
            "adj_mean_delta": mean_val,
            "adj_stdev_delta": stdev_val,
            "adj_n": float(len(finite_values)),
        }

    return aggregates


def load_normal_amplification_from_adj_csv(
    data_root: str,
    model_filter: Optional[str],
    prompt_lang_filter: Optional[str],
    adj_lang_filter: Optional[str],
) -> dict[tuple[str, str, str, str], dict[str, float]]:
    pattern = os.path.join(
        data_root,
        "adj_lang_method_logit_change",
        "*",
        "prompt_*__adj_*",
        "adj_lang_method_logit_change.csv",
    )

    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)

    for path in glob.glob(pattern):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                intervention_type = (row.get("intervention_type") or "combined").strip()
                if intervention_type != "normal_amplification":
                    continue

                model = (row.get("model") or "").strip()
                prompt_lang = (row.get("prompt_lang") or "").strip()
                adj_lang = (row.get("adj_lang") or "").strip()
                amplification_method = (row.get("amplification_method") or "").strip()

                if model_filter is not None and model != model_filter:
                    continue
                if prompt_lang_filter is not None and prompt_lang != prompt_lang_filter:
                    continue
                if adj_lang_filter is not None and adj_lang != adj_lang_filter:
                    continue
                if amplification_method not in METHOD_TO_INTERVENTION:
                    continue

                raw_delta = row.get("delta_combined_logit")
                if raw_delta in (None, ""):
                    continue

                try:
                    delta = float(raw_delta)
                except ValueError:
                    continue
                if not math.isfinite(delta):
                    continue

                key = (model, prompt_lang, adj_lang, amplification_method)
                grouped[key].append(delta)

    aggregates: dict[tuple[str, str, str, str], dict[str, float]] = {}
    for key, values in grouped.items():
        if not values:
            continue
        mean_val = statistics.fmean(values)
        if len(values) > 1:
            variance = sum((v - mean_val) ** 2 for v in values) / (len(values) - 1)
            stdev_val = math.sqrt(variance)
        else:
            stdev_val = 0.0

        aggregates[key] = {
            "normal_mean_delta": mean_val,
            "normal_stdev_delta": stdev_val,
            "normal_n": float(len(values)),
        }

    return aggregates


def load_intervention_metrics(
    data_root: str,
    model_filter: Optional[str],
    prompt_lang_filter: Optional[str],
    adj_lang_filter: Optional[str],
) -> dict[tuple[str, str, str, str, str], dict[str, float]]:
    pattern = os.path.join(data_root, "interventions", "*", "*", "*", "all_interventions.csv")

    metrics: dict[tuple[str, str, str, str, str], dict[str, float]] = {}

    for path in glob.glob(pattern):
        rel = os.path.relpath(path, data_root)
        # rel: interventions/<model>/<prompt_lang>/<adj_lang>/all_interventions.csv
        parts = rel.split(os.sep)
        if len(parts) < 5:
            continue

        model = parts[1]
        prompt_lang = parts[2]
        adj_lang = parts[3]

        if model_filter is not None and model != model_filter:
            continue
        if prompt_lang_filter is not None and prompt_lang != prompt_lang_filter:
            continue
        if adj_lang_filter is not None and adj_lang != adj_lang_filter:
            continue

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                experiment = (row.get("Experiment") or "").strip()
                if experiment not in {
                    "ablation",
                    "amplification",
                    "non-distractor amplification",
                }:
                    continue

                method = (row.get("Method") or "").strip()
                run_lang = (row.get("Run") or "").strip()

                # We compare against adj_lang rows from interventions.
                if run_lang != adj_lang:
                    continue

                raw_mean = row.get("mean")
                if raw_mean in (None, ""):
                    continue

                raw_stdev = row.get("stdev")

                try:
                    mean_val = float(raw_mean)
                except ValueError:
                    continue

                stdev_val = 0.0
                if raw_stdev not in (None, ""):
                    try:
                        stdev_val = float(raw_stdev)
                    except ValueError:
                        stdev_val = 0.0

                key = (model, prompt_lang, adj_lang, experiment, method)
                metrics[key] = {
                    "intervention_mean": mean_val,
                    "intervention_stdev": stdev_val,
                }

    return metrics


def default_output_path(data_root: str) -> str:
    out_dir = os.path.join(data_root, "adj_lang_method_logit_change", "analysis")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, "adj_logit_vs_interventions.csv")


def default_summary_plot_csv_path(data_root: str) -> str:
    out_dir = os.path.join(data_root, "adj_lang_method_logit_change", "analysis")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, "adj_logit_plot_summary.csv")


def default_plot_dir(data_root: str) -> str:
    out_dir = os.path.join(data_root, "adj_lang_method_logit_change", "analysis", "plots")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def aggregate_adj_for_plot(
    adj_aggs: dict[tuple[str, str, str, str, str], dict[str, float]]
) -> dict[tuple[str, str, str], dict[str, float]]:
    grouped: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)

    for (model, prompt_lang, adj_lang, ablation_method, amplification_method), vals in adj_aggs.items():
        key = (model, prompt_lang, adj_lang)
        grouped[key][f"combined_pair:zero_ablation={ablation_method},target_amplification={amplification_method}"] = vals["adj_mean_delta"]

    return grouped


def build_plot_rows(
    adj_aggs: dict[tuple[str, str, str, str, str], dict[str, float]],
    normal_amp_adj_aggs: dict[tuple[str, str, str, str], dict[str, float]],
    intervention_metrics: dict[tuple[str, str, str, str, str], dict[str, float]],
    normal_amplification_experiment: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    adj_plot_aggs = aggregate_adj_for_plot(adj_aggs)

    for (model, prompt_lang, adj_lang), values in sorted(adj_plot_aggs.items()):
        # 3 types from normal amplification only (adj-lang-only by default).
        for method in METHOD_ORDER:
            adj_key = (model, prompt_lang, adj_lang, method)
            normal_metrics = normal_amp_adj_aggs.get(adj_key)
            if normal_metrics is not None:
                rows.append(
                    {
                        "model": model,
                        "prompt_lang": prompt_lang,
                        "adj_lang": adj_lang,
                        "plot_type": f"normal_amplification:{method}",
                        "mean_logit_change": normal_metrics["normal_mean_delta"],
                        "source": "adj_lang_method_logit_change:normal_amplification",
                    }
                )
                continue

            # Backward-compatible fallback when old adj csvs do not contain normal rows.
            intervention_method = METHOD_TO_INTERVENTION[method]
            key = (
                model,
                prompt_lang,
                adj_lang,
                normal_amplification_experiment,
                intervention_method,
            )
            metrics = intervention_metrics.get(key)
            if metrics is None:
                continue
            rows.append(
                {
                    "model": model,
                    "prompt_lang": prompt_lang,
                    "adj_lang": adj_lang,
                    "plot_type": f"normal_amplification:{method}",
                    "mean_logit_change": metrics["intervention_mean"],
                    "source": f"normal_interventions:{normal_amplification_experiment}",
                }
            )

        # 9 types from pairwise combined intervention runs.
        for ablation_method in METHOD_ORDER:
            for amplification_method in METHOD_ORDER:
                key_pair = (
                    f"combined_pair:zero_ablation={ablation_method},"
                    f"target_amplification={amplification_method}"
                )
                if key_pair in values:
                    rows.append(
                        {
                            "model": model,
                            "prompt_lang": prompt_lang,
                            "adj_lang": adj_lang,
                            "plot_type": key_pair,
                            "mean_logit_change": values[key_pair],
                            "source": "adj_lang_method_logit_change",
                        }
                    )

    return rows


def write_plot_summary_csv(rows: list[dict[str, object]], output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fieldnames = [
        "model",
        "prompt_lang",
        "adj_lang",
        "plot_type",
        "mean_logit_change",
        "source",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_plots(rows: list[dict[str, object]], plot_dir: str) -> int:
    os.makedirs(plot_dir, exist_ok=True)
    grouped: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)

    for row in rows:
        key = (str(row["model"]), str(row["prompt_lang"]), str(row["adj_lang"]))
        mean_val = row.get("mean_logit_change")
        if mean_val is None:
            continue
        grouped[key][str(row["plot_type"])] = float(str(mean_val))

    plot_order = [
        "normal_amplification:desc",
        "normal_amplification:val",
        "normal_amplification:freq",
        "combined_pair:zero_ablation=desc,target_amplification=desc",
        "combined_pair:zero_ablation=desc,target_amplification=val",
        "combined_pair:zero_ablation=desc,target_amplification=freq",
        "combined_pair:zero_ablation=val,target_amplification=desc",
        "combined_pair:zero_ablation=val,target_amplification=val",
        "combined_pair:zero_ablation=val,target_amplification=freq",
        "combined_pair:zero_ablation=freq,target_amplification=desc",
        "combined_pair:zero_ablation=freq,target_amplification=val",
        "combined_pair:zero_ablation=freq,target_amplification=freq",
    ]

    plot_count = 0
    for (model, prompt_lang, adj_lang), vals in sorted(grouped.items()):
        y_vals = [vals.get(label, float("nan")) for label in plot_order]
        x = list(range(len(plot_order)))

        plt.figure(figsize=(16, 6))
        bars = plt.bar(x, y_vals, color="#2b6cb0")
        for idx, y in enumerate(y_vals):
            if math.isnan(y):
                bars[idx].set_alpha(0.2)
                bars[idx].set_color("#9aa5b1")

        plt.axhline(0, color="black", linewidth=1)
        plt.xticks(x, plot_order, rotation=45, ha="right")
        plt.ylabel("Mean Logit Change")
        plt.xlabel("Intervention Type")
        plt.title(f"{model} | prompt={prompt_lang} | adj={adj_lang}")
        plt.tight_layout()

        model_dir = os.path.join(plot_dir, model)
        os.makedirs(model_dir, exist_ok=True)
        out_path = os.path.join(model_dir, f"prompt_{prompt_lang}__adj_{adj_lang}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        plot_count += 1

    return plot_count


def main() -> None:
    args = parse_args()
    data_root = get_data_root(args.data_root)

    adj_aggs = load_adj_aggregates(
        data_root=data_root,
        model_filter=args.model,
        prompt_lang_filter=args.prompt_lang,
        adj_lang_filter=args.adj_lang,
        ablation_filter=args.ablation_filter,
    )

    normal_amp_adj_aggs = load_normal_amplification_from_adj_csv(
        data_root=data_root,
        model_filter=args.model,
        prompt_lang_filter=args.prompt_lang,
        adj_lang_filter=args.adj_lang,
    )

    intervention_metrics = load_intervention_metrics(
        data_root=data_root,
        model_filter=args.model,
        prompt_lang_filter=args.prompt_lang,
        adj_lang_filter=args.adj_lang,
    )

    out_file = args.output_file or default_output_path(data_root)
    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)

    fieldnames = [
        "model",
        "prompt_lang",
        "adj_lang",
        "ablation_method",
        "amplification_method",
        "intervention_method",
        "baseline_source",
        "adj_mean_delta",
        "adj_stdev_delta",
        "adj_n",
        "intervention_mean",
        "intervention_stdev",
        "difference_adj_minus_intervention",
        "abs_difference",
    ]

    rows_written = 0
    missing_counter = 0

    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for (model, prompt_lang, adj_lang, ablation_method, amplification_method), adj_vals in sorted(adj_aggs.items()):
            intervention_method = METHOD_TO_INTERVENTION[amplification_method]

            normal_adj_key = (model, prompt_lang, adj_lang, amplification_method)
            normal_adj_metrics = normal_amp_adj_aggs.get(normal_adj_key)
            if normal_adj_metrics is not None:
                baseline_mean = normal_adj_metrics["normal_mean_delta"]
                baseline_stdev = normal_adj_metrics["normal_stdev_delta"]
                baseline_source = "adj_lang_method_logit_change:normal_amplification"
            else:
                intervention_key = (model, prompt_lang, adj_lang, args.experiment, intervention_method)
                metrics = intervention_metrics.get(intervention_key)
                if metrics is None:
                    missing_counter += 1
                    continue
                baseline_mean = metrics["intervention_mean"]
                baseline_stdev = metrics["intervention_stdev"]
                baseline_source = f"all_interventions:{args.experiment}"

            diff = adj_vals["adj_mean_delta"] - baseline_mean
            row = {
                "model": model,
                "prompt_lang": prompt_lang,
                "adj_lang": adj_lang,
                "ablation_method": ablation_method,
                "amplification_method": amplification_method,
                "intervention_method": intervention_method,
                "baseline_source": baseline_source,
                "adj_mean_delta": adj_vals["adj_mean_delta"],
                "adj_stdev_delta": adj_vals["adj_stdev_delta"],
                "adj_n": int(adj_vals["adj_n"]),
                "intervention_mean": baseline_mean,
                "intervention_stdev": baseline_stdev,
                "difference_adj_minus_intervention": diff,
                "abs_difference": abs(diff),
            }
            writer.writerow(row)
            rows_written += 1

    summary_plot_csv = args.summary_plot_csv or default_summary_plot_csv_path(data_root)
    plot_rows = build_plot_rows(
        adj_aggs,
        normal_amp_adj_aggs,
        intervention_metrics,
        normal_amplification_experiment=args.normal_amplification_experiment,
    )
    write_plot_summary_csv(plot_rows, summary_plot_csv)

    plot_dir = args.plot_dir or default_plot_dir(data_root)
    plot_count = make_plots(plot_rows, plot_dir)

    print(f"Data root: {data_root}")
    print(f"Wrote comparison rows: {rows_written}")
    print(f"Skipped rows with missing intervention match: {missing_counter}")
    print(f"Output: {out_file}")
    print(f"Plot summary CSV rows: {len(plot_rows)}")
    print(f"Plot summary CSV: {summary_plot_csv}")
    print(f"Plots written: {plot_count}")
    print(f"Plot directory: {plot_dir}")


if __name__ == "__main__":
    main()
