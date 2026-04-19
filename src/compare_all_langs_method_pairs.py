import argparse
import csv
import glob
import os
import statistics
from collections import defaultdict

import matplotlib.pyplot as plt


METHOD_ORDER = ["desc", "val", "freq"]
METHOD_TO_INTERVENTION = {
    "desc": "AnnSel",
    "val": "ValSel",
    "freq": "FreqSel",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate and compare paired zero-ablation / amplification methods "
            "from all_langs_intervention_logit_change CSVs."
        )
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Path to an all_langs_intervention_logit_change CSV file or folder.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help=(
            "Optional summary CSV path. Defaults to "
            "data/all_langs_intervention_logit_change/analysis/method_pair_summary.csv"
        ),
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help=(
            "Optional output directory for comparison plots. Defaults to "
            "data/all_langs_intervention_logit_change/analysis/plots"
        ),
    )
    return parser.parse_args()


def default_input_file() -> str:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(
        repo_root,
        "data",
        "all_langs_intervention_logit_change",
    )


def default_output_file() -> str:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(repo_root, "data", "all_langs_intervention_logit_change", "analysis")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, "method_pair_summary.csv")


def default_plot_dir() -> str:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(repo_root, "data", "all_langs_intervention_logit_change", "analysis", "plots")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def resolve_input_files(input_path: str) -> list[str]:
    if os.path.isfile(input_path):
        return [input_path]

    csv_files: list[str] = []
    for root, _, files in os.walk(input_path):
        for file_name in files:
            if file_name.endswith(".csv"):
                csv_files.append(os.path.join(root, file_name))
    return sorted(csv_files)


def load_intervention_baselines(data_root: str) -> dict[tuple[str, str, str, str, str, str], dict[str, float]]:
    pattern = os.path.join(data_root, "interventions", "*", "*", "*", "all_interventions.csv")
    grouped: dict[tuple[str, str, str, str, str, str], list[float]] = defaultdict(list)

    for csv_path in sorted(glob.glob(pattern)):
        rel_path = os.path.relpath(csv_path, data_root)
        parts = rel_path.split(os.sep)
        if len(parts) < 5:
            continue

        model = parts[1]
        prompt_lang = parts[2]
        adj_lang = parts[3]

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                experiment = (row.get("Experiment") or "").strip()
                if experiment not in {"original", "amplification", "non-distractor amplification"}:
                    continue

                method = (row.get("Method") or "").strip()
                if method not in METHOD_TO_INTERVENTION.values():
                    continue

                measure_lang = (row.get("Run") or "").strip()
                if not measure_lang:
                    continue

                raw_mean = row.get("mean")
                if raw_mean in (None, ""):
                    continue

                try:
                    mean_val = float(raw_mean)
                except ValueError:
                    continue

                baseline_kind = "zero_ablation" if experiment == "original" else "target_amplification"
                key = (model, prompt_lang, adj_lang, baseline_kind, method, measure_lang)
                grouped[key].append(mean_val)

    baselines: dict[tuple[str, str, str, str, str, str], dict[str, float]] = {}
    for key, values in grouped.items():
        finite_values = [value for value in values if value == value]
        if not finite_values:
            continue
        baselines[key] = {
            "mean": statistics.fmean(finite_values),
            "stdev": statistics.stdev(finite_values) if len(finite_values) > 1 else 0.0,
            "n": float(len(finite_values)),
        }

    return baselines


def build_comparison_rows(
    combined_aggs: dict[tuple[str, str, str, str, str, str, str], dict[str, float]],
    baselines: dict[tuple[str, str, str, str, str, str], dict[str, float]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (
        model,
        prompt_lang,
        adj_lang,
        intervention_lang,
        measure_lang,
        ablation_method,
        amplification_method,
    ), combined_metrics in sorted(combined_aggs.items()):
        amp_key = (
            model,
            prompt_lang,
            adj_lang,
            "target_amplification",
            METHOD_TO_INTERVENTION[amplification_method],
            measure_lang,
        )
        # Use diagonal combined pair (ablation==amplification) as zero_ablation:method.
        zero_key = (
            model,
            prompt_lang,
            adj_lang,
            intervention_lang,
            measure_lang,
            ablation_method,
            ablation_method,
        )
        zero_metrics = combined_aggs.get(zero_key)

        amp_metrics = baselines.get(amp_key)
        if zero_metrics is None or amp_metrics is None:
            continue

        combined_mean = combined_metrics["mean_intervened_target_minus_base"]
        original_mean = combined_metrics["mean_baseline_target_minus_base"]
        zero_mean = zero_metrics["mean_intervened_target_minus_base"]
        amp_mean = amp_metrics["mean"]

        rows.append(
            {
                "model": model,
                "prompt_lang": prompt_lang,
                "adj_lang": adj_lang,
                "intervention_lang": intervention_lang,
                "measure_lang": measure_lang,
                "ablation_method": ablation_method,
                "amplification_method": amplification_method,
                "zero_ablation_mean": zero_mean,
                "target_amplification_mean": amp_mean,
                "original_mean": original_mean,
                "combined_mean": combined_mean,
                "combined_minus_original": combined_mean - original_mean,
                "combined_minus_zero_ablation": combined_mean - zero_mean,
                "combined_minus_target_amplification": combined_mean - amp_mean,
                "combined_n": combined_metrics["n"],
                "zero_ablation_n": zero_metrics["n"],
                "target_amplification_n": amp_metrics["n"],
            }
        )

    return rows


def write_comparison_csv(rows: list[dict[str, object]], output_file: str) -> None:
    fieldnames = [
        "model",
        "prompt_lang",
        "adj_lang",
        "intervention_lang",
        "measure_lang",
        "ablation_method",
        "amplification_method",
        "zero_ablation_mean",
        "target_amplification_mean",
        "original_mean",
        "combined_mean",
        "combined_minus_original",
        "combined_minus_zero_ablation",
        "combined_minus_target_amplification",
        "combined_n",
        "zero_ablation_n",
        "target_amplification_n",
    ]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison_rows(rows: list[dict[str, object]], plot_dir: str) -> int:
    os.makedirs(plot_dir, exist_ok=True)
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)

    for row in rows:
        if str(row["intervention_lang"]) != str(row["adj_lang"]):
            continue
        key = (
            str(row["model"]),
            str(row["prompt_lang"]),
            str(row["adj_lang"]),
        )
        grouped[key].append(row)

    plot_count = 0
    for (model, prompt_lang, adj_lang), pair_rows in sorted(grouped.items()):
        pair_rows.sort(key=lambda row: (METHOD_ORDER.index(str(row["ablation_method"])), METHOD_ORDER.index(str(row["amplification_method"]))))

        plot_order = [
            "original",
            "zero_ablation:desc",
            "zero_ablation:val",
            "zero_ablation:freq",
            "target_amplification:desc",
            "target_amplification:val",
            "target_amplification:freq",
            "combined_pair:zero_ablation=desc,target_amplification=val",
            "combined_pair:zero_ablation=desc,target_amplification=freq",
            "combined_pair:zero_ablation=val,target_amplification=desc",
            "combined_pair:zero_ablation=val,target_amplification=freq",
            "combined_pair:zero_ablation=freq,target_amplification=desc",
            "combined_pair:zero_ablation=freq,target_amplification=val",
        ]

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
        grouped_points: dict[str, dict[str, float]] = defaultdict(dict)

        for row in pair_rows:
            measure_lang = str(row["measure_lang"])
            ablation_method = str(row["ablation_method"])
            amplification_method = str(row["amplification_method"])
            key_pair = f"combined_pair:zero_ablation={ablation_method},target_amplification={amplification_method}"
            grouped_points[measure_lang]["original"] = float(str(row["original_mean"]))
            grouped_points[measure_lang][f"zero_ablation:{ablation_method}"] = float(str(row["zero_ablation_mean"]))
            grouped_points[measure_lang][f"target_amplification:{amplification_method}"] = float(str(row["target_amplification_mean"]))
            grouped_points[measure_lang][key_pair] = float(str(row["combined_mean"]))

        x = list(range(len(plot_order)))
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
        for measure_lang in measure_lang_order:
            values_by_label = grouped_points.get(measure_lang, {})
            y_vals = [values_by_label.get(label, float("nan")) for label in plot_order]
            x_vals = [idx + offsets[measure_lang] for idx in x]
            ax.scatter(x_vals, y_vals, color=language_colors[measure_lang], s=45, label=measure_lang, zorder=3)

        ax.axhline(0, color="black", linewidth=1)
        ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
        ax.set_ylabel("Target-Base Logit Difference")
        ax.set_xticks(x)
        ax.set_xticklabels(plot_order, rotation=45, ha="right")
        ax.set_xlabel("Method Pair")
        ax.legend(title="measure_lang", ncol=2, fontsize=9)
        fig.suptitle(f"{model} | prompt={prompt_lang} | adj={adj_lang}", y=0.99)
        fig.tight_layout()

        model_dir = os.path.join(plot_dir, model)
        os.makedirs(model_dir, exist_ok=True)
        out_path = os.path.join(model_dir, f"prompt_{prompt_lang}__adj_{adj_lang}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        plot_count += 1

    return plot_count


def main() -> None:
    args = parse_args()
    input_path = args.input_file or default_input_file()
    output_file = args.output_file or default_output_file()
    plot_dir = args.plot_dir or default_plot_dir()

    input_files = resolve_input_files(input_path)
    if not input_files:
        raise FileNotFoundError(f"No CSV files found under: {input_path}")

    grouped: dict[tuple[str, str, str, str, str, str, str], list[float]] = defaultdict(list)
    baseline_grouped: dict[tuple[str, str, str, str, str, str, str], list[float]] = defaultdict(list)
    row_count = 0

    for csv_path in input_files:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_count += 1
                model = (row.get("model") or "").strip()
                prompt_lang = (row.get("prompt_lang") or "").strip()
                adj_lang = (row.get("adj_lang") or "").strip()
                intervention_lang = (row.get("intervention_lang") or "").strip()
                measure_lang = (row.get("measure_lang") or "").strip()
                ablation_method = (row.get("ablation_method") or "").strip()
                amplification_method = (row.get("amplification_method") or "").strip()

                baseline_raw = row.get("baseline_target_minus_base")
                intervened_raw = row.get("intervened_target_minus_base")
                if baseline_raw in (None, "") or intervened_raw in (None, ""):
                    continue

                try:
                    baseline_val = float(baseline_raw)
                    intervened_val = float(intervened_raw)
                except ValueError:
                    continue

                key = (
                    model,
                    prompt_lang,
                    adj_lang,
                    intervention_lang,
                    measure_lang,
                    ablation_method,
                    amplification_method,
                )
                grouped[key].append(intervened_val)
                baseline_grouped[key].append(baseline_val)

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    combined_aggs: dict[tuple[str, str, str, str, str, str, str], dict[str, float]] = {}
    for key, values in sorted(grouped.items()):
        finite_values = [value for value in values if isinstance(value, float) and value == value]
        baseline_values = [value for value in baseline_grouped.get(key, []) if isinstance(value, float) and value == value]
        if not finite_values:
            continue
        if not baseline_values:
            continue
        mean_val = statistics.fmean(finite_values)
        stdev_val = statistics.stdev(finite_values) if len(finite_values) > 1 else 0.0
        baseline_mean_val = statistics.fmean(baseline_values)
        baseline_stdev_val = statistics.stdev(baseline_values) if len(baseline_values) > 1 else 0.0
        model, prompt_lang, adj_lang, intervention_lang, measure_lang, ablation_method, amplification_method = key
        summary_rows.append(
            {
                "model": model,
                "prompt_lang": prompt_lang,
                "adj_lang": adj_lang,
                "intervention_lang": intervention_lang,
                "measure_lang": measure_lang,
                "ablation_method": ablation_method,
                "amplification_method": amplification_method,
                "mean_baseline_target_minus_base": baseline_mean_val,
                "stdev_baseline_target_minus_base": baseline_stdev_val,
                "mean_intervened_target_minus_base": mean_val,
                "stdev_intervened_target_minus_base": stdev_val,
                "n": len(finite_values),
            }
        )
        combined_aggs[key] = {
            "mean_baseline_target_minus_base": baseline_mean_val,
            "stdev_baseline_target_minus_base": baseline_stdev_val,
            "mean_intervened_target_minus_base": mean_val,
            "stdev_intervened_target_minus_base": stdev_val,
            "n": float(len(finite_values)),
        }

    baselines = load_intervention_baselines(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
    comparison_rows = build_comparison_rows(combined_aggs, baselines)

    fieldnames = [
        "model",
        "prompt_lang",
        "adj_lang",
        "intervention_lang",
        "measure_lang",
        "ablation_method",
        "amplification_method",
        "mean_baseline_target_minus_base",
        "stdev_baseline_target_minus_base",
        "mean_intervened_target_minus_base",
        "stdev_intervened_target_minus_base",
        "n",
    ]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    comparison_output_file = os.path.join(os.path.dirname(output_file), "method_pair_comparison.csv")
    write_comparison_csv(comparison_rows, comparison_output_file)

    plot_count = plot_comparison_rows(comparison_rows, plot_dir)

    print(f"Read {row_count} rows from {len(input_files)} CSV file(s)")
    print(f"Wrote {len(summary_rows)} grouped rows to {output_file}")
    print(f"Wrote {len(comparison_rows)} comparison rows to {comparison_output_file}")
    print(f"Wrote {plot_count} plot(s) to {plot_dir}")


if __name__ == "__main__":
    main()
