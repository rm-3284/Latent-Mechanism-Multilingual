"""Merge per-feature-set evaluation JSON files into one summary CSV."""

from __future__ import annotations

import argparse
import csv
import json
import os
from glob import glob

from hyperparameter_sensitivity import repo_data_dir
from models import hf_model_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge feature_set_evaluations summaries.")
    parser.add_argument("--model", "-m", type=str, default="gemma-2-2b", choices=hf_model_names.keys())
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir or os.path.join(
        repo_data_dir(), "additional_experiments", args.model, "feature_set_evaluations"
    )
    output_csv = args.output_csv or os.path.join(input_dir, "summary_merged.csv")

    rows = []
    for path in sorted(glob(os.path.join(input_dir, "*.json"))):
        if os.path.basename(path) == "manifest.json":
            continue
        with open(path, "r") as f:
            payload = json.load(f)
        row = {
            "feature_set_id": payload.get("feature_set_id"),
            "feature_set_file": payload.get("feature_set_file"),
            "method": payload.get("method"),
            "sweep": payload.get("sweep"),
            "is_default": payload.get("is_default"),
            "result_file": path,
        }
        row.update({f"hp_{k}": v for k, v in payload.get("hyperparameters", {}).items()})
        if "antonyms" in payload:
            row.update({f"antonyms_{k}": v for k, v in payload["antonyms"]["summary"].items()})
        if "enumerations" in payload:
            row.update({f"enumerations_{k}": v for k, v in payload["enumerations"]["summary"].items()})
        rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No evaluation JSON files found in {input_dir}")

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    main()
