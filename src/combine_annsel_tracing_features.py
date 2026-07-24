"""
Combine per-language AnnSel tracing-sensitivity feature files into full multilingual sets.

Each tracing sensitivity run saved only one language (e.g. ...__de.json). This script
groups files that share the same tracing hyperparameters and merges selected_features
across languages so feature-intervention can ablate/amplify both sides.

Example:
  python combine_annsel_tracing_features.py --model gemma-2-2b
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any

from hyperparameter_sensitivity import repo_data_dir
from template import lang_to_flores_key


LANGS = set(lang_to_flores_key.keys())


def parse_stem(stem: str) -> tuple[str, str] | None:
    """Split '<hp_stem>__<lang>' -> (hp_stem, lang)."""
    if "__" not in stem:
        return None
    base, lang = stem.rsplit("__", 1)
    if lang not in LANGS:
        return None
    return base, lang


def combine_group(paths: dict[str, str]) -> dict[str, Any]:
    missing = sorted(LANGS - set(paths))
    if missing:
        raise ValueError(f"Incomplete language set; missing {missing}")

    selected: dict[str, list[str]] = {}
    sources: dict[str, str] = {}
    template: dict[str, Any] | None = None
    for lang in sorted(paths):
        with open(paths[lang], "r") as f:
            payload = json.load(f)
        if template is None:
            template = payload
        elif payload.get("hyperparameters") != template.get("hyperparameters"):
            raise ValueError(
                f"Hyperparameter mismatch for {lang}: {paths[lang]} vs {paths[sorted(paths)[0]]}"
            )
        feats = payload.get("selected_features", {})
        if set(feats.keys()) != {lang}:
            raise ValueError(
                f"Expected only '{lang}' in {paths[lang]}, found {sorted(feats.keys())}"
            )
        selected[lang] = list(feats[lang])
        sources[lang] = paths[lang]

    assert template is not None
    out = {
        "method": template.get("method", "AnnSel"),
        "model": template.get("model"),
        "sweep": template.get("sweep"),
        "hyperparameters": template.get("hyperparameters"),
        "is_default": template.get("is_default", False),
        "selected_features": selected,
        "selection_threshold": template.get("selection_threshold"),
        "combined_from": sources,
        "n_features_per_lang": {lang: len(selected[lang]) for lang in selected},
    }
    return out


def default_input_dir(model: str) -> str:
    return os.path.join(
        repo_data_dir(),
        "additional_experiments",
        model,
        "annsel_tracing_sensitivity",
        "selected_features",
        "AnnSel",
    )


def default_output_dir(model: str) -> str:
    return os.path.join(
        repo_data_dir(),
        "additional_experiments",
        model,
        "annsel_tracing_sensitivity",
        "selected_features_combined",
        "AnnSel",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge per-language AnnSel tracing feature sets by hyperparameter config."
    )
    parser.add_argument("--model", default="gemma-2-2b")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    input_dir = args.input_dir or default_input_dir(args.model)
    output_dir = args.output_dir or default_output_dir(args.model)
    os.makedirs(output_dir, exist_ok=True)

    groups: dict[str, dict[str, str]] = defaultdict(dict)
    for name in sorted(os.listdir(input_dir)):
        if not name.endswith(".json"):
            continue
        parsed = parse_stem(os.path.splitext(name)[0])
        if parsed is None:
            print(f"Skipping non per-lang file: {name}")
            continue
        base, lang = parsed
        groups[base][lang] = os.path.join(input_dir, name)

    written: list[str] = []
    for base, paths in sorted(groups.items()):
        payload = combine_group(paths)
        out_path = os.path.join(output_dir, f"{base}.json")
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        written.append(out_path)
        counts = payload["n_features_per_lang"]
        print(f"Wrote {out_path} ({counts})")

    manifest = {
        "model": args.model,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "n_combined": len(written),
        "files": written,
    }
    # Keep manifest outside AnnSel/ so array exporters that glob *.json skip it.
    manifest_path = os.path.join(os.path.dirname(output_dir), "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"Combined {len(written)} hyperparameter settings -> {output_dir}")


if __name__ == "__main__":
    main()
