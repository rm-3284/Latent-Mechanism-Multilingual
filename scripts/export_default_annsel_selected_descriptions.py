#!/usr/bin/env python3
"""Fetch Neuronpedia descriptions for default AnnSel selected features and export CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict

import requests

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

from lib.models import neuronpedia_urls  # noqa: E402
from lib.template import identifiers, langs_big  # noqa: E402

DEFAULT_SEL = os.path.join(
    REPO,
    "data/additional_experiments/gemma-2-2b/selected_features/AnnSel/selection_threshold_0.1.json",
)
FLORES_DIR = os.path.join(REPO, "data/flores_features/gemma-2-2b")
CACHE_PATH = os.path.join(REPO, "data/cache/neuronpedia_descriptions/gemma-2-2b_descriptions.json")
OUT_DIR = os.path.join(REPO, "data/additional_experiments/gemma-2-2b/selected_features/AnnSel")
MODEL = "gemma-2-2b"


def lang_match(description: str, lang: str) -> bool:
    if not description:
        return False
    return any(token in description for token in identifiers[lang])


def display_description(description: str) -> str:
    return description if description.strip() else "(missing description)"


def load_json(path: str) -> dict | list:
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def fetch_description(
    feat: str,
    cache: dict[str, str],
    headers: dict[str, str],
    url_template: str,
    *,
    refetch: bool,
) -> str:
    cached = cache.get(feat)
    if cached is not None and cached.strip() and not refetch:
        return cached

    layer, feature_idx = feat.split(".")
    url = url_template.format(layer=layer, feature_idx=feature_idx)
    for attempt in range(5):
        try:
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 429:
                time.sleep(min(2**attempt, 30))
                continue
            response.raise_for_status()
            explanations = response.json().get("explanations", [])
            desc = explanations[0]["description"] if explanations else ""
            cache[feat] = desc
            time.sleep(0.35)
            return desc
        except (requests.RequestException, KeyError, IndexError):
            time.sleep(min(2**attempt, 30))
    cache.setdefault(feat, "")
    return cache[feat]


def export_per_language(
    selected: dict[str, list[str]],
    counts_by_lang: dict[str, dict[str, int]],
    descriptions: dict[str, str],
    *,
    lang_name_only: bool,
) -> list[dict[str, str]]:
    os.makedirs(OUT_DIR, exist_ok=True)
    aggregate_rows: list[dict[str, str]] = []
    for lang in sorted(langs_big):
        if lang not in selected:
            continue

        rows = []
        for feat in selected[lang]:
            desc = descriptions.get(feat, "")
            matches = lang_match(desc, lang)
            if lang_name_only and not matches:
                continue
            row = {
                "lang": lang,
                "feature": feat,
                "annotation_match_count": counts_by_lang[lang].get(feat, 0),
                "description": desc,
                "matches_lang_identifier": matches,
            }
            rows.append(row)
            aggregate_rows.append(
                {
                    "lang": lang,
                    "feature": feat,
                    "annotation_match_count": str(row["annotation_match_count"]),
                    "description": desc,
                }
            )
        rows.sort(key=lambda r: (-r["annotation_match_count"], r["feature"]))

        by_count_path = os.path.join(OUT_DIR, f"default_annsel_{lang}_selected_descriptions_by_count.csv")
        fieldnames = ["feature", "annotation_match_count", "description", "matches_lang_identifier"]
        with open(by_count_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows({k: r[k] for k in fieldnames} for r in rows)

        desc_counts: Counter[str] = Counter()
        desc_feats: dict[str, list[str]] = {}
        for row in rows:
            desc_counts[row["description"]] += row["annotation_match_count"]
            desc_feats.setdefault(row["description"], []).append(row["feature"])

        freq_rows = sorted(
            [
                {
                    "description": desc,
                    "total_annotation_match_count": count,
                    "num_features": len(desc_feats[desc]),
                }
                for desc, count in desc_counts.items()
            ],
            key=lambda r: (-r["total_annotation_match_count"], r["description"]),
        )
        freq_path = os.path.join(OUT_DIR, f"default_annsel_{lang}_selected_description_frequency.csv")
        with open(freq_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["description", "total_annotation_match_count", "num_features"],
            )
            writer.writeheader()
            writer.writerows(freq_rows)

        lang_only_rows = [r for r in rows if r["matches_lang_identifier"]]
        lang_by_count_path = os.path.join(OUT_DIR, f"default_annsel_{lang}_descriptions_by_count.csv")
        with open(lang_by_count_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["feature", "annotation_match_count", "description"],
            )
            writer.writeheader()
            writer.writerows(
                {
                    "feature": r["feature"],
                    "annotation_match_count": r["annotation_match_count"],
                    "description": r["description"],
                }
                for r in lang_only_rows
            )

        lang_freq: Counter[str] = Counter()
        lang_freq_feats: dict[str, list[str]] = defaultdict(list)
        for row in lang_only_rows:
            lang_freq[row["description"]] += row["annotation_match_count"]
            lang_freq_feats[row["description"]].append(row["feature"])
        lang_freq_rows = sorted(
            [
                {
                    "description": desc,
                    "total_annotation_match_count": count,
                    "num_features": len(lang_freq_feats[desc]),
                }
                for desc, count in lang_freq.items()
            ],
            key=lambda r: (-r["total_annotation_match_count"], r["description"]),
        )
        lang_freq_path = os.path.join(OUT_DIR, f"default_annsel_{lang}_description_frequency.csv")
        with open(lang_freq_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["description", "total_annotation_match_count", "num_features"],
            )
            writer.writeheader()
            writer.writerows(lang_freq_rows)

        print(
            f"{lang}: wrote {len(rows)} rows "
            f"({sum(1 for r in rows if r['matches_lang_identifier'])} with lang identifier)"
        )

    aggregate_rows.sort(
        key=lambda r: (-int(r["annotation_match_count"]), r["lang"], r["feature"])
    )
    aggregate_path = os.path.join(OUT_DIR, "default_annsel_feature_descriptions_by_count.csv")
    with open(aggregate_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["lang", "feature", "annotation_match_count", "description"],
        )
        writer.writeheader()
        writer.writerows(aggregate_rows)

    desc_meta: dict[str, dict[str, set[str] | int]] = defaultdict(
        lambda: {"count": 0, "features": set(), "langs": set(), "pairs": set()}
    )
    for row in aggregate_rows:
        desc = display_description(row["description"])
        meta = desc_meta[desc]
        meta["count"] = int(meta["count"]) + int(row["annotation_match_count"])
        meta["features"].add(row["feature"])
        meta["langs"].add(row["lang"])
        meta["pairs"].add((row["lang"], row["feature"]))

    freq_aggregate_rows = sorted(
        [
            {
                "description": desc,
                "total_annotation_match_count": meta["count"],
                "num_selected_features": len(meta["features"]),
                "num_langs": len(meta["langs"]),
                "num_lang_feature_pairs": len(meta["pairs"]),
            }
            for desc, meta in desc_meta.items()
        ],
        key=lambda r: (-r["total_annotation_match_count"], r["description"]),
    )
    freq_aggregate_path = os.path.join(OUT_DIR, "default_annsel_descriptions_by_frequency.csv")
    with open(freq_aggregate_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "description",
                "total_annotation_match_count",
                "num_selected_features",
                "num_langs",
                "num_lang_feature_pairs",
            ],
        )
        writer.writeheader()
        writer.writerows(freq_aggregate_rows)

    return aggregate_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection-file",
        default=DEFAULT_SEL,
        help="AnnSel selected-features JSON (default: threshold 0.1)",
    )
    parser.add_argument(
        "--lang-name-only",
        action="store_true",
        help="Only export features whose description contains that language's identifier",
    )
    parser.add_argument(
        "--refetch-all",
        action="store_true",
        help="Re-fetch all descriptions from Neuronpedia",
    )
    args = parser.parse_args()

    payload = load_json(args.selection_file)
    selected: dict[str, list[str]] = payload["selected_features"]

    counts_by_lang = {}
    for lang in langs_big:
        path = os.path.join(FLORES_DIR, f"{lang}_features.json")
        counts_by_lang[lang] = load_json(path) if os.path.exists(path) else {}

    cache: dict[str, str] = load_json(CACHE_PATH) if os.path.exists(CACHE_PATH) else {}
    unique_feats = sorted({feat for feats in selected.values() for feat in feats})
    headers = {"User-Agent": "Mozilla/5.0 (compatible; Latent-Mechanism-Multilingual/1.0)"}
    url_template = neuronpedia_urls[MODEL]

    print(f"Fetching descriptions for {len(unique_feats)} unique selected features...")
    for i, feat in enumerate(unique_feats, 1):
        fetch_description(
            feat,
            cache,
            headers,
            url_template,
            refetch=args.refetch_all,
        )
        if i % 25 == 0 or i == len(unique_feats):
            print(f"  {i}/{len(unique_feats)}")
            save_json(CACHE_PATH, cache)

    save_json(CACHE_PATH, cache)
    export_per_language(selected, counts_by_lang, cache, lang_name_only=args.lang_name_only)
    print(f"Done. Output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
