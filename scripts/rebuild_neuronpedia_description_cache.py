#!/usr/bin/env python3
"""Rebuild the shared Neuronpedia description cache for a model."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time

import requests

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

from lib.models import neuronpedia_urls  # noqa: E402


def load_cache(cache_path: str) -> dict[str, str]:
    if not os.path.exists(cache_path):
        return {}
    with open(cache_path, "r") as f:
        return json.load(f)


def save_cache(cache_path: str, descriptions: dict[str, str]) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp_path = f"{cache_path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(descriptions, f, indent=2)
    os.replace(tmp_path, cache_path)


def default_cache_path(model: str) -> str:
    cache_dir = os.environ.get(
        "NEURONPEDIA_DESCRIPTION_CACHE_DIR",
        os.path.join(REPO, "data", "cache", "neuronpedia_descriptions"),
    )
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{model}_descriptions.json")


def manifest_path(model: str) -> str:
    return os.path.join(
        os.path.dirname(default_cache_path(model)),
        f"{model}_description_keys.json",
    )


def shard_path(output_cache_path: str, shard_id: int) -> str:
    return f"{output_cache_path}.shard.{shard_id:04d}.json"


def collect_feature_keys(model: str, cache_path: str, extra_dirs: list[str]) -> list[str]:
    keys: set[str] = set()
    if os.path.exists(cache_path):
        keys.update(load_cache(cache_path))

    for directory in extra_dirs:
        if not os.path.isdir(directory):
            continue
        for root, _, files in os.walk(directory):
            for name in files:
                if not name.endswith(".json"):
                    continue
                path = os.path.join(root, name)
                try:
                    with open(path, "r") as f:
                        payload = json.load(f)
                except (json.JSONDecodeError, OSError):
                    continue
                if isinstance(payload, dict):
                    for key in payload:
                        if isinstance(key, str) and key.count(".") == 1:
                            parts = key.split(".")
                            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                                keys.add(key)
                    selected = payload.get("selected_features")
                    if isinstance(selected, dict):
                        for feats in selected.values():
                            keys.update(f for f in feats if isinstance(f, str))
                elif isinstance(payload, list):
                    for item in payload:
                        if (
                            isinstance(item, (list, tuple))
                            and len(item) == 2
                            and all(isinstance(v, int) for v in item)
                        ):
                            keys.add(f"{item[0]}.{item[1]}")
    return sorted(keys)


def write_manifest(model: str, cache_path: str, extra_dirs: list[str]) -> list[str]:
    keys = collect_feature_keys(model, cache_path, extra_dirs)
    if not keys:
        raise SystemExit(f"No feature keys found for model {model}")
    path = manifest_path(model)
    save_cache(path, {key: "" for key in keys})
    print(f"Wrote {len(keys)} keys to {path}")
    return keys


def load_manifest(model: str) -> list[str]:
    path = manifest_path(model)
    if not os.path.exists(path):
        raise SystemExit(f"Missing manifest {path}; run with --write-manifest first")
    payload = load_cache(path)
    return sorted(payload.keys())


def shard_keys(keys: list[str], shard_id: int, num_shards: int) -> list[str]:
    return [key for idx, key in enumerate(keys) if idx % num_shards == shard_id]


def fetch_description_unlocked(
    url: str,
    headers: dict[str, str],
    *,
    timeout: float = 30.0,
    max_retries: int = 8,
    min_interval: float = 0.35,
) -> str:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            if response.status_code == 429:
                delay = min(2 ** (attempt + 1), 60) + random.uniform(0, 1)
                time.sleep(delay)
                continue
            response.raise_for_status()
            explanations = response.json().get("explanations", [])
            time.sleep(min_interval)
            if not explanations:
                return ""
            return explanations[0]["description"]
        except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
            last_error = exc
            time.sleep(min(2 ** attempt, 30))
    if last_error is not None:
        raise last_error
    return ""


def seed_shard_from_sources(
    assigned_keys: list[str],
    shard_file: str,
    sources: list[str],
) -> dict[str, str]:
    shard = load_cache(shard_file) if os.path.exists(shard_file) else {}
    assigned = set(assigned_keys)
    for source in sources:
        if not os.path.exists(source):
            continue
        cached = load_cache(source)
        for key in assigned:
            if key in cached and key not in shard:
                shard[key] = cached[key]
    return shard


def run_shard(
    model: str,
    cache_path: str,
    output_cache_path: str,
    shard_id: int,
    num_shards: int,
    resume: bool,
) -> None:
    keys = load_manifest(model)
    assigned = shard_keys(keys, shard_id, num_shards)
    shard_file = shard_path(output_cache_path, shard_id)
    sources = [shard_file]
    if resume:
        sources.extend([output_cache_path, cache_path])
    rebuilt = seed_shard_from_sources(assigned, shard_file, sources) if resume else {}
    pending = [key for key in assigned if key not in rebuilt] if resume else assigned

    url_template = neuronpedia_urls[model]
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; Latent-Mechanism-Multilingual/1.0)"
    }
    print(
        f"Shard {shard_id}/{num_shards}: {len(pending)} pending / {len(assigned)} assigned"
    )
    start = time.time()
    for i, key in enumerate(pending, 1):
        layer, feature_idx = key.split(".")
        url = url_template.format(layer=layer, feature_idx=feature_idx)
        try:
            rebuilt[key] = fetch_description_unlocked(url, headers)
        except Exception as exc:
            print(f"  failed {key}: {exc}")
            rebuilt[key] = ""
        if i % 25 == 0 or i == len(pending):
            save_cache(shard_file, rebuilt)
            elapsed = time.time() - start
            print(f"  shard {shard_id}: {i}/{len(pending)} ({elapsed:.0f}s)")

    save_cache(shard_file, rebuilt)
    nonempty = sum(1 for v in rebuilt.values() if v.strip())
    print(f"Shard {shard_id} done: {len(rebuilt)} entries, {nonempty} non-empty")


def merge_shards(
    model: str,
    cache_path: str,
    output_cache_path: str,
    num_shards: int,
    install: bool,
) -> None:
    merged: dict[str, str] = {}
    for source in [output_cache_path, cache_path]:
        if os.path.exists(source):
            merged.update(load_cache(source))

    shard_files = sorted(glob.glob(f"{output_cache_path}.shard.*.json"))
    if num_shards and len(shard_files) != num_shards:
        print(
            f"Warning: expected {num_shards} shard files, found {len(shard_files)}"
        )
    for path in shard_files:
        merged.update(load_cache(path))

    keys = load_manifest(model)
    missing = [key for key in keys if key not in merged]
    if missing:
        print(f"Warning: {len(missing)} keys still missing after merge")

    save_cache(output_cache_path, {key: merged.get(key, "") for key in keys})
    nonempty = sum(1 for key in keys if merged.get(key, "").strip())
    print(
        f"Merged {len(keys)} keys ({nonempty} non-empty) into {output_cache_path}"
    )
    if install:
        os.replace(output_cache_path, cache_path)
        print(f"Installed cache at {cache_path}")


def run_serial(
    model: str,
    cache_path: str,
    output_cache_path: str,
    extra_dirs: list[str],
    resume: bool,
    backup: bool,
) -> None:
    keys = collect_feature_keys(model, cache_path, extra_dirs)
    if not keys:
        raise SystemExit(f"No feature keys found for model {model}")

    if backup and os.path.exists(cache_path):
        backup_path = f"{cache_path}.clt-hp.bak"
        if not os.path.exists(backup_path):
            with open(cache_path, "r") as src, open(backup_path, "w") as dst:
                dst.write(src.read())
            print(f"Backed up existing cache to {backup_path}")

    rebuilt = load_cache(output_cache_path) if resume else {}
    pending = [key for key in keys if key not in rebuilt] if resume else keys
    url_template = neuronpedia_urls[model]
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; Latent-Mechanism-Multilingual/1.0)"
    }
    print(f"Rebuilding {len(pending)} descriptions for {model}")
    start = time.time()
    for i, key in enumerate(pending, 1):
        layer, feature_idx = key.split(".")
        url = url_template.format(layer=layer, feature_idx=feature_idx)
        try:
            rebuilt[key] = fetch_description_unlocked(url, headers)
        except Exception as exc:
            print(f"  failed {key}: {exc}")
            rebuilt[key] = ""
        if i % 50 == 0 or i == len(pending):
            save_cache(output_cache_path, rebuilt)
            print(f"  {i}/{len(pending)} ({time.time() - start:.0f}s)")

    save_cache(output_cache_path, rebuilt)
    os.replace(output_cache_path, cache_path)
    print(f"Installed cache at {cache_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemma-2-2b", choices=list(neuronpedia_urls))
    parser.add_argument("--cache-path", default=None)
    parser.add_argument("--output-cache-path", default=None)
    parser.add_argument("--extra-dir", action="append", default=[])
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--backup", action="store_true", default=True)
    args = parser.parse_args()

    cache_path = args.cache_path or default_cache_path(args.model)
    output_cache_path = args.output_cache_path or f"{cache_path}.rebuild"
    default_extra = [
        os.path.join(REPO, "data", "flores_features", args.model),
        os.path.join(REPO, "data", "additional_experiments", args.model),
    ]
    extra_dirs = default_extra + args.extra_dir

    if args.write_manifest:
        write_manifest(args.model, cache_path, extra_dirs)
        return

    if args.merge_only:
        if args.num_shards is None:
            raise SystemExit("--merge-only requires --num-shards")
        merge_shards(
            args.model,
            cache_path,
            output_cache_path,
            args.num_shards,
            install=args.install,
        )
        return

    if args.shard_id is not None:
        if args.num_shards is None:
            raise SystemExit("--shard-id requires --num-shards")
        run_shard(
            args.model,
            cache_path,
            output_cache_path,
            args.shard_id,
            args.num_shards,
            resume=args.resume,
        )
        return

    run_serial(
        args.model,
        cache_path,
        output_cache_path,
        extra_dirs,
        resume=args.resume,
        backup=args.backup,
    )


if __name__ == "__main__":
    main()
