import argparse
from concurrent.futures import ProcessPoolExecutor
from datasets import load_dataset
import fcntl
import gc
import json
import os
import pandas as pd
import torch
from typing import Any

from circuit_tracer_import import ReplacementModel, attribute
from circuit_tracer.graph import Graph
from pipeline_data.generic_sentences import alphabet_char, filter_sentences
from device_setup import device
from feature_extraction import (
    build_pruned_adjacency_base,
    distinct_path_max_bottleneck,
    prune_paths_by_first_last,
    pick_last_pos_features,
)
from template import lang_to_flores_key
from models import hf_model_names, hf_transcoder_names

DEFAULT_ATTRIBUTE_KWARGS = {
    "max_n_logits": 5,
    "desired_logit_prob": 0.95,
    "max_feature_nodes": None,
    "batch_size": 4,
    "offload": "cpu",
    "verbose": True,
}


def attribute_prompt(
    model: ReplacementModel,
    prompt: str,
    logit_focus: list[int] | None = None,
    **attribute_kwargs: Any,
) -> Graph:
    kwargs = {**DEFAULT_ATTRIBUTE_KWARGS, **attribute_kwargs}
    return attribute(
        prompt=prompt,
        model=model,
        max_n_logits=kwargs["max_n_logits"],
        desired_logit_prob=kwargs["desired_logit_prob"],
        batch_size=kwargs["batch_size"],
        max_feature_nodes=kwargs["max_feature_nodes"],
        offload=kwargs["offload"],
        verbose=kwargs["verbose"],
    )


def features_from_graph(
    graph: Graph,
    logit_focus: list[int] | None = None,
    throughput_threshold: float = 0.1,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
    MAX_ITERATIONS: int = 75,
    threshold_first: float = 0.5,
    threshold_last: float = 0.25,
) -> list[tuple[int, int]]:
    """Extract traced features from a pre-built attribution graph."""
    if logit_focus is None:
        logit_focus = [0]

    paths: list[list[int]] = []
    pruned_adjacency_base = build_pruned_adjacency_base(
        graph, node_threshold, edge_threshold
    )
    for pos in range(1, graph.n_pos):
        for logit in logit_focus:
            path = distinct_path_max_bottleneck(
                graph,
                pos,
                logit,
                throughput_threshold=throughput_threshold,
                node_threshold=node_threshold,
                edge_threshold=edge_threshold,
                MAX_ITERATIONS=MAX_ITERATIONS,
                pruned_adjacency_base=pruned_adjacency_base,
            )
            paths.extend(path)

    pruned = prune_paths_by_first_last(graph, paths, threshold_first, threshold_last)
    return pick_last_pos_features(graph, pruned)


def _features_from_graph_pt(
    pt_path: str,
    logit_focus: list[int],
    tracing_params: dict[str, float | int],
) -> list[tuple[int, int]]:
    graph = Graph.from_pt(pt_path, map_location="cpu")
    try:
        return features_from_graph(
            graph,
            logit_focus=logit_focus,
            throughput_threshold=tracing_params["throughput_threshold"],
            node_threshold=tracing_params["node_threshold"],
            edge_threshold=tracing_params["edge_threshold"],
            MAX_ITERATIONS=tracing_params["max_iterations"],
            threshold_first=tracing_params["threshold_first"],
            threshold_last=tracing_params["threshold_last"],
        )
    finally:
        del graph


def features_from_graph_cache(
    graph_cache_dir: str,
    num_sentences: int,
    tracing_params: dict[str, float | int],
    logit_focus: list[int] | None = None,
    prune_workers: int = 1,
) -> list[tuple[int, int]]:
    """Load cached attribution graphs and run path pruning only."""
    if logit_focus is None:
        logit_focus = [0]

    pt_paths = [
        os.path.join(graph_cache_dir, f"{idx:04d}.pt") for idx in range(num_sentences)
    ]
    missing = [path for path in pt_paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} cached graph(s) under {graph_cache_dir}; "
            f"first missing: {missing[0]}"
        )

    features: list[tuple[int, int]] = []
    if prune_workers > 1:
        chunksize = max(1, len(pt_paths) // (prune_workers * 4))
        with ProcessPoolExecutor(max_workers=prune_workers) as pool:
            for sentence_features in pool.map(
                _features_from_graph_pt,
                pt_paths,
                [logit_focus] * num_sentences,
                [tracing_params] * num_sentences,
                chunksize=chunksize,
            ):
                features.extend(sentence_features)
        return features

    for pt_path in pt_paths:
        features.extend(
            _features_from_graph_pt(pt_path, logit_focus, tracing_params)
        )
    return features


def graph_cache_manifest_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "manifest.json")


def ensure_attribution_graph_cache(
    model: ReplacementModel,
    lang: str,
    sentences: list[str],
    cache_dir: str,
    logit_focus: list[int] | None = None,
    **attribute_kwargs: Any,
) -> str:
    """Attribute each sentence once and save graphs to cache_dir/{idx:04d}.pt."""
    if logit_focus is None:
        logit_focus = [0]

    os.makedirs(cache_dir, exist_ok=True)
    manifest_path = graph_cache_manifest_path(cache_dir)
    manifest = {
        "lang": lang,
        "num_sentences": len(sentences),
        "sentences": sentences,
    }
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            existing = json.load(f)
        if existing.get("sentences") != sentences:
            raise ValueError(
                f"Graph cache at {cache_dir} was built for different sentences; "
                "delete the directory or use --no-cache to rebuild."
            )
    else:
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    num_cached = sum(
        1 for idx in range(len(sentences))
        if os.path.exists(os.path.join(cache_dir, f"{idx:04d}.pt"))
    )
    if num_cached == len(sentences):
        print(f"  attribution graphs already cached for {lang} ({len(sentences)} sentences)")
        return cache_dir

    print(
        f"  building attribution graph cache for {lang} "
        f"({num_cached}/{len(sentences)} already cached)"
    )
    for idx, prompt in enumerate(sentences):
        pt_path = os.path.join(cache_dir, f"{idx:04d}.pt")
        if os.path.exists(pt_path):
            continue
        lock_path = f"{pt_path}.lock"
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            if os.path.exists(pt_path):
                continue
            print(f"    attributing {lang} sentence {idx + 1}/{len(sentences)}")
            graph = attribute_prompt(
                model, prompt, logit_focus=logit_focus, **attribute_kwargs
            )
            graph.to_pt(pt_path)
            del graph
            torch.cuda.empty_cache()
            gc.collect()

    return cache_dir


def iterate_through_sentences(
        model: ReplacementModel,
        sentences: list[str],
        logit_focus: list[int] = [0],
        throughput_threshold: float = 0.1,
        node_threshold: float = 0.8, edge_threshold: float = 0.98,
        MAX_ITERATIONS: int = 75,
        threshold_first = 0.5, threshold_last = 0.25,
        max_n_logits = 5, desired_logit_prob = 0.95,
        max_feature_nodes = None, batch_size = 4,
        offload = 'cpu', verbose = True,
        ) -> list[tuple[int, int]]:
    tracing_params = {
        "throughput_threshold": throughput_threshold,
        "node_threshold": node_threshold,
        "edge_threshold": edge_threshold,
        "max_iterations": MAX_ITERATIONS,
        "threshold_first": threshold_first,
        "threshold_last": threshold_last,
    }
    attribute_kwargs = {
        "max_n_logits": max_n_logits,
        "desired_logit_prob": desired_logit_prob,
        "max_feature_nodes": max_feature_nodes,
        "batch_size": batch_size,
        "offload": offload,
        "verbose": verbose,
    }
    features = []
    for prompt in sentences:
        graph = attribute_prompt(model, prompt, logit_focus=logit_focus, **attribute_kwargs)
        features.extend(
            features_from_graph(
                graph,
                logit_focus=logit_focus,
                **tracing_params,
            )
        )
        del graph
        torch.cuda.empty_cache()
        gc.collect()

    return features

def argsparse():
    parser = argparse.ArgumentParser(description='Extract features from FLORES dataset')
    parser.add_argument('--model', type=str, default='gemma-2-2b', choices=hf_model_names.keys(), help='Model to use for feature extraction')
    parser.add_argument('--lang', type=str, default=None, choices=lang_to_flores_key.keys(), help='Language to extract features for')
    parser.add_argument('--output-dir', type=str, default=None, help='Override output directory (default: data/flores_features/<model>)')
    parser.add_argument('--num-sentences', type=int, default=100, help='Number of FLORES sentences per language')
    parser.add_argument('--throughput-threshold', type=float, default=0.1)
    parser.add_argument('--node-threshold', type=float, default=0.8)
    parser.add_argument('--edge-threshold', type=float, default=0.98)
    parser.add_argument('--max-iterations', type=int, default=75)
    parser.add_argument('--threshold-first', type=float, default=0.5)
    parser.add_argument('--threshold-last', type=float, default=0.25)
    return parser.parse_args()

if __name__ == "__main__":
    args = argsparse()
    current_file_path = __file__
    current_directory = os.path.dirname(current_file_path)
    absolute_directory = os.path.abspath(current_directory)
    data_directory = args.output_dir or os.path.join(os.path.dirname(absolute_directory), "data", "flores_features", args.model)
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    model_name = args.model
    transcoder_name = hf_transcoder_names[model_name]
    model = ReplacementModel.from_pretrained(hf_model_names[model_name], transcoder_name, device=device, dtype=torch.bfloat16)
    
    for lang, ds_key in lang_to_flores_key.items():
        if args.lang and args.lang != lang:
            continue
        print(f"Loading {ds_key}")
        # Use streaming to save memory
        ds = load_dataset("openlanguagedata/flores_plus", ds_key, split="dev")
        ds = ds.shuffle(seed=42)
        batch = [example['text'] for i, example in enumerate(ds) if i < 150]
        sentences = filter_sentences(batch, alphabet_char[lang], model, num_sentences=args.num_sentences)
        del batch
        features = iterate_through_sentences(
            model,
            sentences,
            max_feature_nodes=None,
            throughput_threshold=args.throughput_threshold,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
            MAX_ITERATIONS=args.max_iterations,
            threshold_first=args.threshold_first,
            threshold_last=args.threshold_last,
        )
        file_name = f'{lang}.json'
        with open(os.path.join(data_directory, file_name), 'w') as f:
            json.dump(features, f)
        del features, sentences
        torch.cuda.empty_cache()
        gc.collect()
