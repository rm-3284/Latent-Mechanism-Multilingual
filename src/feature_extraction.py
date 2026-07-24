import argparse
import copy
import fcntl
import heapq
import json
import logging
import os
import random
import time
import requests
import torch
from collections import Counter
from typing import Optional

from circuit_tracer_import import Graph, ReplacementModel, attribute, prune_graph
from models import neuronpedia_urls
from template import base_strings, langs, langs_big, identifiers

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Graph index helpers
# -----------------------------------------------------------------------------
def token_to_idx(graph: Graph, token: int) -> int:
    """Map a token position to the corresponding graph node index."""
    features = len(graph.selected_features)
    errors = graph.cfg.n_layers * graph.n_pos
    if token >= 0:
        return features + errors + token
    else:
        return features + errors + graph.n_pos + token

def logit_to_idx(graph: Graph, logit: int) -> int:
    """Map a logit index to the corresponding graph node index."""
    features = len(graph.selected_features)
    errors = graph.cfg.n_layers * graph.n_pos
    tokens = graph.n_pos
    return features + errors + tokens + logit

def path_reconstruct(start, last, step_dict) -> list[int]:
    """Reconstruct a path from predecessor links."""
    curr = last
    path = [last]
    while curr != start:
        try:
            curr = step_dict[curr]
            path.insert(0, curr)
        except KeyError:
            raise KeyError('Path is disconnected')
    return path

def path_to_edge_weights(graph: Graph, path: list[int]) -> list[float]:
    """Return edge weights along a path in order."""
    path_copy = copy.deepcopy(path)
    last = path[-1]
    if last < len(graph.logit_tokens):
        path_copy[-1] = logit_to_idx(graph, path[-1])
    n = len(path_copy)
    weights = []
    for i in range(n - 1):
        weight = (graph.adjacency_matrix[path_copy[i+1], path_copy[i]]).item()
        weights.append(weight)
    return weights

# Find distinct high-throughput paths using iterative widest-path extraction.
def build_pruned_adjacency_base(
    graph: Graph,
    node_threshold: float,
    edge_threshold: float,
) -> torch.Tensor:
    """Precompute node/edge-pruned adjacency once per graph for path finding."""
    pruned_adjacency_matrix = graph.adjacency_matrix.clone()
    node_mask, edge_mask, _ = prune_graph(graph, node_threshold, edge_threshold)
    n, _ = pruned_adjacency_matrix.shape

    for i in range(n):
        if not node_mask[i]:
            pruned_adjacency_matrix[i, :] = 0.0
            pruned_adjacency_matrix[:, i] = 0.0
    pruned_adjacency_matrix = pruned_adjacency_matrix * edge_mask.float()

    positive_values = (pruned_adjacency_matrix > 0).float()
    return pruned_adjacency_matrix * positive_values


def distinct_path_max_bottleneck(
        graph: Graph, token_idx: int, logit_idx: int,
        throughput_threshold: float = 0.1,
        node_threshold: float = 0.8, edge_threshold: float = 0.98,
        MAX_ITERATIONS: int = 75,
        pruned_adjacency_base: torch.Tensor | None = None,
        ) -> list[list[int]]:

    if throughput_threshold < 0:
        raise ValueError('The throughput threshold cannot be negative')
    if MAX_ITERATIONS <= 0:
        raise ValueError('The maximum number of iterations have to be positive')

    if pruned_adjacency_base is None:
        pruned_adjacency_matrix = build_pruned_adjacency_base(
            graph, node_threshold, edge_threshold
        )
    else:
        pruned_adjacency_matrix = pruned_adjacency_base.clone()

    start_node_idx = token_to_idx(graph, token_idx)
    target_node_idx = logit_to_idx(graph, logit_idx)

    # Exclude the direct token->logit edge to favor mediated paths.
    pruned_adjacency_matrix[target_node_idx, start_node_idx] = 0

    # This mutable copy removes internal nodes after each discovered path.
    matrix_for_distinct_paths = pruned_adjacency_matrix.clone()
    n, _ = matrix_for_distinct_paths.shape

    paths = []
    iteration_counter = 0

    while True:
        iteration_counter += 1
        if iteration_counter > MAX_ITERATIONS:
            print(f"Hit MAX_ITERATIONS ({MAX_ITERATIONS}). Forced termination.")
            return paths

        # Step 1: Find a widest path in the current graph state.

        # `bottleneck_capacity[v]` stores the best min-edge capacity found from source->v.
        bottleneck_capacity = torch.full((n,), float('-inf'))
        bottleneck_capacity[start_node_idx] = float('inf')

        # Max-heap via negated capacities because heapq is a min-heap.
        pq = [(-float('inf'), start_node_idx)]

        current_path_predecessors = {}

        path_found_in_iteration = False

        while pq:
            neg_current_bottleneck, u = heapq.heappop(pq)
            current_bottleneck = -neg_current_bottleneck

            # Skip stale queue entries.
            if current_bottleneck < bottleneck_capacity[u].item():
                continue

            if u == target_node_idx:
                path_found_in_iteration = True
                break

            outgoing_edges_from_u = matrix_for_distinct_paths[:, u]
            valid_neighbors = (outgoing_edges_from_u > throughput_threshold).nonzero().squeeze(1)

            for v_tensor in valid_neighbors:
                v = v_tensor.item()
                edge_capacity_u_v = outgoing_edges_from_u[v].item()

                new_bottleneck = min(current_bottleneck, edge_capacity_u_v)

                if new_bottleneck > bottleneck_capacity[v].item():
                    bottleneck_capacity[v] = new_bottleneck
                    current_path_predecessors[v] = u
                    heapq.heappush(pq, (-new_bottleneck, v))

        if not path_found_in_iteration or bottleneck_capacity[target_node_idx].item() <= throughput_threshold:
            return paths

        # Step 2: Reconstruct the discovered path.
        reconstructed_path = path_reconstruct(start_node_idx, target_node_idx, current_path_predecessors)
        paths.append(reconstructed_path)

        # Step 3: Remove internal nodes so subsequent paths are node-disjoint.
        for node_in_path in reconstructed_path[1:-1]:
            matrix_for_distinct_paths[node_in_path, :] = 0.0
            matrix_for_distinct_paths[:, node_in_path] = 0.0

# -----------------------------------------------------------------------------
# Feature extraction helpers
# -----------------------------------------------------------------------------
def paths_list(*paths_lsts: list[list[int]]) -> list[list[int]]:
    path_sets = []
    for paths_lst in paths_lsts:
        for path in paths_lst:
            path_sets.append(path)
    return path_sets

def create_feature_dict(
    graph: Graph,
    paths: list[list[int]],
    model: str = "gemma-2-2b",
) -> dict[str, str]:
    feature_dict = dict()
    url_template = neuronpedia_urls[model]

    for path in paths:
        features = path[1:-1]
        for feature in features:
            layer, pos, feature_idx = graph.active_features[graph.selected_features[feature]]
            key = f"{layer.item()}.{feature_idx.item()}"
            if feature_dict.get(key) is None:
                response = requests.get(
                    url_template.format(layer=layer.item(), feature_idx=feature_idx.item())
                )
                explanations = response.json().get("explanations", [])
                description = explanations[0]["description"] if explanations else ""
                feature_dict[key] = description

    return feature_dict

def prune_paths_by_first_last(graph: Graph, paths: list[list[int]], threshold_first: float, threshold_last: float) -> list[list[int]]:
    pruned_paths = []
    for path in paths:
        weights = path_to_edge_weights(graph, path)
        if weights[0] > threshold_first and weights[-1] > threshold_last:
            pruned_paths.append(path)
    return pruned_paths

def paths_set_to_json(
    graph: Graph,
    paths: list[list[int]],
    filename: Optional[str] = None,
    feature_dict: Optional[dict[str, str]] = None,
    model: str = "gemma-2-2b",
) -> dict[str, str]:
    feature_set = set()
    for path in paths:
        middle = path[1:-1]
        for feature in middle:
            feature_set.add(feature)

    description_dict = dict()
    if feature_dict is None:
        feature_dict = dict()

    for feature in feature_set:
        layer, pos, feature_idx = graph.active_features[graph.selected_features[feature]]
        layer = layer.item() if isinstance(layer, torch.Tensor) else layer
        pos = pos.item() if isinstance(pos, torch.Tensor) else pos
        feature_idx = feature_idx.item() if isinstance(feature_idx, torch.Tensor) else feature_idx

        description_key = f'{layer}.{pos}.{feature_idx}'
        feature_dict_key = f'{layer}.{feature_idx}'
        try:
            description = feature_dict[feature_dict_key]
            description_dict[description_key] = description
        except KeyError:
            try:
                url_template = neuronpedia_urls[model]
                response = requests.get(
                    url_template.format(layer=layer, feature_idx=feature_idx)
                )
                explanations = response.json().get("explanations", [])
                description = explanations[0]["description"] if explanations else ""
                feature_dict[feature_dict_key] = description
                description_dict[description_key] = description
            except TypeError:
                raise TypeError(f"Layer {layer}, position {pos}, feature {feature_idx} does not exist")
                
    if filename is not None:
        with open(filename, 'w') as file:
            json.dump(description_dict, file, indent=4)

    return description_dict
    
def find_substring(hay: str, needles: list[str]) -> bool:
    for needle in needles:
        if needle in hay:
            return True
    return False


def neuronpedia_api_lock_path() -> str:
    return os.environ.get(
        "NEURONPEDIA_API_LOCK",
        "/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/data/cache/neuronpedia_api.lock",
    )


def neuronpedia_description_cache_path(model: str) -> str:
    cache_dir = os.environ.get(
        "NEURONPEDIA_DESCRIPTION_CACHE_DIR",
        "/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/data/cache/neuronpedia_descriptions",
    )
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{model}_descriptions.json")


def load_neuronpedia_description_cache(cache_path: str) -> dict[str, str]:
    if not os.path.exists(cache_path):
        return {}
    with open(cache_path, "r") as f:
        return json.load(f)


def lookup_neuronpedia_description_cache(cache_path: str, key: str) -> str | None:
    """Read one cached description, reloading from disk under a shared lock."""
    cache_lock_path = f"{cache_path}.lock"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        return load_neuronpedia_description_cache(cache_path).get(key)


def store_neuronpedia_description_cache(
    cache_path: str,
    key: str,
    description: str,
) -> None:
    """Persist one description so other jobs can reuse it immediately."""
    cache_lock_path = f"{cache_path}.lock"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        merged = load_neuronpedia_description_cache(cache_path)
        merged[key] = description
        save_neuronpedia_description_cache(cache_path, merged)


def save_neuronpedia_description_cache(cache_path: str, descriptions: dict[str, str]) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp_path = f"{cache_path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(descriptions, f, indent=2)
    os.replace(tmp_path, cache_path)


def _retry_delay_seconds(
    attempt: int,
    *,
    response: requests.Response | None = None,
    base_delay: float,
    max_delay: float,
) -> float:
    if response is not None and response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        if retry_after is not None:
            try:
                return min(max(float(retry_after), base_delay), max_delay) + random.uniform(0, 1)
            except ValueError:
                pass
        return min(base_delay * (2 ** (attempt + 2)), max_delay) + random.uniform(0, 2)
    return min(base_delay * (2 ** attempt), max_delay) + random.uniform(0, 1)


def fetch_neuronpedia_description(
    url: str,
    headers: dict[str, str],
    key: str,
    *,
    timeout: float = 30.0,
    max_retries: int = 16,
    base_delay: float = 2.0,
    max_delay: float = 120.0,
    min_interval: float = 0.35,
) -> str:
    """Fetch a feature description from Neuronpedia with global throttling and retries."""
    lock_path = neuronpedia_api_lock_path()
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    last_error: requests.exceptions.RequestException | None = None

    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, timeout=timeout)
                if response.status_code == 429:
                    delay = _retry_delay_seconds(
                        attempt,
                        response=response,
                        base_delay=base_delay,
                        max_delay=max_delay,
                    )
                    logger.warning(
                        "Neuronpedia rate-limited for %s (attempt %d/%d); retrying in %.1fs",
                        key,
                        attempt + 1,
                        max_retries,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                response.raise_for_status()
                response_json = response.json()
                explanations = response_json.get("explanations", [])
                time.sleep(min_interval)
                if not explanations:
                    return ""
                return explanations[0]["description"]
            except requests.exceptions.HTTPError as exc:
                last_error = exc
                if exc.response is not None and exc.response.status_code == 429:
                    if attempt + 1 >= max_retries:
                        break
                    delay = _retry_delay_seconds(
                        attempt,
                        response=exc.response,
                        base_delay=base_delay,
                        max_delay=max_delay,
                    )
                    logger.warning(
                        "Neuronpedia rate-limited for %s (attempt %d/%d); retrying in %.1fs",
                        key,
                        attempt + 1,
                        max_retries,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                if attempt + 1 >= max_retries:
                    break
                delay = _retry_delay_seconds(
                    attempt,
                    response=exc.response,
                    base_delay=base_delay,
                    max_delay=max_delay,
                )
                logger.warning(
                    "Neuronpedia request failed for %s (attempt %d/%d): %s; retrying in %.1fs",
                    key,
                    attempt + 1,
                    max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)
            except requests.exceptions.RequestException as exc:
                last_error = exc
                if attempt + 1 >= max_retries:
                    break
                delay = _retry_delay_seconds(
                    attempt,
                    base_delay=base_delay,
                    max_delay=max_delay,
                )
                logger.warning(
                    "Neuronpedia request failed for %s (attempt %d/%d): %s; retrying in %.1fs",
                    key,
                    attempt + 1,
                    max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)

    assert last_error is not None
    logger.error("HTTP request failed for %s after %d attempts: %s", key, max_retries, last_error)
    raise last_error

def pick_last_pos_features(graph: Graph, paths: list[list[int]]) -> list[tuple[int, int]]:
    feature_list = []
    n_pos = graph.n_pos
    for path in paths:
        middle = path[1:-1]
        for feature in middle:
            layer, pos, feature_idx = graph.active_features[graph.selected_features[feature]]
            layer = layer.item() if isinstance(layer, torch.Tensor) else layer
            pos = pos.item() if isinstance(pos, torch.Tensor) else pos
            feature_idx = feature_idx.item() if isinstance(feature_idx, torch.Tensor) else feature_idx

            if pos == n_pos - 1:
                feature_list.append((layer, feature_idx))

    return feature_list

def choose_language_features(
    features: list[tuple[int, int]],
    language_identifiers: list[str],
    feature_dict: Optional[dict[str, str]] = None,
    model: str = "gemma-2-2b",
    description_cache_path: str | None = None,
) -> tuple[dict[str, int], dict[str, str]]:
    lang_feature_dict = dict()
    cache_path = description_cache_path or neuronpedia_description_cache_path(model)
    feature_description_dict = dict(load_neuronpedia_description_cache(cache_path))
    if feature_dict is not None:
        feature_description_dict.update(feature_dict)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }

    if model not in neuronpedia_urls:
        raise ValueError(
            f"Model '{model}' not found in available models. Available: {list(neuronpedia_urls.keys())}"
        )
    url_template = neuronpedia_urls[model]
    feature_counts = Counter((layer, feature_idx) for layer, feature_idx in features)

    for layer, feature_idx in feature_counts:
        key = f"{layer}.{feature_idx}"

        description = feature_description_dict.get(key)
        if not description:
            cached = lookup_neuronpedia_description_cache(cache_path, key)
            if cached:
                description = cached
                feature_description_dict[key] = description

        if not description:
            try:
                url = url_template.format(layer=layer, feature_idx=feature_idx)
                logger.info(f"Fetching feature description from: {url}")
                description = fetch_neuronpedia_description(url, headers, key)
                logger.info(f"Successfully fetched description for {key}")
                store_neuronpedia_description_cache(cache_path, key, description)
                feature_description_dict[key] = description
            except requests.exceptions.RequestException:
                raise
            except KeyError as e:
                logger.error(
                    "Missing key in API response for %s: %s",
                    key,
                    e,
                )
                raise
            except TypeError:
                raise TypeError(f"Layer {layer}, feature {feature_idx} does not exist")

    for (layer, feature_idx), count in feature_counts.items():
        key = f"{layer}.{feature_idx}"
        description = feature_description_dict[key]
        if find_substring(description, language_identifiers):
            lang_feature_dict[key] = lang_feature_dict.get(key, 0) + count

    return lang_feature_dict, feature_description_dict

def iterate_through_data(
        train_data: list[tuple[dict[str, str], dict[str, list[str]]]],
        model: ReplacementModel,
        lang: str,
        base_prompt: str = base_strings['en'],
        important_pos: list[int] = [2, -4, -1],
        logit_focus: list[int] = [0],
        throughput_threshold: float = 0.1,
        node_threshold: float = 0.8, edge_threshold: float = 0.98,
        MAX_ITERATIONS: int = 75,
        threshold_first = 0.5, threshold_last = 0.25,
        max_n_logits = 5, desired_logit_prob = 0.95,
        max_feature_nodes = None, batch_size = 256,
        offload = 'cpu', verbose = True,
        ) -> list[tuple[int, int]]:
    features = []
    if lang not in langs:
        raise KeyError(f"{lang} is not a valid language for this experiment")
    for adj, _ in train_data:
        prompt = base_prompt.format(adj=adj[lang])
        graph = attribute(
            prompt=prompt,
            model=model,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
            batch_size=batch_size,
            max_feature_nodes=max_feature_nodes,
            offload=offload,
            verbose=verbose,
        )
        paths = []
        for pos in important_pos:
            path = []
            for logit in logit_focus:
                p = distinct_path_max_bottleneck(
                    graph, pos, logit, 
                    throughput_threshold=throughput_threshold, 
                    node_threshold=node_threshold, 
                    edge_threshold=edge_threshold, 
                    MAX_ITERATIONS=MAX_ITERATIONS)
                path.extend(p)
            paths.extend(path)
        pruned = prune_paths_by_first_last(graph, paths, threshold_first, threshold_last)
        last_pos_features = pick_last_pos_features(graph, pruned)
        features.extend(last_pos_features)
    return features

def parse_args():
    parser = argparse.ArgumentParser(description='Extract and process language features from FLORES dataset')
    parser.add_argument('--model', type=str, default='gemma-2-2b', 
                        choices=list(neuronpedia_urls.keys()),
                        help='Model to use for feature extraction')
    parser.add_argument('--lang', type=str, default=None, choices=langs_big, help='Language to extract features for')
    parser.add_argument('--data-dir', type=str, default=None, help='Override flores_features directory')
    return parser.parse_args()

if __name__ == '__main__':
    # Keep logging concise for long feature-fetch loops.
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S'
    )
    
    args = parse_args()
    model = args.model
    lang = args.lang

    import os
    current_file_path = __file__
    current_directory = os.path.dirname(current_file_path)
    absolute_directory = os.path.abspath(current_directory)
    data_directory = args.data_dir or os.path.join(os.path.dirname(absolute_directory), "data", "flores_features", model)
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    feature_descriptions = dict()
    for lang in langs_big:
        if args.lang and args.lang != lang:
            continue
        print(f"Processing {lang} with model {model}...")
        
        # Try to load {lang}.json first, fall back to {lang}_short.json
        primary_file = os.path.join(data_directory, f'{lang}.json')
        fallback_file = os.path.join(data_directory, f'{lang}_short.json')
        
        if os.path.exists(primary_file):
            json_file = primary_file
            print(f"  Loading from {lang}.json")
        elif os.path.exists(fallback_file):
            json_file = fallback_file
            print(f"  Loading from {lang}_short.json (fallback)")
        else:
            print(f"  ERROR: Neither {lang}.json nor {lang}_short.json found. Skipping {lang}...")
            continue
        
        with open(json_file, 'r') as f:
            features = json.load(f)
        lang_features, feature_descriptions = choose_language_features(features, identifiers[lang], feature_descriptions, model=model)
        file_name = lang + "_features.json"
        file_path = os.path.join(data_directory, file_name)
        with open(file_path, 'w') as f:
            json.dump(lang_features, f)
        print(f"  Saved {file_name}")
        