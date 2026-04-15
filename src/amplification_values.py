import argparse
from datasets import load_dataset
import gc
import json
import math
import matplotlib.pyplot as plt
import os
import statistics
import torch
from typing import Optional
from tqdm import tqdm

from pipeline_data.generic_sentences import alphabet_char, filter_sentences
from device_setup import device
from circuit_tracer_import import Graph, Supernode, Feature, ReplacementModel, attribute
from template import lang_to_flores_key
from models import hf_model_names, hf_transcoder_names

def set_features_from_supernodes(*supernodes: Supernode) -> list[Feature]:
    feature_lst = []
    for supernode in supernodes:
        feature_lst.extend(supernode.features)
    return list(set(feature_lst))

def feature_find(graph: Graph, feature: str) -> Optional[int]:
    layer, feature_idx = feature.split(".")
    layer = int(layer)
    feature_idx = int(feature_idx)
    pos = graph.n_pos - 1
    feature_tensor = torch.tensor([layer, pos, feature_idx], device=device)

    element_wise_match = (graph.active_features == feature_tensor)
    row_match = torch.all(element_wise_match, dim=1)
    matching_indices = torch.where(row_match)[0]
    if matching_indices.numel() > 1:
        raise ValueError('Multiple matching rows')
    elif matching_indices.numel() == 0:
        return None
    return matching_indices.item()

def feature_find_for_every_pos(graph: Graph, feature: str) -> list[int]:
    layer, feature_idx = feature.split(".")
    layer = int(layer)
    feature_idx = int(feature_idx)

    idx_list = []
    for pos in range(graph.n_pos):
        feature_tensor = torch.tensor([layer, pos, feature_idx], device=device)
        element_wise_match = (graph.active_features == feature_tensor)
        row_match = torch.all(element_wise_match, dim=1)
        matching_indices = torch.where(row_match)[0]
        if matching_indices.numel() > 1:
            raise ValueError('Multiple matching rows')
        elif matching_indices.numel() !=0:
            idx_list.append(matching_indices.item())
    return idx_list

def get_feature_activation_from_prompt(
        prompt: str, feature_list: list[str], model: ReplacementModel,
        max_n_logits = 5, desired_logit_prob = 0.95,
        max_feature_nodes = None, batch_size = 8,
        offload = 'cpu', verbose = True,
        ) -> list[float]:
    graph = attribute(
        prompt=prompt,
        model=model,
        max_n_logits=max_n_logits,
        desired_logit_prob=desired_logit_prob,
        batch_size=batch_size,
        max_feature_nodes=max_feature_nodes,
        offload=offload,
        verbose=verbose
    )
    activation_list = []
    for feature in feature_list:
        idx = feature_find(graph, feature)
        if idx is None:
            activation_list.append(float('nan'))
        else:
            activation_value = graph.activation_values[idx]
            activation_value = activation_value.item() if isinstance(activation_value, torch.Tensor) else activation_value
            activation_list.append(activation_value)
    del graph
    torch.cuda.empty_cache()
    return activation_list

def get_feature_activation_for_every_pos(
    prompt: str, feature_list: list[str], model: ReplacementModel,
        max_n_logits = 5, desired_logit_prob = 0.95,
        max_feature_nodes = None, batch_size = 4,
        offload = 'cpu', verbose = True,
        ) -> list[list[float]]:
    graph = attribute(
        prompt=prompt,
        model=model,
        max_n_logits=max_n_logits,
        desired_logit_prob=desired_logit_prob,
        batch_size=batch_size,
        max_feature_nodes=max_feature_nodes,
        offload=offload,
        verbose=verbose
    )
    activation_list = []
    for feature in feature_list:
        idx_list = feature_find_for_every_pos(graph, feature)
        if len(idx_list) == 0:
            activation_list.append([float('nan')])
        else:
            activation_list_for_each_feature = []
            for idx in idx_list:
                activation_value = graph.activation_values[idx]
                activation_value = activation_value.item() if isinstance(activation_value, torch.Tensor) else activation_value
                activation_list_for_each_feature.append(activation_value)
            activation_list.append(activation_list_for_each_feature)
    
    del graph
    return activation_list

def iterate_over_sentences(prompts: list[str], feature_list: list[str], model: ReplacementModel) -> dict[str, list[float]]:
    activation_values_dict = dict()
    for feature in feature_list:
        activation_values_dict[feature] = []
    for prompt in prompts:
        activation_list = get_feature_activation_from_prompt(prompt, feature_list, model)
        for i, feature in enumerate(feature_list):
            activation_values_dict[feature].append(activation_list[i])
        torch.cuda.empty_cache()
        gc.collect()
    return activation_values_dict

def iterate_every_pos_feature_activation(prompts: list[str], feature_list: list[str], model: ReplacementModel) -> dict[str, list[float]]:
    activation_values_dict = dict()
    for feature in feature_list:
        activation_values_dict[feature] = []
    for prompt in tqdm(prompts, desc="Prompts (every pos)"):
        activation_list = get_feature_activation_for_every_pos(prompt, feature_list, model)
        for i, feature in enumerate(feature_list):
            activation_values_dict[feature].extend(activation_list[i])
        torch.cuda.empty_cache()
        gc.collect()
    return activation_values_dict

def make_histogram_from_values_dict(data: list[float], bins: int = 30, title: str = "Histogram of Data (NaNs Excluded)", xlabel: str = "Value", ylabel: str = "Frequency") -> None:
    clean_data = []
    nan_count = 0
    
    for x in data:
        if math.isnan(x):
            nan_count += 1
        else:
            clean_data.append(x)

    print(f"Total NaN values found: {nan_count}")
    if not clean_data:
        print("No valid data points to plot after excluding NaNs.")
        return
    
    maximum = max(clean_data)
    minimum = min(clean_data)
    mean = statistics.mean(clean_data)
    median = statistics.median(clean_data)
    print(f'Max {maximum}, Min {minimum}, Mean {mean}, Median {median}')
    
    plt.figure(figsize=(10, 6)) # Set figure size for better readability
    plt.hist(clean_data, bins=bins, edgecolor='black', alpha=0.7) # 'edgecolor' for bin borders, 'alpha' for transparency
    
    # Add labels and title
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Add a text box to display NaN count on the plot
    plt.text(0.95, 0.95, f'NaNs: {nan_count}, Max: {maximum}, Min: {minimum}, Mean: {mean}, Median: {median}', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
             
    plt.grid(axis='y', alpha=0.75) # Add a grid for better readability
    plt.show()
    return

def print_feature(feature: Feature) -> str:
    layer = feature.layer
    feature_idx = feature.feature_idx
    layer = layer.item() if isinstance(layer, torch.Tensor) else layer
    feature_idx = feature_idx.item() if isinstance(feature_idx, torch.Tensor) else feature_idx
    return f'Layer {layer}, feature_idx {feature_idx}'

def summarize(features: list[float]) -> tuple[int, int, float, float, float, float]:
    # return value is number of nans, total number excluding nans, min, max, mean, median
    clean_data = []
    nan_count = 0
    
    for x in features:
        if math.isnan(x):
            nan_count += 1
        else:
            clean_data.append(x)
    
    if len(clean_data) != 0:
        maximum = max(clean_data)
        minimum = min(clean_data)
        mean = statistics.mean(clean_data)
        median = statistics.median(clean_data)
    else:
        maximum = float('nan')
        minimum = float('nan')
        mean = float('nan')
        median = float('nan')
    return nan_count, len(clean_data), minimum, maximum, mean, median

def argsparse():
    parser = argparse.ArgumentParser(description='Calculate amplification values for features extracted from FLORES dataset')
    parser.add_argument('--model', type=str, default='gemma-2-2b', choices=hf_model_names.keys(), help='Model to use for calculating amplification values')
    parser.add_argument('--lang', type=str, default=None, choices=lang_to_flores_key.keys(), help='Language to calculate amplification values for (if not specified, calculates for all languages)')
    parser.add_argument('--start-idx', type=int, default=0, help='Start index for sentence batch (inclusive)')
    parser.add_argument('--end-idx', type=int, default=None, help='End index for sentence batch (exclusive)')
    return parser.parse_args()

if __name__ == "__main__":
    args = argsparse()
    model_name = args.model
    transcoder_name = hf_transcoder_names[model_name]
    model = ReplacementModel.from_pretrained(hf_model_names[model_name], transcoder_name, device=device, dtype=torch.bfloat16)

    current_file_path = __file__
    current_directory = os.path.dirname(current_file_path)
    absolute_directory = os.path.abspath(current_directory)
    data_directory = os.path.join(os.path.dirname(absolute_directory), "data")
    flores_directory = os.path.join(data_directory, "flores_features", args.model)
    lang_specific_directory = os.path.join(data_directory, "language_specific_features", args.model)
    multilingual_features_directory = os.path.join(data_directory, "multilingual_llm_features", args.model)

    amplification_value_directory = os.path.join(data_directory, "amplification_values", args.model)
    os.makedirs(amplification_value_directory, exist_ok=True)

    for lang, ds_key in lang_to_flores_key.items():
        if args.lang is not None and lang != args.lang:
            continue

        file_name = f"{lang}.json"
        if os.path.exists(os.path.join(amplification_value_directory, file_name)):
            continue

        print(f"Loading {ds_key}")
        # Use streaming to save memory
        ds = load_dataset("openlanguagedata/flores_plus", ds_key, split="dev")
        ds = ds.shuffle(seed=42)
        batch = [example['text'] for i, example in enumerate(ds) if i < 150]
        sentences = filter_sentences(batch, alphabet_char[lang], model, num_sentences=30) # only returns 100 sentences
        del batch

        # features
        file_name = f"{lang}.json"
        feature_set = set()
        #if os.path.exists(os.path.join(flores_directory, file_name)):
        with open(os.path.join(flores_directory, file_name), 'r') as f:
                flores_features = json.load(f)
        print(f"Loaded features from {flores_directory} for {lang} in {file_name}")
        #else:
        #    with open(os.path.join(flores_directory, f"{lang}_short.json"), 'r') as f:
        #        flores_features = json.load(f)
        #    print(f"Loaded features from {flores_directory} for {lang} in {lang}_short.json")
        for layer, feature_idx in flores_features:
            key = f"{layer}.{feature_idx}"
            feature_set.add(key)
        
        # Updated to read from sparse format
        sparse_file_name = f"{lang}_sparse_long.json"
        with open(os.path.join(lang_specific_directory, sparse_file_name), 'r') as f:
            sparse_data = json.load(f)
            lang_specific_features = sparse_data['max_values']
        for key in lang_specific_features.keys():
            feature_set.add(key)
        
        file_name = f"{lang}_long.json"
        with open(os.path.join(multilingual_features_directory, file_name), 'r') as f:
            multilingual_features = json.load(f)
        for key in multilingual_features.keys():
            feature_set.add(key)

        file_name = f"{lang}.json"
        feature_list = list(feature_set)
        activation_dict = iterate_over_sentences(sentences, feature_list, model)
        with open(os.path.join(amplification_value_directory, file_name), 'w') as f:
            json.dump(activation_dict, f)
        
        del activation_dict, sentences, feature_list, feature_set
        torch.cuda.empty_cache()
        gc.collect()

    for lang in lang_to_flores_key.keys():
        if args.lang is not None and lang != args.lang:
            continue

        file_name = f"{lang}.json"
        summarized_file_name = f"{lang}_summary.json"
        if os.path.exists(os.path.join(amplification_value_directory, summarized_file_name)):
            continue

        with open(os.path.join(amplification_value_directory, file_name), 'r') as f:
            feature_dict = json.load(f)
        
        feature_summarized = dict()
        for key, features in feature_dict.items():
            summarized = summarize(features)
            feature_summarized[key] = summarized
        
        with open(os.path.join(amplification_value_directory, summarized_file_name), 'w') as f:
            json.dump(feature_summarized, f)
        
        del feature_dict, feature_summarized
        torch.cuda.empty_cache()
        gc.collect()
    
    # Cleanup after first two loops
    torch.cuda.empty_cache()
    gc.collect()
    
    # Reload model for next phase
    #model = ReplacementModel.from_pretrained(hf_model_names[model_name], transcoder_name, device=device, dtype=torch.bfloat16)

    for lang, ds_key in lang_to_flores_key.items():
        if args.lang is not None and lang != args.lang:
            continue

        file_name = f"{lang}.json"
        # Determine output file name based on batch indices
        start_idx = args.start_idx
        # We'll determine end_idx after batch is created
        print(f"Loading {ds_key}")
        # Use streaming to save memory
        ds = load_dataset("openlanguagedata/flores_plus", ds_key, split="dev")
        ds = ds.shuffle(seed=42)
        max_sentence_length = 100  # character limit
        batch = [example['text'] for example in ds if len(example['text']) < max_sentence_length]
        end_idx = args.end_idx if args.end_idx is not None else len(batch)
        batch = batch[start_idx:end_idx]
        destination_file_name = f"{lang}_every_pos_{start_idx}_{end_idx}.json"
        if os.path.exists(os.path.join(amplification_value_directory, destination_file_name)):
            continue
        print(f"Processing sentences {start_idx} to {end_idx} for {lang}")
        print(batch)
        feature_set = set()

        if os.path.exists(os.path.join(flores_directory, file_name)):
            with open(os.path.join(flores_directory, file_name), 'r') as f:
                flores_features = json.load(f)
            print(f"Loaded features from {flores_directory} for {lang} in {file_name}")
        else:
            with open(os.path.join(flores_directory, f"{lang}_short.json"), 'r') as f:
                flores_features = json.load(f)
            print(f"Loaded features from {flores_directory} for {lang} in {lang}_short.json")
        for layer, feature_idx in flores_features:
            key = f"{layer}.{feature_idx}"
            feature_set.add(key)
        
        # Updated to read from sparse format
        sparse_file_name = f"{lang}_sparse.json"
        with open(os.path.join(lang_specific_directory, sparse_file_name), 'r') as f:
            sparse_data = json.load(f)
            lang_specific_features = sparse_data['max_values']
        for key in lang_specific_features.keys():
            feature_set.add(key)
        
        with open(os.path.join(multilingual_features_directory, file_name), 'r') as f:
            multilingual_features = json.load(f)
        for key in multilingual_features.keys():
            feature_set.add(key)

        feature_list = list(feature_set)
        activation_dict = iterate_every_pos_feature_activation(batch, feature_list, model)
        with open(os.path.join(amplification_value_directory, destination_file_name), 'w') as f:
            json.dump(activation_dict, f)
        
        del activation_dict, batch, feature_list, feature_set
        torch.cuda.empty_cache()
        gc.collect()

    """
    for lang in lang_to_flores_key.keys():
        if args.lang is not None and lang != args.lang:
            continue

        file_name = f"{lang}_every_pos.json"
        summarized_file_name = f"{lang}_pos_summary.json"
        if os.path.exists(os.path.join(amplification_value_directory, summarized_file_name)):
            continue

        with open(os.path.join(amplification_value_directory, file_name), 'r') as f:
            feature_dict = json.load(f)
        
        feature_summarized = dict()
        for key, features in feature_dict.items():
            summarized = summarize(features)
            feature_summarized[key] = summarized
        
        with open(os.path.join(amplification_value_directory, summarized_file_name), 'w') as f:
            json.dump(feature_summarized, f)
        
        del feature_dict, feature_summarized
        torch.cuda.empty_cache()
        gc.collect()
    """
    # Final cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()
