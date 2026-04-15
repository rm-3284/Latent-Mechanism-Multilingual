"""
Compute cosine similarity between feature extraction methods.

This script:
1. Extracts features from three methods (description-based, value-based, frequency-based)
2. Uses the decoding weight matrix (W_dec) from ReplacementModel to convert features to activation space
3. Computes average direction vectors for each layer
4. Calculates cosine similarity between methods
"""

import argparse
import json
import os
import torch
import torch.nn.functional as F

from ablation_amplification_intervention import (
    description_based_features,
    mean_value_based_features,
    freq_based_features,
)
from circuit_tracer_import import ReplacementModel
from device_setup import device
from template import lang_to_flores_key
from models import hf_model_names, hf_transcoder_names, layer_num


def extract_features_from_method(
    feature_dict: dict[str, list[str]],
    langs: list[str]
) -> dict[str, torch.Tensor]:
    """
    Convert feature strings (layer.feature_idx) to tensors of feature indices per layer.
    
    Returns:
        dict[str, torch.Tensor]: Maps language to (n_features,) tensor of feature indices
    """
    lang_features = {}
    for lang in langs:
        features_by_layer = {}
        for feature_str in feature_dict[lang]:
            parts = feature_str.split('.')
            layer = int(parts[0])
            feature_idx = int(parts[1])
            
            if layer not in features_by_layer:
                features_by_layer[layer] = []
            features_by_layer[layer].append(feature_idx)
        
        lang_features[lang] = features_by_layer
    
    return lang_features


def get_feature_vectors_in_activation_space(
    model: ReplacementModel,
    lang_features: dict[str, dict[int, list[int]]],
    num_layers: int
) -> dict[str, dict[int, torch.Tensor]]:
    """
    Convert feature indices to vectors in activation space using W_dec.
    
    For each language and layer:
    - Extract the relevant rows from W_dec (decoder weight matrix)
    - These rows form the basis for the feature directions in activation space
    
    Returns:
        dict[str, dict[int, torch.Tensor]]: 
            Maps language -> layer -> (n_features, d_model) activation space vectors
    """
    feature_vectors = {}
    
    for lang in lang_features.keys():
        feature_vectors[lang] = {}
        
        for layer_idx in range(num_layers):
            if layer_idx not in lang_features[lang]:
                feature_vectors[lang][layer_idx] = None
                continue
            
            feature_indices = lang_features[lang][layer_idx]
            transcoder = model.transcoders[layer_idx]
            W_dec = transcoder.W_dec  # (n_features, d_model)
            
            # Extract rows corresponding to the selected features
            feature_vecs = W_dec[feature_indices]  # (n_selected_features, d_model)
            feature_vectors[lang][layer_idx] = feature_vecs
    
    return feature_vectors


def compute_average_direction_per_layer(
    feature_vectors: dict[str, dict[int, torch.Tensor]],
    num_layers: int
) -> dict[str, dict[int, torch.Tensor]]:
    """
    Compute average direction vector for each layer.
    
    Returns:
        dict[str, dict[int, torch.Tensor]]: 
            Maps language -> layer -> (d_model,) average direction vector
    """
    avg_directions = {}
    
    for lang, layer_vectors in feature_vectors.items():
        avg_directions[lang] = {}
        
        for layer_idx in range(num_layers):
            if layer_idx not in layer_vectors or layer_vectors[layer_idx] is None:
                avg_directions[lang][layer_idx] = None
                continue
            
            feature_vecs = layer_vectors[layer_idx]  # (n_features, d_model)
            # Average across features
            avg_dir = feature_vecs.mean(dim=0)  # (d_model,)
            avg_directions[lang][layer_idx] = avg_dir
    
    return avg_directions


def compute_cosine_similarity(
    avg_directions1: dict[int, torch.Tensor],
    avg_directions2: dict[int, torch.Tensor],
    num_layers: int
) -> dict[int, float]:
    """
    Compute cosine similarity between two sets of average direction vectors.
    
    Returns:
        dict[int, float]: Maps layer index to cosine similarity
    """
    similarities = {}
    
    for layer_idx in range(num_layers):
        if (layer_idx not in avg_directions1 or avg_directions1[layer_idx] is None or
            layer_idx not in avg_directions2 or avg_directions2[layer_idx] is None):
            similarities[layer_idx] = None
            continue
        
        vec1 = avg_directions1[layer_idx]
        vec2 = avg_directions2[layer_idx]
        
        # Compute cosine similarity
        cos_sim = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()
        similarities[layer_idx] = cos_sim
    
    return similarities


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute cosine similarity between feature extraction methods",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--model',
        '-m',
        type=str,
        default='gemma-2-2b',
        choices=hf_model_names.keys(),
        help='Model to use for the experiment',
    )
    parser.add_argument(
        '--output-dir',
        '-o',
        type=str,
        default="../data/additional_experiments/",
        help='Output directory to save results',
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Load the model
    model_name = args.model
    transcoder_name = hf_transcoder_names.get(model_name, "gemma")
    print(f"Loading model {model_name}...")
    model = ReplacementModel.from_pretrained(
        hf_model_names[model_name], 
        transcoder_name, 
        device=device, 
        dtype=torch.bfloat16
    )
    
    # Setup directories
    current_file_path = __file__
    current_directory = os.path.dirname(current_file_path)
    absolute_directory = os.path.abspath(current_directory)
    data_directory = os.path.join(os.path.dirname(absolute_directory), "data")
    flores_directory = os.path.join(data_directory, "flores_features", model_name)
    lang_specific_directory = os.path.join(data_directory, "language_specific_features", model_name)
    multilingual_features_directory = os.path.join(data_directory, "multilingual_llm_features", model_name)
    
    # Get number of layers
    num_layers = layer_num[model_name]
    langs = list(lang_to_flores_key.keys())
    
    # Extract features using the three methods
    print("Extracting features from three methods...")
    desc_features = description_based_features(flores_directory, langs, 0.1)
    val_features = mean_value_based_features(multilingual_features_directory, langs, 50)
    freq_features = freq_based_features(lang_specific_directory, langs)
    
    # Convert to activation space vectors
    print("Converting features to activation space vectors...")
    desc_lang_features = extract_features_from_method(desc_features, langs)
    val_lang_features = extract_features_from_method(val_features, langs)
    freq_lang_features = extract_features_from_method(freq_features, langs)
    
    desc_feature_vectors = get_feature_vectors_in_activation_space(
        model, desc_lang_features, num_layers
    )
    val_feature_vectors = get_feature_vectors_in_activation_space(
        model, val_lang_features, num_layers
    )
    freq_feature_vectors = get_feature_vectors_in_activation_space(
        model, freq_lang_features, num_layers
    )
    
    # Compute average direction vectors per layer
    print("Computing average direction vectors...")
    desc_avg_dirs = compute_average_direction_per_layer(desc_feature_vectors, num_layers)
    val_avg_dirs = compute_average_direction_per_layer(val_feature_vectors, num_layers)
    freq_avg_dirs = compute_average_direction_per_layer(freq_feature_vectors, num_layers)
    
    # Compute pairwise cosine similarities
    print("Computing cosine similarities...")
    desc_vs_val = {}
    desc_vs_freq = {}
    val_vs_freq = {}
    
    for lang in langs:
        desc_vs_val[lang] = compute_cosine_similarity(
            desc_avg_dirs[lang], val_avg_dirs[lang], num_layers
        )
        desc_vs_freq[lang] = compute_cosine_similarity(
            desc_avg_dirs[lang], freq_avg_dirs[lang], num_layers
        )
        val_vs_freq[lang] = compute_cosine_similarity(
            val_avg_dirs[lang], freq_avg_dirs[lang], num_layers
        )
    
    # Compute summary statistics
    print("\n" + "="*80)
    print("COSINE SIMILARITY RESULTS")
    print("="*80 + "\n")
    
    # Per-language and per-layer summary
    for lang in langs:
        print(f"\nLanguage: {lang}")
        print("-" * 80)
        print(f"{'Layer':<10} {'Desc vs Val':<20} {'Desc vs Freq':<20} {'Val vs Freq':<20}")
        print("-" * 80)
        
        for layer_idx in range(num_layers):
            desc_val = desc_vs_val[lang][layer_idx]
            desc_freq = desc_vs_freq[lang][layer_idx]
            val_freq = val_vs_freq[lang][layer_idx]
            
            desc_val_str = f"{desc_val:.4f}" if desc_val is not None else "N/A"
            desc_freq_str = f"{desc_freq:.4f}" if desc_freq is not None else "N/A"
            val_freq_str = f"{val_freq:.4f}" if val_freq is not None else "N/A"
            
            print(f"{layer_idx:<10} {desc_val_str:<20} {desc_freq_str:<20} {val_freq_str:<20}")
    
    # Summary across languages and layers
    print("\n" + "="*80)
    print("AVERAGE SIMILARITIES ACROSS LANGUAGES AND LAYERS")
    print("="*80 + "\n")
    
    all_desc_val = []
    all_desc_freq = []
    all_val_freq = []
    
    for lang in langs:
        for layer_idx in range(num_layers):
            if desc_vs_val[lang][layer_idx] is not None:
                all_desc_val.append(desc_vs_val[lang][layer_idx])
            if desc_vs_freq[lang][layer_idx] is not None:
                all_desc_freq.append(desc_vs_freq[lang][layer_idx])
            if val_vs_freq[lang][layer_idx] is not None:
                all_val_freq.append(val_vs_freq[lang][layer_idx])
    
    print(f"Description vs Value-based:   {sum(all_desc_val)/len(all_desc_val):.4f}")
    print(f"Description vs Frequency-based: {sum(all_desc_freq)/len(all_desc_freq):.4f}")
    print(f"Value-based vs Frequency-based: {sum(all_val_freq)/len(all_val_freq):.4f}")
    
    # Per-layer average
    print("\n" + "-"*80)
    print("Average similarity per layer (across languages)")
    print("-"*80)
    print(f"{'Layer':<10} {'Desc vs Val':<20} {'Desc vs Freq':<20} {'Val vs Freq':<20}")
    print("-"*80)
    
    for layer_idx in range(num_layers):
        layer_desc_val = [desc_vs_val[lang][layer_idx] for lang in langs 
                         if desc_vs_val[lang][layer_idx] is not None]
        layer_desc_freq = [desc_vs_freq[lang][layer_idx] for lang in langs 
                          if desc_vs_freq[lang][layer_idx] is not None]
        layer_val_freq = [val_vs_freq[lang][layer_idx] for lang in langs 
                         if val_vs_freq[lang][layer_idx] is not None]
        
        desc_val_str = f"{sum(layer_desc_val)/len(layer_desc_val):.4f}" if layer_desc_val else "N/A"
        desc_freq_str = f"{sum(layer_desc_freq)/len(layer_desc_freq):.4f}" if layer_desc_freq else "N/A"
        val_freq_str = f"{sum(layer_val_freq)/len(layer_val_freq):.4f}" if layer_val_freq else "N/A"
        
        print(f"{layer_idx:<10} {desc_val_str:<20} {desc_freq_str:<20} {val_freq_str:<20}")
    
    # Optionally save results to JSON
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        
        results = {
            "desc_vs_val": {lang: {str(k): v for k, v in desc_vs_val[lang].items()} 
                           for lang in langs},
            "desc_vs_freq": {lang: {str(k): v for k, v in desc_vs_freq[lang].items()} 
                            for lang in langs},
            "val_vs_freq": {lang: {str(k): v for k, v in val_vs_freq[lang].items()} 
                           for lang in langs},
        }
        
        output_file = os.path.join(args.output_dir, f"{model_name}_cosine_similarities.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")
