import argparse
from datasets import load_dataset
import gc
import glob
import json
import os
import requests
import torch
import torch.nn.functional as F
from collections import defaultdict

from circuit_tracer_import import attribute, ReplacementModel
from device_setup import device
from template import lang_to_flores_key, langs_big, identifiers
from models import hf_model_names, hf_transcoder_names, neuronpedia_urls, layer_num, n_features

def get_activation_vector(
        prompt: str, 
        model: ReplacementModel,
        n_layers = 26,
        max_n_logits = 5,
        desired_logit_prob = 0.95,
        max_feature_nodes = None,
        batch_size = 4,
        offload = 'cpu',
        verbose = True,
        ) -> tuple[dict[str, int], int, dict[str, float]]:
    """
    Sparse dictionary version - no memory limits needed!
    Returns activation counts as dict instead of dense tensor.
    """
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
    active_features = graph.active_features # (n_active_features, 3) containing (layer, pos, feature_idx)
    activation_values = graph.activation_values
    
    # Use sparse dictionaries instead of dense tensors
    activation_counts = defaultdict(int)
    max_values_dict = dict()
    n_pos = graph.n_pos
    
    for i, (layer, pos, feature_idx) in enumerate(active_features):
        layer = int(layer) if isinstance(layer, torch.Tensor) else layer
        feature_idx = int(feature_idx) if isinstance(feature_idx, torch.Tensor) else feature_idx
        
        key = f"{layer}.{feature_idx}"
        activation_counts[key] += 1
        
        activation_value = activation_values[i]
        activation_value = activation_value.item() if isinstance(activation_value, torch.Tensor) else activation_value
        max_values_dict[key] = max(max_values_dict.get(key, 0), activation_value)
    
    del graph, active_features, activation_values
    torch.cuda.empty_cache()
    gc.collect()
    
    return dict(activation_counts), n_pos, max_values_dict

def get_lang_activation_vector(
        prompts: list[str],
        model: ReplacementModel,
        n_layers = 26,
        max_feature_nodes=None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """
    Sparse dictionary version - returns dicts instead of tensors.
    Returns:
        activation_per_pos: dict mapping feature to average activation per position
        active_example_ratio: dict mapping feature to ratio of examples where active
        max_values_dict: dict mapping feature to max activation value
    """
    activation_total = defaultdict(int)  # total count across all examples
    active_in_examples = defaultdict(int)  # number of examples where feature is active
    n_pos_total = 0
    max_values_dict = dict()
    
    for prompt in prompts:
        counts, n_pos, values_dict = get_activation_vector(prompt, model, n_layers, max_feature_nodes=max_feature_nodes)
        n_pos_total += n_pos
        
        # Accumulate counts
        for key, count in counts.items():
            activation_total[key] += count
            active_in_examples[key] += 1  # This example has this feature active
        
        # Track max values
        for key, val in values_dict.items():
            max_values_dict[key] = max(max_values_dict.get(key, 0), val)
        
        del counts, values_dict
        torch.cuda.empty_cache()
        gc.collect()
    
    # Normalize
    activation_per_pos = {key: count / n_pos_total for key, count in activation_total.items()}
    active_example_ratio = {key: count / len(prompts) for key, count in active_in_examples.items()}
    
    return activation_per_pos, active_example_ratio, max_values_dict

def normalize_sparse(lang_activation_vec: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """
    Normalize sparse dictionaries - for each feature, normalize across languages.
    Returns dict[lang][feature] = normalized_value
    """
    # Collect all features across all languages
    all_features = set()
    for lang_dict in lang_activation_vec.values():
        all_features.update(lang_dict.keys())
    
    normalized = {lang: {} for lang in lang_activation_vec.keys()}
    
    for feature in all_features:
        # Get values for this feature across all languages
        values = [lang_activation_vec[lang].get(feature, 0) for lang in lang_activation_vec.keys()]
        total = sum(values)
        
        if total > 0:
            for lang, val in zip(lang_activation_vec.keys(), values):
                normalized[lang][feature] = val / total
    
    return normalized

def choose_language_specific_features(
        langs: list[str], 
        active_tokens: dict[str, dict[str, float]], 
        active_examples: dict[str, dict[str, float]],
        cross_lingual_thres: float,
        example_thres: float,
        token_thres: float = 0.1,
        ) -> dict[str, list[tuple[int, int]]]:
    """
    Sparse version - works with dictionaries instead of tensors.
    No memory limits needed!
    """
    # Collect all features that appear in any language
    all_features = set()
    for lang in langs:
        all_features.update(active_tokens[lang].keys())
    
    language_specific_features = {lang: [] for lang in langs}
    
    for feature_key in all_features:
        # Check if feature meets threshold in at least one language
        passes_threshold = False
        for lang in langs:
            token_val = active_tokens[lang].get(feature_key, 0)
            example_val = active_examples[lang].get(feature_key, 0)
            if token_val > token_thres and example_val > example_thres:
                passes_threshold = True
                break
        
        if not passes_threshold:
            continue
        
        # Get values for all languages
        vals = [active_tokens[lang].get(feature_key, 0) for lang in langs]
        max_val = max(vals)
        
        if max_val == 0:
            continue
        
        # Check which languages are active
        active_langs = [i for i, val in enumerate(vals) if val >= max_val * cross_lingual_thres]
        
        # Only keep if specific to one language
        if len(active_langs) == 1:
            lang = langs[active_langs[0]]
            layer, feature_idx = feature_key.split('.')
            language_specific_features[lang].append((int(layer), int(feature_idx)))
    
    return language_specific_features

def scale_steer_to_A(
        language_features: dict[str, list[tuple[int, int]]],
        max_activation: dict[str, dict[str, float]],
        lang_A: str,
        prompt: str,
        model: ReplacementModel,
        alpha = 0.2,
        p=0.9,
        max_new_tokens = 64,
        max_n_logits = 5,
        desired_logit_prob = 0.95,
        max_feature_nodes = None,
        batch_size = 4,
        offload = 'cpu',
        verbose = True,
        ) -> str:
    lang_A_features = language_features[lang_A]

    generated = prompt
    for _ in range(max_new_tokens):
        graph = attribute(
                prompt=generated,
                model=model,
                max_n_logits=max_n_logits,
                desired_logit_prob=desired_logit_prob,
                batch_size=batch_size,
                max_feature_nodes=max_feature_nodes,
                offload=offload,
                verbose=verbose,
            )
        active_features = graph.active_features # (n_active_features, 3) containing (layer, pos, feature_idx)
        active_features = active_features.detach().cpu()
        activation_values = graph.activation_values
        n_pos = graph.n_pos

        interventions = []
        for layer, feature_idx in lang_A_features:
            pos = -1
            target_row = torch.tensor((layer, n_pos + pos, feature_idx))

            matches = (active_features == target_row)
            row_matches_all = torch.all(matches, dim=1)
            indices = torch.nonzero(row_matches_all, as_tuple=False)
            original_activation = 0
            if indices.numel() > 0:
                index = indices.item()
                print(f"{layer}.{feature_idx} was active in prompt {generated}")
                original_activation = activation_values[index].detach().cpu()
                original_activation = original_activation.item() if isinstance(original_activation, torch.Tensor) else original_activation
            
            activation_value = original_activation + alpha * max_activation[lang_A][f"{layer}.{feature_idx}"]
            # tuple of layer, position, feature_idx, value
            intervention = (layer, pos, feature_idx, activation_value)
            interventions.append(intervention)
        
        new_logits, _ = model.feature_intervention(generated, interventions)
        # top-p decoding
        next_token_logits = new_logits[0, -1, :]
        probs = F.softmax(next_token_logits.float(), dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        indices_to_remove = cumulative_probs > p
        indices_to_remove[..., 1:] = indices_to_remove[..., :-1].clone()
        indices_to_remove[..., 0] = False
        sorted_probs[indices_to_remove] = 0.0

        next_token_index = torch.multinomial(sorted_probs, num_samples=1)
        next_token_id = sorted_indices[next_token_index].item()
        token = model.tokenizer.decode([next_token_id], skip_special_tokens=False)

        generated += token
        print(generated)
        
        del graph, active_features, activation_values, interventions, new_logits, next_token_logits, probs
        torch.cuda.empty_cache()

    return generated

def argsparse():
    parser = argparse.ArgumentParser(description='Extract language specific features and perform interventions')
    parser.add_argument('--model', type=str, default='gemma-2-2b', choices=hf_model_names.keys(), help='Model to use for feature extraction and intervention')
    parser.add_argument('--lang', type=str, default=None, choices=lang_to_flores_key.keys(), help='Optional language to process (default: all languages)')
    return parser.parse_args()

if __name__ == "__main__":
    args = argsparse()
    current_file_path = __file__
    current_directory = os.path.dirname(current_file_path)
    absolute_directory = os.path.abspath(current_directory)
    data_directory = os.path.join(os.path.dirname(absolute_directory), "data", "language_specific_features", args.model)
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    
    model_name = args.model
    transcoder_name = hf_transcoder_names[model_name]
    print(f"Loading model {model_name} with {n_features[model_name]} features per layer")
    print(f"Using sparse dictionary approach - no feature limits needed!")
    model = ReplacementModel.from_pretrained(hf_model_names[model_name], transcoder_name, device=device, dtype=torch.bfloat16)

    lang_activation_vec = dict()
    lang_active_examples = dict()
    lang_max_vals = dict()
    for lang, ds_key in lang_to_flores_key.items():
        if args.lang is not None and lang != args.lang:
            continue
        print(f"Processing {lang}...")
        json_name = f"{lang}_sparse_long.json"
        json_path = os.path.join(data_directory, json_name)
        
        if os.path.exists(json_path):
            print(f"  Loading cached data for {lang}")
            with open(json_path, 'r') as f:
                data = json.load(f)
            lang_activation_vec[lang] = data['activation_per_pos']
            lang_active_examples[lang] = data['active_example_ratio']
            lang_max_vals[lang] = data['max_values']
        else:
            print(f"  Computing activations for {lang}")
            # Use streaming to save memory
            ds = load_dataset("openlanguagedata/flores_plus", ds_key, split="dev")
            ds = ds.shuffle(seed=42)
            # Skip longer sentences to avoid OOM - longer sentences have more features
            max_sentence_length = 100  # character limit
            batch = [example['text'] for example in ds if len(example['text']) < max_sentence_length][:100]
            print(batch)
            activation_per_pos, active_example_ratio, max_values_dict = get_lang_activation_vector(batch, model, n_layers=layer_num[model_name], max_feature_nodes=None)
            del batch
            
            # Save as JSON (sparse format)
            data = {
                'activation_per_pos': activation_per_pos,
                'active_example_ratio': active_example_ratio,
                'max_values': max_values_dict
            }
            with open(json_path, 'w') as f:
                json.dump(data, f)
            
            lang_activation_vec[lang] = activation_per_pos
            lang_active_examples[lang] = active_example_ratio
            lang_max_vals[lang] = max_values_dict
            print(f"  {lang}: {len(lang_activation_vec[lang])} active features")
            torch.cuda.empty_cache()
            gc.collect()
    
    example_thres = 0.98
    file_name = f"features_{example_thres}.json"
    full_path = os.path.join(data_directory, file_name)
    selected_langs = [args.lang] if args.lang is not None else list(langs_big)
    if os.path.exists(full_path):
        with open(full_path, 'r') as f:
            language_specific_features = json.load(f)
    else:
        language_specific_features = choose_language_specific_features(
            selected_langs, lang_activation_vec, lang_active_examples, 0.8, example_thres
        ) # example_thres is 0.98 for the original paper
        with open(full_path, 'w') as f:
            json.dump(language_specific_features, f)
    """
    max_new_tokens = 16
    data_list = []
    alphas = [0.1, 0.3, 0.4, 0.5, 0.8]
    ps = [0.0, 0.5, 0.8, 0.9, 0.95] # 0.0 is greedy decoding
    for lang in langs_big:
        for alpha in alphas:
            for p in ps:
                output = scale_steer_to_A(language_specific_features, lang_max_vals, lang, "", model, alpha, p, max_new_tokens)
                record = [lang, alpha, p, output]
                data_list.append(record)
    
    file_name = f"text_generation_{max_new_tokens}.jsonl"
    full_path = os.path.join(data_directory, file_name)
    with open(full_path, 'w', encoding='utf-8') as file:
        for record in data_list:
            json_line = json.dumps(record, ensure_ascii=False)
            file.write(json_line + '\n')
    """

    for lang, val in language_specific_features.items():
        file_name = f'{lang}_description.json'
        file_path = os.path.join(data_directory, file_name)
        if os.path.exists(file_path):
            continue

        description_dict = dict()
        for layer, feature_idx in val:
            response = requests.get(neuronpedia_urls[model_name].format(layer=layer, feature_idx=feature_idx))
            explanations = response.json()['explanations']
            try:
                description = explanations[0]['description']
            except IndexError:
                print(f"layer{layer}, feature{feature_idx}, no description")
                description = ""
            key = f'{layer}.{feature_idx}'
            description_dict[key] = description

        with open(file_path, 'w') as f:
            json.dump(description_dict, f)
    
    language_name_in_description = dict()
    patterns = '??_description.json'
    matching_files = glob.glob(os.path.join(data_directory, patterns))
    for file_path in matching_files:
        filename_with_ext = os.path.basename(file_path)
        filename_only, file_extension = os.path.splitext(filename_with_ext)
        first_two_letters = filename_only[:2]

        identifier = identifiers[first_two_letters]
        with open(file_path, 'r') as f:
            descriptions = json.load(f)
        
        total_count = 0
        included = 0
        for _, description in descriptions.items():
            is_included = any(sub in description for sub in identifier)
            total_count += 1
            if is_included:
                included += 1

        language_name_in_description[first_two_letters] = (included, total_count)
    
    file_name = 'summary.json'
    file_path = os.path.join(data_directory, file_name)
    with open(file_path, 'w') as f:
        json.dump(language_name_in_description, f)
    

