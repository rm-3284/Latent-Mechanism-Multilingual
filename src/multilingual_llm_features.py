import argparse
from datasets import load_dataset
import gc
import glob
import json
import matplotlib.pyplot as plt
import math
import os
import pandas as pd
import requests
import seaborn as sns
import torch

from circuit_tracer_import import attribute, ReplacementModel
from device_setup import device, num_gpus
from template import lang_to_flores_key, identifiers, lang_dict
from models import hf_model_names, hf_transcoder_names, neuronpedia_urls

def get_activation(
        prompt: str, 
        model: ReplacementModel,
        max_n_logits = 5,
        desired_logit_prob = 0.95,
        max_feature_nodes = None,
        batch_size = 4,
        offload = 'cpu',
        verbose = True,
        ) -> tuple[dict[str, float], int]:
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
    n_active_features = active_features.shape[0]
    activation_values = graph.activation_values
    n_pos = graph.n_pos
    activation_values_sum = dict()
    for i in range(n_active_features):
        layer, pos, feature_idx = active_features[i, :]
        layer = layer.item() if isinstance(layer, torch.Tensor) else layer
        feature_idx = feature_idx.item() if isinstance(feature_idx, torch.Tensor) else feature_idx
        key = f"{layer}.{feature_idx}"
        activation_value = activation_values[i]
        activation_value = activation_value.item() if isinstance(activation_value, torch.Tensor) else activation_value
        try:
            val = activation_values_sum[key]
            activation_values_sum[key] = val + activation_value
        except KeyError:
            activation_values_sum[key] = activation_value
    for key, val in activation_values_sum.items():
        activation_values_sum[key] = val.item() if isinstance(val, torch.Tensor) else val
    del graph
    torch.cuda.empty_cache()
    return activation_values_sum, n_pos

def get_mean_activation(
        prompts: list[str],
        model: ReplacementModel,
        max_feature_nodes = None,
) -> dict[str, float]:
    mean_activation_dict = dict()
    n_pos_total = 0
    for prompt in prompts:
        activation_values_sum, n_pos = get_activation(prompt, model, max_feature_nodes=max_feature_nodes)
        n_pos_total += n_pos
        for key, val in activation_values_sum.items():
            try:
                current_val = mean_activation_dict[key]
                current_val = current_val.item() if isinstance(current_val, torch.Tensor) else current_val
                mean_activation_dict[key] = current_val + val
            except KeyError:
                mean_activation_dict[key] = val
        del activation_values_sum
        torch.cuda.empty_cache()
        gc.collect()
    
    for key, val in mean_activation_dict.items():
        mean_activation_dict[key] = val / n_pos_total
    return mean_activation_dict

def calculate_v(per_lang_mean_activation_dict: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = per_lang_mean_activation_dict.keys()
    v_dict = dict()
    for lang, sub_dict in per_lang_mean_activation_dict.items():
        v_dict[lang] = dict()
        for feature, val in sub_dict.items():
            gamma = 0
            for key in keys:
                if key == lang:
                    continue
                try:
                    v = per_lang_mean_activation_dict[key][feature]
                    gamma += v
                except KeyError:
                    gamma += 0
            gamma /= len(keys) - 1
            v = val - gamma
            v_dict[lang][feature] = v
    return v_dict

def histogram_v_values(data: dict[str, float], save_path: str):
    if not data:
        print("The dictionary is empty. No plot to generate.")
        return
    
    sorted_items = sorted(data.items(), key=lambda item: item[1], reverse=True)[:100]
    labels, values = zip(*sorted_items)
    if len(values) == 1:
        # If there's only one bar, it's both max and min
        colors = ['blue']
        max_val = values[0]
        min_val = values[0]
    else:
        # Default color 'gray' for all bars
        colors = ['#C0C0C0'] * len(labels)  # Using a hex code for standard gray
        # Set the first bar (max) to green
        colors[0] = '#2ca02c'  # Default matplotlib green
        # Set the last bar (min) to red
        colors[-1] = '#d62728' # Default matplotlib red
        
        max_val = values[0]
        min_val = values[-1]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color=colors)
    plt.xlabel("Categories")
    plt.ylabel("Values")
    plt.title(
        f"Bar Chart in Decreasing Order\n"
        f"Max (Green): {max_val:.2f} | Min (Red): {min_val:.2f}"
    )
    if len(labels) > 5:
        plt.xticks(rotation=45, ha='right')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    plt.savefig(save_path)
    print(f"Bar chart saved to {save_path}")
    return

def last_pos_feature_find(key: str, n_pos: int, active_features: torch.Tensor) -> int:
    pos = n_pos - 1
    layer, feature_idx = key.split('.')
    layer = int(layer)
    feature_idx = int(feature_idx)

    target_row = torch.tensor((layer, pos, feature_idx))
    matches = (active_features == target_row)
    row_matches_all = torch.all(matches, dim=1)
    indices = torch.nonzero(row_matches_all, as_tuple=False)

    if indices.numel() == 0:
        return -1
    else:
        return indices.item()


def steering_from_A_to_B(
        per_lang_mean_activation_dict: dict[str, dict[str, float]], 
        lang_A: str, # source
        lang_B: str, # target
        prompt: str,
        model: ReplacementModel,
        topk = 10,
        max_n_logits = 5,
        desired_logit_prob = 0.95,
        max_feature_nodes = None,
        batch_size = 16,
        offload = 'cpu',
        verbose = True,
        ) -> tuple[torch.Tensor, torch.Tensor]:
    intervening_features1 = list(sorted(per_lang_mean_activation_dict[lang_A].items(), key=lambda item: item[1], reverse=True)[:topk])
    intervening_features2 = list(sorted(per_lang_mean_activation_dict[lang_B].items(), key=lambda item: item[1], reverse=True)[:topk])
    combined_intervening_features = intervening_features1 + intervening_features2

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
    active_features = active_features.detach().cpu()
    activation_values = graph.activation_values
    n_pos = graph.n_pos

    interventions = list()
    for key, _ in combined_intervening_features:
        index = last_pos_feature_find(key, n_pos, active_features)
        if index == -1:
            original_activation = 0
        else:
            original_activation = activation_values[index].detach().cpu()
            original_activation = original_activation.item() if isinstance(original_activation, torch.Tensor) else original_activation

        langA_activation = per_lang_mean_activation_dict[lang_A].get(key, 0)
        langB_activation = per_lang_mean_activation_dict[lang_B].get(key, 0)
        diff = langB_activation - langA_activation

        # tuple of layer, position, feature_idx, value
        intervention = (layer, pos, feature_idx, original_activation + diff)
        interventions.append(intervention)
    
    new_logits, new_activations = model.feature_intervention(prompt, interventions)
    return new_logits, new_activations

def code_switch_analysis(
        per_lang_mean_activation_dict: dict[str, dict[str, float]], 
        lang_A,
        lang_B,
        prompt_list,
        model: ReplacementModel,
        topk = 10,
        max_n_logits = 5,
        desired_logit_prob = 0.95,
        max_feature_nodes = None,
        batch_size = 16,
        offload = 'cpu',
        verbose = True,
        ) -> dict[str, tuple[float, float, float]]: 
        # lang_B noun, lang_A prefix lang_B noun, lang_A prefix lang_A noun
    top_features = list(sorted(per_lang_mean_activation_dict[lang_A].items(), key=lambda item: item[1], reverse=True)[:topk])
    activation_diff = dict()
    for key, _ in top_features:
        activation_diff[key] = [[], [], []]
    
    for prompt in prompt_list:
        ori_lan = prompt["ori_lan"]
        target_lan = prompt["target_lan"]
        ori_sentence = prompt["ori_sentence"]
        sentence = prompt["sentence"]
        if ori_lan != lang_A:
            continue
        if target_lan != lang_A and target_lan != lang_B:
            continue
        graph = attribute(
                prompt=sentence,
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
        if target_lan == lang_A:
            for key, val in activation_diff.items():
                index = last_pos_feature_find(key, n_pos, active_features)
                if index == -1:
                    val[2].append(0)
                else:
                    activation_value = activation_values[index]
                    activation_value = activation_value.item() if isinstance(activation_value, torch.Tensor) else activation_value
                    val[2].append(activation_value)
        elif target_lan == lang_B:
            for key, val in activation_diff.items():
                index = last_pos_feature_find(key, n_pos, active_features)
                if index == -1:
                    val[1].append(0)
                else:
                    activation_value = activation_values[index]
                    activation_value = activation_value.item() if isinstance(activation_value, torch.Tensor) else activation_value
                    val[1].append(activation_value)
            
            # lang_B noun
            prompt_inputs = model.tokenizer.encode(sentence)
            ori_prompt_inputs = model.tokenizer.encode(ori_sentence)
            noun = prompt_inputs[:1] + prompt_inputs[len(ori_prompt_inputs):]
            graph = attribute(
                prompt=noun,
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

            for key, val in activation_diff.items():
                index = last_pos_feature_find(key, n_pos, active_features)
                if index == -1:
                    val[0].append(0)
                else:
                    activation_value = activation_values[index]
                    activation_value = activation_value.item() if isinstance(activation_value, torch.Tensor) else activation_value
                    val[0].append(activation_value)

    result = dict()
    for key, val in activation_diff.items():
        list1, list2, list3 = val
        mean1 = sum(list1) / len(list1)
        mean2 = sum(list2) / len(list2)
        mean3 = sum(list3) / len(list3)
        result[key] = (mean1, mean2, mean3)
    return result

def plot_linguistic_activations_on_ax(
    ax,  # <--- NEW ARGUMENT
    activation_data: dict[str, tuple[float, float, float]],
    feature_type_label: str = "Feature name",
    title: str = "Feature Activations" # Added title argument so each subplot can have a name
):
    """
    Draws the activation plot onto the provided matplotlib Axes 'ax'.
    Does NOT save or close the figure.
    """
    
    context_map = {
        0: "Lang B Noun",
        1: "Lang A Prefix + Lang B Noun",
        2: "Lang A Prefix + Lang A Noun",
    }

    # Flatten data
    rows = []
    for key, activations in activation_data.items():
        for i in range(3):
            rows.append({
                feature_type_label: key,
                'Context': context_map[i],
                'Activation Value': activations[i]
            })
    df = pd.DataFrame(rows)

    # Use Seaborn style (apply globally or locally)
    sns.set_theme(style="whitegrid")

    # --- Main Plotting Call ---
    # Note the `ax=ax` argument! This is critical.
    sns.stripplot(
        data=df, 
        x=feature_type_label, 
        y='Activation Value', 
        hue='Context',
        jitter=0.25,
        size=8,
        alpha=0.7,
        palette='Set1',
        dodge=True,
        ax=ax  # <--- Draw on the specific subplot
    )

    # Formatting specific to this subplot
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Activation Value", fontsize=10)
    ax.set_xlabel(feature_type_label, fontsize=10)
    
    # Rotate X-axis labels locally
    ax.tick_params(axis='x', rotation=45)
    
    # Add reference line
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)

    # Remove the individual legend to save space (we will add a global one later)
    # OR keep it if you want legends everywhere. Usually, global is better.
    if ax.get_legend():
        ax.get_legend().remove()

def plot_linguistic_activations_multiple(all_datasets, titles, filename, lang):
    num_plots = len(all_datasets)
    cols = 2
    rows = math.ceil(num_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(20, 6 * rows), constrained_layout=True)
    axes = axes.flatten()

    for i, (data, label) in enumerate(zip(all_datasets, titles)):
        # Call your function, passing the specific 'ax'
        plot_linguistic_activations_on_ax(
            ax=axes[i], 
            activation_data=data, 
            title=label
        )

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), 
           ncol=3, title=f"Linguistic Context on {lang} features", fontsize=12)

    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    return

def argsparse():
    parser = argparse.ArgumentParser(description='Extract multilingual LLM features and perform analysis')
    parser.add_argument('--model', type=str, default='gemma-2-2b', choices=hf_model_names.keys(), help='Model to use for feature extraction and analysis')
    parser.add_argument('--lang', type=str, default=None, choices=lang_to_flores_key.keys(), help='Optional language to process (default: all languages)')
    return parser.parse_args()

if __name__ == "__main__":
    args = argsparse()
    current_file_path = __file__
    current_directory = os.path.dirname(current_file_path)
    absolute_directory = os.path.abspath(current_directory)
    data_directory = os.path.join(os.path.dirname(absolute_directory), "data", "multilingual_llm_features", args.model)
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    model_name = args.model
    transcoder_name = hf_transcoder_names[model_name]
    print(f"Loading model {model_name}")
    print("Using sparse dictionary approach - no feature limits needed!")
    model = ReplacementModel.from_pretrained(hf_model_names[model_name], transcoder_name, device=device, dtype=torch.bfloat16)

    lang_mean_activation_dict = dict()
    for lang, ds_key in lang_to_flores_key.items():
        if args.lang is not None and lang != args.lang:
            continue
        file_name = f"{lang}_long.json"
        full_path = os.path.join(data_directory, file_name)
        if os.path.exists(full_path):
            with open(full_path, 'r') as f:
                mean_activation = json.load(f)
            lang_mean_activation_dict[lang] = mean_activation
        else:
            # Use streaming to save memory
            ds = load_dataset("openlanguagedata/flores_plus", ds_key, split="dev")
            ds = ds.shuffle(seed=42)
            max_sentence_length = 100  # character limit
            batch = [example['text'] for example in ds if len(example['text']) < max_sentence_length][:100]
            print(batch)
            mean_activation = get_mean_activation(batch, model, max_feature_nodes=None)  # No limits!
            del batch
            with open(full_path, 'w') as f:
                json.dump(mean_activation, f)
            lang_mean_activation_dict[lang] = mean_activation
            torch.cuda.empty_cache()
            gc.collect()
    
    full_path = os.path.join(data_directory, 'v_values.json')
    if os.path.exists(full_path):
        with open(full_path, 'r') as f:
            v_dict = json.load(f)
    else:
        v_dict = calculate_v(lang_mean_activation_dict)
        with open(full_path, 'w') as f:
            json.dump(v_dict, f)

    for lang, data in lang_mean_activation_dict.items():
        file_name = f"{lang}_vplot.png"
        full_path = os.path.join(data_directory, file_name)
        if not os.path.exists(full_path):
           histogram_v_values(data, full_path)
    
    # steering experiments
    """
    full_path = os.path.join(data_directory, 'forced_code_switch.jsonl')
    prompt_list = []
    with open(full_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                json_object = json.loads(line.strip())
                prompt_list.append(json_object)
    """
    data_list = list()
    topk = 50
    """
    for prompt in prompt_list:
        ori_sentence = prompt["ori_sentence"]
        ori_lan = prompt["ori_lan"]
        target_lan = prompt["target_lan"]
        if not (ori_lan in lang_to_flores_key.keys() and target_lan in lang_to_flores_key.keys()):
            continue
        logits, activation = steering_from_A_to_B(v_dict, ori_lan, target_lan, ori_sentence, model, topk)
        top_outputs = get_top_outputs(logits, model)
        record = {"sentence": ori_sentence, "source_lang": ori_lan, "target_lang": target_lan, "top_outputs": top_outputs}
        data_list.append(record)
    
    file_name = f"cross_lingual_continuation_{topk}.jsonl"
    full_path = os.path.join(data_directory, file_name)
    with open(full_path, 'w', encoding='utf-8') as file:
        for record in data_list:
            json_line = json.dumps(record, ensure_ascii=False)
            file.write(json_line + '\n')
    """
    """
    langs = ['en', 'es', 'fr', 'ja', 'ko', 'zh']
    for lang_A in langs:
        for lang_B in langs:
            if lang_A == lang_B:
                continue
            if os.path.exists(full_path):
                continue
            file_name = f"code_switch_analysis_{lang_A}_{lang_B}.json"
            full_path = os.path.join(data_directory, file_name)
            result = code_switch_analysis(v_dict, lang_A, lang_B, prompt_list, model, topk)
            with open(full_path, 'w') as f:
                json.dump(result, f)
    """
    for lang, val in v_dict.items():
        file_name = f'{lang}_description.json'
        file_path = os.path.join(data_directory, file_name)
        if os.path.exists(file_path):
            continue

        top_features = list(sorted(val.items(), key=lambda item: item[1], reverse=True)[:topk])
        description_dict = dict()
        for key, _ in top_features:
            layer, feature_idx = key.split('.')
            response = requests.get(neuronpedia_urls[model_name].format(layer=layer, feature_idx=feature_idx))
            explanations = response.json()['explanations']
            try:
                description = explanations[0]['description']
            except IndexError:
                print(f"layer{layer}, feature{feature_idx}, no description")
                description = ""
            description_dict[key] = description

        with open(file_path, 'w') as f:
            json.dump(description_dict, f)

    file_name = 'summary.json'
    file_path = os.path.join(data_directory, file_name)
    if not os.path.exists(file_path):
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
        
        with open(file_path, 'w') as f:
            json.dump(language_name_in_description, f)
    
    # plots
    """
    lang_names = lang_dict['en']
    for prompt_lang in langs:
        data_list = list()
        titles = list()
        for test_lang in langs:
            if test_lang == prompt_lang:
                continue
            file = os.path.join(data_directory, f"code_switch_analysis_{prompt_lang}_{test_lang}.json")
            with open(file, 'r') as f:
                data = json.load(f)
            data_list.append(data)
            titles.append(f"{lang_names[prompt_lang]}-{lang_names[test_lang]}")
        filename = f"code_switch_activations_{prompt_lang}.png"
        plot_linguistic_activations_multiple(data_list, titles, os.path.join(data_directory, filename), lang_names[prompt_lang])
    """
