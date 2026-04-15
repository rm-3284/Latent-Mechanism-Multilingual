import argparse
import json
import os
import torch
from typing import Literal

from circuit_tracer_import import Feature, ReplacementModel
from pipeline_data.adjectives import big_data
from device_setup import device
from intervention import (
    amplification, get_best_base, get_best_rank,
    get_top_outputs, logit_diff_single,
    )
from template import lang_to_flores_key, base_strings
from models import hf_model_names, hf_transcoder_names

def direction_ablation_helper(model: ReplacementModel, layer_idx: int, target_features: list[int], prompt):
    # the projection can be calculated as follows. v = W_dec[feature, :]
    # W_all @ feature_activation = original_vector
    # We want original_vector - original_vector \cdot v / || v || for several v
    # V = W_dec [features, :] --> original_vector \cdot v is V^T @ original vector
    # Also, to convert it back, we need W_all^{-1}

    """
    W_all = model.transcoders[layer_idx].W_dec # (n_features, d_model)
    #print(W_all.shape)
    W_I = W_all.T[:, target_features]
    # (W_I^T * W_I)^-1 * W_I^T
    W_I_pinv = torch.linalg.pinv(W_I.to(torch.float32)).to(torch.bfloat16) # (k, d_model)
    #print(W_I_pinv.shape)
    _, activation_cache = model.get_activations(prompt) # (layer, pos, n_features)
    z = activation_cache[layer_idx, ...] # (pos, n_features)
    #print(z.shape)
    h_hat = torch.matmul(z, W_all) # (pos, d_model)
    #print(h_hat.shape)
    proj_z_I = torch.matmul(h_hat, W_I_pinv.T) # (seq, k)
    adjusted_activations = z[..., target_features] - proj_z_I # (seq, k)

    intervention_list = list()
    for i, k in enumerate(target_features):
        val = adjusted_activations[-1, i]
        val = val.item() if isinstance(val, torch.Tensor) else val
        element = (layer_idx, -1, k, val)
        intervention_list.append(element)
    return intervention_list"""

    transcoder = model.transcoders[layer_idx]
    W_all = transcoder.W_dec # (n_features, d_model)
    b_dec = transcoder.b_dec # (d_model)
    _, activation_cache = model.get_activations(prompt) # (layer, pos, n_features)

    z = activation_cache[layer_idx, -1, :] # (n_features)
    W_I = W_all.T[:, target_features] # (d_model, k)
    # (W_I^T W_I)^{-1} W_I^T
    W_I_pinv = torch.linalg.pinv(W_I.to(torch.float32)).to(W_all.dtype)
    h_hat = (z @ W_all) + b_dec
    proj_coefficients = torch.matmul(h_hat, W_I_pinv.T)
    current_target_acts = z[target_features]
    adjusted_activations = current_target_acts - proj_coefficients
    
    intervention_list = []
    for i, feature_idx in enumerate(target_features):
        val = adjusted_activations[i].item()
        intervention_list.append((layer_idx, -1, feature_idx, val))
    return intervention_list
    
def direction_ablation_layer_determine(features: dict[str, list[tuple[Feature, float]]], lang, num_layers: int) -> tuple[int, list[int]]:
    features_list = features[lang]
    features_set = dict()
    for i in range(num_layers):
        features_set[i] = set()
    for f, _ in features_list:
        features_set[f.layer].add(f.feature_idx)
    
    max_idx = -1
    max_length = -1
    for k, v in features_set.items():
        if len(v) >= max_length:
            max_idx = k
    return max_idx, list(features_set[max_idx])
    

def description_based_features(dir_name: str, languages: list[str], threshold: float=0.1) -> dict[str, list[str]]:
    lang_features = dict()
    for lang in languages:
        file_name = f"{lang}_features.json"
        with open(os.path.join(dir_name, file_name), 'r') as f:
            features_freq = json.load(f)
        max_val = max(features_freq.values())
        features = list()
        for key, val in features_freq.items():
            if val >= max_val * threshold:
                features.append(key)
        lang_features[lang] = features
    return lang_features

def mean_value_based_features(dir_name: str, languages: list[str], topk: int=50) -> dict[str, list[str]]:
    lang_features = dict()
    for lang in languages:
        file_name = f"{lang}.json"
        with open(os.path.join(dir_name, file_name), 'r') as f:
            feature_val = json.load(f)
        sorted_features = sorted(feature_val.items(), key=lambda item: item[1], reverse=True)[:topk]
        features = list()
        for feature, _ in sorted_features:
            features.append(feature)
        lang_features[lang] = features
    return lang_features

def freq_based_features(dir_name: str, languages: list[str]) -> dict[str, list[str]]:
    lang_features = dict()
    file_name = "features_0.98.json"
    with open(os.path.join(dir_name, file_name), 'r') as f:
        feature_dict = json.load(f)
    for lang in languages:
        features = feature_dict[lang]
        features_str = list()
        for key, feature_idx in features:
            string = f"{key}.{feature_idx}"
            features_str.append(string)
        lang_features[lang] = features_str
    return lang_features

def model_run(prompt: str, model: ReplacementModel) -> tuple[list[tuple[str, float]], torch.Tensor]:
    logits, _ = model.get_activations(prompt)
    return get_top_outputs(logits, model), logits

def model_intervention(prompt: str, model: ReplacementModel, interventions: list[tuple[int, int, int, float]]) -> tuple[list[tuple[str, float]], torch.Tensor]:
    logits, _ = model.feature_intervention(prompt, interventions)
    return get_top_outputs(logits, model), logits

def map_from_mode_to_idx(mode: str) -> Literal[2, 3, 4, 5]:
    if mode == 'minimum':
        return 2
    elif mode == 'maximum':
        return 3
    elif mode == 'mean':
        return 4
    elif mode == 'median':
        return 5
    else:
        raise KeyError(f"{mode} is not a valid argument. Options are ['minimum', 'maximum', 'mean', 'median']")
    

def activation_dict(lang_feature_dict: dict[str, list[str]], dir_path: str, langs: list[str], mode: str = 'median') -> dict[str, list[tuple[Feature, float]]]:
    idx = map_from_mode_to_idx(mode)
    
    feature_val_by_langs = dict()
    for lang in langs:
        file_name = f"{lang}_pos_summary.json"
        with open(os.path.join(dir_path, file_name)) as f:
            feature_values = json.load(f)
        
        feature_val_list = list()
        for key in lang_feature_dict[lang]:
            layer, feature_idx = key.split('.')
            layer = int(layer)
            feature_idx = int(feature_idx)
            feature = Feature(layer=layer, pos=-1, feature_idx=feature_idx)
            val = feature_values[key][idx]
            if val == float('nan'):
                val = 0
            feature_val_list.append((feature, val))
        feature_val_by_langs[lang] = feature_val_list
    return feature_val_by_langs

def ablation_and_amplification(ablation, amplification):
    ablation_features = set()
    for layer, _, feature_idx, _ in ablation:
        ablation_features.add((layer, feature_idx))
    amplification_features = set()
    for layer, _, feature_idx, _ in amplification:
        amplification_features.add((layer, feature_idx))
    intersection = ablation_features.intersection(amplification_features)

    intervention_list = []
    for layer, pos, feature_idx, val in ablation:
        if (layer, feature_idx) not in intersection:
            intervention_list.append((layer, pos, feature_idx, val))
    for layer, pos, feature_idx, val in amplification:
        if (layer, feature_idx) not in intersection:
            intervention_list.append((layer, pos, feature_idx, val))
    return intervention_list

def perform_intervention(
        prompt: str, model: ReplacementModel, logits: torch.Tensor, 
        adj_lang: str, ans, ablation, amplification, langs):
    # info that should be stored is after each intervention, 
    # for answer in each language, how the logit changed and rank changed
    # the base should be the adjective language
    base = get_best_base(logits, ans[adj_lang], model)

    ablations_dict = dict()
    ablations_outputs = dict()
    for intervention_lang in langs:
        ablations_dict[intervention_lang] = dict()
        #ablation_outputs, ablation_logits = model_intervention(prompt, model, ablation[intervention_lang])
        layer, features = ablation[intervention_lang]
        intervention_list = direction_ablation_helper(model, layer, features, prompt)
        ablation_outputs, ablation_logits = model_intervention(prompt, model, intervention_list)

        ablations_outputs[intervention_lang] = ablation_outputs
        for measure_lang in langs:
            target = get_best_base(ablation_logits, ans[measure_lang], model)
            o_diff, n_diff, _ = logit_diff_single(logits, ablation_logits, target, base, model)
            n_rank = get_best_rank(ablation_logits, ans[measure_lang], model)

            ablations_dict[intervention_lang][measure_lang] = (o_diff, n_diff, n_rank)

    amplifications_dict = dict()
    amplifications_outputs = dict()
    #for intervention_lang in langs:
    #    amplifications_dict[intervention_lang] = dict()
    #    amplification_outputs, amplification_logits = model_intervention(prompt, model, amplification[intervention_lang])
    #    amplifications_outputs[intervention_lang] = amplification_outputs
    #    for measure_lang in langs:
    #        target = get_best_base(amplification_logits, ans[measure_lang], model)
    #        o_diff, n_diff, _ = logit_diff_single(logits, amplification_logits, target, base, model)
    #        n_rank = get_best_rank(amplification_logits, ans[measure_lang], model)
    #
    #       amplifications_dict[intervention_lang][measure_lang] = (o_diff, n_diff, n_rank)

    interventions_dict = dict()
    interventions_outputs = dict()
    for ablation_lang in langs:
        interventions_dict[ablation_lang] = dict()
        interventions_outputs[ablation_lang] = dict()
        for amplification_lang in langs:
            #intervention_list = ablation_and_amplification(ablation[ablation_lang], amplification[amplification_lang])
            layer, features = ablation[ablation_lang]
            intervention_list = ablation_and_amplification(direction_ablation_helper(model, layer, features, prompt), amplification[amplification_lang])

            interventions_dict[ablation_lang][amplification_lang] = dict()
            intervention_outputs, intervention_logits = model_intervention(prompt, model, intervention_list)
            interventions_outputs[amplification_lang] = intervention_outputs
            for measure_lang in langs:
                target = get_best_base(intervention_logits, ans[measure_lang], model)
                o_diff, n_diff, _ = logit_diff_single(logits, intervention_logits, target, base, model)
                n_rank = get_best_rank(intervention_logits, ans[measure_lang], model)
                interventions_dict[ablation_lang][amplification_lang][measure_lang] = (o_diff, n_diff, n_rank)
    
    return ablations_dict, ablations_outputs, amplifications_dict, amplifications_outputs, interventions_outputs, interventions_dict

def combine_except_one(intervention: dict[str, list[tuple]], exclude:str):
    already_seen = set()
    duplicates = set()
    for key, val in intervention.items():
        if key == exclude:
            continue
        for layer, _, feature_idx, _ in val:
            item = (layer, feature_idx)
            if item in already_seen:
                duplicates.add(item)
            else:
                already_seen.add(item)
    interventions = list()
    for key, vals in intervention.items():
        if key == exclude:
            continue
        for layer, pos, feature_idx, val in vals:
            item = (layer, feature_idx)
            if item not in duplicates:
                interventions.append((layer, pos, feature_idx, val))
    return interventions


def perform_everything_intervention(
        prompt: str, model: ReplacementModel, logits: torch.Tensor,
        adj_lang: str, ans, ablation, amplification, langs):
    base = get_best_base(logits, ans[adj_lang], model)
    
    ablations_dict = dict()
    ablations_outputs = dict()
    for intervention_lang in langs:
        ablations_dict[intervention_lang] = dict()
        ablation_intervention = combine_except_one(ablation, intervention_lang)
        ablation_outputs, ablation_logits = model_intervention(prompt, model, ablation_intervention)
        ablations_outputs[intervention_lang] = ablation_outputs
        for measure_lang in langs:
            target = get_best_base(ablation_logits, ans[measure_lang], model)
            o_diff, n_diff, _ = logit_diff_single(logits, ablation_logits, target, base, model)
            n_rank = get_best_rank(ablation_logits, ans[measure_lang], model)

            ablations_dict[intervention_lang][measure_lang] = (o_diff, n_diff, n_rank)

    amplifications_dict = dict()
    amplifications_outputs = dict()
    for intervention_lang in langs:
        amplifications_dict[intervention_lang] = dict()
        amplification_intervention = combine_except_one(amplification, intervention_lang)
        amplification_outputs, amplification_logits = model_intervention(prompt, model, amplification_intervention)
        amplifications_outputs[intervention_lang] = amplification_outputs
        for measure_lang in langs:
            target = get_best_base(amplification_logits, ans[measure_lang], model)
            o_diff, n_diff, _ = logit_diff_single(logits, amplification_logits, target, base, model)
            n_rank = get_best_rank(amplification_logits, ans[measure_lang], model)

            amplifications_dict[intervention_lang][measure_lang] = (o_diff, n_diff, n_rank)
    
    return ablations_dict, ablations_outputs, amplifications_dict, amplifications_outputs

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prompt language",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # 2. Add a string argument
    # 'message' is the variable name inside the script
    # '--input-string' is the flag used on the command line
    parser.add_argument(
        '--lang',
        '-l',
        type=str,
        default=None,
        help='Prompt language',
    )
    parser.add_argument(
        '--model',
        '-m',
        type=str,
        default='gemma-2-2b',
        choices=hf_model_names.keys(),
        help='Model to use for intervention',
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # load the model
    model_name = args.model
    transcoder_name = hf_transcoder_names.get(args.model)
    model = ReplacementModel.from_pretrained(hf_model_names[model_name], transcoder_name, device=device, dtype=torch.bfloat16)

    direction_ablation_helper(model, 25, [452, 24522], "Hello world.")

    # relevant directories
    current_file_path = __file__
    current_directory = os.path.dirname(current_file_path)
    absolute_directory = os.path.abspath(current_directory)
    data_directory = os.path.join(os.path.dirname(absolute_directory), "data")
    flores_directory = os.path.join(data_directory, "flores_features", model_name)
    lang_specific_directory = os.path.join(data_directory, "language_specific_features", model_name)
    multilingual_features_directory = os.path.join(data_directory, "multilingual_llm_features", model_name)
    amplification_values_directory = os.path.join(data_directory, "amplification_values", model_name)

    langs = list(lang_to_flores_key.keys())

    # get the features + amplification values
    desc_features = description_based_features(flores_directory, langs, 0.1)
    val_features = mean_value_based_features(multilingual_features_directory, langs, 50)
    freq_features = freq_based_features(lang_specific_directory, langs)

    desc_interventions = activation_dict(desc_features, amplification_values_directory, langs)
    val_interventions = activation_dict(val_features, amplification_values_directory, langs)
    freq_interventions = activation_dict(freq_features, amplification_values_directory, langs)

    desc_ablations = dict()
    desc_amplifications = dict()
    val_ablations = dict()
    val_amplifications = dict()
    freq_ablations = dict()
    freq_amplifications = dict()
    for lang in langs:
        #desc_ablations[lang] = ablation(desc_interventions, lang)
        desc_ablations[lang] = direction_ablation_layer_determine(desc_interventions, lang, model_name)
        desc_amplifications[lang] = amplification(desc_interventions, lang)
        #val_ablations[lang] = ablation(val_interventions, lang)
        val_ablations[lang] = direction_ablation_layer_determine(val_interventions, lang, model_name)
        val_amplifications[lang] = amplification(val_interventions, lang)
        #freq_ablations[lang] = ablation(freq_interventions, lang)
        freq_ablations[lang] = direction_ablation_layer_determine(freq_interventions, lang, model_name)
        freq_amplifications[lang] = amplification(freq_interventions, lang)

    print(desc_ablations)
    print(val_ablations)
    print(freq_ablations)

    # ablation + amplification experiments
    output_dir = os.path.join(data_directory, "interventions", model_name)
    for prompt_lang in langs:
        if args.lang is not None:
            if prompt_lang != args.lang:
                continue

        lang_out_dir = os.path.join(output_dir, prompt_lang)

        base = base_strings[prompt_lang]
        for adj_lang in langs:
            adj_lang_out_dir = os.path.join(lang_out_dir, adj_lang)
            os.makedirs(adj_lang_out_dir, exist_ok=True)

            #desc_based = {'ablation': {'outputs': {}, 'logits_and_ranks': {}}, 'amplification': {'outputs': {}, 'logits_and_ranks': {}}, 'intervention': {'outputs': {}, 'logits_and_ranks': {}}, 'direction_ablation_everything_except': {'outputs': {}, 'logits_and_ranks': {}}, 'amplification_everything_except': {'outputs': {}, 'logits_and_ranks': {}}}
            #val_based = {'ablation': {'outputs': {}, 'logits_and_ranks': {}}, 'amplification': {'outputs': {}, 'logits_and_ranks': {}}, 'intervention': {'outputs': {}, 'logits_and_ranks': {}}, 'direction_ablation_everything_except': {'outputs': {}, 'logits_and_ranks': {}}, 'amplification_everything_except': {'outputs': {}, 'logits_and_ranks': {}}}
            #freq_based = {'ablation': {'outputs': {}, 'logits_and_ranks': {}}, 'amplification': {'outputs': {}, 'logits_and_ranks': {}}, 'intervention': {'outputs': {}, 'logits_and_ranks': {}}, 'direction_ablation_everything_except': {'outputs': {}, 'logits_and_ranks': {}}, 'amplification_everything_except': {'outputs': {}, 'logits_and_ranks': {}}}
            desc_based = {'direction_ablation': {'outputs': {}, 'logits_and_ranks': {}}, 'direction_intervention': {'outputs': {}, 'logits_and_ranks': {}}}
            val_based = {'direction_ablation': {'outputs': {}, 'logits_and_ranks': {}}, 'direction_intervention': {'outputs': {}, 'logits_and_ranks': {}}}
            freq_based = {'direction_ablation': {'outputs': {}, 'logits_and_ranks': {}}, 'direction_intervention': {'outputs': {}, 'logits_and_ranks': {}}}


            before_intervention = {'outputs': {}, 'ranks': {}}
            for adj, ans in big_data:
                adjective = adj[adj_lang]
                prompt = base.format(adj=adjective)
                
                base_line, logits = model_run(prompt, model)
                before_intervention['outputs'][prompt] = base_line
                before_intervention['ranks'][prompt] = dict()
                for measure_lang in langs:
                    rank = get_best_rank(logits, ans[measure_lang], model)
                    before_intervention['ranks'][prompt][measure_lang] = rank
                
                if (get_best_rank(logits, ans['en'], model) >= 10 and 
                    get_best_rank(logits, ans[adj_lang], model) >= 10 and
                    get_best_rank(logits, ans[prompt_lang], model) >= 10):
                    # the model does not understand the correct meaning of the adjective
                    print(f"skipping prompt {prompt}")
                    continue


                
                ( 
                 ablations_dict, ablations_outputs, 
                 amplifications_dict, amplifications_outputs, 
                 interventions_outputs, interventions_dict
                 ) = perform_intervention(prompt, model, logits, adj_lang, ans, desc_ablations, desc_amplifications, langs)
                
                #(
                # ablations_et_dict, ablations_et_outputs,
                # amplification_et_dict, amplification_et_outputs
                # ) = perform_everything_intervention(prompt, model, logits, adj_lang, ans, desc_ablations, desc_amplifications, langs)
                
                desc_based['direction_ablation']['outputs'][prompt] = ablations_outputs
                desc_based['direction_ablation']['logits_and_ranks'][prompt] = ablations_dict
                #desc_based['amplification']['outputs'][prompt] = amplifications_outputs
                #desc_based['amplification']['logits_and_ranks'][prompt] = amplifications_dict
                desc_based['direction_intervention']['outputs'][prompt] = interventions_outputs
                desc_based['direction_intervention']['logits_and_ranks'][prompt] = interventions_dict
                #desc_based['ablation_everything_except']['outputs'][prompt] = ablations_et_outputs
                #desc_based['ablation_everything_except']['logits_and_ranks'][prompt] = ablations_et_dict
                #desc_based['amplification_everything_except']['outputs'][prompt] = amplification_et_outputs
                #desc_based['amplification_everything_except']['logits_and_ranks'][prompt] = amplification_et_dict
                
                (
                 ablations_dict, ablations_outputs, 
                 amplifications_dict, amplifications_outputs, 
                 interventions_outputs, interventions_dict
                 ) = perform_intervention(prompt, model, logits, adj_lang, ans, val_ablations, val_amplifications, langs)
                #(
                # ablations_et_dict, ablations_et_outputs,
                # amplification_et_dict, amplification_et_outputs
                # ) = perform_everything_intervention(prompt, model, logits, adj_lang, ans, val_ablations, val_amplifications, langs)
                
                val_based['direction_ablation']['outputs'][prompt] = ablations_outputs
                val_based['direction_ablation']['logits_and_ranks'][prompt] = ablations_dict
                #val_based['amplification']['outputs'][prompt] = amplifications_outputs
                #val_based['amplification']['logits_and_ranks'][prompt] = amplifications_dict
                val_based['direction_intervention']['outputs'][prompt] = interventions_outputs
                val_based['direction_intervention']['logits_and_ranks'][prompt] = interventions_dict
                #val_based['ablation_everything_except']['outputs'][prompt] = ablations_et_outputs
                #val_based['ablation_everything_except']['logits_and_ranks'][prompt] = ablations_et_dict
                #val_based['amplification_everything_except']['outputs'][prompt] = amplification_et_outputs
                #val_based['amplification_everything_except']['logits_and_ranks'][prompt] = amplification_et_dict
                
                (
                 ablations_dict, ablations_outputs, 
                 amplifications_dict, amplifications_outputs, 
                 interventions_outputs, interventions_dict
                 ) = perform_intervention(prompt, model, logits, adj_lang, ans, freq_ablations, freq_amplifications, langs)
                #(
                # ablations_et_dict, ablations_et_outputs,
                # amplification_et_dict, amplification_et_outputs
                # ) = perform_everything_intervention(prompt, model, logits, adj_lang, ans, val_ablations, val_amplifications, langs)
                
                freq_based['direction_ablation']['outputs'][prompt] = ablations_outputs
                freq_based['direction_ablation']['logits_and_ranks'][prompt] = ablations_dict
                #freq_based['amplification']['outputs'][prompt] = amplifications_outputs
                #freq_based['amplification']['logits_and_ranks'][prompt] = amplifications_dict
                freq_based['direction_intervention']['outputs'][prompt] = interventions_outputs
                freq_based['direction_intervention']['logits_and_ranks'][prompt] = interventions_dict
                #freq_based['ablation_everything_except']['outputs'][prompt] = ablations_et_outputs
                #freq_based['ablation_everything_except']['logits_and_ranks'][prompt] = ablations_et_dict
                #freq_based['amplification_everything_except']['outputs'][prompt] = amplification_et_outputs
                #freq_based['amplification_everything_except']['logits_and_ranks'][prompt] = amplification_et_dict

            for key, val in before_intervention.items():
                file_name = f'before_intervention_{key}.json'
                with open(os.path.join(adj_lang_out_dir, file_name), 'w') as f:
                    json.dump(val, f, indent=4)

            for key, val in desc_based.items():
                for key2, val2 in val.items():
                    file_name = f'description_based_{key}_{key2}.json'
                    with open(os.path.join(adj_lang_out_dir, file_name), 'w') as f:
                        json.dump(val2, f, indent=4)

            for key, val in val_based.items():
                for key2, val2 in val.items():
                    file_name = f'value_based_{key}_{key2}.json'
                    with open(os.path.join(adj_lang_out_dir, file_name), 'w') as f:
                        json.dump(val2, f, indent=4)

            for key, val in freq_based.items():
                for key2, val2 in val.items():
                    file_name = f'frequency_based_{key}_{key2}.json'
                    with open(os.path.join(adj_lang_out_dir, file_name), 'w') as f:
                        json.dump(val2, f, indent=4)


