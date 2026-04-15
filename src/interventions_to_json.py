
import argparse
import gc
import json
import nnsight
import os
import torch
import torch.nn.functional as F
import transformers
from tqdm import tqdm
from typing import Any

from ablation_amplification_intervention import (
    activation_dict,
    ablation_and_amplification,
    combine_except_one,
    description_based_features,
    direction_ablation_layer_determine,
    direction_ablation_helper,
    freq_based_features,
    mean_value_based_features,
    model_intervention,
    model_run,
)
from circuit_tracer_import import ReplacementModel
from pipeline_data.adjectives import big_data
from device_setup import device
from direction_ablation import (
    interventions_to_dict, interventions_to_dict_everything_ablation, 
    run_ablation_experiment
    )
from intervention import (
    ablation, amplification,
    )
from template import lang_to_flores_key, base_strings
from models import hf_model_names, hf_transcoder_names, layer_num, use_bos

def get_logit_and_rank(logits: torch.Tensor, target: str, tokenizer: Any, model_name: str) -> tuple[float, int, float]:
    # returns logit, rank, prob
    # use bos or not
    if use_bos.get(model_name, False):
        idx = 1
    else:
        idx = 0

    l = logits.squeeze(0)[-1]
    t = tokenizer.encode(target)[idx] # 1 if they use bos token
    lg = l[t]
    lg = lg.item() if isinstance(lg, torch.Tensor) else lg

    _, indices = torch.sort(l, dim=-1, descending=True)
    mask = (indices == t)
    rank = torch.argmax(mask.int(), dim=-1)
    rank = rank.item() if isinstance(rank, torch.Tensor) else rank

    probs = F.softmax(l, dim=-1)
    prob = probs[t]
    prob = prob.item() if isinstance(prob, torch.Tensor) else prob
    return lg, rank, prob

def get_logits_and_ranks(logit: torch.Tensor, ans: dict[str, list[str]], tokenizer: Any, model_name: str) -> dict[str, dict[str, tuple[float, int, float]]]:
    result = dict()
    for key, value in ans.items():
        result[key] = dict()
        for v in value:
            logit_and_rank = get_logit_and_rank(logit, v, tokenizer, model_name)
            result[key][v] = logit_and_rank
    return result

def get_top_outputs_from_tokenizer(logits: torch.Tensor, tokenizer: Any, k: int = 10):
    top_probs, top_token_ids = logits.squeeze(0)[-1].softmax(-1).topk(k)
    top_tokens = [tokenizer.decode(token_id) for token_id in top_token_ids]
    top_outputs = list(zip(top_tokens, top_probs.tolist()))
    return top_outputs

def feature_interventions(prompt: str, model: ReplacementModel, ans: dict[str, list[str]], intervention: dict[str, list[tuple[int, int, int, float]]], langs: list[str], model_name: str):
    results = dict()
    for intervened_lang in langs:
        results[intervened_lang] = dict()

        new_outputs, new_logits = model_intervention(prompt, model, intervention[intervened_lang])
        result = get_logits_and_ranks(new_logits, ans, model.tokenizer, model_name)
        results[intervened_lang]['output'] = new_outputs
        results[intervened_lang]['langs'] = result
    return results

def direction_ablate(prompt: str, tokenizer: Any,
    ans, interventions, langs, nnsight_model, device=device, model_name: str = "gemma-2-2b"
    ):
    results = dict()
    for intervention_lang in langs:
        results[intervention_lang] = dict()
        ablation_logits = run_ablation_experiment(nnsight_model, prompt, interventions[intervention_lang], device=device)
        result = get_logits_and_ranks(ablation_logits, ans, tokenizer, model_name)
        outputs = get_top_outputs_from_tokenizer(ablation_logits, tokenizer)
        results[intervention_lang]['output'] = outputs
        results[intervention_lang]['langs'] = result
    return results

def feature_ablation_and_amplification(prompt: str, model: ReplacementModel, ans: dict[str, list[str]], ablation, amplification, langs, model_name: str = "gemma-2-2b"):
    results = dict()
    for ablation_lang in langs:
        results[ablation_lang] = dict()
        for amplification_lang in langs:
            results[ablation_lang][amplification_lang] = dict()
            interventions = ablation_and_amplification(ablation[ablation_lang], amplification[amplification_lang])
            new_outputs, logits = model_intervention(prompt, model, interventions)
            result = get_logits_and_ranks(logits, ans, model.tokenizer, model_name)
            results[ablation_lang][amplification_lang]['output'] = new_outputs
            results[ablation_lang][amplification_lang]['langs'] = result
    return results

def load_replacement_model(model_name: str) -> ReplacementModel:
    transcoder_name = hf_transcoder_names.get(model_name, "gemma")
    return ReplacementModel.from_pretrained(hf_model_names[model_name], transcoder_name, device=device, dtype=torch.bfloat16)

def load_nnsight_model(model_name: str, use_cpu: bool = False):
    device_map = "cpu" if use_cpu else device
    return nnsight.LanguageModel(hf_model_names[model_name], device_map=device_map)

def patch_check_model_inputs_for_qwen3(model_name: str):
    if not model_name.lower().startswith("qwen"):
        return
    try:
        qwen3_model = transformers.models.qwen3.modeling_qwen3.Qwen3Model
        if hasattr(qwen3_model.forward, "__wrapped__"):
            print("[Patch] Removing strict signature check from Qwen3Model...")
            qwen3_model.forward = qwen3_model.forward.__wrapped__
    except Exception as exc:
        print(f"[Patch Warning] Could not patch Qwen3Model.forward: {exc}")

def move_direction_maps_to_cpu(direction_maps: dict[str, dict[int, torch.Tensor]]) -> dict[str, dict[int, torch.Tensor]]:
    moved_maps = dict()
    for lang, layer_map in direction_maps.items():
        moved_maps[lang] = dict()
        for layer, directions in layer_map.items():
            moved_maps[lang][layer] = directions.detach().cpu()
    return moved_maps

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prompt language",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # 2. Add a string argument
    # 'message' is the variable name inside the script
    # '--input-string' is the flag used on the command line
    parser.add_argument(
        '--prompt_lang',
        '-pl',
        type=str,
        default=None,
        help='Prompt language',
    )
    parser.add_argument(
        '--adj_lang',
        '-al',
        type=str,
        default=None,
        help='Adjective language',
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
        '--skip_direction_ablation',
        action='store_true',
        help='Skip nnsight-based multi-layer direction ablation experiments (useful for low VRAM runs).',
    )
    parser.add_argument(
        '--nnsight_cpu',
        action='store_true',
        help='Run nnsight-based multi-layer direction ablation on CPU to reduce CUDA memory usage.',
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(f"Running with prompt language: {args.prompt_lang}, adjective language: {args.adj_lang}, model: {args.model}")

    model_name = args.model

    patch_check_model_inputs_for_qwen3(model_name)

    if args.nnsight_cpu and not args.skip_direction_ablation:
        print("nnsight direction ablation will run on CPU.")

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

    desc_ablations = {'feature': dict(), 'one-layer-direction': dict(), 'feature_everything': dict(), 'one-layer-direction_everything': dict(), 'direction-ablation': dict(), 'direction-ablation-everything': dict()}
    desc_amplifications = {'normal': dict(), 'everything': dict()}
    val_ablations = {'feature': dict(), 'one-layer-direction': dict(), 'feature_everything': dict(), 'one-layer-direction_everything': dict(), 'direction-ablation': dict(), 'direction-ablation-everything': dict()}
    val_amplifications = {'normal': dict(), 'everything': dict()}
    freq_ablations = {'feature': dict(), 'one-layer-direction': dict(), 'feature_everything': dict(), 'one-layer-direction_everything': dict(), 'direction-ablation': dict(), 'direction-ablation-everything': dict()}
    freq_amplifications = {'normal': dict(), 'everything': dict()}
    for lang in langs:
        desc_ablations['feature'][lang] = ablation(desc_interventions, lang)
        desc_amplifications['normal'][lang] = amplification(desc_interventions, lang)
        val_ablations['feature'][lang] = ablation(val_interventions, lang)
        val_amplifications['normal'][lang] = amplification(val_interventions, lang)
        freq_ablations['feature'][lang] = ablation(freq_interventions, lang)
        freq_amplifications['normal'][lang] = amplification(freq_interventions, lang)

    for lang in langs:
        desc_ablations['feature_everything'][lang] = combine_except_one(desc_ablations['feature'], lang)
        desc_amplifications['everything'][lang] = combine_except_one(desc_amplifications['normal'], lang)
        val_ablations['feature_everything'][lang] = combine_except_one(val_ablations['feature'], lang)
        val_amplifications['everything'][lang] = combine_except_one(val_amplifications['normal'], lang)
        freq_ablations['feature_everything'][lang] = combine_except_one(freq_ablations['feature'], lang)
        freq_amplifications['everything'][lang] = combine_except_one(freq_amplifications['normal'], lang)

    if not args.skip_direction_ablation:
        direction_setup_model = load_replacement_model(model_name)
        for lang in langs:
            desc_ablations['direction-ablation'][lang] = interventions_to_dict(desc_interventions, lang, direction_setup_model)
            desc_ablations['direction-ablation-everything'][lang] = interventions_to_dict_everything_ablation(desc_interventions, lang, direction_setup_model)
            val_ablations['direction-ablation'][lang] = interventions_to_dict(val_interventions, lang, direction_setup_model)
            val_ablations['direction-ablation-everything'][lang] = interventions_to_dict_everything_ablation(val_interventions, lang, direction_setup_model)
            freq_ablations['direction-ablation'][lang] = interventions_to_dict(freq_interventions, lang, direction_setup_model)
            freq_ablations['direction-ablation-everything'][lang] = interventions_to_dict_everything_ablation(freq_interventions, lang, direction_setup_model)

        desc_ablations['direction-ablation'] = move_direction_maps_to_cpu(desc_ablations['direction-ablation'])
        desc_ablations['direction-ablation-everything'] = move_direction_maps_to_cpu(desc_ablations['direction-ablation-everything'])
        val_ablations['direction-ablation'] = move_direction_maps_to_cpu(val_ablations['direction-ablation'])
        val_ablations['direction-ablation-everything'] = move_direction_maps_to_cpu(val_ablations['direction-ablation-everything'])
        freq_ablations['direction-ablation'] = move_direction_maps_to_cpu(freq_ablations['direction-ablation'])
        freq_ablations['direction-ablation-everything'] = move_direction_maps_to_cpu(freq_ablations['direction-ablation-everything'])

        del direction_setup_model
        gc.collect()
        torch.cuda.empty_cache()

    one_layer_ablation = {'desc': dict(), 'val': dict(), 'freq': dict()}
    for lang in langs:
        one_layer_ablation['desc'][lang] = direction_ablation_layer_determine(desc_interventions, lang, num_layers=layer_num[model_name])
        one_layer_ablation['val'][lang] = direction_ablation_layer_determine(val_interventions, lang, num_layers=layer_num[model_name])
        one_layer_ablation['freq'][lang] = direction_ablation_layer_determine(freq_interventions, lang, num_layers=layer_num[model_name])


    # ablation + amplification experiments
    output_dir = os.path.join(data_directory, "interventions", model_name)
    for prompt_lang in tqdm(langs, desc="Prompt Languages"):
        if args.prompt_lang != None:
            if prompt_lang != args.prompt_lang:
                continue

        lang_out_dir = os.path.join(output_dir, prompt_lang)

        base = base_strings[prompt_lang]
        for adj_lang in tqdm(langs, desc=f"Adj Languages ({prompt_lang})"):
            if args.adj_lang is not None:
                if adj_lang != args.adj_lang:
                    continue

            model = load_replacement_model(model_name)
            tokenizer = model.tokenizer

            adj_lang_out_dir = os.path.join(lang_out_dir, adj_lang)
            os.makedirs(adj_lang_out_dir, exist_ok=True)

            desc_based = {
                'original': {}, 'distractor ablation': {}, 'ablation': {}, 'distractor one-layer direction ablation': {}, 
                'one-layer direction ablation': {}, 'distractor multi-layer direction ablation': {}, 
                'multi-layer direction ablation': {}, 'amplification': {}, 'non-distractor amplification': {}, 
                'feature-intervention': {}, 'one-layer direction intervention': {}, }
            val_based = {
                'original': {}, 'distractor ablation': {}, 'ablation': {}, 'distractor one-layer direction ablation': {}, 
                'one-layer direction ablation': {}, 'distractor multi-layer direction ablation': {}, 
                'multi-layer direction ablation': {}, 'amplification': {}, 'non-distractor amplification': {}, 
                'feature-intervention': {}, 'one-layer direction intervention': {}, }
            freq_based = {
                'original': {}, 'distractor ablation': {}, 'ablation': {}, 'distractor one-layer direction ablation': {}, 
                'one-layer direction ablation': {}, 'distractor multi-layer direction ablation': {}, 
                'multi-layer direction ablation': {}, 'amplification': {}, 'non-distractor amplification': {}, 
                'feature-intervention': {}, 'one-layer direction intervention': {}, }

            for adj, ans in tqdm(big_data, desc=f"Adjectives non-nnsight ({prompt_lang}, {adj_lang})"):
                adjective = adj[adj_lang]
                prompt = base.format(adj=adjective)

                # calculate the direction ablations
                
                for lang in langs:
                    layer, features = one_layer_ablation['desc'][lang]
                    interventions = direction_ablation_helper(model, layer, features, prompt)
                    desc_ablations['one-layer-direction'][lang] = interventions
                for lang in langs:
                    desc_ablations['one-layer-direction_everything'][lang] = combine_except_one(desc_ablations['one-layer-direction'], lang)
                
                for lang in langs:
                    layer, features = one_layer_ablation['val'][lang]
                    interventions = direction_ablation_helper(model, layer, features, prompt)
                    val_ablations['one-layer-direction'][lang] = interventions
                for lang in langs:
                    val_ablations['one-layer-direction_everything'][lang] = combine_except_one(val_ablations['one-layer-direction'], lang)
                
                for lang in langs:
                    layer, features = one_layer_ablation['freq'][lang]
                    interventions = direction_ablation_helper(model, layer, features, prompt)
                    freq_ablations['one-layer-direction'][lang] = interventions
                for lang in langs:
                    freq_ablations['one-layer-direction_everything'][lang] = combine_except_one(freq_ablations['one-layer-direction'], lang)
    
                
                # original
                base_line, logits = model_run(prompt, model)
                before_intervention = get_logits_and_ranks(logits, ans, model.tokenizer, model_name)
                desc_based['original'][prompt] = {'output': base_line, 'langs': before_intervention}
                val_based['original'][prompt] = {'output': base_line, 'langs': before_intervention}
                freq_based['original'][prompt] = {'output': base_line, 'langs': before_intervention}
                
                # distractor ablation
                desc_based['distractor ablation'][prompt] = feature_interventions(prompt, model, ans, desc_ablations['feature'], langs, model_name)
                val_based['distractor ablation'][prompt] = feature_interventions(prompt, model, ans, val_ablations['feature'], langs, model_name)
                freq_based['distractor ablation'][prompt] = feature_interventions(prompt, model, ans, freq_ablations['feature'], langs, model_name)

                # ablation
                desc_based['ablation'][prompt] = feature_interventions(prompt, model, ans, desc_ablations['feature_everything'], langs, model_name)
                val_based['ablation'][prompt] = feature_interventions(prompt, model, ans, val_ablations['feature_everything'], langs, model_name)
                freq_based['ablation'][prompt] = feature_interventions(prompt, model, ans, freq_ablations['feature_everything'], langs, model_name)

                # distractor one-layer direction
                desc_based['distractor one-layer direction ablation'][prompt] = feature_interventions(prompt, model, ans, desc_ablations['one-layer-direction'], langs, model_name)
                val_based['distractor one-layer direction ablation'][prompt] = feature_interventions(prompt, model, ans, val_ablations['one-layer-direction'], langs, model_name)
                freq_based['distractor one-layer direction ablation'][prompt] = feature_interventions(prompt, model, ans, freq_ablations['one-layer-direction'], langs, model_name)

                # one-layer direction ablation
                desc_based['one-layer direction ablation'][prompt] = feature_interventions(prompt, model, ans, desc_ablations['one-layer-direction_everything'], langs, model_name)
                val_based['one-layer direction ablation'][prompt] = feature_interventions(prompt, model, ans, val_ablations['one-layer-direction_everything'], langs, model_name)
                freq_based['one-layer direction ablation'][prompt] = feature_interventions(prompt, model, ans, freq_ablations['one-layer-direction_everything'], langs, model_name)

                # amplification
                desc_based['amplification'][prompt] = feature_interventions(prompt, model, ans, desc_amplifications['everything'], langs, model_name)
                val_based['amplification'][prompt] = feature_interventions(prompt, model, ans, val_amplifications['everything'], langs, model_name)
                freq_based['amplification'][prompt] = feature_interventions(prompt, model, ans, freq_amplifications['everything'], langs, model_name)

                # non-distractor amplification
                desc_based['non-distractor amplification'][prompt] = feature_interventions(prompt, model, ans, desc_amplifications['normal'], langs, model_name)
                val_based['non-distractor amplification'][prompt] = feature_interventions(prompt, model, ans, val_amplifications['normal'], langs, model_name)
                freq_based['non-distractor amplification'][prompt] = feature_interventions(prompt, model, ans, freq_amplifications['normal'], langs, model_name)

                # feature-intervention
                desc_based['feature-intervention'][prompt] = feature_ablation_and_amplification(prompt, model, ans, desc_ablations['feature'], desc_amplifications['normal'], langs, model_name)
                val_based['feature-intervention'][prompt] = feature_ablation_and_amplification(prompt, model, ans, val_ablations['feature'], val_amplifications['normal'], langs, model_name)
                freq_based['feature-intervention'][prompt] = feature_ablation_and_amplification(prompt, model, ans, freq_ablations['feature'], freq_amplifications['normal'], langs, model_name)

                # one-layer direction intervention
                desc_based['one-layer direction intervention'][prompt] = feature_ablation_and_amplification(prompt, model, ans, desc_ablations['one-layer-direction'], desc_amplifications['normal'], langs, model_name)
                val_based['one-layer direction intervention'][prompt] = feature_ablation_and_amplification(prompt, model, ans, val_ablations['one-layer-direction'], val_amplifications['normal'], langs, model_name)
                freq_based['one-layer direction intervention'][prompt] = feature_ablation_and_amplification(prompt, model, ans, freq_ablations['one-layer-direction'], freq_amplifications['normal'], langs, model_name)

            del model
            gc.collect()
            torch.cuda.empty_cache()

            if not args.skip_direction_ablation:
                nnsight_model = load_nnsight_model(model_name, use_cpu=args.nnsight_cpu)
                nnsight_device = "cpu" if args.nnsight_cpu else device
                for adj, ans in tqdm(big_data, desc=f"Adjectives nnsight ({prompt_lang}, {adj_lang})"):
                    adjective = adj[adj_lang]
                    prompt = base.format(adj=adjective)

                    desc_based['distractor multi-layer direction ablation'][prompt] = direction_ablate(prompt, tokenizer, ans, desc_ablations['direction-ablation'], langs, nnsight_model, device=nnsight_device, model_name=model_name)
                    val_based['distractor multi-layer direction ablation'][prompt] = direction_ablate(prompt, tokenizer, ans, val_ablations['direction-ablation'], langs, nnsight_model, device=nnsight_device, model_name=model_name)
                    freq_based['distractor multi-layer direction ablation'][prompt] = direction_ablate(prompt, tokenizer, ans, freq_ablations['direction-ablation'], langs, nnsight_model, device=nnsight_device, model_name=model_name)

                    desc_based['multi-layer direction ablation'][prompt] = direction_ablate(prompt, tokenizer, ans, desc_ablations['direction-ablation-everything'], langs, nnsight_model, device=nnsight_device, model_name=model_name)
                    val_based['multi-layer direction ablation'][prompt] = direction_ablate(prompt, tokenizer, ans, val_ablations['direction-ablation-everything'], langs, nnsight_model, device=nnsight_device, model_name=model_name)
                    freq_based['multi-layer direction ablation'][prompt] = direction_ablate(prompt, tokenizer, ans, freq_ablations['direction-ablation-everything'], langs, nnsight_model, device=nnsight_device, model_name=model_name)

                del nnsight_model
                gc.collect()
                torch.cuda.empty_cache()

            filename = "interventions_and_results_description.json"
            with open(os.path.join(adj_lang_out_dir, filename), 'w') as f:
                json.dump(desc_based, f, indent=4)
            
            filename = "interventions_and_results_value.json"
            with open(os.path.join(adj_lang_out_dir, filename), 'w') as f:
                json.dump(val_based, f, indent=4)

            filename = "interventions_and_results_frequency.json"
            with open(os.path.join(adj_lang_out_dir, filename), 'w') as f:
                json.dump(freq_based, f, indent=4)
