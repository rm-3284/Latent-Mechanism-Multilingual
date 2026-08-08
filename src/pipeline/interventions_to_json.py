
import argparse
import gc
import json
import os
import torch
import torch.nn.functional as F
import transformers
from tqdm import tqdm
from typing import Any

from lib.ablation_amplification_intervention import (
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
from lib.circuit_tracer_import import ReplacementModel
from lib.pipeline_data.adjectives import big_data
from lib.device_setup import device
from lib.direction_ablation import (
    interventions_to_dict, interventions_to_dict_everything_ablation, 
    run_ablation_experiment
    )
from lib.intervention import (
    ablation,
    amplification,
    load_nnsight_model,
)
from lib.template import lang_to_flores_key, base_strings
from lib.models import hf_model_names, hf_transcoder_names, layer_num, use_bos

METHOD_FILE_STEMS = {
    "description": "description",
    "value": "value",
    "frequency": "frequency",
}
EMPTY_EXPERIMENT_KEYS = (
    "original",
    "distractor ablation",
    "ablation",
    "distractor one-layer direction ablation",
    "one-layer direction ablation",
    "distractor multi-layer direction ablation",
    "multi-layer direction ablation",
    "amplification",
    "non-distractor amplification",
    "feature-intervention",
    "one-layer direction intervention",
)


def empty_results_dict() -> dict:
    return {key: {} for key in EMPTY_EXPERIMENT_KEYS}


def fill_one_layer_direction_ablations(
    model,
    prompt: str,
    langs: list[str],
    one_layer_ablation_for_method: dict,
    ablations: dict,
) -> None:
    for lang in langs:
        layer, features = one_layer_ablation_for_method[lang]
        ablations["one-layer-direction"][lang] = direction_ablation_helper(
            model, layer, features, prompt
        )
    for lang in langs:
        ablations["one-layer-direction_everything"][lang] = combine_except_one(
            ablations["one-layer-direction"], lang
        )


def run_feature_experiments_for_methods(
    prompt: str,
    model,
    ans: dict,
    langs: list[str],
    model_name: str,
    method_bundles: list[tuple[dict, dict, dict]],
) -> None:
    """Run shared non-nnsight experiments for each (results, ablations, amplifications)."""
    feature_specs = [
        ("distractor ablation", "feature", "ablation"),
        ("ablation", "feature_everything", "ablation"),
        ("distractor one-layer direction ablation", "one-layer-direction", "ablation"),
        ("one-layer direction ablation", "one-layer-direction_everything", "ablation"),
        ("amplification", "everything", "amplification"),
        ("non-distractor amplification", "normal", "amplification"),
    ]
    combo_specs = [
        ("feature-intervention", "feature", "normal"),
        ("one-layer direction intervention", "one-layer-direction", "normal"),
    ]
    for results, ablations, amplifications in method_bundles:
        for exp_name, key, source in feature_specs:
            interv = ablations[key] if source == "ablation" else amplifications[key]
            results[exp_name][prompt] = feature_interventions(
                prompt, model, ans, interv, langs, model_name
            )
        for exp_name, abl_key, amp_key in combo_specs:
            results[exp_name][prompt] = feature_ablation_and_amplification(
                prompt,
                model,
                ans,
                ablations[abl_key],
                amplifications[amp_key],
                langs,
                model_name,
            )


def run_nnsight_experiments_for_methods(
    prompt: str,
    tokenizer,
    ans: dict,
    langs: list[str],
    nnsight_model,
    nnsight_device,
    model_name: str,
    method_bundles: list[tuple[dict, dict, dict]],
) -> None:
    for results, ablations, _amplifications in method_bundles:
        results["distractor multi-layer direction ablation"][prompt] = direction_ablate(
            prompt,
            tokenizer,
            ans,
            ablations["direction-ablation"],
            langs,
            nnsight_model,
            device=nnsight_device,
            model_name=model_name,
        )
        results["multi-layer direction ablation"][prompt] = direction_ablate(
            prompt,
            tokenizer,
            ans,
            ablations["direction-ablation-everything"],
            langs,
            nnsight_model,
            device=nnsight_device,
            model_name=model_name,
        )


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
    from lib.paths import data_dir_str
    data_directory = data_dir_str()
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

            desc_based = empty_results_dict()
            val_based = empty_results_dict()
            freq_based = empty_results_dict()
            method_bundles = [
                (desc_based, desc_ablations, desc_amplifications),
                (val_based, val_ablations, val_amplifications),
                (freq_based, freq_ablations, freq_amplifications),
            ]
            one_layer_by_method = [
                (one_layer_ablation["desc"], desc_ablations),
                (one_layer_ablation["val"], val_ablations),
                (one_layer_ablation["freq"], freq_ablations),
            ]

            for adj, ans in tqdm(big_data, desc=f"Adjectives non-nnsight ({prompt_lang}, {adj_lang})"):
                adjective = adj[adj_lang]
                prompt = base.format(adj=adjective)

                for ola_map, ablations in one_layer_by_method:
                    fill_one_layer_direction_ablations(
                        model, prompt, langs, ola_map, ablations
                    )

                base_line, logits = model_run(prompt, model)
                before_intervention = get_logits_and_ranks(
                    logits, ans, model.tokenizer, model_name
                )
                for results, _ablations, _amplifications in method_bundles:
                    results["original"][prompt] = {
                        "output": base_line,
                        "langs": before_intervention,
                    }

                run_feature_experiments_for_methods(
                    prompt, model, ans, langs, model_name, method_bundles
                )

            del model
            gc.collect()
            torch.cuda.empty_cache()

            if not args.skip_direction_ablation:
                nnsight_model = load_nnsight_model(
                    model_name, device_map="cpu" if args.nnsight_cpu else str(device)
                )
                nnsight_device = "cpu" if args.nnsight_cpu else device
                for adj, ans in tqdm(big_data, desc=f"Adjectives nnsight ({prompt_lang}, {adj_lang})"):
                    adjective = adj[adj_lang]
                    prompt = base.format(adj=adjective)
                    run_nnsight_experiments_for_methods(
                        prompt,
                        tokenizer,
                        ans,
                        langs,
                        nnsight_model,
                        nnsight_device,
                        model_name,
                        method_bundles,
                    )

                del nnsight_model
                gc.collect()
                torch.cuda.empty_cache()

            for method_name, results in (
                ("description", desc_based),
                ("value", val_based),
                ("frequency", freq_based),
            ):
                filename = f"interventions_and_results_{METHOD_FILE_STEMS[method_name]}.json"
                with open(os.path.join(adj_lang_out_dir, filename), "w") as f:
                    json.dump(results, f, indent=4)
