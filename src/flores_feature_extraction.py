import argparse
from datasets import load_dataset
import gc
import json
import pandas as pd
import torch
import os

from circuit_tracer_import import ReplacementModel, attribute
from pipeline_data.generic_sentences import alphabet_char, filter_sentences
from device_setup import device
from feature_extraction import distinct_path_max_bottleneck, prune_paths_by_first_last, pick_last_pos_features
from template import lang_to_flores_key
from models import hf_model_names, hf_transcoder_names

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
    features = []
    for prompt in sentences:
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
        decoded = model.tokenizer.encode(prompt)
        for pos in range (1, len(decoded)): # the first one is <bos>
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
        del graph, paths, pruned, last_pos_features
        torch.cuda.empty_cache()
        gc.collect()
    
    return features

def argsparse():
    parser = argparse.ArgumentParser(description='Extract features from FLORES dataset')
    parser.add_argument('--model', type=str, default='gemma-2-2b', choices=hf_model_names.keys(), help='Model to use for feature extraction')
    parser.add_argument('--lang', type=str, default=None, choices=lang_to_flores_key.keys(), help='Language to extract features for')
    return parser.parse_args()

if __name__ == "__main__":
    args = argsparse()
    current_file_path = __file__
    current_directory = os.path.dirname(current_file_path)
    absolute_directory = os.path.abspath(current_directory)
    data_directory = os.path.join(os.path.dirname(absolute_directory), "data", "flores_features", args.model)
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
        sentences = filter_sentences(batch, alphabet_char[lang], model, num_sentences=100) # only returns 100 sentences
        del batch
        features = iterate_through_sentences(model, sentences, max_feature_nodes=None)  # No limits!
        file_name = f'{lang}.json'
        with open(os.path.join(data_directory, file_name), 'w') as f:
            json.dump(features, f)
        del features, sentences
        torch.cuda.empty_cache()
        gc.collect()
