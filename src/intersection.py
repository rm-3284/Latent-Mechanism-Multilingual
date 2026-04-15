import argparse
import json
import os

from ablation_amplification_intervention import (
    description_based_features,
    mean_value_based_features,
    freq_based_features,
)
from models import hf_model_names
from template import lang_to_flores_key, identifiers

def intersection_across_langs(d):
    keys = sorted(list(d.keys()))
    result = dict()
    for base_lang in keys:
        result[base_lang] = dict()
        for intersection_lang in keys:
            base = set(d[base_lang])
            comparison = set(d[intersection_lang])
            intersection = base.intersection(comparison)
            result[base_lang][intersection_lang] = (len(base), len(intersection))
    return result

def intersection_across_dictionary(annSel, valSel, freqSel):
    langs = sorted(list(annSel.keys()))
    dictionary = {"AnnSel": annSel, "ValSel": valSel, "FreqSel": freqSel}
    result = dict()
    for method1, d1 in dictionary.items():
        result[method1] = dict()
        for method2, d2 in dictionary.items():
            result[method1][method2] = dict()
            for lang1 in langs:
                result[method1][method2][lang1] = dict()
                for lang2 in langs:
                    set1 = set(d1[lang1])
                    set2 = set(d2[lang2])
                    intersection = set1.intersection(set2)
                    result[method1][method2][lang1][lang2] = (len(set1), len(intersection))
    return result

def number_of_features_with_lang_name(feature_dict, description_dict):
    counts_dict = dict()
    for lang, features in feature_dict.items():
        identifier = identifiers[lang]
        count = 0
        for feature in features:
            try:
                description = description_dict[feature]            
            except KeyError:
                print(f"Feature {feature} not found in description dictionary.")
                continue    
            for item in identifier:
                if item in description:
                    count += 1
                    break
        counts_dict[lang] = (len(features), count)
    return counts_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Compute feature-overlap summaries across methods/languages")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="qwen3-4b",
        choices=hf_model_names.keys(),
        help="Model to use for loading feature data",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    model_name = args.model
    # relevant directories
    current_file_path = __file__
    current_directory = os.path.dirname(current_file_path)
    absolute_directory = os.path.abspath(current_directory)
    data_directory = os.path.join(os.path.dirname(absolute_directory), "data")
    flores_directory = os.path.join(data_directory, "flores_features", model_name)
    lang_specific_directory = os.path.join(data_directory, "language_specific_features", model_name)
    multilingual_features_directory = os.path.join(data_directory, "multilingual_llm_features", model_name)
    amplification_values_directory = os.path.join(data_directory, "amplification_values", model_name)
    output_directory = os.path.join(data_directory, "additional_experiments", model_name)
    os.makedirs(output_directory, exist_ok=True)

    langs = list(lang_to_flores_key.keys())

    # get the features + amplification values (dict[lang, list[str]])
    desc_features = description_based_features(flores_directory, langs, 0.1)
    val_features = mean_value_based_features(multilingual_features_directory, langs, 50)
    freq_features = freq_based_features(lang_specific_directory, langs)

    """
    annSelLang = intersection_across_langs(desc_features)
    valSelLang = intersection_across_langs(val_features)
    freqSelLang = intersection_across_langs(freq_features)
    across_dict = intersection_across_dictionary(desc_features, val_features, freq_features)
    with open(os.path.join(output_directory, f"intersections_{model_name}.json"), 'w') as f:
        json.dump("AnnSel intersection between languages", f, indent=4)
        f.write('\n')
        json.dump(annSelLang, f, indent=4)
        f.write('\n')
        json.dump("ValSel intersection between languages", f, indent=4)
        f.write('\n')
        json.dump(valSelLang, f, indent=4)
        f.write('\n')
        json.dump("FreqSel intersection between languages", f, indent=4)
        f.write('\n')
        json.dump(freqSelLang, f, indent=4)
        f.write('\n')
        json.dump("Intersection between different methods and languages", f, indent=4)
        f.write('\n')
        json.dump(across_dict, f, indent=4)
        f.write('\n')
    """
    desc_counts = dict()
    for lang, features in desc_features.items():
        desc_counts[lang] = (len(features), len(features))
    
    val_description_dict = dict()
    for lang in val_features.keys():
        filename = f"{lang}_description.json"
        with open(os.path.join(multilingual_features_directory, filename), 'r') as f:
            description_dict = json.load(f)
        for key, value in description_dict.items():
            val_description_dict[key] = value
    val_counts = number_of_features_with_lang_name(val_features, val_description_dict)

    freq_description_dict = dict()
    for lang in freq_features.keys():
        filename = f"{lang}_description.json"
        with open(os.path.join(lang_specific_directory, filename), 'r') as f:
            description_dict = json.load(f)
        for key, value in description_dict.items():
            freq_description_dict[key] = value
    freq_counts = number_of_features_with_lang_name(freq_features, freq_description_dict)

    with open(os.path.join(output_directory, f"lang_name_counts_{model_name}.json"), 'w') as f:
        json.dump("AnnSel counts", f, indent=4)
        f.write('\n')
        json.dump(desc_counts, f, indent=4)
        f.write('\n')
        json.dump("ValSel counts", f, indent=4)
        f.write('\n')
        json.dump(val_counts, f, indent=4)
        f.write('\n')
        json.dump("FreqSel counts", f, indent=4)
        f.write('\n')
        json.dump(freq_counts, f, indent=4)
