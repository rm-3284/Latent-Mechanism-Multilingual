#!/usr/bin/env python3
"""Recompute NaN logprob entries in interventions_multiple_words JSON files.

Uses the same computation functions as multiple_words_intervention.py. Only cells
that are currently NaN are overwritten; other values are left unchanged.

Example:
  python recompute_multiple_words_nans.py --model qwen3-4b --prompt-lang en --list-lang de
  python recompute_multiple_words_nans.py --model qwen3-4b --method description --dry-run
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from typing import Any

import torch

from ablation_amplification_intervention import (
    activation_dict,
    combine_except_one,
    description_based_features,
    direction_ablation_helper,
    direction_ablation_layer_determine,
    freq_based_features,
    mean_value_based_features,
)
from circuit_tracer_import import ReplacementModel
from device_setup import device
from direction_ablation import interventions_to_dict, interventions_to_dict_everything_ablation
from intervention import ablation, amplification
from models import hf_model_names, hf_transcoder_names, layer_num
from multiple_words_intervention import (
    ablation_and_amplification,
    base_prompts,
    categories,
    direction_ablate_logprob,
    feature_ablation_and_amplification_logprob,
    feature_interventions_logprob,
    get_logprob_with_intervention,
    number_of_choices_to_display,
    options,
    transform_intervention,
)
from template import lang_to_flores_key

METHOD_TO_FILE_STEM = {
    "description": "description",
    "frequency": "frequency",
    "value": "value",
}

DIRECTION_EXPERIMENTS = frozenset(
    {
        "distractor multi-layer direction ablation",
        "multi-layer direction ablation",
    }
)

NEEDS_ONE_LAYER_UPDATE = frozenset(
    {
        "distractor one-layer direction ablation",
        "one-layer direction ablation",
        "one-layer direction intervention",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute NaN logprob values in multiple_words intervention JSON files.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="qwen3-4b",
        choices=hf_model_names.keys(),
        help="Model name (directory under data/interventions_multiple_words/).",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=tuple(METHOD_TO_FILE_STEM.keys()),
        help="Feature-selection method file to update (default: all three).",
    )
    parser.add_argument(
        "--prompt-lang",
        "-l",
        type=str,
        default=None,
        help="Only recompute files under this prompt language (e.g. en).",
    )
    parser.add_argument(
        "--list-lang",
        type=str,
        default=None,
        help="Only recompute this list/display language (e.g. de).",
    )
    parser.add_argument(
        "--nnsight-device",
        type=str,
        default="cpu",
        choices=("auto", "cuda", "cpu"),
        help="Device for nnsight (multi-layer direction ablation only).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report NaN counts without loading the model or writing files.",
    )
    parser.add_argument(
        "--regenerate-normalized",
        action="store_true",
        help="After patching, rewrite *_normalized.json using the same rule as data/.../average.py.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run logprob diagnostics on the first NaN prompt before recomputing.",
    )
    return parser.parse_args()


def count_nans(obj: Any) -> int:
    if isinstance(obj, dict):
        return sum(count_nans(v) for v in obj.values())
    if isinstance(obj, float) and math.isnan(obj):
        return 1
    return 0


def normalize_intervention_list(
    intervention: list[tuple[int, int, int, float]],
) -> list[tuple[int, int, int, float]]:
    normalized: list[tuple[int, int, int, float]] = []
    for layer, pos, feature_idx, val in intervention:
        if isinstance(val, torch.Tensor):
            val = val.item()
        fv = float(val)
        if math.isfinite(fv):
            normalized.append((layer, pos, feature_idx, fv))
    return normalized


def normalize_intervention_by_lang(
    intervention_by_lang: dict[str, list[tuple[int, int, int, float]]],
) -> dict[str, list[tuple[int, int, int, float]]]:
    return {lang: normalize_intervention_list(intervention_by_lang[lang]) for lang in intervention_by_lang}


def iter_nan_paths(obj: Any, path: tuple = ()) -> list[tuple]:
    paths: list[tuple] = []
    if isinstance(obj, dict):
        for key, val in obj.items():
            paths.extend(iter_nan_paths(val, path + (key,)))
    elif isinstance(obj, float) and math.isnan(obj):
        paths.append(path)
    return paths


def get_at_path(root: Any, path: tuple) -> Any:
    cur = root
    for part in path[:-1]:
        cur = cur[part]
    return cur[path[-1]]


def set_at_path(root: Any, path: tuple, value: float) -> None:
    cur = root
    for part in path[:-1]:
        cur = cur[part]
    cur[path[-1]] = value


def merge_recomputed_block(dst: dict, src: dict) -> int:
    """Replace dst with src when dst subtree has NaNs and src is clean; else merge leaves."""
    if count_nans(dst) == 0:
        return 0
    if count_nans(src) == 0:
        n_fixed = count_nans(dst)
        dst.clear()
        dst.update(copy.deepcopy(src))
        return n_fixed
    patched = 0
    if isinstance(dst, dict) and isinstance(src, dict):
        for key, src_val in src.items():
            if key not in dst:
                dst[key] = copy.deepcopy(src_val)
                continue
            dst_val = dst[key]
            if isinstance(dst_val, float) and math.isnan(dst_val):
                if isinstance(src_val, float) and math.isfinite(src_val):
                    dst[key] = src_val
                    patched += 1
            elif isinstance(dst_val, dict) and isinstance(src_val, dict):
                patched += merge_recomputed_block(dst_val, src_val)
    return patched


def write_json_atomic(path: str, data: dict) -> None:
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, allow_nan=False)
    os.replace(tmp_path, path)


def load_json_or_restore(path: str) -> dict:
    """Load JSON; if truncated/corrupt, restore from .bak when present."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        backup_path = path + ".bak"
        if os.path.isfile(backup_path):
            with open(backup_path, encoding="utf-8") as f:
                data = json.load(f)
            shutil.copy2(backup_path, path)
            print(f"  restored corrupt file from backup: {path}")
            return data
        raise RuntimeError(
            f"Corrupt JSON at {path}: {exc}. "
            "Likely truncated by a failed prior write. "
            f"Restore with: git checkout -- {path}"
        ) from exc


def validate_json_paths(json_paths: list[tuple[str, str, str, str]]) -> None:
    """Fail fast before model load if any input JSON is unreadable."""
    corrupt: list[str] = []
    for json_path, _method, _pl, _ll in json_paths:
        try:
            load_json_or_restore(json_path)
        except (json.JSONDecodeError, RuntimeError):
            corrupt.append(json_path)
    if corrupt:
        print("ERROR: corrupt JSON (cannot parse). Restore from git or .bak before re-running:")
        for path in corrupt[:10]:
            print(f"  {path}")
        if len(corrupt) > 10:
            print(f"  ... and {len(corrupt) - 10} more")
        raise SystemExit(1)


def recompute_single_cell(
    experiment: str,
    prompt: str,
    path: tuple,
    model: ReplacementModel,
    ablations: dict,
    amplifications: dict,
) -> float:
    """Recompute one NaN leaf using the candidate string key stored in the JSON."""
    if len(path) == 3:
        ilang, clang, candidate_key = path
        intervention_map = _intervention_map_for_experiment(experiment, ablations, amplifications)
        interventions = normalize_intervention_list(intervention_map[ilang])
        return get_logprob_with_intervention(prompt, candidate_key, model, interventions)
    if len(path) == 4:
        abl_lang, amp_lang, _clang, candidate_key = path
        if experiment == "feature-intervention":
            ablation_map = ablations["feature"]
        elif experiment == "one-layer direction intervention":
            ablation_map = ablations["one-layer-direction"]
        else:
            raise ValueError(f"Unexpected 4-level experiment: {experiment}")
        interventions = ablation_and_amplification(
            normalize_intervention_list(ablation_map[abl_lang]),
            normalize_intervention_list(amplifications["normal"][amp_lang]),
        )
        return get_logprob_with_intervention(prompt, candidate_key, model, interventions)
    raise ValueError(f"Unexpected NaN path depth for {experiment}: {path}")


def _intervention_map_for_experiment(
    experiment: str,
    ablations: dict,
    amplifications: dict,
    ablation_only: bool = False,
) -> dict:
    if ablation_only:
        return ablations["feature"]
    mapping = {
        "distractor ablation": ablations["feature"],
        "ablation": ablations["feature_everything"],
        "distractor one-layer direction ablation": ablations["one-layer-direction"],
        "one-layer direction ablation": ablations["one-layer-direction_everything"],
        "amplification": amplifications["everything"],
        "non-distractor amplification": amplifications["normal"],
    }
    if experiment not in mapping:
        raise KeyError(f"No intervention map for {experiment}")
    return mapping[experiment]


def prompts_with_nan_by_experiment(data: dict) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for experiment, prompts in data.items():
        if experiment == "original":
            continue
        for prompt, block in prompts.items():
            if count_nans(block) > 0:
                out[experiment].add(prompt)
    return out


def _build_fill_in(display_choices: list[str], list_lang: str) -> str:
    if list_lang in ("ja", "zh"):
        return ",".join(display_choices) + ","
    return ", ".join(display_choices) + ","


def _build_ans(options_key: str, n_display: int, langs: list[str]) -> dict[str, str]:
    ans = {lang: options[options_key][lang][n_display:] for lang in langs}
    for lang in langs:
        if lang in ("ja", "zh"):
            ans[lang] = ",".join(ans[lang])
        else:
            ans[lang] = " " + ", ".join(ans[lang])
    return ans


def resolve_ans_from_json(data: dict, prompt: str, langs: list[str]) -> dict[str, str] | None:
    """Use continuation strings stored under original logprobs (matches existing JSON keys)."""
    original = data.get("original", {}).get(prompt)
    if not original:
        return None
    logprobs = original.get("logprobs")
    if not logprobs:
        return None
    ans: dict[str, str] = {}
    for lang in langs:
        lang_probs = logprobs.get(lang)
        if not lang_probs:
            return None
        keys = list(lang_probs.keys())
        if len(keys) != 1:
            return None
        ans[lang] = keys[0]
    return ans


def resolve_ans_for_prompt(
    prompt: str, prompt_lang: str, list_lang: str, langs: list[str], data: dict | None = None
) -> dict[str, str]:
    """Match prompt/ans construction in multiple_words_intervention.py."""
    if data is not None:
        ans = resolve_ans_from_json(data, prompt, langs)
        if ans is not None:
            return ans

    base = base_prompts[prompt_lang]
    for category_key, category_by_lang in categories.items():
        category_text = category_by_lang[prompt_lang]
        n_display = number_of_choices_to_display[category_key]
        display_choices = options[category_key][list_lang][:n_display]
        candidate_prompt = base.format(
            category=category_text, choices=_build_fill_in(display_choices, list_lang)
        )
        if candidate_prompt == prompt:
            return _build_ans(category_key, n_display, langs)
    raise ValueError(
        f"Prompt not found in enumeration definitions for prompt_lang={prompt_lang!r}, "
        f"list_lang={list_lang!r}: {prompt[:80]!r}..."
    )


def load_nnsight_model(model_name: str, device_map: str):
    import nnsight

    kwargs: dict = {}
    hf_id = hf_model_names[model_name]
    if "qwen" in hf_id.lower():
        kwargs["trust_remote_code"] = True
    if device_map == "cpu":
        kwargs["low_cpu_mem_usage"] = False
        kwargs["torch_dtype"] = torch.float32
    else:
        kwargs["torch_dtype"] = torch.bfloat16
    return nnsight.LanguageModel(hf_id, device_map=device_map, **kwargs)


def resolve_nnsight_device_map(model_name: str, requested: str) -> str:
    if requested == "cuda":
        return str(device)
    # Default and "auto": keep nnsight on CPU (avoids two full models on GPU for qwen3-4b).
    return "cpu"


def setup_global_interventions(model_name: str, model: ReplacementModel, langs: list[str], data_directory: str):
    flores_directory = os.path.join(data_directory, "flores_features", model_name)
    lang_specific_directory = os.path.join(data_directory, "language_specific_features", model_name)
    multilingual_features_directory = os.path.join(data_directory, "multilingual_llm_features", model_name)
    amplification_values_directory = os.path.join(data_directory, "amplification_values", model_name)

    desc_features = description_based_features(flores_directory, langs, 0.1)
    val_features = mean_value_based_features(multilingual_features_directory, langs, 50)
    freq_features = freq_based_features(lang_specific_directory, langs)

    desc_interventions = activation_dict(desc_features, amplification_values_directory, langs)
    val_interventions = activation_dict(val_features, amplification_values_directory, langs)
    freq_interventions = activation_dict(freq_features, amplification_values_directory, langs)

    def _build_feature_and_amp(intervention_dict):
        abl_feature = {lang: ablation(intervention_dict, lang) for lang in langs}
        amp_normal = {lang: amplification(intervention_dict, lang) for lang in langs}
        abl_everything = {lang: combine_except_one(abl_feature, lang) for lang in langs}
        amp_everything = {lang: combine_except_one(amp_normal, lang) for lang in langs}
        return abl_feature, abl_everything, amp_normal, amp_everything

    def _build_ablations(intervention_dict):
        abl_feature, abl_everything, _, _ = _build_feature_and_amp(intervention_dict)
        return {
            "feature": abl_feature,
            "feature_everything": abl_everything,
            "one-layer-direction": {},
            "one-layer-direction_everything": {},
            "direction-ablation": {
                lang: interventions_to_dict(intervention_dict, lang, model) for lang in langs
            },
            "direction-ablation-everything": {
                lang: interventions_to_dict_everything_ablation(intervention_dict, lang, model)
                for lang in langs
            },
        }

    def _build_amplifications(intervention_dict):
        _, _, amp_normal, amp_everything = _build_feature_and_amp(intervention_dict)
        return {"normal": amp_normal, "everything": amp_everything}

    desc_ablations = _build_ablations(desc_interventions)
    desc_amplifications = _build_amplifications(desc_interventions)
    val_ablations = _build_ablations(val_interventions)
    val_amplifications = _build_amplifications(val_interventions)
    freq_ablations = _build_ablations(freq_interventions)
    freq_amplifications = _build_amplifications(freq_interventions)

    one_layer_ablation = {
        "desc": {
            lang: direction_ablation_layer_determine(desc_interventions, lang, num_layers=layer_num[model_name])
            for lang in langs
        },
        "val": {
            lang: direction_ablation_layer_determine(val_interventions, lang, num_layers=layer_num[model_name])
            for lang in langs
        },
        "freq": {
            lang: direction_ablation_layer_determine(freq_interventions, lang, num_layers=layer_num[model_name])
            for lang in langs
        },
    }

    return {
        "desc": (desc_ablations, desc_amplifications),
        "val": (val_ablations, val_amplifications),
        "freq": (freq_ablations, freq_amplifications),
        "one_layer_ablation": one_layer_ablation,
    }


def update_per_prompt_one_layer(
    model: ReplacementModel,
    prompt: str,
    method_prefix: str,
    ablations: dict,
    one_layer_ablation: dict,
    langs: list[str],
) -> None:
    for lang in langs:
        layer, features = one_layer_ablation[method_prefix][lang]
        ablations["one-layer-direction"][lang] = direction_ablation_helper(model, layer, features, prompt)
    for lang in langs:
        ablations["one-layer-direction_everything"][lang] = combine_except_one(
            ablations["one-layer-direction"], lang
        )


def recompute_experiment(
    experiment: str,
    prompt: str,
    ans: dict[str, str],
    model: ReplacementModel,
    ablations: dict,
    amplifications: dict,
    langs: list[str],
    nnsight_model,
) -> Any:
    feature_like = (
        "feature",
        "feature_everything",
        "one-layer-direction",
        "one-layer-direction_everything",
    )
    norm_ablations = {
        k: (normalize_intervention_by_lang(v) if k in feature_like else v)
        for k, v in ablations.items()
    }
    norm_amplifications = {
        k: normalize_intervention_by_lang(v) for k, v in amplifications.items()
    }

    if experiment == "distractor ablation":
        return feature_interventions_logprob(prompt, model, ans, norm_ablations["feature"], langs)
    if experiment == "ablation":
        return feature_interventions_logprob(prompt, model, ans, norm_ablations["feature_everything"], langs)
    if experiment == "distractor one-layer direction ablation":
        return feature_interventions_logprob(prompt, model, ans, norm_ablations["one-layer-direction"], langs)
    if experiment == "one-layer direction ablation":
        return feature_interventions_logprob(
            prompt, model, ans, norm_ablations["one-layer-direction_everything"], langs
        )
    if experiment == "distractor multi-layer direction ablation":
        return direction_ablate_logprob(
            prompt, model, ans, ablations["direction-ablation"], langs, nnsight_model
        )
    if experiment == "multi-layer direction ablation":
        return direction_ablate_logprob(
            prompt, model, ans, ablations["direction-ablation-everything"], langs, nnsight_model
        )
    if experiment == "amplification":
        return feature_interventions_logprob(prompt, model, ans, norm_amplifications["everything"], langs)
    if experiment == "non-distractor amplification":
        return feature_interventions_logprob(prompt, model, ans, norm_amplifications["normal"], langs)
    if experiment == "feature-intervention":
        return feature_ablation_and_amplification_logprob(
            prompt, model, ans, norm_ablations["feature"], norm_amplifications["normal"], langs
        )
    if experiment == "one-layer direction intervention":
        return feature_ablation_and_amplification_logprob(
            prompt,
            model,
            ans,
            norm_ablations["one-layer-direction"],
            norm_amplifications["normal"],
            langs,
        )
    raise KeyError(f"Unknown experiment: {experiment}")


def run_debug_probe(
    prompt: str,
    target: str,
    model: ReplacementModel,
    desc_interventions: dict,
    langs: list[str],
) -> None:
    """Quick check: Tensor intervention values -> NaN logits; float values -> OK."""
    from intervention import ablation

    print("  [debug] probe target:", repr(target[:60]))
    for ilang in ("en", "fr"):
        raw = ablation(desc_interventions, ilang)
        norm = normalize_intervention_list(raw)
        for label, interv in [(f"{ilang} raw Tensor", raw), (f"{ilang} float", norm)]:
            prompt_ids = model.tokenizer.encode(prompt, add_special_tokens=False)
            target_ids = model.tokenizer.encode(target, add_special_tokens=False)
            full_input = model.tokenizer.decode(prompt_ids + target_ids)
            transformed = transform_intervention(
                interv, len(prompt_ids), len(prompt_ids + target_ids) - 1
            )
            logits, _ = model.feature_intervention(full_input, transformed, return_activations=False)
            sl = logits[0, len(prompt_ids) - 1 : len(prompt_ids + target_ids) - 1, :]
            has_nan = bool(torch.isnan(sl).any().item())
            val_types = {type(t[3]).__name__ for t in interv[:3]} if interv else set()
            print(
                f"  [debug] {label}: len={len(interv)} transformed={len(transformed)} "
                f"val_types={val_types} logits_nan={has_nan}"
            )


def process_file(
    json_path: str,
    method: str,
    prompt_lang: str,
    list_lang: str,
    model: ReplacementModel,
    global_state: dict,
    langs: list[str],
    nnsight_model,
    model_name: str,
    dry_run: bool,
    regenerate_normalized: bool,
    debug: bool = False,
) -> tuple[int, int]:
    backup_path = json_path + ".bak"
    if not os.path.isfile(backup_path):
        shutil.copy2(json_path, backup_path)
        print(f"  backup: {backup_path}")

    data = load_json_or_restore(json_path)

    before = count_nans(data)
    if before == 0:
        print(f"  {json_path}: no NaNs, skipping")
        return 0, 0

    nan_by_exp = prompts_with_nan_by_experiment(data)
    print(f"  {json_path}: {before} NaNs across {sum(len(v) for v in nan_by_exp.values())} experiment-prompt pairs")

    if dry_run:
        for experiment, prompts in sorted(nan_by_exp.items()):
            print(f"    {experiment}: {len(prompts)} prompts")
        return before, 0

    method_prefix = {"description": "desc", "frequency": "freq", "value": "val"}[method]
    ablations, amplifications = global_state[method_prefix]

    if debug and nan_by_exp:
        first_exp = next(iter(nan_by_exp))
        first_prompt = sorted(nan_by_exp[first_exp])[0]
        ans = resolve_ans_for_prompt(first_prompt, prompt_lang, list_lang, langs, data=data)
        from ablation_amplification_intervention import (
            activation_dict,
            description_based_features,
            freq_based_features,
            mean_value_based_features,
        )

        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(repo, "data")
        flores = os.path.join(data_dir, "flores_features", model_name)
        amp_dir = os.path.join(data_dir, "amplification_values", model_name)
        lang_spec = os.path.join(data_dir, "language_specific_features", model_name)
        multi = os.path.join(data_dir, "multilingual_llm_features", model_name)
        if method_prefix == "desc":
            feats = description_based_features(flores, langs, 0.1)
        elif method_prefix == "freq":
            feats = freq_based_features(lang_spec, langs)
        else:
            feats = mean_value_based_features(multi, langs, 50)
        interv_dict = activation_dict(feats, amp_dir, langs)
        print(f"  [debug] first NaN block: {first_exp} / {first_prompt[:50]}...")
        run_debug_probe(first_prompt, ans["en"], model, interv_dict, langs)

    patched_total = 0
    for experiment, prompts in nan_by_exp.items():
        for prompt in sorted(prompts):
            ans = resolve_ans_for_prompt(prompt, prompt_lang, list_lang, langs, data=data)
            if experiment in NEEDS_ONE_LAYER_UPDATE:
                update_per_prompt_one_layer(
                    model, prompt, method_prefix, ablations, global_state["one_layer_ablation"], langs
                )
            fresh = recompute_experiment(
                experiment, prompt, ans, model, ablations, amplifications, langs, nnsight_model
            )
            block = data[experiment][prompt]
            patched_total += merge_recomputed_block(block, fresh)
            fresh_nans = count_nans(fresh)
            if fresh_nans:
                print(
                    f"    WARNING: {experiment} / {prompt[:50]}... "
                    f"still has {fresh_nans} NaNs in fresh recompute"
                )

    # Per-cell fallback using candidate keys from the JSON (handles key mismatches).
    for experiment, prompts in nan_by_exp.items():
        for prompt in sorted(prompts):
            block = data[experiment][prompt]
            for path in iter_nan_paths(block):
                try:
                    value = recompute_single_cell(
                        experiment, prompt, path, model, ablations, amplifications
                    )
                except Exception as exc:
                    print(f"    ERROR recomputing {experiment} {path}: {exc}")
                    continue
                if math.isfinite(value):
                    set_at_path(block, path, value)
                    patched_total += 1
                else:
                    print(f"    WARNING: non-finite value for {experiment} {path}: {value}")

    after = count_nans(data)
    if after > 0:
        sample_paths = []
        for experiment, prompts in nan_by_exp.items():
            for prompt in list(prompts)[:1]:
                for path in iter_nan_paths(data[experiment][prompt])[:3]:
                    sample_paths.append((experiment, prompt[:40], path))
        print(f"  ERROR: {after} NaNs remain in {json_path}; not writing. Samples: {sample_paths}")
        return before, after

    write_json_atomic(json_path, data)
    print(f"  wrote {json_path}: patched {patched_total} cells, {before} -> {after} NaNs")

    if regenerate_normalized:
        regenerate_normalized_file(json_path, hf_model_names[model_name])

    return before, after


def regenerate_normalized_file(json_path: str, hf_id: str) -> None:
    from transformers import AutoTokenizer

    normalized_path = json_path.replace(".json", "_normalized.json")
    if not os.path.isfile(normalized_path):
        return

    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code="qwen" in hf_id.lower())
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    for experiment, prompts in data.items():
        for _prompt, vals in prompts.items():
            if "intervention" not in experiment:
                for _key1, val1 in vals.items():
                    for _key2, val2 in val1.items():
                        for options, val in list(val2.items()):
                            token_num = tokenizer(options, return_tensors="pt").input_ids.shape[1] - 1
                            if token_num > 0 and not (isinstance(val, float) and math.isnan(val)):
                                val2[options] = val / token_num
            else:
                for _key1, val1 in vals.items():
                    for _key2, val2 in val1.items():
                        for _key3, val3 in val2.items():
                            for options, val in list(val3.items()):
                                token_num = tokenizer(options, return_tensors="pt").input_ids.shape[1] - 1
                                if token_num > 0 and not (isinstance(val, float) and math.isnan(val)):
                                    val3[options] = val / token_num

    with open(normalized_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, allow_nan=False)
    print(f"  regenerated {normalized_path}")


def main() -> int:
    args = parse_args()

    src_dir = os.path.dirname(os.path.abspath(__file__))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    repo_root = os.path.dirname(src_dir)
    data_directory = os.path.join(repo_root, "data")
    output_dir = os.path.join(data_directory, "interventions_multiple_words", args.model)
    langs = list(lang_to_flores_key.keys())

    methods = [args.method] if args.method else list(METHOD_TO_FILE_STEM.keys())

    json_paths: list[tuple[str, str, str, str]] = []
    for prompt_lang in langs:
        if args.prompt_lang is not None and prompt_lang != args.prompt_lang:
            continue
        for list_lang in langs:
            if args.list_lang is not None and list_lang != args.list_lang:
                continue
            pair_dir = os.path.join(output_dir, prompt_lang, list_lang)
            for method in methods:
                stem = METHOD_TO_FILE_STEM[method]
                json_path = os.path.join(pair_dir, f"interventions_and_results_{stem}.json")
                if os.path.isfile(json_path):
                    json_paths.append((json_path, method, prompt_lang, list_lang))

    if not json_paths:
        print("No matching JSON files found.")
        return 1

    validate_json_paths(json_paths)

    total_before = 0
    for json_path, method, prompt_lang, list_lang in json_paths:
        total_before += count_nans(load_json_or_restore(json_path))

    print(f"Found {len(json_paths)} files with up to {total_before} total NaNs (counted separately per file).")

    if args.dry_run:
        for json_path, method, prompt_lang, list_lang in json_paths:
            data = load_json_or_restore(json_path)
            n = count_nans(data)
            if n:
                print(f"{prompt_lang}/{list_lang} [{method}]: {n} NaNs")
        return 0

    print(f"Loading {args.model} ReplacementModel on {device}...")
    model = ReplacementModel.from_pretrained(
        hf_model_names[args.model],
        hf_transcoder_names[args.model],
        device=device,
        dtype=torch.bfloat16,
    )

    nnsight_device_map = resolve_nnsight_device_map(args.model, args.nnsight_device)
    nnsight_model = None
    needs_nnsight = False
    for json_path, _method, _pl, _ll in json_paths:
        nan_by_exp = prompts_with_nan_by_experiment(load_json_or_restore(json_path))
        if DIRECTION_EXPERIMENTS & nan_by_exp.keys():
            needs_nnsight = True
            break
    if needs_nnsight:
        print(f"Loading nnsight model (device_map={nnsight_device_map!r})...")
        nnsight_model = load_nnsight_model(args.model, nnsight_device_map)

    print("Building global intervention dictionaries...")
    global_state = setup_global_interventions(args.model, model, langs, data_directory)

    remaining = 0
    for json_path, method, prompt_lang, list_lang in json_paths:
        print(f"Processing {prompt_lang}/{list_lang} [{method}]...")
        _, after = process_file(
            json_path,
            method,
            prompt_lang,
            list_lang,
            model,
            global_state,
            langs,
            nnsight_model,
            args.model,
            dry_run=False,
            regenerate_normalized=args.regenerate_normalized,
            debug=args.debug,
        )
        remaining += after

    print(f"Done. Remaining NaNs across processed files: {remaining}")
    return 0 if remaining == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
