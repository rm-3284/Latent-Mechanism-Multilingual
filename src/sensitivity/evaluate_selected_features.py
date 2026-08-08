"""
Evaluate hyperparameter-sensitivity feature sets on antonym and enumeration benchmarks.

Loads selected_features/*.json produced by hyperparameter_sensitivity.py (or
annsel_tracing_sensitivity.py), runs interventions, and writes per-set results
plus an aggregate summary CSV.

Example:
  python evaluate_selected_features.py --model gemma-2-2b --benchmark both
  python evaluate_selected_features.py --features-file ../data/.../ValSel/topk_50.json --benchmark antonyms

Feature-intervention ablates prompt-language features and amplifies adjective-language
features (antonyms) or list-language features (enumerations). Same-language pairs are
skipped because ablation and amplification target the same features and cancel out.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from glob import glob
from typing import Any

import torch
from tqdm import tqdm

from lib.ablation_amplification_intervention import (
    ablation_and_amplification,
    activation_dict,
    model_intervention,
    model_run,
)
from lib.circuit_tracer_import import ReplacementModel
from lib.device_setup import device
from sensitivity.hyperparameter_sensitivity import repo_data_dir
from lib.intervention import ablation, amplification
from pipeline.interventions_to_json import feature_interventions, get_logits_and_ranks
from lib.models import hf_model_names, hf_transcoder_names
from pipeline.multiple_words_intervention import (
    base_prompts,
    categories,
    feature_interventions_logprob,
    get_logprob_with_intervention,
    get_logprobs_for_candidates,
    number_of_choices_to_display,
    options,
)
from lib.pipeline_data.adjectives import big_data
from lib.template import base_strings, lang_to_flores_key


def resolve_repo_relative_path(path: str) -> str:
    """Resolve a path; relative paths are anchored at the repo root."""
    if os.path.isabs(path):
        return path
    repo_root = os.path.dirname(repo_data_dir())
    return os.path.join(repo_root, path)


def discover_feature_files(
    features_dir: str,
    features_file: str | None,
    defaults_only: bool,
) -> list[str]:
    if features_file is not None:
        return [resolve_repo_relative_path(features_file)]
    pattern = os.path.join(features_dir, "**", "*.json")
    paths = sorted(glob(pattern, recursive=True))
    # Skip legacy duplicate filenames from an earlier naming scheme.
    paths = [p for p in paths if "threshold_threshold_" not in os.path.basename(p)]
    if not defaults_only:
        return paths
    selected = []
    for path in paths:
        with open(path, "r") as f:
            payload = json.load(f)
        if payload.get("is_default", False):
            selected.append(path)
    return selected


def load_feature_payload(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def feature_set_id(payload: dict[str, Any], path: str) -> str:
    method = payload.get("method", "unknown")
    sweep = payload.get("sweep", "unknown")
    hp = payload.get("hyperparameters", {})
    if len(hp) == 1:
        key, value = next(iter(hp.items()))
        tag = f"{value:g}" if isinstance(value, float) else str(value)
        return f"{method}__{sweep}__{key}_{tag}"
    stem = os.path.splitext(os.path.basename(path))[0]
    return f"{method}__{stem}"


def build_intervention_maps(
    features_by_lang: dict[str, list[str]],
    amplification_values_directory: str,
    langs: list[str],
) -> tuple[
    dict[str, list[tuple[int, int, int, float]]],
    dict[str, list[tuple[int, int, int, float]]],
]:
    feature_values = activation_dict(
        {lang: features_by_lang.get(lang, []) for lang in langs},
        amplification_values_directory,
        langs,
    )
    ablations = {lang: ablation(feature_values, lang) for lang in langs}
    amplifications = {lang: amplification(feature_values, lang) for lang in langs}
    return ablations, amplifications


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def resolve_output_dir(path: str | None, model_name: str) -> str:
    """Resolve output directory; relative paths are anchored at the repo root."""
    if path is None:
        return os.path.join(
            repo_data_dir(), "additional_experiments", model_name, "feature_set_evaluations"
        )
    return resolve_repo_relative_path(path)


def build_cross_lang_intervention(
    ablations: dict[str, list[tuple[int, int, int, float]]],
    amplifications: dict[str, list[tuple[int, int, int, float]]],
    ablation_lang: str,
    amplification_lang: str,
) -> list[tuple[int, int, int, float]]:
    """Ablate one language's features and amplify another's (no same-lang overlap)."""
    return ablation_and_amplification(
        ablations[ablation_lang], amplifications[amplification_lang]
    )


def run_feature_interventions_for_type(
    prompt: str,
    model: ReplacementModel,
    ans: dict[str, list[str]],
    langs: list[str],
    model_name: str,
    intervention_type: str,
    ablations: dict[str, list[tuple[int, int, int, float]]],
    amplifications: dict[str, list[tuple[int, int, int, float]]],
    ablation_lang: str | None = None,
    amplification_lang: str | None = None,
) -> dict[str, dict[str, Any]]:
    if intervention_type == "distractor ablation":
        return feature_interventions(prompt, model, ans, ablations, langs, model_name)
    if intervention_type == "amplification":
        return feature_interventions(prompt, model, ans, amplifications, langs, model_name)
    if intervention_type == "feature-intervention":
        if ablation_lang is None or amplification_lang is None:
            raise ValueError(
                "feature-intervention requires ablation_lang and amplification_lang"
            )
        interventions = build_cross_lang_intervention(
            ablations, amplifications, ablation_lang, amplification_lang
        )
        _, new_logits = model_intervention(prompt, model, interventions)
        result = get_logits_and_ranks(new_logits, ans, model.tokenizer, model_name)
        return {
            amplification_lang: {
                "output": None,
                "langs": result,
                "ablation_lang": ablation_lang,
                "amplification_lang": amplification_lang,
            }
        }
    raise ValueError(f"Unknown intervention type: {intervention_type}")


def run_feature_interventions_logprob_for_type(
    prompt: str,
    model: ReplacementModel,
    ans: dict[str, str],
    langs: list[str],
    intervention_type: str,
    ablations: dict[str, list[tuple[int, int, int, float]]],
    amplifications: dict[str, list[tuple[int, int, int, float]]],
    ablation_lang: str | None = None,
    amplification_lang: str | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    if intervention_type == "distractor ablation":
        return feature_interventions_logprob(prompt, model, ans, ablations, langs)
    if intervention_type == "amplification":
        return feature_interventions_logprob(prompt, model, ans, amplifications, langs)
    if intervention_type == "feature-intervention":
        if ablation_lang is None or amplification_lang is None:
            raise ValueError(
                "feature-intervention requires ablation_lang and amplification_lang"
            )
        interventions = build_cross_lang_intervention(
            ablations, amplifications, ablation_lang, amplification_lang
        )
        results: dict[str, dict[str, dict[str, float]]] = {
            amplification_lang: {lang: {} for lang in langs}
        }
        for lang in langs:
            candidate = ans[lang]
            results[amplification_lang][lang][candidate] = get_logprob_with_intervention(
                prompt, candidate, model, interventions
            )
        return results
    raise ValueError(f"Unknown intervention type: {intervention_type}")


def evaluate_antonyms(
    model: ReplacementModel,
    model_name: str,
    ablations: dict[str, list[tuple[int, int, int, float]]],
    amplifications: dict[str, list[tuple[int, int, int, float]]],
    langs: list[str],
    intervention_types: list[str],
    prompt_lang: str | None,
    adj_lang: str | None,
    max_items: int | None,
) -> dict[str, Any]:
    dataset = big_data if max_items is None else big_data[:max_items]
    rows: list[dict[str, Any]] = []

    prompt_langs = [prompt_lang] if prompt_lang else langs
    adj_langs = [adj_lang] if adj_lang else langs

    total_samples = len(prompt_langs) * len(adj_langs) * len(dataset)
    sample_bar = tqdm(total=total_samples, desc="Antonym prompts", leave=False)

    for pl in prompt_langs:
        base = base_strings[pl]
        for al in adj_langs:
            for sample_idx, (adj, ans) in enumerate(dataset):
                if al not in adj:
                    sample_bar.update(1)
                    continue
                prompt = base.format(adj=adj[al])
                _, baseline_logits = model_run(prompt, model)
                baseline = get_logits_and_ranks(baseline_logits, ans, model.tokenizer, model_name)

                for intervention_type in intervention_types:
                    if intervention_type == "feature-intervention" and pl == al:
                        continue
                    if intervention_type == "feature-intervention":
                        all_results = run_feature_interventions_for_type(
                            prompt,
                            model,
                            ans,
                            langs,
                            model_name,
                            intervention_type,
                            ablations,
                            amplifications,
                            ablation_lang=pl,
                            amplification_lang=al,
                        )
                        intervention_langs = [al]
                    else:
                        all_results = run_feature_interventions_for_type(
                            prompt,
                            model,
                            ans,
                            langs,
                            model_name,
                            intervention_type,
                            ablations,
                            amplifications,
                        )
                        intervention_langs = langs
                    for intervention_lang in intervention_langs:
                        result = all_results[intervention_lang]
                        for measure_lang in langs:
                            if measure_lang not in ans:
                                continue
                            for target_word in ans[measure_lang]:
                                base_logit = baseline[measure_lang][target_word][0]
                                int_logit = result["langs"][measure_lang][target_word][0]
                                base_rank = baseline[measure_lang][target_word][1]
                                int_rank = result["langs"][measure_lang][target_word][1]
                                on_language_pair = (
                                    measure_lang == al
                                    if intervention_type == "feature-intervention"
                                    else intervention_lang == measure_lang
                                )
                                rows.append(
                                    {
                                        "benchmark": "antonyms",
                                        "sample_idx": sample_idx,
                                        "intervention_type": intervention_type,
                                        "prompt_lang": pl,
                                        "adj_lang": al,
                                        "ablation_lang": (
                                            pl if intervention_type == "feature-intervention" else None
                                        ),
                                        "amplification_lang": (
                                            al if intervention_type == "feature-intervention" else None
                                        ),
                                        "intervention_lang": intervention_lang,
                                        "measure_lang": measure_lang,
                                        "target_word": target_word,
                                        "baseline_logit": base_logit,
                                        "intervened_logit": int_logit,
                                        "delta_logit": int_logit - base_logit,
                                        "baseline_rank": base_rank,
                                        "intervened_rank": int_rank,
                                        "delta_rank": int_rank - base_rank,
                                        "on_language_pair": on_language_pair,
                                    }
                                )
                sample_bar.update(1)

    sample_bar.close()

    on_lang = [r["delta_logit"] for r in rows if r["on_language_pair"]]
    off_lang = [r["delta_logit"] for r in rows if not r["on_language_pair"]]
    summary = {
        "n_rows": len(rows),
        "mean_delta_logit_on_language": mean(on_lang),
        "mean_delta_logit_off_language": mean(off_lang),
        "mean_abs_delta_logit_on_language": mean([abs(v) for v in on_lang]),
    }
    return {"summary": summary, "rows": rows}


def build_enumeration_prompt(
    prompt_lang: str,
    list_lang: str,
    category_key: str,
) -> tuple[str, dict[str, str]]:
    base = base_prompts[prompt_lang]
    category_text = categories[category_key][prompt_lang]
    n_display = number_of_choices_to_display[category_key]
    display_choices = options[category_key][list_lang][:n_display]
    predict_choices = options[category_key][list_lang][n_display:]
    if list_lang not in {"ja", "zh"}:
        fill_in = ", ".join(display_choices) + ","
    else:
        fill_in = ",".join(display_choices) + ","
    prompt = base.format(category=category_text, choices=fill_in)
    ans: dict[str, str] = {}
    for lang in lang_to_flores_key:
        rest = options[category_key][lang][n_display:]
        if lang not in {"ja", "zh"}:
            ans[lang] = " " + ", ".join(rest)
        else:
            ans[lang] = ",".join(rest)
    return prompt, ans


def evaluate_enumerations(
    model: ReplacementModel,
    ablations: dict[str, list[tuple[int, int, int, float]]],
    amplifications: dict[str, list[tuple[int, int, int, float]]],
    langs: list[str],
    intervention_types: list[str],
    prompt_lang: str | None,
    list_lang: str | None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    prompt_langs = [prompt_lang] if prompt_lang else langs
    list_langs = [list_lang] if list_lang else langs

    total_prompts = len(prompt_langs) * len(list_langs) * len(categories)
    prompt_bar = tqdm(total=total_prompts, desc="Enumeration prompts", leave=False)

    for pl in prompt_langs:
        for ll in list_langs:
            for category_key in categories:
                prompt, ans = build_enumeration_prompt(pl, ll, category_key)
                baseline = get_logprobs_for_candidates(prompt, ans, model)

                for intervention_type in intervention_types:
                    if intervention_type == "feature-intervention" and pl == ll:
                        continue
                    if intervention_type == "feature-intervention":
                        all_results = run_feature_interventions_logprob_for_type(
                            prompt,
                            model,
                            ans,
                            langs,
                            intervention_type,
                            ablations,
                            amplifications,
                            ablation_lang=pl,
                            amplification_lang=ll,
                        )
                        intervention_langs = [ll]
                    else:
                        all_results = run_feature_interventions_logprob_for_type(
                            prompt,
                            model,
                            ans,
                            langs,
                            intervention_type,
                            ablations,
                            amplifications,
                        )
                        intervention_langs = langs
                    for intervention_lang in intervention_langs:
                        intervened = all_results[intervention_lang]
                        for measure_lang in langs:
                            candidate = ans[measure_lang]
                            base_lp = baseline[measure_lang][candidate]
                            int_lp = intervened[measure_lang][candidate]
                            on_language_pair = (
                                measure_lang == ll
                                if intervention_type == "feature-intervention"
                                else intervention_lang == measure_lang
                            )
                            rows.append(
                                {
                                    "benchmark": "enumerations",
                                    "category": category_key,
                                    "intervention_type": intervention_type,
                                    "prompt_lang": pl,
                                    "list_lang": ll,
                                    "ablation_lang": (
                                        pl if intervention_type == "feature-intervention" else None
                                    ),
                                    "amplification_lang": (
                                        ll if intervention_type == "feature-intervention" else None
                                    ),
                                    "intervention_lang": intervention_lang,
                                    "measure_lang": measure_lang,
                                    "candidate": candidate,
                                    "baseline_logprob": base_lp,
                                    "intervened_logprob": int_lp,
                                    "delta_logprob": int_lp - base_lp,
                                    "on_language_pair": on_language_pair,
                                }
                            )
                prompt_bar.update(1)

    prompt_bar.close()

    on_lang = [r["delta_logprob"] for r in rows if r["on_language_pair"]]
    off_lang = [r["delta_logprob"] for r in rows if not r["on_language_pair"]]
    summary = {
        "n_rows": len(rows),
        "mean_delta_logprob_on_language": mean(on_lang),
        "mean_delta_logprob_off_language": mean(off_lang),
        "mean_abs_delta_logprob_on_language": mean([abs(v) for v in on_lang]),
    }
    return {"summary": summary, "rows": rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved feature sets on antonym and enumeration benchmarks."
    )
    parser.add_argument("--model", "-m", type=str, default="gemma-2-2b", choices=hf_model_names.keys())
    parser.add_argument(
        "--features-dir",
        type=str,
        default=None,
        help="Directory with selected_features JSON files",
    )
    parser.add_argument(
        "--features-file",
        type=str,
        default=None,
        help="Evaluate a single feature-set JSON (for array jobs)",
    )
    parser.add_argument(
        "--defaults-only",
        action="store_true",
        help="Only evaluate feature sets marked is_default=true",
    )
    parser.add_argument(
        "--benchmark",
        choices=["antonyms", "enumerations", "both"],
        default="both",
    )
    parser.add_argument(
        "--intervention-types",
        nargs="+",
        default=["distractor ablation"],
        choices=["distractor ablation", "amplification", "feature-intervention"],
    )
    parser.add_argument("--prompt-lang", type=str, default=None, choices=lang_to_flores_key.keys())
    parser.add_argument("--adj-lang", type=str, default=None, choices=lang_to_flores_key.keys())
    parser.add_argument("--list-lang", type=str, default=None, choices=lang_to_flores_key.keys())
    parser.add_argument("--max-items", type=int, default=None, help="Cap antonym adjective pairs")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Output root (default: data/additional_experiments/<model>/feature_set_evaluations)",
    )
    parser.add_argument(
        "--save-rows",
        action="store_true",
        help="Save full per-sample rows inside each result JSON (larger files)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_name = args.model
    langs = list(lang_to_flores_key.keys())

    features_dir = args.features_dir or os.path.join(
        repo_data_dir(), "additional_experiments", model_name, "selected_features"
    )
    output_dir = resolve_output_dir(args.output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)

    feature_files = discover_feature_files(features_dir, args.features_file, args.defaults_only)
    if not feature_files:
        raise FileNotFoundError(f"No feature-set JSON files found under {features_dir}")

    print(f"Loading model {model_name}")
    transcoder_name = hf_transcoder_names[model_name]
    model = ReplacementModel.from_pretrained(
        hf_model_names[model_name],
        transcoder_name,
        device=device,
        dtype=torch.bfloat16,
    )

    amplification_values_directory = os.path.join(
        repo_data_dir(), "amplification_values", model_name
    )

    summary_rows: list[dict[str, Any]] = []

    for feature_path in tqdm(feature_files, desc="Feature sets"):
        payload = load_feature_payload(feature_path)
        features_by_lang = payload["selected_features"]
        set_id = feature_set_id(payload, feature_path)
        out_path = os.path.join(output_dir, f"{set_id}.json")
        ablations, amplifications = build_intervention_maps(
            features_by_lang, amplification_values_directory, langs
        )

        result: dict[str, Any] = {
            "feature_set_id": set_id,
            "feature_set_file": feature_path,
            "method": payload.get("method"),
            "sweep": payload.get("sweep"),
            "hyperparameters": payload.get("hyperparameters"),
            "is_default": payload.get("is_default", False),
            "model": model_name,
            "intervention_types": args.intervention_types,
        }

        if args.benchmark in {"antonyms", "both"}:
            print(f"  running antonyms for {set_id}")
            antonym_eval = evaluate_antonyms(
                model=model,
                model_name=model_name,
                ablations=ablations,
                amplifications=amplifications,
                langs=langs,
                intervention_types=args.intervention_types,
                prompt_lang=args.prompt_lang,
                adj_lang=args.adj_lang,
                max_items=args.max_items,
            )
            result["antonyms"] = {"summary": antonym_eval["summary"]}
            if args.save_rows:
                result["antonyms"]["rows"] = antonym_eval["rows"]
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)

        if args.benchmark in {"enumerations", "both"}:
            print(f"  running enumerations for {set_id}")
            enum_eval = evaluate_enumerations(
                model=model,
                ablations=ablations,
                amplifications=amplifications,
                langs=langs,
                intervention_types=args.intervention_types,
                prompt_lang=args.prompt_lang,
                list_lang=args.list_lang,
            )
            result["enumerations"] = {"summary": enum_eval["summary"]}
            if args.save_rows:
                result["enumerations"]["rows"] = enum_eval["rows"]

        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        summary_row = {
            "feature_set_id": set_id,
            "feature_set_file": feature_path,
            "method": payload.get("method"),
            "sweep": payload.get("sweep"),
            "is_default": payload.get("is_default", False),
            **{f"hp_{k}": v for k, v in payload.get("hyperparameters", {}).items()},
            "result_file": out_path,
        }
        if "antonyms" in result:
            summary_row.update(
                {f"antonyms_{k}": v for k, v in result["antonyms"]["summary"].items()}
            )
        if "enumerations" in result:
            summary_row.update(
                {f"enumerations_{k}": v for k, v in result["enumerations"]["summary"].items()}
            )
        summary_rows.append(summary_row)

        torch.cuda.empty_cache()
        gc.collect()

    summary_csv = os.path.join(output_dir, "summary.csv")
    if summary_rows:
        fieldnames = sorted({key for row in summary_rows for key in row.keys()})
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"Evaluated {len(feature_files)} feature set(s)")
    print(f"Wrote per-set JSON files under {output_dir}")
    print(f"Wrote aggregate summary to {summary_csv}")


if __name__ == "__main__":
    main()
