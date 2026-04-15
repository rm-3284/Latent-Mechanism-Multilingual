# Latent Mechanism Multilingual

This repository identifies language-related latent features with three methods and evaluates interventions built from those features.

- Annotation-based: uses Neuronpedia feature descriptions.
- Value-based: uses mean activation differences across languages.
- Frequency-based: uses activation frequency patterns.

## 1. Environment Setup (venv)

From the repository root [README.md](README.md):

1. Create a virtual environment.
python3 -m venv .venv

2. Activate it.
source .venv/bin/activate

3. Upgrade pip and install dependencies.
pip install --upgrade pip
pip install -r requirements.txt

Notes:
- [requirements.txt](requirements.txt) includes circuit-tracer via Git.
- The scripts read and write experiment artifacts under [data](data).

## 2. Core Pipeline Overview

Run the core scripts in this order from the repository root.

1. python src/flores_feature_extraction.py --model gemma-2-2b
2. python src/language_specific_features.py --model gemma-2-2b
3. python src/multilingual_llm_features.py --model gemma-2-2b
4. python src/amplification_values.py --model gemma-2-2b
5. python src/interventions_to_json.py --model gemma-2-2b
6. python src/intersection.py --model gemma-2-2b
7. python src/compute_feature_similarity.py --model gemma-2-2b

Optional alternate experiment:
- python src/multiple_words_intervention.py --model gemma-2-2b

## 3. Script-by-Script Arguments and Outputs

### [src/flores_feature_extraction.py](src/flores_feature_extraction.py)
Purpose: Extract FLORES-derived feature candidates per language.

Arguments:
- --model: model key from [src/models.py](src/models.py)
- --lang: optional single language code

Outputs:
- data/flores_features/<model>/<lang>.json

### [src/language_specific_features.py](src/language_specific_features.py)
Purpose: Build sparse language-specific feature sets and fetch descriptions.

Arguments:
- --model: model key
- --lang: optional single language code

Outputs (under data/language_specific_features/<model>/):
- <lang>_sparse_long.json
- features_0.98.json
- <lang>_description.json
- summary.json

### [src/multilingual_llm_features.py](src/multilingual_llm_features.py)
Purpose: Compute multilingual activation/value statistics and metadata.

Arguments:
- --model: model key
- --lang: optional single language code

Outputs (under data/multilingual_llm_features/<model>/):
- <lang>_long.json
- v_values.json
- <lang>_vplot.png
- <lang>_description.json
- summary.json

### [src/amplification_values.py](src/amplification_values.py)
Purpose: Estimate amplification values used by intervention scripts.

Arguments:
- --model: model key
- --lang: optional single language code
- --start-idx: start index for every-position batch
- --end-idx: end index for every-position batch

Outputs (under data/amplification_values/<model>/):
- <lang>.json
- <lang>_summary.json
- <lang>_every_pos_<start>_<end>.json

### [src/interventions_to_json.py](src/interventions_to_json.py)
Purpose: Run adjective-based intervention suite and save structured outputs.

Arguments:
- --model or -m: model key
- --prompt_lang or -pl: optional prompt language filter
- --adj_lang or -al: optional adjective language filter
- --skip_direction_ablation: disable multi-layer nnsight ablation
- --nnsight_cpu: run nnsight direction ablation on CPU

Outputs (under data/interventions/<model>/<prompt_lang>/<adj_lang>/):
- interventions_and_results_description.json
- interventions_and_results_value.json
- interventions_and_results_frequency.json

### [src/multiple_words_intervention.py](src/multiple_words_intervention.py)
Purpose: Category-list intervention benchmark with multi-word targets.

Arguments:
- --model or -m: model key
- --lang or -l: prompt language filter
- --list_lang: comma-separated list languages

Outputs (under data/interventions_multiple_words/<model>/<prompt_lang>/<list_lang>/):
- interventions_and_results_description.json
- interventions_and_results_value.json
- interventions_and_results_frequency.json

### [src/intersection.py](src/intersection.py)
Purpose: Summarize language-name matches across feature methods.

Arguments:
- --model or -m: model key

Outputs (under data/additional_experiments/<model>/):
- lang_name_counts_<model>.json
- intersections_<model>.json (code path exists but block is currently commented)

### [src/compute_feature_similarity.py](src/compute_feature_similarity.py)
Purpose: Compute cosine similarity between method-level feature directions.

Arguments:
- --model or -m: model key
- --output-dir or -o: output directory (default resolves to data/additional_experiments)

Outputs:
- <output-dir>/<model>_cosine_similarities.json

## 4. Helper Files (Definitions and Shared Utilities)

- [src/models.py](src/models.py): central model name mappings, layer counts, and Neuronpedia URL templates.
- [src/template.py](src/template.py): shared language keys, prompt templates, and identifier dictionaries.
- [src/device_setup.py](src/device_setup.py): device selection and optional Hugging Face auth setup.
- [src/circuit_tracer_import.py](src/circuit_tracer_import.py): re-export layer for circuit-tracer objects and local named tuples.
- [src/intervention.py](src/intervention.py): common intervention primitives and analysis/plot helpers.
- [src/ablation_amplification_intervention.py](src/ablation_amplification_intervention.py): shared ablation-amplification experiment logic.
- [src/direction_ablation.py](src/direction_ablation.py): direction-ablation utilities and Gemma compatibility patching.
- [src/feature_extraction.py](src/feature_extraction.py): graph/path utilities and feature description selection helpers.
- [src/pipeline_data/adjectives/adjectives.py](src/pipeline_data/adjectives/adjectives.py): adjective datasets used in intervention experiments.
- [src/pipeline_data/generic_sentences/dataset_prepare.py](src/pipeline_data/generic_sentences/dataset_prepare.py): generic-sentence filtering utilities and dataset prep helpers.
