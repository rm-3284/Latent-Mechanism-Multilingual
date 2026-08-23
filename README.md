# Latent Mechanisms of Code-Switching in Multilingual LMs

Code for the paper *Latent Mechanisms of Code-Switching in Multilingual Language Models*.

We find language-controlling latents in cross-layer transcoders with three selectors, then steer generation on two code-switching benchmarks.

| Paper name | What it does |
|---|---|
| **ValSel** | Mean activation difference vs other languages |
| **FreqSel** | Language-specific firing frequency |
| **AnnSel** | Circuit-tracer paths + Neuronpedia descriptions that mention the language |

Models: `gemma-2-2b` (Gemma-2-2B-pt CLT) and `qwen3-4b`. Languages: `en`, `fr`, `de`, `es`, `zh`, `ja`, `ko`.

## Evaluation datasets (Antonyms and Enumerations)

These live at the **repo root** under `datasets/`, and again as symlinks under `experiments/antonyms/` and `experiments/enumerations/`. The `data/` folder is experiment *outputs*, not these eval sets.

| Paper benchmark | Location | Task |
|---|---|---|
| **Antonyms** | [`datasets/antonyms.py`](datasets/antonyms.py) · [`experiments/antonyms/`](experiments/antonyms/) | Cue adjective in \(l_0\); score the antonym in target \(l\) |
| **Enumerations** | [`datasets/enumerations.json`](datasets/enumerations.json) · [`experiments/enumerations/`](experiments/enumerations/) | Start a list; continue remaining items in target \(l\) |

Details: [`datasets/README.md`](datasets/README.md).

Python still imports Antonyms as `from lib.pipeline_data.adjectives import big_data` (shim to `datasets/antonyms.py`).

## Repository layout

```text
datasets/                 # paper eval sets (Antonyms, Enumerations)
experiments/              # paper-section folders + dataset symlinks
paper.tex
src/
  lib/                    # models, intervention primitives, paths
  pipeline/               # latent selection + Antonyms/Enumerations runs
  analysis/               # logit-change, method comparison, isolation
  sensitivity/            # hyperparams, matched budgets, knock-out metrics
  flores_steering/        # held-out FLORES+ continuation (appendix)
data/                     # extracted latents and intervention artifacts
scripts/maintenance/      # Neuronpedia cache / NaN repair
```

Paper section → code:

| Paper | Scripts |
|---|---|
| § Latent selection (ValSel / FreqSel / AnnSel) | `pipeline.flores_feature_extraction`, `pipeline.language_specific_features` (AnnSel), `pipeline.multilingual_llm_features` (ValSel & FreqSel), `pipeline.amplification_values` |
| Antonyms interventions | `pipeline.interventions_to_json`, `analysis.adj_lang_method_logit_change`, `analysis.all_langs_intervention_logit_change` |
| Enumerations interventions | `pipeline.multiple_words_intervention` |
| Redundancy / knock-out | `analysis.projected_amplification_isolation`, `pipeline.intersection`, `pipeline.compute_feature_similarity` |
| Hyperparameter sensitivity | `sensitivity.hyperparameter_sensitivity`, `sensitivity.annsel_tracing_sensitivity` |
| FLORES+ continuation | `flores_steering.prepare_flores_heldout`, `flores_steering.flores_continuation_steering` |

## 1. Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
export PYTHONPATH=src
```

`requirements.txt` installs circuit-tracer from Git. Artifacts go under `data/`.

## 2. How to run each experiment

Replace `gemma-2-2b` with `qwen3-4b` where needed. Default paper settings: ValSel \(K=50\); FreqSel \(T=0.8\), \(N=98\), \(M=10\).

### 2.1 Select latents from FLORES+

Used to build the three latent sets (not the Antonyms/Enumerations eval data).

```bash
export PYTHONPATH=src
python -m pipeline.flores_feature_extraction --model gemma-2-2b
python -m pipeline.language_specific_features --model gemma-2-2b   # AnnSel
python -m pipeline.multilingual_llm_features --model gemma-2-2b    # ValSel + FreqSel stats
python -m pipeline.amplification_values --model gemma-2-2b
```

Outputs:

- `data/flores_features/<model>/`
- `data/language_specific_features/<model>/` (AnnSel)
- `data/multilingual_llm_features/<model>/` (ValSel / FreqSel)
- `data/amplification_values/<model>/`

### 2.2 Antonyms benchmark

Cue/target pairs: [`datasets/antonyms.py`](datasets/antonyms.py).

```bash
python -m pipeline.interventions_to_json --model gemma-2-2b
# optional: --prompt_lang en --adj_lang fr
```

Writes `data/interventions/<model>/<prompt_lang>/<adj_lang>/interventions_and_results_{description,value,frequency}.json` (AnnSel / ValSel / FreqSel).

Logit-margin tables (paper \(\Delta m_l\)):

```bash
python -m analysis.adj_lang_method_logit_change --model gemma-2-2b
python -m analysis.all_langs_intervention_logit_change --model gemma-2-2b
```

### 2.3 Enumerations benchmark

Lists: [`datasets/enumerations.json`](datasets/enumerations.json). The runner also has the same categories inlined in `src/pipeline/multiple_words_intervention.py`.

```bash
python -m pipeline.multiple_words_intervention --model gemma-2-2b
# optional: --lang en --list_lang ja
```

Writes `data/interventions_multiple_words/<model>/<prompt_lang>/<list_lang>/`.

### 2.4 Intervention strategies (Zero, 1L, multi-layer, Amp, Zero+Amp)

The Antonyms and Enumerations runners already sweep:

- distractor / full ablation (zero)
- one-layer and multi-layer direction ablation
- amplification
- feature-intervention (Zero+Amp style)

Paper main comparison uses **Zero+Amp** (ablate context language \(l_0\), amplify target \(l\)).

### 2.5 Redundancy / knock-out

```bash
python -m pipeline.intersection --model gemma-2-2b
python -m pipeline.compute_feature_similarity --model gemma-2-2b
python -m analysis.projected_amplification_isolation --model gemma-2-2b
python -m analysis.plot_projected_amplification_isolation --model gemma-2-2b
```

Per-language metrics from saved interventions:

```bash
python -m sensitivity.extract_original_per_lang_metrics --model gemma-2-2b
python -m sensitivity.analyze_per_lang_metrics --model gemma-2-2b
```

### 2.6 Hyperparameter sensitivity

```bash
python -m sensitivity.hyperparameter_sensitivity --model gemma-2-2b
python -m sensitivity.annsel_tracing_sensitivity --model gemma-2-2b
python -m sensitivity.budget_matched_features --model gemma-2-2b
python -m sensitivity.evaluate_selected_features --model gemma-2-2b
```

### 2.7 FLORES+ continuation (appendix)

Held-out prefixes, then steer with selected latents:

```bash
python -m flores_steering.prepare_flores_heldout --model gemma-2-2b
python -m flores_steering.flores_continuation_steering --model gemma-2-2b
```

Outputs under `data/flores_heldout/` and `data/flores_continuation/`.

## 3. Core module arguments

### `pipeline.flores_feature_extraction`
- `--model`, `--lang` (optional)
- Out: `data/flores_features/<model>/<lang>.json`

### `pipeline.language_specific_features` (AnnSel)
- `--model`, `--lang` (optional)
- Out: `data/language_specific_features/<model>/`

### `pipeline.multilingual_llm_features` (ValSel / FreqSel)
- `--model`, `--lang` (optional)
- Out: `data/multilingual_llm_features/<model>/`

### `pipeline.amplification_values`
- `--model`, `--lang`, `--start-idx`, `--end-idx`
- Out: `data/amplification_values/<model>/`

### `pipeline.interventions_to_json` (Antonyms)
- `--model` / `-m`, `--prompt_lang` / `-pl`, `--adj_lang` / `-al`
- `--skip_direction_ablation`, `--nnsight_cpu`

### `pipeline.multiple_words_intervention` (Enumerations)
- `--model` / `-m`, `--lang` / `-l`, `--list_lang`

## 4. Shared code

- [`src/lib/models.py`](src/lib/models.py) — model ids, layers, Neuronpedia URLs
- [`src/lib/template.py`](src/lib/template.py) — languages and prompt frames
- [`src/lib/intervention.py`](src/lib/intervention.py) — ablation / amplification
- [`src/lib/paths.py`](src/lib/paths.py) — `datasets_dir()`, `antonyms_path()`, `enumerations_path()`
