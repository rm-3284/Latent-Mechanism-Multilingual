# Antonyms

Paper § Datasets. Cue adjective in context language \(l_0\); score the antonym in target language \(l\).

**Dataset file (source of truth):** [`antonyms.py`](antonyms.py) (symlink to [`../../datasets/antonyms.py`](../../datasets/antonyms.py)).

Import in code: `from lib.pipeline_data.adjectives import big_data`.

```bash
export PYTHONPATH=src
python -m pipeline.interventions_to_json --model gemma-2-2b
# optional: --prompt_lang en --adj_lang fr

python -m analysis.adj_lang_method_logit_change --model gemma-2-2b
python -m analysis.all_langs_intervention_logit_change --model gemma-2-2b
```

Outputs: `data/interventions/<model>/<prompt_lang>/<adj_lang>/`.
