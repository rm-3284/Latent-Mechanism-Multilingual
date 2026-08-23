# Enumerations

Paper § Datasets. Start an ordered list in \(l_0\); continue remaining items in target \(l\). Metric: length-normalized logprob.

**Dataset file (source of truth):** [`enumerations.json`](enumerations.json) (symlink to [`../../datasets/enumerations.json`](../../datasets/enumerations.json)).

The runner still uses matching inline tables in `src/pipeline/multiple_words_intervention.py`; keep those in sync if you edit the JSON.

```bash
export PYTHONPATH=src
python -m pipeline.multiple_words_intervention --model gemma-2-2b
# optional: --lang en --list_lang ja
```

Outputs: `data/interventions_multiple_words/<model>/<prompt_lang>/<list_lang>/`.
