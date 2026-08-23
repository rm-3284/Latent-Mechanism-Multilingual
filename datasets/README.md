# Evaluation datasets (paper)

These are the two code-switching benchmarks from the paper.

| Paper name | File | Used by |
|---|---|---|
| **Antonyms** | [`antonyms.py`](antonyms.py) | adjective intervention scripts (`pipeline.interventions_to_json`, analysis logit-change, etc.) |
| **Enumerations** | [`enumerations.json`](enumerations.json) | category-list / multi-word intervention (`pipeline.multiple_words_intervention`) |

Languages: English (`en`), French (`fr`), German (`de`), Spanish (`es`), Chinese (`zh`), Japanese (`ja`), Korean (`ko`).

## Antonyms

- 100 adjective/antonym pairs per language (same items as Appendix *Adjectives Used in Antonyms*).
- Prompt shape: context in language \(l_0\), target antonym in language \(l\).
- Main Python object: `big_data` (list of `[cue_dict, antonym_dict]` pairs).
- Also exported: `small_data`, `train_data`, `test_data`.

Code still imports this as:

```python
from lib.pipeline_data.adjectives import big_data
```

## Enumerations

- Ordered lists (months, numbers, weekdays, seasons, …) with a split index.
- Prompt shape: category frame + first items, then continue the list in the target language.
- JSON fields: `items[]` with `category`, `split_at`, and `translations[lang]` (`frame`, `sequence`).

The intervention runner currently also has inline category tables in `src/pipeline/multiple_words_intervention.py`; keep those in sync with this JSON if you edit the lists.
