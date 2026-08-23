# Selection-method comparison tables

Paper: *Selection method efficacy*; appendix detailed logit tables.

```bash
export PYTHONPATH=src
python -m analysis.adj_lang_method_logit_change --model gemma-2-2b
python -m analysis.all_langs_intervention_logit_change --model gemma-2-2b
python -m analysis.compare_all_langs_method_pairs --model gemma-2-2b
python -m analysis.compare_adj_logit_vs_interventions --model gemma-2-2b
```
