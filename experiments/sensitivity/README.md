# Hyperparameter sensitivity (appendix)

Paper appendix: fixed-budget comparison; ValSel / FreqSel / AnnSel sensitivity.

```bash
export PYTHONPATH=src
python -m sensitivity.hyperparameter_sensitivity --model gemma-2-2b
python -m sensitivity.annsel_tracing_sensitivity --model gemma-2-2b
python -m sensitivity.budget_matched_features --model gemma-2-2b
python -m sensitivity.evaluate_selected_features --model gemma-2-2b
```
