# Redundancy / knock-out

Paper findings on latent overlap; appendix *Additional Redundancy Analysis*.

```bash
export PYTHONPATH=src
python -m pipeline.intersection --model gemma-2-2b
python -m pipeline.compute_feature_similarity --model gemma-2-2b
python -m analysis.projected_amplification_isolation --model gemma-2-2b
python -m analysis.plot_projected_amplification_isolation --model gemma-2-2b
python -m sensitivity.extract_original_per_lang_metrics --model gemma-2-2b
python -m sensitivity.analyze_per_lang_metrics --model gemma-2-2b
```
