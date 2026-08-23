# Latent selection (ValSel, FreqSel, AnnSel)

Paper: *Manipulating Generation Language* / Experiments setup. Latents are extracted from **FLORES+**, not from Antonyms/Enumerations.

```bash
export PYTHONPATH=src
python -m pipeline.flores_feature_extraction --model gemma-2-2b
python -m pipeline.multilingual_llm_features --model gemma-2-2b    # ValSel + FreqSel
python -m pipeline.language_specific_features --model gemma-2-2b   # AnnSel
python -m pipeline.amplification_values --model gemma-2-2b
```

Outputs: `data/flores_features/`, `data/multilingual_llm_features/`, `data/language_specific_features/`, `data/amplification_values/`.
