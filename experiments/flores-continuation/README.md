# Continued generation (appendix)

Paper appendix *Continued Generation*: held-out FLORES+ prefixes, then steer with selected latents.

```bash
export PYTHONPATH=src
python -m flores_steering.prepare_flores_heldout --model gemma-2-2b
python -m flores_steering.flores_continuation_steering --model gemma-2-2b
```

Outputs: `data/flores_heldout/`, `data/flores_continuation/`.
