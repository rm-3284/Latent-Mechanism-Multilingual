# Experiments (paper → code)

This folder mirrors the **Experiments** and appendix sections of `paper.tex`.
Runnable modules live under `src/`; evaluation sets live under [`datasets/`](../datasets/).

From the repo root:

```bash
export PYTHONPATH=src
```

| Paper | This folder | Command |
|---|---|---|
| Latent selection (ValSel, FreqSel, AnnSel) + FLORES+ | [`latent-selection/`](latent-selection/) | `python -m pipeline.{flores_feature_extraction,multilingual_llm_features,language_specific_features,amplification_values}` |
| **Antonyms** dataset + interventions | [`antonyms/`](antonyms/) | `python -m pipeline.interventions_to_json` |
| **Enumerations** dataset + interventions | [`enumerations/`](enumerations/) | `python -m pipeline.multiple_words_intervention` |
| Intervention strategies (Zero, 1L, Amp, Zero+Amp, …) | [`intervention-strategies/`](intervention-strategies/) | included in the two runners above |
| Selection-method comparison / logit tables | [`analysis-tables/`](analysis-tables/) | `python -m analysis.adj_lang_method_logit_change` |
| Redundancy / knock-out | [`redundancy/`](redundancy/) | `python -m analysis.projected_amplification_isolation` |
| Hyperparameter sensitivity (appendix) | [`sensitivity/`](sensitivity/) | `python -m sensitivity.hyperparameter_sensitivity` |
| Continued generation on FLORES+ (appendix) | [`flores-continuation/`](flores-continuation/) | `python -m flores_steering.flores_continuation_steering` |

Models: `gemma-2-2b`, `qwen3-4b`. Languages: `en`, `fr`, `de`, `es`, `zh`, `ja`, `ko`.

Default selectors: ValSel \(K=50\); FreqSel \(T=0.8\), \(N=98\), \(M=10\).

Full command reference: [../README.md](../README.md).
