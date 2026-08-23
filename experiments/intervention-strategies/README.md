# Intervention strategies

Paper: *Intervention strategy effectiveness* (Zero ablation, one-layer / multi-layer direction ablation, amplification, **Zero+Amp**, 1L+Amp).

These are **not** separate scripts. Both `pipeline.interventions_to_json` (Antonyms) and `pipeline.multiple_words_intervention` (Enumerations) sweep the strategies and write them into the JSON under `data/interventions*` keys such as `distractor ablation`, `amplification`, `feature-intervention` (Zero+Amp).

Main paper comparison uses **Zero+Amp**: ablate context language \(l_0\), amplify target \(l\).
