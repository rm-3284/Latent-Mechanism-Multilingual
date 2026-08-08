#!/bin/bash
# Submit baseline prewarm (7 jobs) then by-value sweep (203 jobs) with a dependency.
#
# Usage:
#   bash jobscripts/submit_annsel_tracing_fine_arrays.sh
#   ARRAY_CONCURRENCY=50 bash jobscripts/submit_annsel_tracing_fine_arrays.sh

set -eo pipefail

REPO_ROOT="/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual"
cd "${REPO_ROOT}"

BASELINE_ID="$(sbatch --parsable jobscripts/run_annsel_tracing_baseline_array.sh)"
echo "Submitted baseline array: ${BASELINE_ID}"

VALUE_ID="$(sbatch --parsable --dependency=afterok:"${BASELINE_ID}" jobscripts/run_annsel_tracing_sensitivity_by_value_array.sh)"
echo "Submitted value array (after baseline): ${VALUE_ID}"

EVAL_ID="$(sbatch --parsable --dependency=afterok:"${VALUE_ID}" jobscripts/run_evaluate_annsel_tracing_array.sh)"
echo "Submitted eval array (after tracing values): ${EVAL_ID}"
echo "Monitor: squeue -j ${BASELINE_ID},${VALUE_ID},${EVAL_ID}"
