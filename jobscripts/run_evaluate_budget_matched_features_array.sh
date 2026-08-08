#!/bin/bash
#SBATCH --job-name=eval_budget
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --partition=all
#SBATCH --array=0-2
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=rm4411@princeton.edu
#SBATCH --output=/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/jobscripts/logs/%x_%A_%a.out
#SBATCH --error=/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/jobscripts/logs/%x_%A_%a.err

# Array job: evaluate budget-matched feature sets.
#
# For the min_default_counts experiment (recommended), use instead:
#   sbatch jobscripts/run_evaluate_min_default_counts_array.sh
#
# Regenerate feature sets first:
#   sbatch jobscripts/run_budget_matched_features.sh
#
# Usage:
#   sbatch jobscripts/run_evaluate_budget_matched_features_array.sh
#   BENCHMARK=antonyms sbatch jobscripts/run_evaluate_budget_matched_features_array.sh
#
# After all array tasks finish, merge summaries:
#   python -m sensitivity.merge_feature_set_eval_summaries \
#     --model gemma-2-2b \
#     --input-dir data/additional_experiments/gemma-2-2b/budget_matched_evaluations \
#     --output-csv data/additional_experiments/gemma-2-2b/budget_matched_evaluations/summary_merged.csv

set -eo pipefail

REPO_ROOT="/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual"
MODEL="${MODEL:-gemma-2-2b}"
BENCHMARK="${BENCHMARK:-both}"
FEATURES_DIR="${FEATURES_DIR:-${REPO_ROOT}/data/additional_experiments/${MODEL}/budget_matched_features}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/data/additional_experiments/${MODEL}/budget_matched_evaluations}"

module purge
module load anaconda3/2024.02
CONDA_ENV="${CONDA_ENV:-latent-mechanism-multilingual}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

if ! python -c "import torch" >/dev/null 2>&1; then
	echo "ERROR: torch is not available in CONDA_ENV=${CONDA_ENV}" >&2
	echo "Python executable: $(command -v python)" >&2
	exit 1
fi

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export TMPDIR="${TMPDIR:-/tmp}"
mkdir -p "${TMPDIR}" 2>/dev/null || export TMPDIR="/tmp"
mkdir -p "${REPO_ROOT}/jobscripts/logs"

mapfile -t FEATURE_FILES < <(
	find "${FEATURES_DIR}" \( -name 'min_default_counts*.json' -o -name 'per_lang_min_pool*.json' -o -name 'budget_k*.json' \) | sort
)
NUM_FILES="${#FEATURE_FILES[@]}"
if [[ "${NUM_FILES}" -eq 0 ]]; then
	echo "No budget-matched feature JSON files under ${FEATURES_DIR}" >&2
	echo "Run: sbatch jobscripts/run_budget_matched_features.sh" >&2
	exit 1
fi
if [[ "${SLURM_ARRAY_TASK_ID}" -ge "${NUM_FILES}" ]]; then
	echo "Array task ${SLURM_ARRAY_TASK_ID} >= ${NUM_FILES} files; exiting."
	exit 0
fi

FEATURE_FILE="${FEATURE_FILES[$SLURM_ARRAY_TASK_ID]}"
echo "Task ${SLURM_ARRAY_TASK_ID}/${NUM_FILES}: ${FEATURE_FILE}"

export PYTHONPATH="${REPO_ROOT}/src"
cd "${REPO_ROOT}"
python -m sensitivity.evaluate_selected_features \
	--model "${MODEL}" \
	--features-file "${FEATURE_FILE}" \
	--benchmark "${BENCHMARK}" \
	--output-dir "${OUTPUT_DIR}" \
	"$@"
