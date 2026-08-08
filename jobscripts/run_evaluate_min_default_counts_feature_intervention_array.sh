#!/bin/bash
#SBATCH --job-name=eval_mindef_fi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --partition=all
#SBATCH --array=0-2
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=rm4411@princeton.edu
#SBATCH --output=/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/jobscripts/logs/%x_%A_%a.out
#SBATCH --error=/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/jobscripts/logs/%x_%A_%a.err

# Re-run feature-intervention only for min_default_counts (3 methods).
# Ablates prompt language, amplifies adjective/list language.
#
# Usage:
#   sbatch jobscripts/run_evaluate_min_default_counts_feature_intervention_array.sh
#
# Output:
#   data/additional_experiments/<model>/budget_matched_evaluations/min_default_counts/feature-intervention/

set -eo pipefail

REPO_ROOT="/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual"
MODEL="${MODEL:-gemma-2-2b}"
BENCHMARK="${BENCHMARK:-both}"
FEATURES_DIR="${FEATURES_DIR:-${REPO_ROOT}/data/additional_experiments/${MODEL}/budget_matched_features}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/data/additional_experiments/${MODEL}/budget_matched_evaluations/min_default_counts/feature-intervention}"

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
mkdir -p "${REPO_ROOT}/jobscripts/logs" "${OUTPUT_DIR}"

mapfile -t FEATURE_FILES < <(find "${FEATURES_DIR}" -name 'min_default_counts*.json' | sort)
NUM_FILES="${#FEATURE_FILES[@]}"
if [[ "${NUM_FILES}" -eq 0 ]]; then
	echo "No min_default_counts feature JSON files under ${FEATURES_DIR}" >&2
	exit 1
fi
if [[ "${SLURM_ARRAY_TASK_ID}" -ge "${NUM_FILES}" ]]; then
	echo "Array task ${SLURM_ARRAY_TASK_ID} >= ${NUM_FILES} files; exiting."
	exit 0
fi

FEATURE_FILE="${FEATURE_FILES[$SLURM_ARRAY_TASK_ID]}"
echo "Task ${SLURM_ARRAY_TASK_ID}/${NUM_FILES}: ${FEATURE_FILE}"
echo "  intervention: feature-intervention (ablate prompt lang, amplify adj/list lang)"
echo "  output dir: ${OUTPUT_DIR}"

export PYTHONPATH="${REPO_ROOT}/src"
cd "${REPO_ROOT}"
python -m sensitivity.evaluate_selected_features \
	--model "${MODEL}" \
	--features-file "${FEATURE_FILE}" \
	--benchmark "${BENCHMARK}" \
	--intervention-types feature-intervention \
	--output-dir "${OUTPUT_DIR}" \
	"$@"
