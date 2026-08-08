#!/bin/bash
#SBATCH --job-name=flores_cont
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --partition=all
#SBATCH --array=0-2
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=rm4411@princeton.edu
#SBATCH --output=/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/jobscripts/logs/%x_%A_%a.out
#SBATCH --error=/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/jobscripts/logs/%x_%A_%a.err

# Steer held-out FLORES+ truncated sentences with per-language latents from each
# method (AnnSel, ValSel, FreqSel) and continue the sentence.
#
# Prerequisites:
#   1. sbatch jobscripts/run_prepare_flores_heldout.sh
#   2. sbatch jobscripts/run_budget_matched_features.sh  (or existing feature JSONs)
#   3. sbatch jobscripts/run_amplification_values.sh
#
# Usage:
#   sbatch jobscripts/run_flores_continuation_steering_array.sh
#   sbatch --dependency=afterok:<prep_jobid> jobscripts/run_flores_continuation_steering_array.sh
#
# Outputs:
#   data/flores_continuation/<model>/<feature_stem>/{AnnSel,ValSel,FreqSel}__continuations.json

set -eo pipefail

REPO_ROOT="/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual"
MODEL="${MODEL:-gemma-2-2b}"
FEATURES_DIR="${FEATURES_DIR:-${REPO_ROOT}/data/additional_experiments/${MODEL}/budget_matched_features}"
HELDOUT_FILE="${HELDOUT_FILE:-${REPO_ROOT}/data/flores_heldout/${MODEL}/heldout_sentences.json}"

module purge
module load anaconda3/2024.02
CONDA_ENV="${CONDA_ENV:-latent-mechanism-multilingual}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

if ! python -c "import torch" >/dev/null 2>&1; then
	echo "ERROR: torch is not available in CONDA_ENV=${CONDA_ENV}" >&2
	exit 1
fi

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export TMPDIR="${TMPDIR:-/tmp}"
mkdir -p "${TMPDIR}" 2>/dev/null || export TMPDIR="/tmp"
mkdir -p "${REPO_ROOT}/jobscripts/logs"

if [[ ! -f "${HELDOUT_FILE}" ]]; then
	echo "Held-out file not found: ${HELDOUT_FILE}" >&2
	echo "Run: sbatch jobscripts/run_prepare_flores_heldout.sh" >&2
	exit 1
fi

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

export PYTHONPATH="${REPO_ROOT}/src"
cd "${REPO_ROOT}"
python -m flores_steering.flores_continuation_steering \
	--model "${MODEL}" \
	--features-file "${FEATURE_FILE}" \
	--heldout-file "${HELDOUT_FILE}" \
	"$@"
