#!/bin/bash
#SBATCH --job-name=flores_def
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

# Steer held-out FLORES+ continuations with full unconstrained default latent sets
# (is_default=true from hyperparameter_sensitivity.py), one method per array task.
#
# Default feature files:
#   AnnSel: selection_threshold_0.1  (threshold=0.1)
#   ValSel: topk_50                   (topk=50)
#   FreqSel: example_thres 0.98 / cross_lingual_thres 0.8 / token_thres 0.1
#
# Prerequisites:
#   1. sbatch jobscripts/run_prepare_flores_heldout.sh
#   2. sbatch jobscripts/run_hyperparameter_sensitivity.sh  (selected_features/)
#   3. sbatch jobscripts/run_amplification_values.sh
#
# Usage:
#   sbatch jobscripts/run_flores_continuation_steering_defaults_array.sh
#   TOP_P=0.5 sbatch jobscripts/run_flores_continuation_steering_defaults_array.sh
#
# Outputs:
#   data/flores_continuation/<model>/default_features/{AnnSel,FreqSel,ValSel}__continuations.json
#   TOP_P=0.5 -> .../default_features/top_p_0.5/{AnnSel,FreqSel,ValSel}__continuations.json

set -eo pipefail

REPO_ROOT="/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual"
MODEL="${MODEL:-gemma-2-2b}"
SELECTED_FEATURES_DIR="${SELECTED_FEATURES_DIR:-${REPO_ROOT}/data/additional_experiments/${MODEL}/selected_features}"
HELDOUT_FILE="${HELDOUT_FILE:-${REPO_ROOT}/data/flores_heldout/${MODEL}/heldout_sentences.json}"
TOP_P="${TOP_P:-0.9}"
if [[ -z "${OUTPUT_DIR:-}" ]]; then
	if [[ "${TOP_P}" == "0.9" ]]; then
		OUTPUT_DIR="${REPO_ROOT}/data/flores_continuation/${MODEL}/default_features"
	else
		OUTPUT_DIR="${REPO_ROOT}/data/flores_continuation/${MODEL}/default_features/top_p_${TOP_P}"
	fi
fi

FEATURE_FILES=(
	"${SELECTED_FEATURES_DIR}/AnnSel/selection_threshold_0.1.json"
	"${SELECTED_FEATURES_DIR}/FreqSel/example_thres__cross_lingual_thres_0.8__example_thres_0.98__token_thres_0.1.json"
	"${SELECTED_FEATURES_DIR}/ValSel/topk_50.json"
)

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
mkdir -p "${REPO_ROOT}/jobscripts/logs" "${OUTPUT_DIR}"

if [[ ! -f "${HELDOUT_FILE}" ]]; then
	echo "Held-out file not found: ${HELDOUT_FILE}" >&2
	echo "Run: sbatch jobscripts/run_prepare_flores_heldout.sh" >&2
	exit 1
fi

NUM_FILES="${#FEATURE_FILES[@]}"
for f in "${FEATURE_FILES[@]}"; do
	if [[ ! -f "${f}" ]]; then
		echo "Feature file not found: ${f}" >&2
		echo "Run: sbatch jobscripts/run_hyperparameter_sensitivity.sh" >&2
		exit 1
	fi
done

if [[ "${SLURM_ARRAY_TASK_ID}" -ge "${NUM_FILES}" ]]; then
	echo "Array task ${SLURM_ARRAY_TASK_ID} >= ${NUM_FILES} files; exiting."
	exit 0
fi

FEATURE_FILE="${FEATURE_FILES[$SLURM_ARRAY_TASK_ID]}"
echo "Task ${SLURM_ARRAY_TASK_ID}/${NUM_FILES}: ${FEATURE_FILE}"
echo "TOP_P=${TOP_P} OUTPUT_DIR=${OUTPUT_DIR}"

export PYTHONPATH="${REPO_ROOT}/src"
cd "${REPO_ROOT}"
python -m flores_steering.flores_continuation_steering \
	--model "${MODEL}" \
	--features-file "${FEATURE_FILE}" \
	--heldout-file "${HELDOUT_FILE}" \
	--output-dir "${OUTPUT_DIR}" \
	--top-p "${TOP_P}" \
	"$@"
