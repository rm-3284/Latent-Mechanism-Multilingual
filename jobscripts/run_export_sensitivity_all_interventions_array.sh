#!/bin/bash
#SBATCH --job-name=export_sens_arr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --partition=all
#SBATCH --array=0-57
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=rm4411@princeton.edu
#SBATCH --output=/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/jobscripts/logs/%x_%A_%a.out
#SBATCH --error=/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/jobscripts/logs/%x_%A_%a.err

# Array export: one sensitivity feature set per task -> all_interventions.csv per lang pair.
#
# Usage:
#   sbatch jobscripts/run_export_sensitivity_all_interventions_array.sh
#   FEATURES_DIR=.../selected_features/FreqSel sbatch --array=0-19 ...
#
# Single file (override array):
#   sbatch --array=0 jobscripts/run_export_sensitivity_all_interventions_array.sh \
#     --features-file data/.../FreqSel/example_thres__....json

set -eo pipefail

REPO_ROOT="/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual"
MODEL="${MODEL:-gemma-2-2b}"
BENCHMARK="${BENCHMARK:-both}"
FEATURES_DIR="${FEATURES_DIR:-${REPO_ROOT}/data/additional_experiments/${MODEL}/selected_features}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/data/additional_experiments/${MODEL}/hyperparameter_sensitivity_interventions}"

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

SINGLE_FEATURE_FILE=""
REMAINING_ARGS=()
while [[ $# -gt 0 ]]; do
	case "$1" in
	--features-file)
		SINGLE_FEATURE_FILE="$2"
		shift 2
		;;
	--features-file=*)
		SINGLE_FEATURE_FILE="${1#*=}"
		shift
		;;
	*)
		REMAINING_ARGS+=("$1")
		shift
		;;
	esac
done

if [[ -n "${SINGLE_FEATURE_FILE}" ]]; then
	FEATURE_FILE="$(readlink -f "${SINGLE_FEATURE_FILE}")"
	if [[ ! -f "${FEATURE_FILE}" ]]; then
		echo "Feature file not found: ${SINGLE_FEATURE_FILE}" >&2
		exit 1
	fi
	if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "${SLURM_ARRAY_TASK_ID}" != "0" ]]; then
		echo "Single-file mode: skipping array task ${SLURM_ARRAY_TASK_ID}."
		exit 0
	fi
else
	mapfile -t FEATURE_FILES < <(find "${FEATURES_DIR}" -name '*.json' | sort)
	NUM_FILES="${#FEATURE_FILES[@]}"
	if [[ "${NUM_FILES}" -eq 0 ]]; then
		echo "No feature JSON files under ${FEATURES_DIR}" >&2
		exit 1
	fi
	if [[ "${SLURM_ARRAY_TASK_ID}" -ge "${NUM_FILES}" ]]; then
		echo "Array task ${SLURM_ARRAY_TASK_ID} >= ${NUM_FILES} files; exiting."
		exit 0
	fi
	FEATURE_FILE="${FEATURE_FILES[$SLURM_ARRAY_TASK_ID]}"
fi

echo "Exporting ${FEATURE_FILE}"

export PYTHONPATH="${REPO_ROOT}/src"
cd "${REPO_ROOT}"
python -m sensitivity.export_matched_budget_all_interventions \
	--model "${MODEL}" \
	--features-file "${FEATURE_FILE}" \
	--benchmark "${BENCHMARK}" \
	--output-dir "${OUTPUT_ROOT}" \
	--nest-under-feature-id \
	--write-merged-csv \
	"${REMAINING_ARGS[@]}"
