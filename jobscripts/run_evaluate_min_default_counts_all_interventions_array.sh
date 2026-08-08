#!/bin/bash
#SBATCH --job-name=eval_mindef_all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --partition=all
#SBATCH --array=3-8
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=rm4411@princeton.edu
#SBATCH --output=/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/jobscripts/logs/%x_%A_%a.out
#SBATCH --error=/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/jobscripts/logs/%x_%A_%a.err

# Evaluate min_default_counts feature sets for all intervention types.
# Array layout: 3 methods x 3 intervention types = 9 tasks (array 0-8).
#
#   task % 3 -> method index (AnnSel, ValSel, FreqSel)
#   task / 3 -> intervention type (distractor ablation, amplification, feature-intervention)
#
# Default array is 3-8 (amplification + feature-intervention only) so existing
# distractor-ablation JSONs in min_default_counts/ are not touched.
# To include distractor ablation in subdirs: ARRAY=0-8 sbatch ...
#
# Existing outputs are skipped when SKIP_IF_EXISTS=1 (default).
#
# Usage (note: pass --array to sbatch, not ARRAY= env var):
#   sbatch --array=3-8 jobscripts/run_evaluate_min_default_counts_all_interventions_array.sh
#   sbatch --array=0-8 jobscripts/run_evaluate_min_default_counts_all_interventions_array.sh
#   sbatch --array=6-8 --export=ALL,SKIP_IF_EXISTS=0 jobscripts/run_evaluate_min_default_counts_all_interventions_array.sh
#
# Or use the dedicated feature-intervention script (recommended):
#   sbatch jobscripts/run_evaluate_min_default_counts_feature_intervention_array.sh
#
# Outputs:
#   data/additional_experiments/<model>/budget_matched_evaluations/min_default_counts/
#     distractor_ablation/
#     amplification/
#     feature-intervention/
#
# After all tasks finish:
#   bash jobscripts/merge_min_default_counts_all_interventions.sh

set -eo pipefail

REPO_ROOT="/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual"
MODEL="${MODEL:-gemma-2-2b}"
BENCHMARK="${BENCHMARK:-both}"
FEATURES_DIR="${FEATURES_DIR:-${REPO_ROOT}/data/additional_experiments/${MODEL}/budget_matched_features}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-${REPO_ROOT}/data/additional_experiments/${MODEL}/budget_matched_evaluations/min_default_counts}"
SKIP_IF_EXISTS="${SKIP_IF_EXISTS:-1}"

INTERVENTION_TYPES=(
	"distractor ablation"
	"amplification"
	"feature-intervention"
)
INTERVENTION_SLUGS=(
	"distractor_ablation"
	"amplification"
	"feature-intervention"
)

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

mapfile -t FEATURE_FILES < <(find "${FEATURES_DIR}" -name 'min_default_counts*.json' | sort)
NUM_METHODS="${#FEATURE_FILES[@]}"
NUM_INTERVENTIONS="${#INTERVENTION_TYPES[@]}"
NUM_TASKS=$((NUM_METHODS * NUM_INTERVENTIONS))

if [[ "${NUM_METHODS}" -eq 0 ]]; then
	echo "No min_default_counts feature JSON files under ${FEATURES_DIR}" >&2
	echo "Run: sbatch jobscripts/run_budget_matched_features.sh" >&2
	exit 1
fi
if [[ "${SLURM_ARRAY_TASK_ID}" -ge "${NUM_TASKS}" ]]; then
	echo "Array task ${SLURM_ARRAY_TASK_ID} >= ${NUM_TASKS} tasks; exiting."
	exit 0
fi

METHOD_IDX=$((SLURM_ARRAY_TASK_ID % NUM_METHODS))
INTERVENTION_IDX=$((SLURM_ARRAY_TASK_ID / NUM_METHODS))
FEATURE_FILE="${FEATURE_FILES[$METHOD_IDX]}"
INTERVENTION_TYPE="${INTERVENTION_TYPES[$INTERVENTION_IDX]}"
INTERVENTION_SLUG="${INTERVENTION_SLUGS[$INTERVENTION_IDX]}"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${INTERVENTION_SLUG}"
mkdir -p "${OUTPUT_DIR}"

METHOD_NAME="$(basename "$(dirname "${FEATURE_FILE}")")"
EXPECTED_GLOB="${METHOD_NAME}__min_default_counts*.json"

if [[ "${SKIP_IF_EXISTS}" == "1" ]]; then
	if compgen -G "${OUTPUT_DIR}/${EXPECTED_GLOB}" >/dev/null; then
		EXISTING=( "${OUTPUT_DIR}"/${EXPECTED_GLOB} )
		echo "Output already exists in ${OUTPUT_DIR}: ${EXISTING[0]}; skipping."
		exit 0
	fi
	if [[ "${INTERVENTION_SLUG}" == "distractor_ablation" ]]; then
		if compgen -G "${BASE_OUTPUT_DIR}/${EXPECTED_GLOB}" >/dev/null; then
			EXISTING=( "${BASE_OUTPUT_DIR}"/${EXPECTED_GLOB} )
			echo "Distractor-ablation result already exists at ${EXISTING[0]}; skipping."
			exit 0
		fi
	fi
fi

echo "Task ${SLURM_ARRAY_TASK_ID}/${NUM_TASKS}"
echo "  method file: ${FEATURE_FILE}"
echo "  intervention: ${INTERVENTION_TYPE}"
echo "  output dir: ${OUTPUT_DIR}"

export PYTHONPATH="${REPO_ROOT}/src"
cd "${REPO_ROOT}"
python -m sensitivity.evaluate_selected_features \
	--model "${MODEL}" \
	--features-file "${FEATURE_FILE}" \
	--benchmark "${BENCHMARK}" \
	--intervention-types "${INTERVENTION_TYPE}" \
	--output-dir "${OUTPUT_DIR}" \
	"$@"
