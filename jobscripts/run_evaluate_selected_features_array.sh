#!/bin/bash
#SBATCH --job-name=eval_feat_arr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --partition=all
#SBATCH --array=0-57
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=rm4411@princeton.edu
#SBATCH --output=/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/jobscripts/logs/%x_%A_%a.out
#SBATCH --error=/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/jobscripts/logs/%x_%A_%a.err

# Array job: evaluate one feature-set JSON per task (all 58 sensitivity settings).
#
# Usage:
#   sbatch jobscripts/run_evaluate_selected_features_array.sh
#   BENCHMARK=antonyms sbatch jobscripts/run_evaluate_selected_features_array.sh
#
# Single feature set (use --array=0 so Slurm does not spawn 0-57):
#   sbatch --array=0 jobscripts/run_evaluate_selected_features_array.sh \
#     --features-file data/.../FreqSel/example_thres__....json \
#     --intervention-types amplification \
#     --output-dir /n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/data/.../feature_set_evaluations_amplification
#
# All feature sets in parallel (58 tasks):
#   python -m sensitivity.merge_feature_set_eval_summaries --model gemma-2-2b \
#     --input-dir data/additional_experiments/gemma-2-2b/feature_set_evaluations_feature_intervention

set -eo pipefail

REPO_ROOT="/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual"
MODEL="${MODEL:-gemma-2-2b}"
BENCHMARK="${BENCHMARK:-both}"
FEATURES_DIR="${FEATURES_DIR:-${REPO_ROOT}/data/additional_experiments/${MODEL}/selected_features}"

module purge
module load anaconda3/2024.02
CONDA_ENV="${CONDA_ENV:-latent-mechanism-multilingual}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export TMPDIR="${TMPDIR:-/tmp}"
mkdir -p "${TMPDIR}" 2>/dev/null || export TMPDIR="/tmp"
mkdir -p "${REPO_ROOT}/jobscripts/logs"

# Single-file mode: pass --features-file on the command line (no array needed).
#   sbatch --array=0 jobscripts/run_evaluate_selected_features_array.sh \
#     --features-file /path/to/one.json --intervention-types amplification
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
set -- "${REMAINING_ARGS[@]}"

if [[ -n "${SINGLE_FEATURE_FILE}" ]]; then
	if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "${SLURM_ARRAY_TASK_ID}" != "0" ]]; then
		echo "Single-file mode: skipping array task ${SLURM_ARRAY_TASK_ID} (only task 0 runs)."
		exit 0
	fi
	FEATURE_FILE="$(readlink -f "${SINGLE_FEATURE_FILE}")"
	if [[ ! -f "${FEATURE_FILE}" ]]; then
		echo "Feature file not found: ${SINGLE_FEATURE_FILE}" >&2
		exit 1
	fi
	echo "Single-file mode: ${FEATURE_FILE}"
	export PYTHONPATH="${REPO_ROOT}/src"
cd "${REPO_ROOT}"
	python -m sensitivity.evaluate_selected_features \
		--model "${MODEL}" \
		--features-file "${FEATURE_FILE}" \
		--benchmark "${BENCHMARK}" \
		"$@"
	exit 0
fi

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
echo "Task ${SLURM_ARRAY_TASK_ID}/${NUM_FILES}: ${FEATURE_FILE}"

export PYTHONPATH="${REPO_ROOT}/src"
cd "${REPO_ROOT}"
python -m sensitivity.evaluate_selected_features \
	--model "${MODEL}" \
	--features-file "${FEATURE_FILE}" \
	--benchmark "${BENCHMARK}" \
	"$@"
