#!/bin/bash
#SBATCH --job-name=eval_trace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --partition=all
#SBATCH --array=0-202%50
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=rm4411@princeton.edu
#SBATCH --output=/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/jobscripts/logs/%x_%A_%a.out
#SBATCH --error=/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/jobscripts/logs/%x_%A_%a.err

# Evaluate one AnnSel tracing feature set on antonyms + enumerations (203 tasks).
# Task index matches run_annsel_tracing_sensitivity_by_value_array.sh.
#
# Usage:
#   sbatch --dependency=afterok:<tracing_value_job_id> jobscripts/run_evaluate_annsel_tracing_array.sh

set -eo pipefail

REPO_ROOT="/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual"
MODEL="${MODEL:-gemma-2-2b}"
BENCHMARK="${BENCHMARK:-both}"
FEATURES_DIR="${FEATURES_DIR:-${REPO_ROOT}/data/additional_experiments/${MODEL}/annsel_tracing_sensitivity/selected_features}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/data/additional_experiments/${MODEL}/annsel_tracing_sensitivity/feature_set_evaluations}"

LANGS=(en fr de es zh ja ko)
VALUES=(
	0.05 0.075 0.1 0.15 0.2 0.25
	0.6 0.7 0.75 0.8 0.85 0.9 0.95
	0.9 0.92 0.95 0.98 0.99
	0.3 0.4 0.5 0.6 0.7
	0.1 0.15 0.2 0.25 0.3 0.35
)

NUM_LANGS="${#LANGS[@]}"
NUM_SETTINGS="${#VALUES[@]}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "${TASK}" -ge $((NUM_LANGS * NUM_SETTINGS)) ]]; then
	echo "Array task ${TASK} >= $((NUM_LANGS * NUM_SETTINGS)); exiting."
	exit 0
fi

LANG_IDX=$((TASK / NUM_SETTINGS))
SETTING_IDX=$((TASK % NUM_SETTINGS))
LANG="${LANGS[$LANG_IDX]}"
VALUE="${VALUES[$SETTING_IDX]}"

declare -A AXIS_FOR_IDX=(
	[0]=throughput_threshold [1]=throughput_threshold [2]=throughput_threshold
	[3]=throughput_threshold [4]=throughput_threshold [5]=throughput_threshold
	[6]=node_threshold [7]=node_threshold [8]=node_threshold [9]=node_threshold
	[10]=node_threshold [11]=node_threshold [12]=node_threshold
	[13]=edge_threshold [14]=edge_threshold [15]=edge_threshold [16]=edge_threshold
	[17]=edge_threshold
	[18]=threshold_first [19]=threshold_first [20]=threshold_first [21]=threshold_first
	[22]=threshold_first
	[23]=threshold_last [24]=threshold_last [25]=threshold_last [26]=threshold_last
	[27]=threshold_last [28]=threshold_last
)
AXIS="${AXIS_FOR_IDX[$SETTING_IDX]}"

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

export PYTHONPATH="${REPO_ROOT}/src"
cd "${REPO_ROOT}"
FEATURE_FILE="$(python -c "
from annsel_tracing_sensitivity import selected_features_path
print(selected_features_path('${FEATURES_DIR}', '${AXIS}', ${VALUE}, '${LANG}'))
")"

if [[ ! -f "${FEATURE_FILE}" ]]; then
	echo "Feature file not found (tracing job may have failed): ${FEATURE_FILE}" >&2
	exit 1
fi

echo "Task ${TASK}: lang=${LANG} axis=${AXIS} value=${VALUE}"
echo "  features=${FEATURE_FILE}"
echo "  benchmark=${BENCHMARK}"

python -m sensitivity.evaluate_selected_features \
	--model "${MODEL}" \
	--features-file "${FEATURE_FILE}" \
	--benchmark "${BENCHMARK}" \
	--output-dir "${OUTPUT_DIR}"
