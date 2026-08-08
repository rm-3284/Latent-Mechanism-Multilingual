#!/bin/bash
#SBATCH --job-name=annsel_val
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --partition=all
#SBATCH --array=0-202%50
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=rm4411@princeton.edu
#SBATCH --output=/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/jobscripts/logs/%x_%A_%a.out
#SBATCH --error=/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/jobscripts/logs/%x_%A_%a.err

# One (language, hyperparameter value) per array task: 7 langs x 29 values = 203 jobs.
# Each job sweeps a single tracing setting (baseline loaded from cache if prewarmed).
#
# Recommended workflow:
#   sbatch jobscripts/run_annsel_tracing_baseline_array.sh
#   sbatch --dependency=afterok:<baseline_job_id> jobscripts/run_annsel_tracing_sensitivity_by_value_array.sh
#
# Or submit both via:
#   bash jobscripts/submit_annsel_tracing_fine_arrays.sh
#
# Usage:
#   sbatch jobscripts/run_annsel_tracing_sensitivity_by_value_array.sh
#   sbatch --array=0-6 jobscripts/run_annsel_tracing_sensitivity_by_value_array.sh  # en only

set -eo pipefail

REPO_ROOT="/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual"
MODEL="${MODEL:-gemma-2-2b}"
NUM_SENTENCES="${NUM_SENTENCES:-100}"
PRUNE_WORKERS="${PRUNE_WORKERS:-12}"

LANGS=(en fr de es zh ja ko)
AXES=(
	throughput_threshold
	node_threshold
	edge_threshold
	threshold_first
	threshold_last
)
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

echo "Task ${TASK}: lang=${LANG} axis=${AXIS} value=${VALUE} num_sentences=${NUM_SENTENCES} prune_workers=${PRUNE_WORKERS}"

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
python -m sensitivity.annsel_tracing_sensitivity \
	--model "${MODEL}" \
	--lang "${LANG}" \
	--axes "${AXIS}" \
	--axis-values "${VALUE}" \
	--num-sentences "${NUM_SENTENCES}" \
	--prune-workers "${PRUNE_WORKERS}"
