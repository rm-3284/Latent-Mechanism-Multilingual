#!/bin/bash
#SBATCH --job-name=annsel_axis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --partition=all
#SBATCH --array=0-34%20
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=rm4411@princeton.edu
#SBATCH --output=/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/jobscripts/logs/%x_%A_%a.out
#SBATCH --error=/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/jobscripts/logs/%x_%A_%a.err

# One (language, axis) pair per array task: 7 langs x 5 axes = 35 jobs.
#
# Uses shared attribution-graph cache (attribute once per sentence per language).
# Parallel jobs for the same language coordinate via file locks when building graphs.
# Neuronpedia annotation is globally throttled via a shared API lock + description cache.
#
# Array index layout:
#   task / 5 -> language index
#   task % 5 -> axis index
#
#   0=en+throughput   6=fr+throughput   12=de+throughput ...
#   1=en+node         7=fr+node         ...
#
# Usage:
#   sbatch jobscripts/run_annsel_tracing_sensitivity_by_axis_array.sh
#   PRUNE_WORKERS=12 sbatch jobscripts/run_annsel_tracing_sensitivity_by_axis_array.sh
#   sbatch --array=12 jobscripts/run_annsel_tracing_sensitivity_by_axis_array.sh  # de + throughput only
#
# After all tasks finish, per-language reports are at:
#   data/additional_experiments/gemma-2-2b/annsel_tracing_sensitivity/
#     annsel_tracing_sensitivity_gemma-2-2b_{lang}.json

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

NUM_LANGS="${#LANGS[@]}"
NUM_AXES="${#AXES[@]}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "${TASK}" -ge $((NUM_LANGS * NUM_AXES)) ]]; then
	echo "Array task ${TASK} >= $((NUM_LANGS * NUM_AXES)); exiting."
	exit 0
fi

LANG_IDX=$((TASK / NUM_AXES))
AXIS_IDX=$((TASK % NUM_AXES))
LANG="${LANGS[$LANG_IDX]}"
AXIS="${AXES[$AXIS_IDX]}"

echo "Task ${TASK}/${NUM_LANGS}x${NUM_AXES}: lang=${LANG} axis=${AXIS}"
echo "  num_sentences=${NUM_SENTENCES} prune_workers=${PRUNE_WORKERS}"

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

export PYTHONPATH="${REPO_ROOT}/src"
cd "${REPO_ROOT}"
python -m sensitivity.annsel_tracing_sensitivity \
	--model "${MODEL}" \
	--lang "${LANG}" \
	--axes "${AXIS}" \
	--num-sentences "${NUM_SENTENCES}" \
	--prune-workers "${PRUNE_WORKERS}"
