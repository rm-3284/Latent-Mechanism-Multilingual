#!/bin/bash
#SBATCH --job-name=annsel_base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --partition=all
#SBATCH --array=0-6
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=rm4411@princeton.edu
#SBATCH --output=/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/jobscripts/logs/%x_%A_%a.out
#SBATCH --error=/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/jobscripts/logs/%x_%A_%a.err

# One language per task: prewarm baseline traced features + Neuronpedia annotation.
# Run this before the by-value sweep array so each value job stays short.
#
# Usage:
#   sbatch jobscripts/run_annsel_tracing_baseline_array.sh

set -eo pipefail

REPO_ROOT="/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual"
MODEL="${MODEL:-gemma-2-2b}"
NUM_SENTENCES="${NUM_SENTENCES:-100}"
PRUNE_WORKERS="${PRUNE_WORKERS:-12}"

LANGS=(en fr de es zh ja ko)
TASK="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "${TASK}" -ge "${#LANGS[@]}" ]]; then
	echo "Array task ${TASK} >= ${#LANGS[@]}; exiting."
	exit 0
fi
LANG="${LANGS[$TASK]}"

echo "Task ${TASK}: baseline-only lang=${LANG} num_sentences=${NUM_SENTENCES} prune_workers=${PRUNE_WORKERS}"

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
	--baseline-only \
	--num-sentences "${NUM_SENTENCES}" \
	--prune-workers "${PRUNE_WORKERS}"
