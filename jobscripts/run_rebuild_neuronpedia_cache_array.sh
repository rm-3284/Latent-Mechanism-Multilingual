#!/bin/bash
#SBATCH --job-name=np_cache
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --partition=all
#SBATCH --array=0-31
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=rm4411@princeton.edu
#SBATCH --output=/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/jobscripts/logs/%x_%A_%a.out
#SBATCH --error=/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/jobscripts/logs/%x_%A_%a.err

# Parallel Neuronpedia description fetch for gemma-2-2b.
# Submit via jobscripts/submit_rebuild_neuronpedia_cache.sh

set -eo pipefail

REPO_ROOT="/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual"
MODEL="${MODEL:-gemma-2-2b}"
NUM_SHARDS="${NUM_SHARDS:-32}"
TASK="${SLURM_ARRAY_TASK_ID:?SLURM array task id required}"

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

cd "${REPO_ROOT}"
python3 scripts/rebuild_neuronpedia_description_cache.py \
	--model "${MODEL}" \
	--shard-id "${TASK}" \
	--num-shards "${NUM_SHARDS}" \
	--resume

echo "Shard ${TASK}/${NUM_SHARDS} complete"
