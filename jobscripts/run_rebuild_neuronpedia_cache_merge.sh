#!/bin/bash
#SBATCH --job-name=np_merge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --partition=all
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=rm4411@princeton.edu
#SBATCH --output=/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/jobscripts/logs/%x_%j.out
#SBATCH --error=/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/jobscripts/logs/%x_%j.err

set -eo pipefail

REPO_ROOT="/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual"
MODEL="${MODEL:-gemma-2-2b}"
NUM_SHARDS="${NUM_SHARDS:-32}"

module purge
module load anaconda3/2024.02
CONDA_ENV="${CONDA_ENV:-latent-mechanism-multilingual}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

unset PYTHONPATH
export PYTHONNOUSERSITE=1
mkdir -p "${REPO_ROOT}/jobscripts/logs"

cd "${REPO_ROOT}"
python3 scripts/rebuild_neuronpedia_description_cache.py \
	--model "${MODEL}" \
	--merge-only \
	--num-shards "${NUM_SHARDS}" \
	--install

echo "Merged and installed Neuronpedia cache for ${MODEL}"
