#!/bin/bash
#SBATCH --job-name=prep_flores_ho
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=all
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=rm4411@princeton.edu
#SBATCH --output=/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/jobscripts/logs/%x_%j.out
#SBATCH --error=/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/jobscripts/logs/%x_%j.err

# Prepare 5 held-out FLORES+ sentences (indices 150-154) with parallel translations
# and midpoint truncation. No GPU required.
#
# Usage:
#   sbatch jobscripts/run_prepare_flores_heldout.sh
#   NUM_SENTENCES=5 sbatch jobscripts/run_prepare_flores_heldout.sh

set -eo pipefail

REPO_ROOT="/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual"
MODEL="${MODEL:-gemma-2-2b}"
NUM_SENTENCES="${NUM_SENTENCES:-5}"
HELDOUT_START="${HELDOUT_START:-150}"

module purge
module load anaconda3/2024.02
CONDA_ENV="${CONDA_ENV:-latent-mechanism-multilingual}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

unset PYTHONPATH
export PYTHONNOUSERSITE=1
mkdir -p "${REPO_ROOT}/jobscripts/logs"

export PYTHONPATH="${REPO_ROOT}/src"
cd "${REPO_ROOT}"
python -m flores_steering.prepare_flores_heldout \
	--model "${MODEL}" \
	--num-sentences "${NUM_SENTENCES}" \
	--heldout-start "${HELDOUT_START}" \
	"$@"
