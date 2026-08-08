#!/bin/bash
#SBATCH --job-name=recompute_nans
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --partition=all
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=rm4411@princeton.edu
#SBATCH --output=/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/jobscripts/logs/%x_%j.out
#SBATCH --error=/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/jobscripts/logs/%x_%j.err

# Recompute NaN logprobs in interventions_multiple_words JSON files.
#
# Single language pair (all methods: description, frequency, value):
#   sbatch jobscripts/run_recompute_multiple_words_nans.sh \
#     --model qwen3-4b --prompt-lang en --list-lang de --regenerate-normalized
#
# All pairs for one model in one job (long; loads model once per file batch):
#   sbatch jobscripts/run_recompute_multiple_words_nans.sh \
#     --model qwen3-4b --regenerate-normalized
#
# Parallel over all 49 prompt_lang/list_lang pairs:
#   sbatch jobscripts/run_recompute_multiple_words_nans_array.sh

set -eo pipefail

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

REPO_ROOT="/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual"
export PYTHONPATH="${REPO_ROOT}/src"
export PYTHONNOUSERSITE=1
export TMPDIR="${TMPDIR:-/tmp}"
mkdir -p "${TMPDIR}" 2>/dev/null || export TMPDIR="/tmp"

cd "${REPO_ROOT}"
python scripts/maintenance/recompute_multiple_words_nans.py "$@"
