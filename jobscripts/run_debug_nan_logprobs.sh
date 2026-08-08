#!/bin/bash
#SBATCH --job-name=debug_nan_lp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --partition=all
#SBATCH --output=/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/jobscripts/logs/%x_%j.out
#SBATCH --error=/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/jobscripts/logs/%x_%j.err

set -eo pipefail

module purge
module load anaconda3/2024.02
CONDA_ENV="${CONDA_ENV:-latent-mechanism-multilingual}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

REPO_ROOT="/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual"
export PYTHONPATH="${REPO_ROOT}/src"
export PYTHONNOUSERSITE=1

cd "${REPO_ROOT}"
python scripts/maintenance/debug_nan_logprobs.py --model qwen3-4b --prompt-lang en --list-lang en \
  --report jobscripts/logs/debug_nan_logprobs_report.txt "$@"
