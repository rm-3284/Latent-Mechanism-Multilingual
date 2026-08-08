#!/bin/bash
#SBATCH --job-name=export_mb_allint
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --partition=all
#SBATCH --array=0-2
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=rm4411@princeton.edu
#SBATCH --output=/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/jobscripts/logs/%x_%A_%a.out
#SBATCH --error=/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/jobscripts/logs/%x_%A_%a.err

# Export all_interventions.csv + all_enumerations_normalized.csv for matched-budget feature sets.
# Array 0-2 -> AnnSel, ValSel, FreqSel. After all tasks finish:
#   bash jobscripts/merge_matched_budget_all_interventions.sh

set -euo pipefail

REPO="/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual"
export PYTHONPATH="${REPO}/src"
cd "${REPO}"

module load anaconda3/2024.02
CONDA_ENV="${CONDA_ENV:-latent-mechanism-multilingual}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

METHODS=(AnnSel ValSel FreqSel)
FEATURE_FILES=(
  "/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/data/additional_experiments/gemma-2-2b/budget_matched_features/AnnSel/min_default_counts__budget_mode_min_default_counts__total_features_220.json"
  "/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/data/additional_experiments/gemma-2-2b/budget_matched_features/ValSel/min_default_counts__budget_mode_min_default_counts__total_features_220.json"
  "/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/data/additional_experiments/gemma-2-2b/budget_matched_features/FreqSel/min_default_counts__budget_mode_min_default_counts__total_features_220.json"
)

TASK=${SLURM_ARRAY_TASK_ID:-0}
METHOD="${METHODS[$TASK]}"
FEATURE_FILE="${FEATURE_FILES[$TASK]}"
OUT_DIR="/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/data/additional_experiments/gemma-2-2b/budget_matched_interventions/min_default_counts"

python -m sensitivity.export_matched_budget_all_interventions \
  --model gemma-2-2b \
  --method "${METHOD}" \
  --features-file "${FEATURE_FILE}" \
  --benchmark both \
  --output-dir "${OUT_DIR}"

echo "Finished ${METHOD} export to ${OUT_DIR}"
