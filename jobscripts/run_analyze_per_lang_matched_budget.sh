#!/bin/bash
#SBATCH --job-name=per_lang_metrics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --partition=all
#SBATCH --array=0-8
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=rm4411@princeton.edu
#SBATCH --output=/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/jobscripts/logs/%x_%A_%a.out
#SBATCH --error=/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/jobscripts/logs/%x_%A_%a.err

# Per-language antonym margin + enumeration normalized logprob for matched-budget feature sets.
# Array: 3 methods x 3 intervention types = 9 tasks (0-8).

set -euo pipefail

REPO="/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual"
export PYTHONPATH="${REPO}/src"
cd "${REPO}"

module load anaconda3/2024.02
CONDA_ENV="${CONDA_ENV:-latent-mechanism-multilingual}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

METHODS=(AnnSel ValSel FreqSel)
INTERVENTIONS=("distractor ablation" amplification feature-intervention)
FEATURE_FILES=(
  "/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/data/additional_experiments/gemma-2-2b/budget_matched_features/AnnSel/min_default_counts__budget_mode_min_default_counts__total_features_220.json"
  "/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/data/additional_experiments/gemma-2-2b/budget_matched_features/ValSel/min_default_counts__budget_mode_min_default_counts__total_features_220.json"
  "/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/data/additional_experiments/gemma-2-2b/budget_matched_features/FreqSel/min_default_counts__budget_mode_min_default_counts__total_features_220.json"
)

TASK=${SLURM_ARRAY_TASK_ID:-0}
METHOD_IDX=$((TASK % 3))
INT_IDX=$((TASK / 3))
METHOD="${METHODS[$METHOD_IDX]}"
INTERVENTION="${INTERVENTIONS[$INT_IDX]}"
FEATURE_FILE="${FEATURE_FILES[$METHOD_IDX]}"

OUT_DIR="../data/additional_experiments/gemma-2-2b/per_lang_metrics/matched_budget"
mkdir -p "${OUT_DIR}"
OUT_CSV="${OUT_DIR}/${METHOD}__${INTERVENTION// /_}.csv"

if [[ "${SKIP_IF_EXISTS:-1}" == "1" && -f "${OUT_CSV}" ]]; then
  echo "Skipping existing ${OUT_CSV}"
  exit 0
fi

python -m sensitivity.analyze_per_lang_metrics \
  --model gemma-2-2b \
  --features-file "${FEATURE_FILE}" \
  --budget-label matched_budget \
  --intervention-types "${INTERVENTION}" \
  --output-csv "${OUT_CSV}"

echo "Wrote ${OUT_CSV}"
