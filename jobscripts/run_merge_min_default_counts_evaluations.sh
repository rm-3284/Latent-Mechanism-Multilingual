#!/bin/bash
#SBATCH --job-name=merge_mindef
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --partition=all
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=rm4411@princeton.edu
#SBATCH --output=/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/jobscripts/logs/%x_%j.out
#SBATCH --error=/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/jobscripts/logs/%x_%j.err

# Merge min_default_counts evaluation JSONs into one summary CSV.
#
# Usage:
#   sbatch jobscripts/run_merge_min_default_counts_evaluations.sh
#   sbatch --dependency=afterok:<array_jobid> jobscripts/run_merge_min_default_counts_evaluations.sh
#
# Output:
#   data/additional_experiments/<model>/budget_matched_evaluations/min_default_counts/summary_merged.csv

set -eo pipefail

REPO_ROOT="/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual"
MODEL="${MODEL:-gemma-2-2b}"
INPUT_DIR="${INPUT_DIR:-${REPO_ROOT}/data/additional_experiments/${MODEL}/budget_matched_evaluations/min_default_counts}"
OUTPUT_CSV="${OUTPUT_CSV:-${INPUT_DIR}/summary_merged.csv}"

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
python -m sensitivity.merge_feature_set_eval_summaries \
	--model "${MODEL}" \
	--input-dir "${INPUT_DIR}" \
	--output-csv "${OUTPUT_CSV}"

echo ""
echo "Merged summary: ${OUTPUT_CSV}"
