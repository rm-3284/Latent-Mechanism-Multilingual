#!/bin/bash
#SBATCH --job-name=budget_match
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=all
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=rm4411@princeton.edu
#SBATCH --output=/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/jobscripts/logs/%x_%j.out
#SBATCH --error=/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/jobscripts/logs/%x_%j.err

# Generate budget-matched feature sets for AnnSel, ValSel, and FreqSel.
#
# Default: K_L = min default selection count across methods per language
# (e.g. ko=7 from AnnSel, de=50 from ValSel).
#
# Usage:
#   sbatch jobscripts/run_budget_matched_features.sh
#   sbatch jobscripts/run_budget_matched_features.sh --report-only
#   BUDGET_MODE=per_lang_min sbatch jobscripts/run_budget_matched_features.sh
#   BUDGET_MODE=uniform UNIFORM_BUDGETS=7,10 sbatch jobscripts/run_budget_matched_features.sh
#
# Outputs (under data/additional_experiments/<model>/budget_matched_features/):
#   candidate_pool_report.json
#   manifest.json
#   summary.csv
#   AnnSel/min_default_counts_total_features_220.json
#   ValSel/min_default_counts_total_features_220.json
#   FreqSel/min_default_counts_total_features_220.json
#
# Then evaluate:
#   sbatch jobscripts/run_evaluate_min_default_counts_array.sh

set -eo pipefail

REPO_ROOT="/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual"
MODEL="${MODEL:-gemma-2-2b}"
BUDGET_MODE="${BUDGET_MODE:-min_default_counts}"
UNIFORM_BUDGETS="${UNIFORM_BUDGETS:-5,7,10,11}"

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
python -m sensitivity.budget_matched_features \
	--model "${MODEL}" \
	--budget-mode "${BUDGET_MODE}" \
	--uniform-budgets "${UNIFORM_BUDGETS}" \
	"$@"

echo ""
echo "Done. Feature sets:"
echo "  ${REPO_ROOT}/data/additional_experiments/${MODEL}/budget_matched_features/"
