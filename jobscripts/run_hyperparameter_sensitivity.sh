#!/bin/bash
#SBATCH --job-name=hp_sensitivity
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

# Offline hyperparameter sensitivity for AnnSel, ValSel, and FreqSel (gemma-2-2b).
# Saves summary metrics plus per-setting selected feature lists for later
# antonym / enumeration intervention runs.
#
# Usage:
#   sbatch jobscripts/run_hyperparameter_sensitivity.sh
#   MODEL=gemma-2-2b sbatch jobscripts/run_hyperparameter_sensitivity.sh
#   sbatch jobscripts/run_hyperparameter_sensitivity.sh --methods annsel,valsel
#
# Outputs (under data/additional_experiments/<model>/):
#   hyperparameter_sensitivity_<model>.json
#   hyperparameter_sensitivity_<model>.csv
#   selected_features/
#     AnnSel/selection_threshold_0.1.json
#     ValSel/topk_50.json
#     FreqSel/example_thres_0.98.json
#     ...
#
# For AnnSel tracing hyperparameters (GPU), use:
#   sbatch jobscripts/run_annsel_tracing_sensitivity.sh

set -eo pipefail

REPO_ROOT="/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual"
MODEL="${MODEL:-gemma-2-2b}"

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
python -m sensitivity.hyperparameter_sensitivity --model "${MODEL}" "$@"

echo ""
echo "Done. Results:"
echo "  ${REPO_ROOT}/data/additional_experiments/${MODEL}/hyperparameter_sensitivity_${MODEL}.json"
echo "  ${REPO_ROOT}/data/additional_experiments/${MODEL}/hyperparameter_sensitivity_${MODEL}.csv"
echo "  ${REPO_ROOT}/data/additional_experiments/${MODEL}/selected_features/"
