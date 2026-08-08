#!/bin/bash
#SBATCH --job-name=export_sens_allint
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --partition=all
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=rm4411@princeton.edu
#SBATCH --output=/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/jobscripts/logs/%x_%j.out
#SBATCH --error=/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/jobscripts/logs/%x_%j.err

# Export all_interventions.csv + all_enumerations_normalized.csv for one sensitivity feature set.
#
# Usage:
#   sbatch jobscripts/run_export_sensitivity_all_interventions.sh \
#     --features-file data/additional_experiments/gemma-2-2b/selected_features/FreqSel/example_thres__....json
#
# Outputs (per prompt_lang / adj_lang or list_lang pair):
#   data/additional_experiments/<model>/hyperparameter_sensitivity_interventions/
#     <feature_set_id>/<prompt_lang>/<pair_lang>/all_interventions.csv
#     <feature_set_id>/<prompt_lang>/<pair_lang>/all_enumerations_normalized.csv

set -eo pipefail

REPO_ROOT="/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual"
MODEL="${MODEL:-gemma-2-2b}"
BENCHMARK="${BENCHMARK:-both}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/data/additional_experiments/${MODEL}/hyperparameter_sensitivity_interventions}"

module purge
module load anaconda3/2024.02
CONDA_ENV="${CONDA_ENV:-latent-mechanism-multilingual}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

if ! python -c "import torch" >/dev/null 2>&1; then
	echo "ERROR: torch is not available in CONDA_ENV=${CONDA_ENV}" >&2
	exit 1
fi

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export TMPDIR="${TMPDIR:-/tmp}"
mkdir -p "${TMPDIR}" 2>/dev/null || export TMPDIR="/tmp"
mkdir -p "${REPO_ROOT}/jobscripts/logs"

export PYTHONPATH="${REPO_ROOT}/src"
cd "${REPO_ROOT}"
python -m sensitivity.export_matched_budget_all_interventions \
	--model "${MODEL}" \
	--benchmark "${BENCHMARK}" \
	--output-dir "${OUTPUT_ROOT}" \
	--nest-under-feature-id \
	--write-merged-csv \
	"$@"
