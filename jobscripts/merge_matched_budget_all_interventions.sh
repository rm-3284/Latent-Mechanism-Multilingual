#!/bin/bash
# Merge partial method CSVs into all_interventions.csv and all_enumerations_normalized.csv.

set -euo pipefail

REPO="/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual"
ROOT="${1:-${REPO}/data/additional_experiments/gemma-2-2b/budget_matched_interventions/min_default_counts}"

export PYTHONPATH="${REPO}/src"
cd "${REPO}"

module load anaconda3/2024.02 2>/dev/null || true
CONDA_ENV="${CONDA_ENV:-latent-mechanism-multilingual}"
if [[ -f "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}" 2>/dev/null || true
fi

count=0
while IFS= read -r pair_dir; do
  if compgen -G "${pair_dir}/all_interventions_*.csv" > /dev/null \
    || compgen -G "${pair_dir}/all_enumerations_normalized_*.csv" > /dev/null; then
    python -m sensitivity.export_matched_budget_all_interventions \
      --merge-methods \
      --input-dir "${pair_dir}"
    count=$((count + 1))
  fi
done < <(find "${ROOT}" -mindepth 2 -maxdepth 2 -type d | sort)

echo "Merged CSVs in ${count} prompt/pair directories under ${ROOT}"
