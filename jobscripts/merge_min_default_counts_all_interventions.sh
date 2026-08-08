#!/bin/bash
# Merge per-intervention evaluation summaries for min_default_counts runs.
# Includes existing distractor-ablation JSONs from the parent directory.
#
# Usage:
#   bash jobscripts/merge_min_default_counts_all_interventions.sh

set -eo pipefail

REPO_ROOT="/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual"
MODEL="${MODEL:-gemma-2-2b}"
BASE_DIR="${BASE_DIR:-${REPO_ROOT}/data/additional_experiments/${MODEL}/budget_matched_evaluations/min_default_counts}"

module purge
module load anaconda3/2024.02 2>/dev/null || true
CONDA_ENV="${CONDA_ENV:-latent-mechanism-multilingual}"
if [[ -f "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" ]]; then
	source "$(conda info --base)/etc/profile.d/conda.sh"
	conda activate "${CONDA_ENV}" 2>/dev/null || true
fi

for slug in distractor_ablation amplification feature-intervention; do
	if [[ "${slug}" == "distractor_ablation" ]]; then
		input_dir="${BASE_DIR}"
	else
		input_dir="${BASE_DIR}/${slug}"
	fi
	output_csv="${BASE_DIR}/${slug}/summary_merged.csv"
	mkdir -p "${BASE_DIR}/${slug}"

	if [[ "${slug}" != "distractor_ablation" && ! -d "${input_dir}" ]]; then
		echo "Skipping ${slug}: directory not found (${input_dir})"
		continue
	fi

	export PYTHONPATH="${REPO_ROOT}/src"
cd "${REPO_ROOT}"
	python -m sensitivity.merge_feature_set_eval_summaries \
		--model "${MODEL}" \
		--input-dir "${input_dir}" \
		--output-csv "${output_csv}"
	echo "Wrote ${output_csv}"
done
