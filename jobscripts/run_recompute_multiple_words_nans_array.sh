#!/bin/bash
#SBATCH --job-name=recompute_nans
#SBATCH --array=0-48
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
#SBATCH --output=/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/jobscripts/logs/%x_%A_%a.out
#SBATCH --error=/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual/jobscripts/logs/%x_%A_%a.err

# Recompute NaN logprobs for one (prompt_lang, list_lang) pair per array task.
# Covers all 7x7 language pairs for MODEL (default: qwen3-4b).
#
# Runtime (typical): ~2-4 h per task (7 prompts x 6 NaN experiments x 3 methods).
# Wall time 08:00:00 is a buffer; full original multi_words runs use 30:00:00.
#
# Submit the full array:
#   sbatch jobscripts/run_recompute_multiple_words_nans_array.sh
#
# Optional environment overrides:
#   MODEL=gemma-2-2b sbatch jobscripts/run_recompute_multiple_words_nans_array.sh
#   REGENERATE_NORM=0 sbatch ...   # skip *_normalized.json refresh
#   METHOD=description sbatch ...  # only one method file per pair
#
# Array task ID maps to prompt_lang/list_lang:
#   idx = prompt_lang_index * 7 + list_lang_index

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

LANGS=(en de es fr zh ja ko)
IDX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}"
if (( IDX < 0 || IDX >= 49 )); then
	echo "ERROR: SLURM_ARRAY_TASK_ID=${IDX} out of range 0-48" >&2
	exit 1
fi

PROMPT_LANG="${LANGS[$((IDX / 7))]}"
LIST_LANG="${LANGS[$((IDX % 7))]}"
MODEL="${MODEL:-qwen3-4b}"

ARGS=(--model "${MODEL}" --prompt-lang "${PROMPT_LANG}" --list-lang "${LIST_LANG}" --nnsight-device cpu)

if [[ -n "${METHOD:-}" ]]; then
	ARGS+=(--method "${METHOD}")
fi

if [[ "${REGENERATE_NORM:-1}" == "1" ]]; then
	ARGS+=(--regenerate-normalized)
fi

echo "Array task ${IDX}: model=${MODEL} prompt_lang=${PROMPT_LANG} list_lang=${LIST_LANG}"
echo "Command: python scripts/maintenance/recompute_multiple_words_nans.py ${ARGS[*]}"

cd "${REPO_ROOT}"
python scripts/maintenance/recompute_multiple_words_nans.py "${ARGS[@]}"
