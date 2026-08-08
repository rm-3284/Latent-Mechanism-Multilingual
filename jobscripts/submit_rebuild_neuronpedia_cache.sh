#!/bin/bash
# Write manifest, submit 32-way CPU array, merge, then downstream AnnSel rebuild.
set -eo pipefail

REPO_ROOT="/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual"
MODEL="${MODEL:-gemma-2-2b}"
NUM_SHARDS="${NUM_SHARDS:-32}"

cd "${REPO_ROOT}"
mkdir -p jobscripts/logs

echo "Stopping any in-process serial rebuild..."
pkill -f "rebuild_neuronpedia_description_cache.py --model ${MODEL}" 2>/dev/null || true
sleep 2

echo "Writing key manifest..."
python3 scripts/rebuild_neuronpedia_description_cache.py \
	--model "${MODEL}" \
	--write-manifest

ARRAY_ID=$(sbatch --parsable --export=ALL,MODEL="${MODEL}",NUM_SHARDS="${NUM_SHARDS}" \
	jobscripts/run_rebuild_neuronpedia_cache_array.sh)
echo "Submitted array ${ARRAY_ID} (${NUM_SHARDS} shards)"

MERGE_ID=$(sbatch --parsable --dependency="afterok:${ARRAY_ID}" \
	--export=ALL,MODEL="${MODEL}",NUM_SHARDS="${NUM_SHARDS}" \
	jobscripts/run_rebuild_neuronpedia_cache_merge.sh)
echo "Submitted merge ${MERGE_ID} (afterok:${ARRAY_ID})"

CHAIN_ID=$(sbatch --parsable --dependency="afterok:${MERGE_ID}" \
	jobscripts/run_rebuild_annsel_artifacts_after_cache.sh)
echo "Submitted downstream chain ${CHAIN_ID} (afterok:${MERGE_ID})"

echo ""
echo "Monitor:"
echo "  tail -f jobscripts/logs/np_cache_${ARRAY_ID}_*.out"
echo "  tail -f jobscripts/logs/np_merge_${MERGE_ID}.out"
echo "  tail -f jobscripts/logs/rebuild_annsel_artifacts.out"
