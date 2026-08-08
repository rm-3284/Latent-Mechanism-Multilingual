#!/usr/bin/env bash
# Rebuild downstream AnnSel artifacts after the Neuronpedia cache refresh finishes.
set -euo pipefail

REPO="/n/fs/vision-mix/rm4411/Latent-Mechanism-Multilingual"
LOG="$REPO/jobscripts/logs/rebuild_annsel_artifacts.out"
CACHE="$REPO/data/cache/neuronpedia_descriptions/gemma-2-2b_descriptions.json"
REBUILD_OUT="$REPO/data/cache/neuronpedia_descriptions/gemma-2-2b_descriptions.json.rebuild"
REBUILD_LOG="$REPO/jobscripts/logs/rebuild_neuronpedia_cache_gemma-2-2b.out"

exec > >(tee -a "$LOG") 2>&1

if pgrep -f "rebuild_neuronpedia_description_cache.py --model gemma-2-2b" >/dev/null; then
  echo "[$(date -Is)] waiting for in-process Neuronpedia cache rebuild to finish"
  while pgrep -f "rebuild_neuronpedia_description_cache.py --model gemma-2-2b" >/dev/null; do
    if [[ -f "$REBUILD_LOG" ]]; then
      tail -n 1 "$REBUILD_LOG" || true
    fi
    sleep 120
  done
else
  echo "[$(date -Is)] no local rebuild process; assuming cache already installed"
fi

if [[ ! -f "$CACHE" ]]; then
  echo "Cache missing at $CACHE; aborting"
  exit 1
fi

if [[ -f "$REBUILD_OUT" ]]; then
  echo "Found leftover rebuild file but no installer step ran; aborting"
  exit 1
fi

echo "[$(date -Is)] rebuilding flores annotation counts"
export PYTHONPATH="$REPO/src"
cd "$REPO"
python -m lib.feature_extraction --model gemma-2-2b

echo "[$(date -Is)] rebuilding AnnSel selected feature sets"
python -m sensitivity.hyperparameter_sensitivity --model gemma-2-2b --methods annsel

echo "[$(date -Is)] exporting default AnnSel description CSVs"
python3 scripts/export_default_annsel_selected_descriptions.py --refetch-all

echo "[$(date -Is)] invalidating tracing annotation caches"
find "$REPO/data/additional_experiments/gemma-2-2b/annsel_tracing_sensitivity/cache" \
  -name '*_features.json' -delete

echo "[$(date -Is)] done"
