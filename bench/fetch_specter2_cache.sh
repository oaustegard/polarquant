#!/usr/bin/env bash
# Fetch precomputed SPECTER2 embeddings into bench/.specter2_cache/ so that
# `python bench/specter2_eval.py --cached` works without rerunning the
# ~50-minute CPU encode of the SPECTER2 transformer model.
#
# Cache contents (~45 MB total):
#   specter2_nlp_broad.npy         — (10000, 768) float32 SPECTER2 embeddings
#   specter2_nlp_broad_texts.json  — original "title [SEP] abstract" strings
#
# Source: oaustegard/claude-container-layers releases (free GH-hosted asset).
# Re-run safe; --clobber-equivalent overwrite via gh release download -O.
set -euo pipefail

REPO=${SPECTER2_CACHE_REPO:-oaustegard/claude-container-layers}
TAG=${SPECTER2_CACHE_TAG:-specter2-nlp-broad-10k}
CACHE_DIR="$(cd "$(dirname "$0")" && pwd)/.specter2_cache"

mkdir -p "$CACHE_DIR"

if ! command -v gh >/dev/null 2>&1; then
  echo "error: gh CLI not found. Install: https://cli.github.com/" >&2
  exit 1
fi

for asset in specter2_nlp_broad.npy specter2_nlp_broad_texts.json; do
  out="$CACHE_DIR/$asset"
  echo "fetching $asset → $out"
  gh release download "$TAG" --repo "$REPO" --pattern "$asset" \
    --output "$out" --clobber
done

echo "done. run: python bench/specter2_eval.py --cached"
