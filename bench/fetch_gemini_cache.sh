#!/usr/bin/env bash
# Fetch precomputed Gemini gemini-embedding-001 embeddings into
# bench/.gemini_cache/ for parallel-encoder comparisons against the
# SPECTER2 baseline (see fetch_specter2_cache.sh).
#
# Both partitions use IDENTICAL text inputs as their SPECTER2 counterparts
# (the same `title [SEP] abstract` strings from the Semantic Scholar
# bulk-search API), so vectors are 1:1 comparable across encoders.
#
# Cache contents (~240 MB total):
#   gemini_nlp_broad.npy            — (10000, 3072) float32, broad NLP
#   gemini_nlp_broad_meta.json      — model id, taskType, dim, truncation count
#   gemini_nlp_narrow.npy           — (10000, 3072) float32, narrow subfield
#                                     (transformer attention mechanism)
#   gemini_nlp_narrow_meta.json     — same schema as broad meta
#
# Source texts are NOT duplicated here — pull them with
# fetch_specter2_cache.sh (writes to bench/.specter2_cache/) since the
# string content is byte-identical.
#
# Source: oaustegard/claude-container-layers releases.
set -euo pipefail

REPO=${GEMINI_CACHE_REPO:-oaustegard/claude-container-layers}
BROAD_TAG=${GEMINI_BROAD_TAG:-gemini-nlp-broad-10k}
NARROW_TAG=${GEMINI_NARROW_TAG:-gemini-nlp-narrow-10k}
CACHE_DIR="$(cd "$(dirname "$0")" && pwd)/.gemini_cache"

mkdir -p "$CACHE_DIR"

if ! command -v gh >/dev/null 2>&1; then
  echo "error: gh CLI not found. Install: https://cli.github.com/" >&2
  exit 1
fi

fetch_assets() {
  local tag="$1"; shift
  for asset in "$@"; do
    out="$CACHE_DIR/$asset"
    echo "fetching $asset → $out"
    gh release download "$tag" --repo "$REPO" --pattern "$asset" \
      --output "$out" --clobber
  done
}

fetch_assets "$BROAD_TAG"  gemini_nlp_broad.npy  gemini_nlp_broad_meta.json
fetch_assets "$NARROW_TAG" gemini_nlp_narrow.npy gemini_nlp_narrow_meta.json

echo "done. embeddings in $CACHE_DIR"
echo "(text inputs are in bench/.specter2_cache/ — same strings as SPECTER2)"
