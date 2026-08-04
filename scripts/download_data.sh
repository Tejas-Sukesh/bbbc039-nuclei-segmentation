#!/usr/bin/env bash
# Fetch BBBC039 (Broad Bioimage Benchmark Collection) into data/raw/.
# ~80 MB total. Idempotent: skips archives that are already present.
set -euo pipefail

BASE="https://data.broadinstitute.org/bbbc/BBBC039"
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/raw"
mkdir -p "$DEST"
cd "$DEST"

for f in images masks metadata; do
  if [[ -f "$f.zip" ]]; then
    echo "== $f.zip already downloaded, skipping"
  else
    echo "== downloading $f.zip"
    curl -fSL -O "$BASE/$f.zip"
  fi
  echo "== unzipping $f.zip"
  unzip -q -o "$f.zip"
done

# The Broad archives are zipped on macOS and carry AppleDouble sidecar files
# (__MACOSX/._name). Left in place they double every glob's file count.
rm -rf __MACOSX

echo
echo "images: $(find images -name '*.tif' | wc -l | tr -d ' ') tif"
echo "masks:  $(find masks  -name '*.png' | wc -l | tr -d ' ') png"
echo "splits: $(for s in training validation test; do
  printf '%s=%s ' "$s" "$(grep -c . metadata/$s.txt)"; done)"
echo "Expected: 200 tif / 200 png / training=100 validation=50 test=50"
