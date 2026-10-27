#!/usr/bin/env bash
# Pack the downloaded data and the results into the Zenodo deposit.
# Usage: ./make_release.sh            (from the repository root, after reproduce.sh steps 1-4)
set -euo pipefail
OUT=release_data/hk-cocitation-data
HERE="$(cd "$(dirname "$0")" && pwd)"
rm -rf "$OUT"; mkdir -p "$OUT/citing" "$OUT/results_cocite"
# data/README.md describes the deposit; take it from the repository if the data folder has none
if [ -f data/README.md ]; then cp data/README.md "$OUT/"
elif [ -f "$HERE/data/README.md" ]; then cp "$HERE/data/README.md" "$OUT/"
else echo "warning: data/README.md not found, deposit will have no README" >&2; fi
cp data/targets.jsonl.gz "$OUT/"
cp data/citing/batch_*.jsonl.gz "$OUT/citing/"
for f in cocite.csv cocite_placebo_nb.csv cocite_placebo.csv; do
  [ -f "data/$f" ] && cp "data/$f" "$OUT/"
done
cp results_cocite/cocite_numbers.json "$OUT/results_cocite/"
[ -f results_cocite.log ] && cp results_cocite.log "$OUT/results_cocite/"
( cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 shasum -a 256 > SHA256SUMS )
( cd release_data && zip -qr hk-cocitation-data.zip hk-cocitation-data )
echo "deposit: release_data/hk-cocitation-data.zip"
du -sh release_data/hk-cocitation-data.zip
