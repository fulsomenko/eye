#!/usr/bin/env bash
set -euo pipefail

declare -A layer=(
  [eye-core]=0
  [eye-geometry]=1 [eye-platform]=1
  [eye-capture]=2 [eye-detect]=2 [eye-estimate]=2 [eye-calibration]=2 [eye-filter]=2 [eye-overlay]=2
  [eye]=3
  [eye-bench]=4
  [eye-app]=5 [eye-lab]=5
)

meta=$(cargo metadata --no-deps --format-version 1 --offline)
status=0

while read -r pkg; do
  if [[ ! -v "layer[$pkg]" ]]; then
    echo "error: workspace crate '$pkg' has no layer in scripts/check-crate-layers.sh"
    status=1
  fi
done < <(jq -r '.packages[].name' <<<"$meta")

while IFS=$'\t' read -r from to kind; do
  [[ -v "layer[$to]" && -v "layer[$from]" ]] || continue
  [[ "$kind" == "dev" ]] && continue
  if (( ${layer[$to]} >= ${layer[$from]} )); then
    echo "error: $from (L${layer[$from]}) depends on $to (L${layer[$to]}); dependencies must point to a lower layer"
    status=1
  fi
  if [[ "$from" == "eye-geometry" && "$to" != "eye-core" ]]; then
    echo "error: eye-geometry is pure math and may depend only on eye-core, found $to"
    status=1
  fi
done < <(jq -r '.packages[] | .name as $from | .dependencies[] | select(.path != null) | [$from, .name, (.kind // "normal")] | @tsv' <<<"$meta")

exit "$status"
