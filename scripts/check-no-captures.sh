#!/usr/bin/env bash
set -euo pipefail

found=$(find . \( -path ./target -o -path ./.git -o -path ./recordings -o -path ./.direnv -o -path './result*' \
                  -o -path ./assets -o -path './crates/*/tests/fixtures' \) -prune -o -type f \
             \( -iname '*.pgm' -o -iname '*.ppm' -o -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \
                -o -iname '*.raw' -o -iname '*.mkv' -o -iname '*.mp4' -o -iname '*.mjpg' -o -iname '*.yuv' \) -print)

if [[ -n "$found" ]]; then
  echo "error: image or video files outside assets/ and crates/*/tests/fixtures/ (G10, D11):"
  echo "$found"
  exit 1
fi
