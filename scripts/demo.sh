#!/usr/bin/env bash
# Demo helper. Keeps demo state in data/demo-app (separate from your own library database).
#   scripts/demo.sh start   start MediaIndex on the demo state dir (http://127.0.0.1:8765)
#   scripts/demo.sh setup   (in a second terminal) add + index the three CC0 demo folders
#   scripts/demo.sh reset   delete the demo state dir only (database, thumbnails). Media files are never touched.
set -euo pipefail
cd "$(dirname "$0")/.."
STATE="$PWD/data/demo-app"
case "${1:-}" in
  start) MEDIAINDEX_DATA_DIR="$STATE" exec uv run python -m mediaindex ;;
  setup) exec uv run python scripts/demo_setup.py ;;
  reset)
    if pgrep -f "[Pp]ython -m mediaindex" >/dev/null; then echo "stop the running MediaIndex server first"; exit 1; fi
    rm -rf "$STATE" && echo "removed $STATE" ;;
  *) sed -n 2,5p "$0"; exit 1 ;;
esac
