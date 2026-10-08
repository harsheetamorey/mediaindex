#!/usr/bin/env bash
# Build dist-app/MediaIndex.app (Apple silicon). Unsigned: it runs on this Mac; other Macs need right-click → Open.
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --extra app
(cd frontend && npm ci && npm run build)
uv run --extra app pyinstaller packaging/macos/MediaIndex.spec --noconfirm --clean \
  --distpath dist-app --workpath build-app
echo "built: dist-app/MediaIndex.app ($(du -sh dist-app/MediaIndex.app | cut -f1))"
