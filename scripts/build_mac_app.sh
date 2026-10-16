#!/usr/bin/env bash
# Build dist-app/MediaIndex.app (Apple silicon) and dist-app/MediaIndex-<version>.zip for a GitHub release.
# The app includes FFmpeg and the photo editing code; model weights are downloaded on first launch, after asking.
# Unsigned: on other Macs, open it with right-click → Open (or System Settings → Privacy & Security → Open Anyway).
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --extra app --extra create
(cd frontend && npm ci && npm run build)

# Static FFmpeg for arm64 (system frameworks only), pinned and checked: https://ffmpeg.martin-riedl.de
FF=build-app/ffmpeg
FF_URL=https://ffmpeg.martin-riedl.de/download/macos/arm64/1789931890_9.0.2
mkdir -p "$FF"
for pair in ffmpeg:c8ed4c4e6978a03c485edbfe4e0a5dc2380f8a30bba5150531b31b094492d924 \
            ffprobe:fcbe839537485eaee7a7a8bc5cbc0f90d53617e80943e8a5b2e31cb851197ea6; do
  name=${pair%%:*} sum=${pair#*:}
  if [ ! -x "$FF/$name" ]; then
    curl -fsSL -o "$FF/$name.zip" "$FF_URL/$name.zip"
    echo "$sum  $FF/$name.zip" | shasum -a 256 -c -
    unzip -o -q "$FF/$name.zip" -d "$FF" && rm "$FF/$name.zip"
  fi
done
cat > "$FF/LICENSE-ffmpeg.txt" <<'TXT'
ffmpeg and ffprobe 9.0.2, static arm64 builds by Martin Riedl (https://ffmpeg.martin-riedl.de), built with --enable-gpl
--enable-version3, so they are licensed under the GNU GPL version 3. They run as separate programs; MediaIndex itself
is Apache-2.0. FFmpeg source code: https://ffmpeg.org/releases/ffmpeg-9.0.2.tar.xz and
https://github.com/FFmpeg/FFmpeg, build scripts: https://git.martin-riedl.de/ffmpeg/build-script
TXT

uv run --extra app --extra create pyinstaller packaging/macos/MediaIndex.spec --noconfirm --clean \
  --distpath dist-app --workpath build-app/pyinstaller
APP=dist-app/MediaIndex.app
codesign --force --deep --sign - "$APP"  # ad-hoc signature, required for arm64
VERSION=$(/usr/libexec/PlistBuddy -c "Print CFBundleShortVersionString" "$APP/Contents/Info.plist")
rm -f "dist-app/MediaIndex-$VERSION-macos-arm64.zip"
ditto -c -k --sequesterRsrc --keepParent "$APP" "dist-app/MediaIndex-$VERSION-macos-arm64.zip"
echo "built: $APP ($(du -sh "$APP" | cut -f1)), dist-app/MediaIndex-$VERSION-macos-arm64.zip ($(du -sh "dist-app/MediaIndex-$VERSION-macos-arm64.zip" | cut -f1))"
