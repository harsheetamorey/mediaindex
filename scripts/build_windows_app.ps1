# Build dist-app\MediaIndex\MediaIndex.exe and dist-app\MediaIndex-<version>-windows-x64.zip (Windows 10/11, x64).
# The app includes FFmpeg and the photo editing code; model weights are downloaded on first launch, after asking.
# Unsigned: Windows SmartScreen may say "Windows protected your PC" the first time: More info -> Run anyway.
# Run from PowerShell in the repository: powershell -ExecutionPolicy Bypass -File scripts\build_windows_app.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
uv sync --extra app --extra create
if ($LASTEXITCODE) { throw "uv sync failed" }
Push-Location frontend; npm ci; npm run build; if ($LASTEXITCODE) { throw "UI build failed" }; Pop-Location

# FFmpeg 9.0.2 essentials build by Gyan Doshi (https://www.gyan.dev/ffmpeg/builds/), pinned and checked
$ff = "build-app\ffmpeg-win"
$url = "https://github.com/GyanD/codexffmpeg/releases/download/9.0.2/ffmpeg-9.0.2-essentials_build.zip"
$sha = "60f467265b1e312373dbcd92200c2618a74850f98d3d078e94296bb3fa2047ba"
New-Item -ItemType Directory -Force $ff | Out-Null
if (-not (Test-Path "$ff\ffmpeg.exe")) {
  $zip = "$ff\ffmpeg.zip"
  Invoke-WebRequest -Uri $url -OutFile $zip
  if ((Get-FileHash $zip -Algorithm SHA256).Hash.ToLower() -ne $sha) { throw "FFmpeg download checksum mismatch" }
  Expand-Archive $zip -DestinationPath "$ff\x" -Force
  Copy-Item "$ff\x\ffmpeg-9.0.2-essentials_build\bin\ffmpeg.exe", "$ff\x\ffmpeg-9.0.2-essentials_build\bin\ffprobe.exe" $ff
  Copy-Item "$ff\x\ffmpeg-9.0.2-essentials_build\LICENSE" "$ff\LICENSE-GPLv3.txt"
  Remove-Item -Recurse -Force "$ff\x", $zip
}
@"
ffmpeg.exe and ffprobe.exe 9.0.2, essentials build by Gyan Doshi (https://www.gyan.dev/ffmpeg/builds/), built with
--enable-gpl --enable-version3, so they are licensed under the GNU GPL version 3 (LICENSE-GPLv3.txt). They run as separate
programs; MediaIndex itself is Apache-2.0. FFmpeg source code: https://ffmpeg.org/releases/ffmpeg-9.0.2.tar.xz and
https://github.com/FFmpeg/FFmpeg; build: https://github.com/GyanD/codexffmpeg/releases/tag/9.0.2
"@ | Set-Content -Encoding utf8 "$ff\LICENSE-ffmpeg.txt"

uv run --extra app --extra create pyinstaller packaging\windows\MediaIndex.spec --noconfirm --clean `
  --distpath dist-app --workpath build-app\pyinstaller-win
if ($LASTEXITCODE) { throw "PyInstaller failed" }
Copy-Item "$ff\LICENSE-GPLv3.txt" "dist-app\MediaIndex\_internal\ffmpeg_bin\" -ErrorAction SilentlyContinue
$version = (Select-String -Path pyproject.toml -Pattern '^version = "(.+)"').Matches[0].Groups[1].Value
$out = "dist-app\MediaIndex-$version-windows-x64.zip"
if (Test-Path $out) { Remove-Item $out }
# tar.exe (bsdtar, in Windows 10+) writes "/" in zip paths as the format requires; Windows PowerShell's
# Compress-Archive writes "\", which other unzip tools turn into odd file names.
tar.exe -a -c -f $out -C dist-app MediaIndex
if ($LASTEXITCODE) { throw "zip failed" }
Write-Host "built: dist-app\MediaIndex\MediaIndex.exe, $out ($([math]::Round((Get-Item $out).Length / 1MB)) MB)"
