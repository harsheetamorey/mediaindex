# PyInstaller spec for MediaIndex (Windows 10/11, x64). Build with: scripts/build_windows_app.ps1
# The model weights are NOT bundled: the app downloads them once on first launch, after asking.
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, copy_metadata

ROOT = Path(SPECPATH).resolve().parents[1]
datas = [(str(ROOT / "frontend" / "dist"), "frontend_dist")]
FFMPEG = ROOT / "build-app" / "ffmpeg-win"  # FFmpeg 9.0.2 essentials (GPL), fetched and checked by the build script
binaries = [(str(FFMPEG / name), "ffmpeg_bin") for name in ("ffmpeg.exe", "ffprobe.exe")]
datas += [(str(FFMPEG / "LICENSE-ffmpeg.txt"), "ffmpeg_bin")]
hiddenimports = ["mediaindex.desktop", "webview.platforms.edgechromium", "webview.platforms.winforms", "clr"]
for pkg in ("mediaindex", "sentence_transformers", "transformers", "tokenizers", "safetensors", "huggingface_hub",
            "torchvision", "torchcodec", "webview", "uvicorn", "fastapi", "starlette", "multipart", "soundfile",
            "_soundfile_data", "diffusers", "peft", "pillow_heif", "pythonnet", "clr_loader"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h
for dist in ("tqdm", "regex", "httpx", "packaging", "filelock", "numpy", "pyyaml", "torch", "pillow", "fastapi",
             "pydantic", "uvicorn", "starlette", "python-multipart", "diffusers", "peft", "accelerate", "psutil", "requests",
             "safetensors", "huggingface-hub", "tokenizers", "transformers"):
    try:  # transformers checks some dependency versions at import time
        datas += copy_metadata(dist)
    except Exception:
        pass

a = Analysis([str(ROOT / "packaging" / "windows" / "launch.py")], pathex=[str(ROOT / "backend")], binaries=binaries,
             datas=datas, hiddenimports=hiddenimports, excludes=["tkinter", "matplotlib", "IPython", "pytest"],
             noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="MediaIndex", console=False, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="MediaIndex", upx=False)
