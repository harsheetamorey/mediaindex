# PyInstaller spec for MediaIndex.app (macOS, Apple silicon). Build with: scripts/build_mac_app.sh
# The model weights are NOT bundled: the app downloads them once on first launch, after asking.
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, copy_metadata

ROOT = Path(SPECPATH).resolve().parents[1]
datas = [(str(ROOT / "frontend" / "dist"), "frontend_dist")]
binaries, hiddenimports = [], ["mediaindex.desktop", "webview.platforms.cocoa"]
for pkg in ("mediaindex", "sentence_transformers", "transformers", "tokenizers", "safetensors", "huggingface_hub",
            "torchvision", "torchcodec", "webview", "uvicorn", "fastapi", "starlette", "multipart", "soundfile",
            "_soundfile_data"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h
for dist in ("tqdm", "regex", "httpx", "packaging", "filelock", "numpy", "pyyaml", "torch", "pillow", "fastapi",
             "pydantic", "uvicorn", "starlette", "python-multipart"):
    try:  # transformers checks some dependency versions at import time
        datas += copy_metadata(dist)
    except Exception:
        pass

a = Analysis([str(ROOT / "packaging" / "macos" / "launch.py")], pathex=[str(ROOT / "backend")], binaries=binaries,
             datas=datas, hiddenimports=hiddenimports, excludes=["tkinter", "matplotlib", "IPython", "pytest"],
             noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="MediaIndex", console=False, target_arch="arm64",
          upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="MediaIndex", upx=False)
app = BUNDLE(coll, name="MediaIndex.app", bundle_identifier="io.github.harsheetamorey.mediaindex",
             info_plist={"CFBundleShortVersionString": "0.1.0", "CFBundleDisplayName": "MediaIndex",
                         "NSHighResolutionCapable": True, "LSMinimumSystemVersion": "13.0",
                         "LSApplicationCategoryType": "public.app-category.photography"})
