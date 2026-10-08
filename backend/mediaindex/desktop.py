"""Desktop entry point (the double-click macOS app): run the local server in-process and show it in a native window.

Same app, same rules as `python -m mediaindex`: loopback only, one model instance, one process per data
directory. The model is never downloaded silently: if it is not in the local cache, the first window asks first.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

APP_SUPPORT = Path.home() / "Library" / "Application Support" / "MediaIndex"
MODEL_CACHE_MB = 1450  # observed size of the pinned model in the Hugging Face cache


def _hf_hub_cache() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def _model_dir() -> Path:
    from .model.profiles import MODEL_ID

    return _hf_hub_cache() / ("models--" + MODEL_ID.replace("/", "--"))


def model_cached() -> bool:
    """Filesystem-only check (no network, no hub import) that the pinned revision's weights are present."""
    from .model.profiles import MODEL_REVISION

    snap = _model_dir() / "snapshots" / MODEL_REVISION
    return snap.is_dir() and any(snap.rglob("*.safetensors")) and (snap / "config.json").exists()


def _free_port(preferred: int) -> int:
    for port in (preferred, 0):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("no free local port")


def _alert(message: str) -> None:
    subprocess.run(["osascript", "-e", f'display alert "MediaIndex" message {json_str(message)} as critical'],
                   capture_output=True)


def json_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def prepare_environment(need_download: bool) -> None:
    if getattr(sys, "frozen", False):  # inside the .app bundle: user data lives in Application Support
        os.environ.setdefault("MEDIAINDEX_DATA_DIR", str(APP_SUPPORT))
        os.environ.setdefault("MEDIAINDEX_UI_DIR", str(Path(getattr(sys, "_MEIPASS", ".")) / "frontend_dist"))
    # Apps opened from Finder get a minimal PATH; add the usual Homebrew locations so FFmpeg is found.
    extra = [p for p in ("/opt/homebrew/bin", "/usr/local/bin") if Path(p).is_dir()]
    os.environ["PATH"] = os.pathsep.join([os.environ.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin"), *extra])
    if not need_download:  # normal use: model files from the local cache only
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["MEDIAINDEX_PORT"] = str(_free_port(int(os.environ.get("MEDIAINDEX_PORT", "8765"))))


FIRST_RUN_HTML = """<!doctype html><html><head><meta charset="utf-8"><style>
body{font-family:-apple-system,system-ui;margin:0;display:grid;place-items:center;height:100vh;background:#f6f6f4;color:#1d1d1f}
main{max-width:460px;padding:24px;text-align:center}h1{font-size:22px}p{color:#555;line-height:1.45}
button{font:inherit;padding:10px 18px;border-radius:8px;border:0;background:#2f5bd3;color:#fff;cursor:pointer}
button:disabled{opacity:.5}#s{margin-top:14px;min-height:1.4em}
@media (prefers-color-scheme:dark){body{background:#1c1c1e;color:#f2f2f7}p{color:#aaa}}
</style></head><body><main><h1>One-time setup</h1>
<p>MediaIndex runs an AI model (Google EmbeddingGemma 2) on this Mac. It needs to download it once, about 1.5 GB,
from Hugging Face. After that, MediaIndex works fully offline and never uploads your files.</p>
<button id="b" onclick="this.disabled=true;document.getElementById('s').textContent='Starting download…';pywebview.api.download()">
Download the model</button><p id="s"></p></main></body></html>"""


class FirstRunApi:
    def __init__(self, url: str):
        self.url = url
        self.window = None

    def download(self) -> None:
        threading.Thread(target=self._download, daemon=True).start()

    def _status(self, text: str) -> None:
        self.window.evaluate_js(f"document.getElementById('s').textContent = {json_str(text)}")

    def _download(self) -> None:
        from huggingface_hub import snapshot_download

        from .model.profiles import MODEL_ID, MODEL_REVISION

        done = threading.Event()

        def progress():
            blobs = _model_dir() / "blobs"
            while not done.wait(1.0):
                mb = sum(f.stat().st_size for f in blobs.glob("*")) / 1e6 if blobs.is_dir() else 0
                self._status(f"Downloaded {mb:,.0f} MB of about {MODEL_CACHE_MB:,} MB…")

        threading.Thread(target=progress, daemon=True).start()
        try:
            snapshot_download(MODEL_ID, revision=MODEL_REVISION)
        except Exception as e:  # noqa: BLE001
            done.set()
            self._status(f"Download failed: {e}. Check your internet connection and reopen MediaIndex.")
            return
        done.set()
        self._status("Done. Opening MediaIndex…")
        self.window.load_url(self.url)


def _selftest(window, query: str, first_run: bool) -> None:
    """MEDIAINDEX_DESKTOP_SELFTEST=<query>: wait for the window to load the UI, run one real search against this
    process's server, print the outcome, then close. (The UI's CSP forbids eval, so no JS is injected.)"""
    import json

    try:
        loaded = window.events.loaded.wait(60)
        if first_run:  # first-run page (no CSP there): press "Download the model"
            window.evaluate_js("document.getElementById('b').click()")
            t0 = time.time()
            while not (window.get_current_url() or "").startswith("http://127.0.0.1"):
                if time.time() - t0 > 3600:
                    raise TimeoutError("model download did not finish within an hour")
                time.sleep(2)
            print("SELFTEST first-run download finished in", round(time.time() - t0), "s", flush=True)
        url = window.get_current_url()
        req = urllib.request.Request(url.rstrip("/") + "/api/search/text", method="POST",
                                     data=json.dumps({"text": query, "media_types": ["image"], "limit": 3}).encode(),
                                     headers={"content-type": "application/json", "Origin": url.rstrip("/")})
        t = time.time()
        with urllib.request.urlopen(req, timeout=300) as r:
            top = [x["asset"]["rel_path"] for x in json.load(r)["results"]]
        print("SELFTEST", json.dumps({"window_loaded": bool(loaded), "url": url, "top": top,
                                      "search_s": round(time.time() - t, 1)}), flush=True)
    except Exception as e:  # noqa: BLE001
        print("SELFTEST failed", repr(e), flush=True)
    finally:
        window.destroy()


def main() -> None:
    from .model.profiles import MODEL_ID  # noqa: F401 (light import: no torch)

    need_download = not model_cached()
    prepare_environment(need_download)

    import uvicorn
    import webview

    from .app import create_app
    from .config import Settings
    from .instance import AlreadyRunning, acquire_instance_lock

    settings = Settings()
    settings.ensure_dirs()
    try:
        lock = acquire_instance_lock(settings.data_dir)
    except AlreadyRunning:
        _alert("MediaIndex is already open.")
        raise SystemExit(1)
    import tempfile

    from .uploads import cleanup_stale_uploads

    spool = settings.data_dir / "tmp_spool"
    spool.mkdir(parents=True, exist_ok=True)
    cleanup_stale_uploads(spool, max_age=0)
    tempfile.tempdir = str(spool)

    server = uvicorn.Server(uvicorn.Config(create_app(settings), host=settings.host, port=settings.port,
                                           workers=1, log_level=os.environ.get("MEDIAINDEX_LOG_LEVEL", "warning"),
                                           timeout_graceful_shutdown=3))
    thread = threading.Thread(target=server.run, name="mediaindex-server", daemon=True)
    thread.start()
    url = f"http://{settings.host}:{settings.port}/"
    for _ in range(300):
        try:
            urllib.request.urlopen(url + "api/health", timeout=1)
            break
        except OSError:
            time.sleep(0.1)
    else:
        _alert("MediaIndex could not start its local server.")
        raise SystemExit(1)

    webview.settings["ALLOW_DOWNLOADS"] = True  # manifest download link
    if need_download:
        api = FirstRunApi(url)
        window = api.window = webview.create_window("MediaIndex", html=FIRST_RUN_HTML, js_api=api, width=1400, height=900)
    else:
        window = webview.create_window("MediaIndex", url, width=1400, height=900, min_size=(900, 600))
    try:
        if os.environ.get("MEDIAINDEX_DESKTOP_SELFTEST"):
            webview.start(_selftest, (window, os.environ["MEDIAINDEX_DESKTOP_SELFTEST"], need_download))
        else:
            webview.start()
    finally:
        # Closing the window quits the app. Open browser connections must not keep the process alive, so give the
        # server a few seconds to finish, then exit for certain. (SQLite commits are durable, and an interrupted
        # import is marked as such and resumed at the next launch.)
        server.should_exit = True
        thread.join(timeout=5)
        lock.close()
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
