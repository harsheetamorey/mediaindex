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
MODEL_CACHE_MB = 1450 + 170  # observed sizes: EmbeddingGemma and the Ask object detector in the Hugging Face cache
EXTRAS_MB = 358 + 196 + 2000 + 128  # SAM, LaMa, Stable Diffusion inpainting and LCM-LoRA (photo editing in Ask)
SKIP_EXTRAS = APP_SUPPORT / "skip-editing-models"  # the user unticked photo editing at first launch


def _hf_hub_cache() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def _model_dir() -> Path:
    from .model.profiles import MODEL_ID

    return _hf_hub_cache() / ("models--" + MODEL_ID.replace("/", "--"))


def model_cached() -> bool:
    """Filesystem-only check (no network) that the pinned revisions of both models are present."""
    from .detect import detector_cached
    from .model.profiles import MODEL_REVISION

    snap = _model_dir() / "snapshots" / MODEL_REVISION
    return (snap.is_dir() and any(snap.rglob("*.safetensors")) and (snap / "config.json").exists()
            and detector_cached())


def extras_cached() -> bool:
    """Photo editing (remove, replace, make images): SAM and LaMa, plus Stable Diffusion when the app includes it."""
    from . import creative, edit

    return edit.models_cached() and creative.available()[0]


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
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, message, "MediaIndex", 0x10)  # MB_ICONERROR
        return
    subprocess.run(["osascript", "-e", f'display alert "MediaIndex" message {json_str(message)} as critical'],
                   capture_output=True)


def json_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def prepare_environment(need_download: bool) -> None:
    if getattr(sys, "frozen", False):  # inside the .app bundle: user data lives in Application Support
        os.environ.setdefault("MEDIAINDEX_DATA_DIR", str(APP_SUPPORT))
        os.environ.setdefault("MEDIAINDEX_UI_DIR", str(Path(getattr(sys, "_MEIPASS", ".")) / "frontend_dist"))
    # The app ships its own FFmpeg; apps opened from Finder get a minimal PATH, so Homebrew's folders are added after it.
    bundled = Path(getattr(sys, "_MEIPASS", "")) / "ffmpeg_bin" if getattr(sys, "frozen", False) else None
    extra = [p for p in ("/opt/homebrew/bin", "/usr/local/bin") if Path(p).is_dir()]
    os.environ["PATH"] = os.pathsep.join([*([str(bundled)] if bundled and bundled.is_dir() else []),
                                          os.environ.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin"), *extra])
    if not need_download:  # normal use: model files from the local cache only
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["MEDIAINDEX_PORT"] = str(_free_port(int(os.environ.get("MEDIAINDEX_PORT", "8765"))))


SETUP_HTML = """<!doctype html><html><head><meta charset="utf-8"><style>
body{font-family:-apple-system,system-ui;margin:0;display:grid;place-items:center;height:100vh;background:#f6f6f4;color:#1d1d1f}
main{max-width:480px;padding:24px;text-align:center}h1{font-size:22px}p{color:#555;line-height:1.45}
label{display:block;text-align:left;background:#fff;border:1px solid #ddd;border-radius:10px;padding:12px 14px;margin:14px 0;
line-height:1.4;color:#333}label small{color:#777}
button{font:inherit;padding:10px 18px;border-radius:8px;border:0;background:#2f5bd3;color:#fff;cursor:pointer}
button:disabled{opacity:.5}#s{margin-top:14px;min-height:1.4em}
@media (prefers-color-scheme:dark){body{background:#1c1c1e;color:#f2f2f7}p{color:#aaa}label{background:#2c2c2e;border-color:#444;color:#ddd}
label small{color:#999}}
</style></head><body><main><h1>__TITLE__</h1>
<p>__INTRO__</p>
__CHOICE__
<button id="b" onclick="this.disabled=true;var c=document.getElementById('x');if(c)c.disabled=true;
document.getElementById('s').textContent='Starting download…';pywebview.api.download(c?c.checked:true)">__BUTTON__</button>
<p id="s"></p></main></body></html>"""

FIRST_RUN_HTML = (SETUP_HTML.replace("__TITLE__", "One-time setup")
                  .replace("__INTRO__", "MediaIndex runs AI models on this Mac: Google EmbeddingGemma 2 for search, and an object "
                           f"detector for counting in Ask. It downloads them once, about {MODEL_CACHE_MB / 1000:.1f} GB, from "
                           "Hugging Face. After that, MediaIndex works fully offline and never uploads your files.")
                  .replace("__CHOICE__", '<label><input type="checkbox" id="x" checked> <b>Also photo editing</b><br><small>'
                           "Remove or replace things in photos and make new images (SAM, LaMa, Stable Diffusion). "
                           f"About {EXTRAS_MB / 1000:.1f} GB more. You can add it later from the Models menu.</small></label>")
                  .replace("__BUTTON__", "Download"))

EXTRAS_HTML = (SETUP_HTML.replace("__TITLE__", "Photo editing models")
               .replace("__INTRO__", "Removing and replacing things in photos and making new images use SAM, LaMa and Stable "
                        f"Diffusion. MediaIndex downloads them once, about {EXTRAS_MB / 1000:.1f} GB, then works offline.")
               .replace("__CHOICE__", "").replace("__BUTTON__", "Download"))


class FirstRunApi:
    def __init__(self, url: str):
        self.url = url
        self.window = None

    def download(self, extras: bool = True) -> None:
        threading.Thread(target=self._download, args=(bool(extras),), daemon=True).start()

    def show_extras(self) -> None:
        """Models menu: download photo editing later (or again, if a download was interrupted)."""
        if self.url and not extras_cached():
            self.window.load_html(EXTRAS_HTML)
        elif self.url:
            _alert("The photo editing models are already downloaded.")

    def _status(self, text: str) -> None:
        self.window.evaluate_js(f"document.getElementById('s').textContent = {json_str(text)}")

    def _download(self, extras: bool) -> None:
        import huggingface_hub.constants as hub
        from huggingface_hub import snapshot_download

        from . import creative, edit
        from .model.profiles import MODEL_ID, MODEL_REVISION

        hub.HF_HUB_OFFLINE = False  # normal use runs offline; downloads happen only here, after the user asked

        done = threading.Event()

        def cache_mb() -> float:  # works for both cache layouts (per-model blobs/ and one shared blobs/)
            hub = _hf_hub_cache()
            return sum(f.stat().st_size for f in hub.glob("**/blobs/**/*") if f.is_file()) / 1e6 if hub.is_dir() else 0

        start = cache_mb()

        def progress():
            while not done.wait(1.0):
                mb = cache_mb() - start
                self._status(f"Downloaded {mb:,.0f} MB of about {total:,} MB…")

        essentials = not model_cached()
        total = (MODEL_CACHE_MB if essentials else 0) + (EXTRAS_MB if extras else 0)
        threading.Thread(target=progress, daemon=True).start()
        try:
            if essentials:
                snapshot_download(MODEL_ID, revision=MODEL_REVISION)
                from .detect import download as download_detector

                download_detector()
            if extras:
                edit.download()
                creative.download()
                SKIP_EXTRAS.unlink(missing_ok=True)
            else:
                SKIP_EXTRAS.touch()
        except Exception as e:  # noqa: BLE001
            done.set()
            self._status(f"Download failed: {e}. Check your internet connection and try again.")
            self.window.evaluate_js("document.getElementById('b').disabled=false")
            return
        finally:
            hub.HF_HUB_OFFLINE = bool(os.environ.get("HF_HUB_OFFLINE"))
        done.set()
        self._status("Done. Opening MediaIndex…")
        self.window.load_url(self.url)


def settings_device() -> str:
    from .config import Settings

    return Settings().resolved_device()


def _selftest(window, query: str, first_run: bool) -> None:
    """MEDIAINDEX_DESKTOP_SELFTEST=<query>: wait for the window to load the UI, run one real search against this
    process's server, print the outcome, then close. (The UI's CSP forbids eval, so no JS is injected into it.)"""
    import json

    try:
        t0 = time.time()
        if first_run:  # first-run page (no CSP there): press "Download the model"
            while not window.evaluate_js("!!document.getElementById('b')"):
                time.sleep(0.5)
            window.evaluate_js("document.getElementById('b').click()")
        while not (window.get_current_url() or "").startswith("http://127.0.0.1"):
            if time.time() - t0 > 3600:
                raise TimeoutError("the app UI did not load")
            time.sleep(0.5)
        if first_run:
            print("SELFTEST first-run download finished in", round(time.time() - t0), "s", flush=True)
        time.sleep(2)
        url = window.get_current_url()
        req = urllib.request.Request(url.rstrip("/") + "/api/search/text", method="POST",
                                     data=json.dumps({"text": query, "media_types": ["image"], "limit": 3}).encode(),
                                     headers={"content-type": "application/json", "Origin": url.rstrip("/")})
        t = time.time()
        with urllib.request.urlopen(req, timeout=300) as r:
            top = [x["asset"]["rel_path"] for x in json.load(r)["results"]]
        search_s = round(time.time() - t, 1)
        base, headers = url.rstrip("/"), {"content-type": "application/json", "Origin": url.rstrip("/")}

        def call(path, body=None):
            req = urllib.request.Request(base + path, method="POST" if body is not None else "GET", headers=headers,
                                         data=json.dumps(body).encode() if body is not None else None)
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.load(r)

        job = call("/api/ask/prepare", {})  # Ask: count objects (loads the bundled detector code), then ask
        while job["status"] in ("queued", "running"):
            time.sleep(1)
            job = call(f"/api/jobs/{job['id']}")
        answer = call("/api/ask", {"message": "how many dogs?", "history": []})["answer"]
        if os.environ.get("MEDIAINDEX_DESKTOP_SELFTEST_EDIT"):  # a photo: remove, then replace, its centre (in-process)
            from PIL import Image

            from .creative import painter_factory
            from .edit import default_editor

            device = settings_device()
            img = Image.open(os.environ["MEDIAINDEX_DESKTOP_SELFTEST_EDIT"]).convert("RGB")
            w, h = img.size
            box = (w * 0.4, h * 0.4, w * 0.6, h * 0.6)
            editor = default_editor(device, painter_factory(device))
            t = time.time()
            removed = editor.remove(img, [box])
            removed_s = round(time.time() - t, 1)
            t = time.time()
            replaced = editor.replace(img, [box], "a red ball")
            print("SELFTEST edit", json.dumps({"remove_s": removed_s, "removed_by": removed.filled_by,
                                               "replace_s": round(time.time() - t, 1), "replace_size": replaced.image.size}),
                  flush=True)
        print("SELFTEST", json.dumps({"ui_loaded_s": round(t - t0, 1), "url": url, "top": top, "search_s": search_s,
                                      "count_objects": job["status"], "count_error": job.get("error"),
                                      "ask": answer[:90]}), flush=True)
    except Exception as e:  # noqa: BLE001
        print("SELFTEST failed", repr(e), flush=True)
    finally:
        window.destroy()


STARTING_HTML = """<!doctype html><html><head><meta charset="utf-8"><style>
body{font-family:-apple-system,system-ui;margin:0;display:grid;place-items:center;height:100vh;background:#f6f6f4;color:#555}
.s{width:28px;height:28px;border:3px solid #ccc;border-top-color:#2f5bd3;border-radius:50%;animation:r 1s linear infinite;margin:0 auto 14px}
@keyframes r{to{transform:rotate(360deg)}}@media (prefers-color-scheme:dark){body{background:#1c1c1e;color:#aaa}}
</style></head><body><div><div class="s"></div>Starting MediaIndex…</div></body></html>"""


def _boot(window, api, need_download: bool, settings, state: dict) -> None:
    """Runs after the window is on screen: start the server (heavy imports happen here), then show the app."""
    try:
        import tempfile

        import uvicorn

        from .app import create_app
        from .uploads import cleanup_stale_uploads

        spool = settings.data_dir / "tmp_spool"
        spool.mkdir(parents=True, exist_ok=True)
        cleanup_stale_uploads(spool, max_age=0)
        tempfile.tempdir = str(spool)
        server = uvicorn.Server(uvicorn.Config(create_app(settings), host=settings.host, port=settings.port, workers=1,
                                               log_level=os.environ.get("MEDIAINDEX_LOG_LEVEL", "warning"),
                                               timeout_graceful_shutdown=3))
        thread = threading.Thread(target=server.run, name="mediaindex-server", daemon=True)
        thread.start()
        state.update(server=server, thread=thread)
        url = f"http://{settings.host}:{settings.port}/"
        for _ in range(300):
            try:
                urllib.request.urlopen(url + "api/health", timeout=1)
                break
            except OSError:
                time.sleep(0.1)
        else:
            raise RuntimeError("the local server did not start")
    except Exception as e:  # noqa: BLE001
        window.load_html(STARTING_HTML.replace("Starting MediaIndex…", f"MediaIndex could not start: {e}"))
        return
    api.url = url
    if need_download:
        window.load_html(FIRST_RUN_HTML if not model_cached() else EXTRAS_HTML)
    else:
        window.load_url(url)
    if os.environ.get("MEDIAINDEX_DESKTOP_SELFTEST"):
        _selftest(window, os.environ["MEDIAINDEX_DESKTOP_SELFTEST"], need_download)


def main() -> None:
    if "--check-models" in sys.argv:  # support: which model groups this app sees as downloaded
        from . import creative, edit

        print({"search_and_ask": model_cached(), "sam_lama": edit.models_cached(), "stable_diffusion": creative.available()})
        try:
            import diffusers  # noqa: F401
            import peft  # noqa: F401
        except ImportError:
            import traceback

            traceback.print_exc()
        return
    need_download = not model_cached()
    prepare_environment(need_download)
    if not need_download and not SKIP_EXTRAS.exists() and not extras_cached():
        need_download = True  # an earlier first-run download stopped part-way: offer photo editing again

    import webview

    from .config import Settings
    from .instance import AlreadyRunning, acquire_instance_lock

    settings = Settings()
    settings.ensure_dirs()
    try:
        lock = acquire_instance_lock(settings.data_dir)
    except AlreadyRunning:
        _alert("MediaIndex is already open.")
        raise SystemExit(1)

    webview.settings["ALLOW_DOWNLOADS"] = True  # manifest download link
    api = FirstRunApi("")
    # Show the window at once; the heavy start-up (PyTorch, the server) happens in _boot while it is on screen.
    window = api.window = webview.create_window("MediaIndex", html=STARTING_HTML, js_api=api, width=1400, height=900,
                                                min_size=(900, 600))
    state: dict = {}
    from webview.menu import Menu, MenuAction

    menu = [Menu("Models", [MenuAction("Download photo editing models…", api.show_extras)])]
    try:
        webview.start(_boot, (window, api, need_download, settings, state), menu=menu)
    finally:
        # Closing the window quits the app. Open browser connections must not keep the process alive, so give the
        # server a few seconds to finish, then exit for certain. (SQLite commits are durable, and an interrupted
        # import is marked as such and resumed at the next launch.)
        if "server" in state:
            state["server"].should_exit = True
            state["thread"].join(timeout=5)
        lock.close()
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
