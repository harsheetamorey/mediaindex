"""Lightweight local relevance-labelling tool for the frozen image evaluation (human judgements only).

Serves http://127.0.0.1:8777 with every query and its pooled candidates (thumbnails), in a fixed order that
hides system rank and scores. Grades are saved to evaluation/labels/labels-v1.json on each click.
Images are read only from the demo pack folder and only for files in the pinned manifest.

Usage: uv run python evaluation/label_server.py --images data/demo/stockimages-cc0 [--labeler "your name"]
"""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
LABELS = HERE / "labels" / "labels-v1.json"


def load(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--labeler", default="anonymous")
    ap.add_argument("--port", type=int, default=8777)
    a = ap.parse_args()
    spec = load(HERE / "queries.json", {})
    pool = load(HERE / "pool-v1.json", {})
    allowed = {i["file"] for i in load(REPO / "manifests" / "stockimages-cc0-500.json", {"items": []})["items"]}
    root = a.images.resolve()
    thumbs: dict[str, bytes] = {}

    def thumb(name: str) -> bytes:
        if name not in thumbs:
            im = Image.open(root / name).convert("RGB")
            im.thumbnail((360, 360))
            b = io.BytesIO()
            im.save(b, "JPEG", quality=82)
            thumbs[name] = b.getvalue()
        return thumbs[name]

    def page() -> str:
        labels = load(LABELS, {})
        rows = []
        done = 0
        for q in spec["queries"]:
            cands = sorted(pool.get(q["id"], []), key=lambda c: hashlib.sha256((q["id"] + c).encode()).hexdigest())
            ql = labels.get(q["id"], {})
            done += all(c in ql for c in cands) and bool(cands)
            desc = html.escape(q.get("text", ""))
            ref = (f'<img class="ref" src="/img/{html.escape(q["reference"])}" title="reference (excluded from results)">'
                   if q.get("reference") else "")
            cells = []
            for c in cands:
                g = ql.get(c)
                btns = "".join(
                    f'<button class="g{v}{" on" if g == v else ""}" onclick="lab(\'{q["id"]}\',\'{c}\',{v},this)">{t}</button>'
                    for v, t in ((0, "not"), (1, "partly"), (2, "relevant")))
                cells.append(f'<div class="c"><img src="/img/{html.escape(c)}" loading="lazy"><div>{btns}</div></div>')
            rows.append(f'<section><h2>{q["id"]} · {q["mode"]} {ref} <span>{desc}</span> '
                        f'<small>{len(ql)}/{len(cands)} judged</small></h2><div class="grid">{"".join(cells)}</div></section>')
        return f"""<!doctype html><meta charset="utf-8"><title>MediaIndex relevance labelling</title>
<style>body{{font-family:system-ui;margin:16px;background:#f6f6f4}}h2{{font-size:16px;display:flex;gap:10px;align-items:center}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:10px}}.c img{{width:100%;aspect-ratio:4/3;object-fit:cover}}
.ref{{height:60px;border:3px solid #2f5bd3}}button{{font-size:12px;margin:2px;border:1px solid #bbb;background:#fff;border-radius:4px}}
button.on.g0{{background:#ddd}}button.on.g1{{background:#ffe08a}}button.on.g2{{background:#7bd88f}}small{{color:#666}}</style>
<h1>Relevance labelling ({done}/{len(spec['queries'])} queries fully judged)</h1>
<p>Judge each image against the query on its own merits. 0 = not relevant, 1 = partially, 2 = clearly relevant. For image and
image+text queries, the blue-bordered reference shows what the user started from. The order is shuffled and scores are hidden.</p>
{''.join(rows)}
<script>function lab(q,c,v,b){{fetch('/label',{{method:'POST',headers:{{'content-type':'application/json'}},
body:JSON.stringify({{q:q,c:c,v:v}})}}).then(r=>{{if(r.ok){{b.parentNode.querySelectorAll('button').forEach(x=>x.classList.remove('on'));b.classList.add('on')}}}})}}</script>"""

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            p = urlparse(self.path).path
            if p == "/":
                body, ctype = page().encode(), "text/html; charset=utf-8"
            elif p.startswith("/img/") and p[5:] in allowed and (root / p[5:]).is_file():
                body, ctype = thumb(p[5:]), "image/jpeg"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("content-type", ctype)
            self.send_header("content-security-policy", "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if urlparse(self.path).path != "/label" or self.headers.get("host", "").split(":")[0] not in ("127.0.0.1", "localhost"):
                self.send_error(403)
                return
            d = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
            if d.get("q") not in pool or d.get("c") not in pool[d["q"]] or d.get("v") not in (0, 1, 2):
                self.send_error(400)
                return
            labels = load(LABELS, {})
            labels.setdefault(d["q"], {})[d["c"]] = d["v"]
            labels["_meta"] = {"labeler": a.labeler, "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                               "scale": spec["protocol"]["relevance_scale"]}
            LABELS.parent.mkdir(parents=True, exist_ok=True)
            LABELS.write_text(json.dumps(labels, indent=1))
            self.send_response(204)
            self.end_headers()

        def log_message(self, *a):
            pass

    print(f"labelling at http://127.0.0.1:{a.port} (Ctrl+C to stop); labels -> {LABELS.relative_to(REPO)}")
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
