"""Run a fixed list of real queries against a running MediaIndex server and print top results.

This is a manual inspection aid (REAL MODEL via the live API), not an accuracy evaluation.

Usage: uv run python scripts/search_check.py text "query one" "query two" [--library ID] [--k 5]
       uv run python scripts/search_check.py image --asset ID [--text "refinement"] [--k 5]
       uv run python scripts/search_check.py image --file path.jpg [--text "refinement"]
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import urllib.request
import uuid
from pathlib import Path

BASE = "http://127.0.0.1:8765"


def post_json(path: str, body: dict) -> dict:
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def post_form(path: str, fields: dict, file: Path | None) -> dict:
    boundary = uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    if file:
        ctype = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{file.name}"\r\n'
                     f"Content-Type: {ctype}\r\n\r\n".encode() + file.read_bytes() + b"\r\n")
    body = b"".join(parts) + f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(BASE + path, data=body, method="POST",
                                 headers={"content-type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def show(label: str, r: dict, k: int) -> None:
    t = r["timing_ms"]
    print(f"\n== {label}  [{r['mode']}; embed {t['query_embedding']:.0f} ms, rank {t['ranking']:.2f} ms, "
          f"{r['candidates_searched']} candidates, index {r['index_state']['state']}]")
    for x in r["results"][:k]:
        src = (x["asset"].get("source") or {}).get("inspection_tags") or ""
        print(f"  {x['rank']:>2}. {x['similarity']:.4f}  {x['asset']['rel_path']:<28} tags: {src[:70]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["text", "image"])
    ap.add_argument("queries", nargs="*")
    ap.add_argument("--library", action="append")
    ap.add_argument("--asset")
    ap.add_argument("--file", type=Path)
    ap.add_argument("--text", default="")
    ap.add_argument("--include-identical", action="store_true")
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    if a.mode == "text":
        for q in a.queries:
            show(repr(q), post_json("/api/search/text", {"text": q, "library_ids": a.library, "limit": a.k}), a.k)
        return 0
    fields = {"limit": a.k, "include_identical": str(a.include_identical).lower()}
    if a.asset:
        fields["asset_id"] = a.asset
    if a.library:
        fields["library_ids"] = ",".join(a.library)
    path = "/api/search/image"
    if a.text:
        fields["text"] = a.text
        path = "/api/search/image-text"
    r = post_form(path, fields, a.file)
    show(f"ref={a.asset or a.file} text={a.text!r}", r, a.k)
    print("  query:", json.dumps(r["query"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
