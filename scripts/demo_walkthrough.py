"""Play the demo storyboard (docs/demo.md) in a real browser against a running server, saving one screenshot per beat.

Uses only the CC0 demo packs. With --video, Playwright also records a silent .webm of the whole run (no narration).
Export beats write copies and a clip into --out/exports (originals are never touched).

Usage: python scripts/demo_walkthrough.py --out data/demo-walkthrough [--port 8765] [--video] [--pause 1.5]
(Playwright is a dev-only tool: `pip install playwright` in a separate venv; it drives installed Chrome.)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import expect, sync_playwright

REPO = Path(__file__).resolve().parents[1]
IMAGES = REPO / "data" / "demo" / "stockimages-cc0"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--pause", type=float, default=1.5, help="seconds to hold on each beat")
    a = ap.parse_args()
    out = a.out.resolve()
    exports = out / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    external: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        ctx = browser.new_context(viewport={"width": 1400, "height": 860},
                                  record_video_dir=str(out) if a.video else None,
                                  record_video_size={"width": 1400, "height": 860})
        page = ctx.new_page()
        page.on("request", lambda r: external.append(r.url)
                if urlparse(r.url).hostname not in ("127.0.0.1", "localhost") and not r.url.startswith(("blob:", "data:"))
                else None)
        hold = lambda: page.wait_for_timeout(int(a.pause * 1000))  # noqa: E731
        cells = page.locator("li.cell")

        def shot(name: str, note: str) -> None:
            hold()
            page.screenshot(path=str(out / f"{name}.png"))
            print(f"{name}: {note}")

        def search(expect_path: str) -> None:
            with page.expect_response(lambda r: expect_path in r.url, timeout=180_000) as resp:
                page.get_by_role("button", name="Search").click()
            assert resp.value.status == 200, f"{expect_path} -> {resp.value.status}"
            page.wait_for_timeout(400)
            expect(cells.first).to_be_visible()

        def titles(n: int = 5) -> list[str]:
            return [cells.nth(i).get_attribute("title") or "" for i in range(min(n, cells.count()))]

        page.goto(f"http://127.0.0.1:{a.port}/")
        expect(page.get_by_role("heading", name="MediaIndex")).to_be_visible()

        # 1. text -> images
        page.get_by_role("radio", name="Images").click()
        page.locator("#q").fill("a snowy mountain landscape")
        search("/api/search/text")
        shot("01-text-search", f"text → images top 5 {titles()}")

        # 2. reference + refinement text
        page.locator('input[type="file"]').set_input_files(str(IMAGES / "stock-00171.jpg"))
        expect(page.locator(".dropzone .ref img")).to_be_visible()
        page.locator("#q").fill("")
        search("/api/search/image")
        shot("02a-reference", f"reference stock-00171 → {titles()}")
        page.locator("#q").fill("at sunset")
        search("/api/search/image-text")
        shot("02b-refined", f"reference + 'at sunset' → {titles()}")
        page.get_by_role("button", name="Remove reference").click()

        # 3. an honest miss: negation is not supported
        page.locator('input[type="file"]').set_input_files(str(IMAGES / "stock-00102.jpg"))
        expect(page.locator(".dropzone .ref img")).to_be_visible()
        page.locator("#q").fill("without any flowers")
        search("/api/search/image-text")
        shot("03-honest-miss", f"viola + 'without any flowers' → {titles()} (still flowers expected)")
        page.get_by_role("button", name="Remove reference").click()

        # 4. text -> sounds, play the matched window
        page.get_by_role("radio", name="Sounds").click()
        page.locator("#q").fill("a dog barking")
        search("/api/search/text")
        cells.first.locator(".cell-play").click()
        shot("04-sounds", f"text → sounds {titles(3)} span {cells.first.locator('.cell-span').inner_text()}")
        cells.first.locator(".cell-play").click()

        # 5. text -> video moments, open the moment
        page.get_by_role("radio", name="Videos").click()
        page.locator("#q").fill("a dog")
        search("/api/search/text")
        spans = [cells.nth(i).locator(".cell-span").inner_text() for i in range(min(3, cells.count()))]
        cells.first.click()
        dialog = page.get_by_role("dialog")
        expect(dialog.locator("#mi-video-player")).to_be_visible()
        shot("05-video-moment", f"text → video moments {list(zip(titles(3), spans))}")

        # 6. clip export of the matched moment
        dialog.get_by_role("button", name="Export clip…").click()
        dialog.get_by_label("Clip destination folder").fill(str(exports))
        dialog.get_by_role("button", name="Preview export").click()
        expect(dialog.locator(".plan")).to_be_visible()
        dialog.get_by_role("button", name="Export clip", exact=True).click()
        expect(dialog.locator(".banner.ok")).to_be_visible(timeout=180_000)
        shot("06-clip-export", dialog.locator(".banner.ok").inner_text().replace("\n", " "))
        page.keyboard.press("Escape")

        # 7. selection -> manifest + copy export
        page.locator(".selections").get_by_role("button", name="+ New").click()
        page.get_by_role("radio", name="Images").click()
        page.locator("#q").fill("a snowy mountain landscape")
        search("/api/search/text")
        for i in range(3):
            cells.nth(i).locator(".cell-add").click()
        expect(page.locator(".cell-add.added")).to_have_count(3)
        page.locator(".selections").get_by_role("button", name="Open").first.click()
        expect(page.locator(".sel-item")).to_have_count(3)
        page.get_by_role("button", name="Copy files to folder…").click()
        page.get_by_label("Destination folder").fill(str(exports))
        page.get_by_role("button", name="Preview").click()
        expect(page.locator(".plan")).to_be_visible()
        page.get_by_role("button", name=re.compile(r"Copy \d+ files")).click()
        expect(page.locator(".export-panel .banner")).to_be_visible(timeout=60_000)
        shot("07-selection-export", page.locator(".export-panel .banner").inner_text().replace("\n", " "))

        ctx.close()
        browser.close()
    print("external requests:", external or "none")
    if a.video:
        print("video:", *sorted(out.glob("*.webm")))
    return 1 if external else 0


if __name__ == "__main__":
    sys.exit(main())
