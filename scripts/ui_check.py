"""Drive the real UI in a headless browser (Playwright + installed Chrome) against a running server.

Steps: add+import folder -> text search -> preview -> use as reference -> image search ->
refinement -> keyboard navigation. Records any non-loopback network request and console errors.

Usage: python scripts/ui_check.py --folder /abs/path --shots OUT_DIR [--skip-import] [--query "..."]
(Playwright is a dev-only tool; install it in a separate venv: `pip install playwright`.)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import expect, sync_playwright

BASE = "http://127.0.0.1:8765/"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder")
    ap.add_argument("--shots", type=Path, required=True)
    ap.add_argument("--skip-import", action="store_true")
    ap.add_argument("--query", default="a zebra")
    ap.add_argument("--refine", default="at night")
    a = ap.parse_args()
    a.shots.mkdir(parents=True, exist_ok=True)
    external, console_errors = [], []

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.on("request", lambda r: external.append(r.url)
                if urlparse(r.url).hostname not in ("127.0.0.1", "localhost") and not r.url.startswith(("blob:", "data:"))
                else None)
        page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
        page.goto(BASE)
        expect(page.get_by_role("heading", name="MediaIndex")).to_be_visible()

        if not a.skip_import:
            page.get_by_role("button", name="+ Add folder").click()
            page.get_by_label("Folder path").fill(a.folder)
            page.get_by_role("button", name="Add library").click()
            name = Path(a.folder).name
            lib = page.locator("li.lib", has_text=name)
            expect(lib).to_be_visible()
            page.screenshot(path=str(a.shots / "1-importing.png"))
            expect(lib.locator(".summary")).to_be_visible(timeout=600_000)
            print("import summary:", lib.locator(".summary summary").inner_text())
            lib.locator(".summary summary").click()
            page.screenshot(path=str(a.shots / "2-imported.png"))

        page.locator("#q").fill(a.query)
        page.get_by_role("button", name="Search").click()
        cells = page.locator("li.cell")
        expect(cells.first).to_be_visible(timeout=120_000)
        print(f"text search '{a.query}': {cells.count()} results; first: {cells.first.get_attribute('title')}")
        assert page.locator(".score").count() == 0, "scores must be hidden by default"
        page.screenshot(path=str(a.shots / "3-text-search.png"))

        # keyboard navigation: focus grid, move right, open with Enter, close with Escape
        cells.first.focus()
        page.keyboard.press("ArrowRight")
        focused_title = page.evaluate("document.activeElement.getAttribute('title')")
        page.keyboard.press("Enter")
        dialog = page.get_by_role("dialog")
        expect(dialog).to_be_visible()
        print("keyboard: ArrowRight focused", focused_title, "and Enter opened", dialog.get_attribute("aria-label"))
        assert dialog.get_attribute("aria-label") == focused_title
        page.screenshot(path=str(a.shots / "4-preview.png"))
        page.keyboard.press("Escape")
        expect(dialog).to_be_hidden()

        # use as reference -> image search
        cells.first.click()
        page.get_by_role("button", name="Use as reference").click()
        expect(page.locator(".dropzone .ref img")).to_be_visible()
        page.locator("#q").fill("")
        expect(page.locator("label.mode")).to_have_text("Similar to reference image")
        page.get_by_role("button", name="Search").click()
        page.wait_for_function("document.querySelector('.results-head') !== null")
        page.wait_for_timeout(500)
        expect(cells.first).to_be_visible(timeout=120_000)
        print("image search first:", cells.first.get_attribute("title"))
        page.screenshot(path=str(a.shots / "5-image-search.png"))

        # refinement
        page.locator("#q").fill(a.refine)
        expect(page.locator("label.mode")).to_have_text("Reference image + refinement text")
        with page.expect_response(lambda r: "/api/search/image-text" in r.url, timeout=120_000) as resp:
            page.get_by_role("button", name="Search").click()
        print("refinement response:", resp.value.status)
        page.wait_for_timeout(300)
        print("refined first:", cells.first.get_attribute("title"))
        page.get_by_label("Show technical scores").check()
        expect(page.locator(".score").first).to_be_visible()
        print("score label sample:", page.locator(".score").first.inner_text(), "| head:",
              page.locator(".results-head").inner_text().replace("\n", " "))
        page.screenshot(path=str(a.shots / "6-refined-with-scores.png"))
        page.get_by_label("Show technical scores").uncheck()

        # upload a reference file through the drop-zone file input
        page.get_by_role("button", name="Remove reference image").click()
        if a.folder:
            broken = Path(a.folder) / "broken.jpg"
            if broken.exists():  # error state: invalid upload must show a readable message
                page.locator('input[type="file"]').set_input_files(str(broken))
                page.get_by_role("button", name="Search").click()
                expect(page.locator(".results .banner.error")).to_be_visible(timeout=60_000)
                print("invalid upload shows:", page.locator(".results .banner.error").inner_text())
                page.screenshot(path=str(a.shots / "7-invalid-upload.png"))
                page.get_by_role("button", name="Remove reference image").click()
                console_errors.clear()  # the expected 422 is logged by the browser
            files = sorted(p for p in Path(a.folder).iterdir()
                           if p.suffix.lower() in (".jpg", ".png", ".webp") and p.name != "broken.jpg")
            if files:
                page.locator('input[type="file"]').set_input_files(str(files[0]))
                with page.expect_response(lambda r: "/api/search/image" in r.url, timeout=120_000) as resp:
                    page.get_by_role("button", name="Search").click()
                print("upload reference search:", resp.value.status, "first:",
                      (page.wait_for_timeout(300), cells.first.get_attribute("title"))[1])

        browser.close()
    print("external requests:", external or "none")
    print("console errors:", console_errors or "none")
    return 0 if not external else 1


if __name__ == "__main__":
    sys.exit(main())
