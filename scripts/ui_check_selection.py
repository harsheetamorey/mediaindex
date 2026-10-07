"""Browser check for selections + exports (Playwright + installed Chrome, against a running server).

Usage: python scripts/ui_check_selection.py --dest "/abs/new folder" --shots DIR [--query zebra] [--phase add|verify]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    ap.add_argument("--shots", type=Path, required=True)
    ap.add_argument("--query", default="a zebra")
    ap.add_argument("--phase", choices=["add", "verify"], default="add")
    a = ap.parse_args()
    a.shots.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        ctx = b.new_context(viewport={"width": 1400, "height": 900}, accept_downloads=True)
        page = ctx.new_page()
        page.goto("http://127.0.0.1:8765/")
        if a.phase == "add":
            page.locator("#q").fill(a.query)
            page.get_by_role("button", name="Search").click()
            cells = page.locator("li.cell")
            expect(cells.first).to_be_visible(timeout=120_000)
            titles = []
            for i in range(4):
                cells.nth(i).hover()
                cells.nth(i).locator(".cell-add").click()
                titles.append(cells.nth(i).get_attribute("title"))
            expect(page.locator(".cell-add.added")).to_have_count(4)
            print("added:", titles)
            cells.nth(4).focus()
            page.keyboard.press("a")  # keyboard add
            expect(page.locator(".cell-add.added")).to_have_count(5)
            page.screenshot(path=str(a.shots / "10-added.png"))
            page.locator(".selections").get_by_role("button", name="Open").first.click()
            items = page.locator(".sel-item")
            expect(items).to_have_count(5)
            before = [items.nth(i).locator(".sel-path").inner_text() for i in range(5)]
            items.nth(1).get_by_role("button", name="Move up").click()
            expect(items.nth(0).locator(".sel-path")).to_have_text(before[1])
            items.nth(4).get_by_role("button", name="Remove").click()
            expect(items).to_have_count(4)
            print("order after move-up/remove:", [items.nth(i).locator(".sel-path").inner_text() for i in range(4)])
            with page.expect_download() as dl:
                page.get_by_role("link", name="Download manifest").click()
            man = json.loads(Path(dl.value.path()).read_text())
            print("manifest:", man["format"], len(man["items"]), "items; first source_path:", man["items"][0]["source_path"])
            print("manifest query_context:", man["items"][0]["query_context"])
            page.get_by_role("button", name="Copy files to folder…").click()
            page.get_by_label("Destination folder").fill("/")
            page.get_by_role("button", name="Preview").click()
            expect(page.locator(".export-panel .error")).to_be_visible()
            print("root dest rejected:", page.locator(".export-panel .error").inner_text())
            page.get_by_label("Destination folder").fill(a.dest)
            page.get_by_role("button", name="Preview").click()
            expect(page.locator(".plan")).to_be_visible()
            print("plan:", page.locator(".plan p").first.inner_text())
            print("plan files:", page.locator(".plan-files").inner_text().replace("\n", " | "))
            page.screenshot(path=str(a.shots / "11-export-preview.png"))
            page.get_by_role("button", name=re.compile(r"Copy \d+ files")).click()
            expect(page.locator(".export-panel .banner")).to_be_visible(timeout=60_000)
            print("export result:", page.locator(".export-panel .banner").inner_text())
            page.screenshot(path=str(a.shots / "12-exported.png"))
        else:
            page.locator(".selections").get_by_role("button", name="Open").first.click()
            items = page.locator(".sel-item")
            expect(items.first).to_be_visible()
            print("after restart items:", [items.nth(i).locator(".sel-path").inner_text() for i in range(items.count())])
            print("status badges:", page.locator(".sel-item .badge.warn").all_inner_texts())
            page.screenshot(path=str(a.shots / "13-after-restart.png"))
        b.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
