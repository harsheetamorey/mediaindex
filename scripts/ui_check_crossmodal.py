"""Browser check for the cross-modal workspace (Playwright + installed Chrome, running server).

Usage: python scripts/ui_check_crossmodal.py --shots DIR
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=Path, required=True)
    a = ap.parse_args()
    a.shots.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        page = b.new_page(viewport={"width": 1400, "height": 1000})
        page.goto("http://127.0.0.1:8765/")
        page.get_by_role("radio", name="Both").click()
        page.locator("#q").fill("a train")
        page.get_by_role("button", name="Search").click()
        expect(page.locator(".section-title").first).to_be_visible(timeout=120_000)
        print("sections:", page.locator(".section-title").all_inner_texts())
        print("first image:", page.get_by_role("listbox", name="Images").locator("li.cell").first.get_attribute("title"),
              "| first sound:", page.get_by_role("listbox", name="Sounds").locator("li.cell").first.get_attribute("title"))
        page.screenshot(path=str(a.shots / "30-both.png"))

        # image result -> use as reference -> target Sounds (experimental)
        page.get_by_role("listbox", name="Images").locator("li.cell").first.click()
        page.get_by_role("dialog").get_by_role("button", name="Use as reference").click()
        page.get_by_role("radio", name="Sounds").click()
        page.locator("#q").fill("")
        print("mode:", page.locator("label.mode").inner_text().replace("\n", " "))
        expect(page.locator("label.mode .badge.exp")).to_be_visible()
        with page.expect_response(lambda r: "/api/search/image" in r.url, timeout=120_000) as resp:
            page.get_by_role("button", name="Search").click()
        body = resp.value.json()
        print("image→sounds:", body["mode"], body["query"].get("mode_status"), "first:", body["results"][0]["asset"]["rel_path"])
        expect(page.locator(".banner.warn", has_text="Experimental cross-media")).to_be_visible()
        page.screenshot(path=str(a.shots / "31-image-to-sound.png"))

        # sound result -> reference -> target Images
        page.locator("li.cell").first.click()
        page.get_by_role("dialog").get_by_role("button", name="Use as reference").click()
        page.get_by_role("radio", name="Images").click()
        with page.expect_response(lambda r: "/api/search/audio" in r.url, timeout=120_000) as resp:
            page.get_by_role("button", name="Search").click()
        body = resp.value.json()
        print("sound→images:", body["mode"], body["query"].get("mode_status"), "first:", body["results"][0]["asset"]["rel_path"])
        page.screenshot(path=str(a.shots / "32-sound-to-image.png"))

        # global ranking toggle (technical) labelled experimental
        page.get_by_role("button", name="Remove reference").click()
        page.get_by_role("radio", name="Both").click()
        page.get_by_label("Show technical scores").check()
        page.get_by_label("Mixed ranking (experimental)").check()
        page.locator("#q").fill("a train")
        with page.expect_response(lambda r: "/api/search/text" in r.url, timeout=120_000) as resp:
            page.get_by_role("button", name="Search").click()
        print("global:", resp.value.json()["grouping"], "| banner:",
              page.locator(".banner.warn", has_text="mixed ranking").first.inner_text()[:80])
        page.screenshot(path=str(a.shots / "33-global.png"))
        b.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
