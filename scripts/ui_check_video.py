"""Browser check for video moments: search, seek-to-moment playback, stop at window end, other moments,
sound suggestions and clip export (Playwright + installed Chrome, running server).

Usage: python scripts/ui_check_video.py --shots DIR --dest "/abs/clip folder"
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=Path, required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--query", default="a dog")
    a = ap.parse_args()
    a.shots.mkdir(parents=True, exist_ok=True)
    external = []
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True, args=["--autoplay-policy=no-user-gesture-required"])
        page = b.new_page(viewport={"width": 1400, "height": 1000})
        page.on("request", lambda r: external.append(r.url) if not r.url.startswith(("http://127.0.0.1", "blob:", "data:")) else None)
        page.goto("http://127.0.0.1:8765/")
        page.get_by_role("radio", name="Videos").click()
        page.locator("#q").fill(a.query)
        page.get_by_role("button", name="Search").click()
        cells = page.locator("li.cell")
        expect(cells.first).to_be_visible(timeout=120_000)
        print("video results:", [(cells.nth(i).get_attribute("title"), cells.nth(i).locator(".cell-span").inner_text())
                                 for i in range(cells.count())])
        page.screenshot(path=str(a.shots / "40-video-results.png"))
        target = cells.filter(has_text="scenes.mp4").first
        span = target.locator(".cell-span").inner_text().split(" of ")[0]
        target.click()
        dialog = page.get_by_role("dialog")
        expect(dialog.locator("#mi-video-player")).to_be_visible()
        v = dialog.locator("#mi-video-player")
        page.wait_for_timeout(1500)
        print("player seeked to", round(v.evaluate("x => x.currentTime"), 2), "for window", span)
        dialog.get_by_role("button", name=re.compile("Play matched moment")).click()
        page.wait_for_timeout(1200)
        t, paused = v.evaluate("x => [x.currentTime, x.paused]")
        print(f"playing: currentTime={t:.2f} paused={paused}")
        for _ in range(40):  # wait (max ~20 s) for the player to stop itself at the window end
            page.wait_for_timeout(500)
            t, paused = v.evaluate("x => [x.currentTime, x.paused]")
            if paused:
                break
        print(f"stopped by itself: currentTime={t:.2f} paused={paused}")
        page.screenshot(path=str(a.shots / "41-video-moment.png"))
        moments = dialog.locator("button.moment")
        if moments.count():
            label = moments.first.inner_text()
            moments.first.click()
            page.wait_for_timeout(1000)
            print(f"other moment {label!r}: currentTime={v.evaluate('x => x.currentTime'):.2f}")
            v.evaluate("x => x.pause()")

        # clip export from the detail panel
        dialog.get_by_role("button", name="Export clip…").click()
        dialog.get_by_label("Clip destination folder").fill(a.dest)
        dialog.get_by_role("button", name="Preview export").click()
        expect(dialog.locator(".plan")).to_be_visible()
        print("clip plan:", dialog.locator(".plan p").first.inner_text())
        dialog.get_by_role("button", name="Export clip", exact=True).click()
        expect(dialog.locator(".banner.ok")).to_be_visible(timeout=120_000)
        print("clip result:", dialog.locator(".banner.ok").inner_text())
        page.screenshot(path=str(a.shots / "42-clip-exported.png"))

        # sound suggestions for the moment (experimental)
        dialog.get_by_role("button", name=re.compile("Find sounds for this moment")).click()
        expect(page.locator(".banner.warn", has_text="Experimental cross-media")).to_be_visible(timeout=60_000)
        print("suggested sounds:", [cells.nth(i).get_attribute("title") for i in range(min(3, cells.count()))])
        page.screenshot(path=str(a.shots / "43-sound-suggestions.png"))
        b.close()
    print("external requests:", external or "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
