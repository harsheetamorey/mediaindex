"""Browser check for sound search + segment playback (Playwright + installed Chrome, running server).

Usage: python scripts/ui_check_audio.py --shots DIR [--query "a dog barking"] [--upload file.mp3]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=Path, required=True)
    ap.add_argument("--query", default="a dog barking")
    ap.add_argument("--upload")
    a = ap.parse_args()
    a.shots.mkdir(parents=True, exist_ok=True)
    external = []
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True, args=["--autoplay-policy=no-user-gesture-required"])
        page = b.new_page(viewport={"width": 1400, "height": 900})
        page.on("request", lambda r: external.append(r.url) if not r.url.startswith(("http://127.0.0.1", "blob:", "data:")) else None)
        page.goto("http://127.0.0.1:8765/")
        page.get_by_role("radio", name="Sounds").click()
        page.locator("#q").fill(a.query)
        page.get_by_role("button", name="Search").click()
        cells = page.locator("li.cell")
        expect(cells.first).to_be_visible(timeout=120_000)
        first = cells.first
        span = first.locator(".cell-span").inner_text()
        print(f"sound search {a.query!r}: {cells.count()} results; first {first.get_attribute('title')} span {span}")
        page.screenshot(path=str(a.shots / "20-sound-results.png"))

        # play the medley result if present (has a non-zero start), else the first
        target = cells.filter(has_text="medley.wav").first if cells.filter(has_text="medley.wav").count() else first
        tspan = target.locator(".cell-span").inner_text()
        target.hover()
        target.locator(".cell-play").click()
        expect(target.locator(".cell-play.on")).to_be_visible()
        page.wait_for_timeout(1500)
        player = page.locator("#mi-segment-player")
        t = player.evaluate("a => [a.currentTime, a.paused, a.src.split('/api/')[1]]")
        print(f"playing {target.get_attribute('title')} window {tspan}; player currentTime={t[0]:.2f}s paused={t[1]} src=/api/{t[2]}")
        start = float(tspan.split("–")[0].split(":")[0]) * 60 + float(tspan.split("–")[0].split(":")[1])
        assert start <= t[0] < start + 10, "playback did not start inside the matched window"
        target.locator(".cell-play").click()  # stop
        expect(target.locator(".cell-play.on")).to_have_count(0)

        target.click()
        dialog = page.get_by_role("dialog")
        expect(dialog).to_be_visible()
        print("detail:", dialog.locator("dl").inner_text().replace("\n", " | ")[:220])
        audio = dialog.locator("audio")
        assert audio.get_attribute("src").endswith("/file")
        dialog.get_by_role("button", name="0:20–0:30").click()
        page.wait_for_timeout(1200)
        t = page.locator("#mi-segment-player").evaluate("a => [a.currentTime, a.paused]")
        print(f"other-window button 0:20–0:30 -> currentTime={t[0]:.2f}s paused={t[1]}")
        assert 20 <= t[0] < 30
        dialog.get_by_role("button", name="0:20–0:30").click()
        page.screenshot(path=str(a.shots / "21-sound-detail.png"))
        dialog.get_by_role("button", name="Use as reference").click()
        expect(page.locator(".ref-audio")).to_be_visible()
        print("reference label:", page.locator(".ref-audio-label").inner_text(), "| mode:", page.locator("label.mode").inner_text())
        assert page.locator("#q").is_disabled()
        with page.expect_response(lambda r: "/api/search/audio" in r.url, timeout=120_000) as resp:
            page.get_by_role("button", name="Search").click()
        body = resp.value.json()
        print("audio-reference search:", resp.value.status, body["query"].get("reference_span"),
              "first:", body["results"][0]["asset"]["rel_path"], body["results"][0]["start_s"], body["results"][0]["end_s"])
        page.wait_for_timeout(400)
        page.screenshot(path=str(a.shots / "22-sound-reference.png"))
        if a.upload:
            page.get_by_role("button", name="Remove reference").click()
            page.locator('input[type="file"]').set_input_files(a.upload)
            with page.expect_response(lambda r: "/api/search/audio" in r.url, timeout=120_000) as resp:
                page.get_by_role("button", name="Search").click()
            body = resp.value.json()
            print("upload reference:", resp.value.status, "first:", body["results"][0]["asset"]["rel_path"])
        page.get_by_role("radio", name="Images").click() if not page.locator(".ref-audio").count() else None
        b.close()
    print("external requests:", external or "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
