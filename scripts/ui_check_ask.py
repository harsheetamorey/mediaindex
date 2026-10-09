"""Browser check for the Ask tab (Playwright + installed Chrome, against a running server with real models).

Usage: python scripts/ui_check_ask.py --shots DIR
Asks a count, a search, a follow-up and a describe question, and clicks a photo; saves screenshots.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=Path, required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8765/")
    a = ap.parse_args()
    a.shots.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        page = b.new_context(viewport={"width": 1400, "height": 900}).new_page()
        page.goto(a.url)
        page.get_by_role("button", name="Ask", exact=True).first.click()
        expect(page.locator(".ask-status")).to_contain_text("photos checked", timeout=30_000)
        print("status:", page.locator(".ask-status").inner_text().replace("\n", " | "))
        page.screenshot(path=str(a.shots / "ask-0-empty.png"))
        answers = page.locator(".msg.assistant:not(:has-text('Thinking…'))")
        for i, q in enumerate(["How many dogs do I have?", "show me a city street at night", "which of those have cars?",
                               "describe the first one"]):
            t = time.time()
            page.get_by_label("Ask about your photos").fill(q)
            page.get_by_role("button", name="Ask", exact=True).last.click()
            expect(answers).to_have_count(i + 1, timeout=180_000)
            last = answers.nth(i)
            print(f"Q: {q} ({time.time() - t:.1f}s)\n  A: {last.locator('.bubble').inner_text()}\n"
                  f"  photos: {last.locator('.msg-photo').count()}")
            page.screenshot(path=str(a.shots / f"ask-{i + 1}.png"))
        t = time.time()
        answers.nth(1).locator(".msg-photo").nth(1).click()  # ask about a photo by clicking it
        expect(answers).to_have_count(5, timeout=180_000)
        print(f"clicked photo 2 ({time.time() - t:.1f}s)\n  A: {answers.nth(4).locator('.bubble').inner_text()}")
        page.screenshot(path=str(a.shots / "ask-5-clicked.png"), full_page=True)
        # Doubting an answer: the evidence (boxes) is shown instead of a guess.
        page.get_by_role("button", name="New chat").click()
        answers = page.locator(".msg.assistant:not(:has-text('Thinking…'))")
        for i, q in enumerate(["How many dogs do I have?", "describe the third one", "then why did you say it's a dog?"]):
            page.get_by_label("Ask about your photos").fill(q)
            page.get_by_role("button", name="Ask", exact=True).last.click()
            expect(answers).to_have_count(i + 1, timeout=180_000)
            last = answers.nth(i)
            print(f"Q: {q}\n  A: {last.locator('.bubble').inner_text()}\n  boxes drawn: {last.locator('.det-box').count()}")
        page.screenshot(path=str(a.shots / "ask-6-evidence.png"), full_page=True)
        b.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
