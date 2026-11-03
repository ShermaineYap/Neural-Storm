"""Capture README screenshots of the running app (streamlit run app.py on :8501).

Usage: python docs/capture.py [base_url]
"""
import sys
import time

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
OUT = "docs/img"


def wait_ready(page):
    page.wait_for_selector("text=HazeAgent bulletin", timeout=180_000)
    time.sleep(8)


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=2)
        page.goto(BASE)
        wait_ready(page)
        page.screenshot(path=f"{OUT}/map.png")
        page.get_by_role("tab", name="Station detail").click()
        page.wait_for_selector("text=Why this forecast", timeout=120_000)
        time.sleep(5)
        page.screenshot(path=f"{OUT}/station.png")
        page.get_by_role("tab", name="Model evidence").click()
        page.wait_for_selector("text=Walk-forward results", timeout=60_000)
        time.sleep(4)
        page.screenshot(path=f"{OUT}/evidence.png", full_page=True)
        b.close()


if __name__ == "__main__":
    main()
