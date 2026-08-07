"""Headless smoke test: run every dashboard page and fail on any exception.

Uses Streamlit's own ``AppTest`` harness, so the pages execute exactly as they
do in the browser — including the sidebar filters, caching and page routing —
without needing a display.

Run with::

    python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

PAGES = [
    "app_pages/overview.py",
    "app_pages/product_analysis.py",
    "app_pages/price_forecast.py",
    "app_pages/inflation_analysis.py",
    "app_pages/model_comparison.py",
    "app_pages/insights.py",
    "app_pages/downloads.py",
]


def run_page(page: str, **session_state: object) -> tuple[bool, list[str]]:
    """Execute one page and return ``(ok, messages)``."""
    app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=180)
    for key, value in session_state.items():
        app.session_state[key] = value
    app.run()
    if page != PAGES[0]:
        app.switch_page(page)
        app.run()

    problems = [f"EXCEPTION: {e.value}" for e in app.exception]
    problems += [f"ERROR: {e.value}" for e in app.error]
    return not problems, problems


def main() -> int:
    """Run every page under a few filter combinations."""
    scenarios: list[dict[str, object]] = [
        {},
        {"filter_product": "Potato", "filter_horizon": 7, "filter_theme": "Dark"},
        {"filter_product": "Soybean Oil", "filter_horizon": 60, "filter_model": "XGBoost"},
    ]

    failures = 0
    for index, state in enumerate(scenarios, start=1):
        label = ", ".join(f"{k}={v}" for k, v in state.items()) or "defaults"
        print(f"\n--- scenario {index}: {label} ---")
        for page in PAGES:
            ok, problems = run_page(page, **state)
            status = "ok " if ok else "FAIL"
            print(f"  [{status}] {page}")
            for problem in problems:
                failures += 1
                print(f"        {problem.splitlines()[0][:220]}")

    print("\nAll pages rendered cleanly." if not failures else f"\n{failures} problem(s) found.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
