#!/usr/bin/env python3
"""Render docs/task3_report.html to the Task 3 PDF with headless Chrome.

    py scripts/make_task3_report.py

The report text lives in docs/task3_report.html (written by hand, numbers
taken from reports/task3/). Figures are read from docs/ and reports/task3/figures/.
The PDF is written to docs/ (next to the Task 1 report) and copied to the
repo root and "task /" (folder name has a trailing space) for submission. The brief allows max 5 pages, so
the page count is checked.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "docs" / "task3_report.html"
PDF = ROOT / "docs" / "Team BORNEO YUK - Task 3 Model Refinement Multimodal Integration.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def main() -> int:
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--allow-file-access-from-files", f"--print-to-pdf={PDF}", HTML.as_uri()],
                   check=True, capture_output=True)
    import pymupdf
    n = len(pymupdf.open(PDF))
    print(f"wrote {PDF.name}: {n} pages")
    for d in (ROOT, ROOT / "task "):          # local copies for submission (not tracked)
        if d.exists():
            shutil.copy(PDF, d / PDF.name)
    if n > 5:
        print("WARNING: the brief allows max 5 pages", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
