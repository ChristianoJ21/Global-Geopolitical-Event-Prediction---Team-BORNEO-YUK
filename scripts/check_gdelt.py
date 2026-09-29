"""Is GDELT willing to talk to us right now?

Run this BEFORE starting a long fetch, and after any run that ended in a 429
storm::

    py scripts/check_gdelt.py

Why it exists
-------------
GDELT rate-limits per IP ("one request every 5 seconds"), and exceeding that
does not merely fail the offending request: the IP enters an extended penalty
window in which *every* request is refused, whatever the spacing. A build
started inside that window makes no progress but looks busy — the progress bar
advances, warnings scroll, and hours disappear.

One cheap request answers the only question that matters before committing to
a multi-hour fetch: is the door open?

This script deliberately issues a single request and does not retry. Retrying
is what dug the hole.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import requests  # noqa: E402

from src.config import load_config  # noqa: E402
from src.gdelt import DOC_API, build_theme_query  # noqa: E402
from src.http_cache import USER_AGENT  # noqa: E402


def check(cfg) -> int:
    """Return 0 if GDELT answered, 1 if throttled, 2 if unreachable."""
    families = cfg.dig("theme_families", default={}) or {}
    family, themes = next(iter(families.items()))
    query = build_theme_query(themes, cfg.dig("gdelt", "languages", default=["english"]))

    sess = requests.Session()
    sess.headers.update({"User-Agent": USER_AGENT})
    params = {
        "query": query, "mode": "artlist", "maxrecords": 250, "format": "json",
        "sort": "datedesc",
        # A fixed historical day: the answer is stable, so repeated checks are
        # comparable and cost GDELT almost nothing.
        "startdatetime": "20210715000000", "enddatetime": "20210715235959",
    }

    started = time.perf_counter()
    try:
        resp = sess.get(DOC_API, params=params, timeout=60)
    except requests.RequestException as exc:
        print(f"UNREACHABLE  ({type(exc).__name__}) — check your network connection.")
        return 2
    elapsed = time.perf_counter() - started

    if resp.status_code == 200:
        try:
            n = len(resp.json().get("articles", []))
        except ValueError:
            n = -1
        print(f"CLEAR        HTTP 200 in {elapsed:.1f}s, {n} articles (family: {family})")
        print()
        print("Safe to start. Run ONE fetch at a time:")
        print("    py -m src.build_dataset --stride 30 --skip-timelines")
        return 0

    if resp.status_code in (429, 503):
        print(f"THROTTLED    HTTP {resp.status_code} in {elapsed:.1f}s")
        print()
        print("The IP is inside GDELT's penalty window. Do NOT start a fetch —")
        print("it will make no progress and lengthen the penalty. Wait, then re-run")
        print("this check; do not poll more than once every few minutes.")
        print()
        print("GDELT's own guidance for heavy use, quoted from the 429 body:")
        print("  - contact kalev.leetaru5@gmail.com for larger queries")
        print("  - or use the bulk ngrams dataset instead of the DOC API:")
        print("    https://blog.gdeltproject.org/using-the-new-web-ngrams-dataset-to-find-relevant-coverage/")
        return 1

    print(f"UNEXPECTED   HTTP {resp.status_code} in {elapsed:.1f}s")
    print(resp.text[:300])
    return 2


if __name__ == "__main__":
    raise SystemExit(check(load_config(ROOT / "config" / "config.yaml")))
