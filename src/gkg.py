"""NEWS (bulk route): GDELT 2.0 Global Knowledge Graph files instead of the DOC API.

Why this module exists
----------------------
The DOC 2.0 API (``gdelt.py``) allows one request every five seconds per IP,
and exceeding that earns an IP-level block that outlasts the burst by many
hours. At ~3,500 days x 6 theme families the API route needs ~21,000 requests;
in September 2026 it got this project's IP blocked for over a day.

GDELT publishes the same index as static 15-minute GKG files on its bulk file
server, which GDELT itself recommends for heavy use and which is not behind the
API quota. Each file lists every English-language article GDELT processed in
that 15 minutes, with URL, source domain, theme codes, tone and mentioned
locations. Retrieval therefore stays theme-driven (report §4.2): the same
``theme_families`` codes select the same kind of article.

What changes relative to the DOC API, and why it is acceptable
--------------------------------------------------------------
* **Headlines only from ~late 2019.** GDELT began storing ``<PAGE_TITLE>`` in
  the GKG Extras field between Sept 2019 and March 2020. Before that we fall
  back to the words of the URL slug ("/china-slaps-tariffs-on-us-goods.html"
  -> "china slaps tariffs on us goods") and record which was used in
  ``title_source``, so Task 2 can ablate or drop the slug era.
* **Time-of-day sampling instead of the 250-record ceiling.** All 96 files a
  day would be ~1 GB/day, so we take ``files_per_day`` evenly spaced slices.
  Within a slice we see *every* matching article, and the per-family daily cap
  is a seeded random draw — not the DOC API's "most recent 250", which biased
  every capped day toward late-evening coverage.
* **``sourcecountry`` holds the most-mentioned location country (FIPS 10-4),
  not the outlet's country**, because GKG does not record where an outlet is
  based. The Stage-2 gate asks whether an article touches a pivotal state,
  which a location code measures directly, and ``actor_countries`` in the
  config are FIPS codes already. (The DOC API returned full country names,
  which never matched those codes at all.)
* **Timelines come from the same files.** The DOC timeline endpoint sits on the
  blocked API, so daily per-family volume (articles per retrieved slice, counted
  *before* the cap) and mean tone are computed here.
"""
from __future__ import annotations

import gzip
import html
import io
import json
import logging
import os
import random
import threading
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlparse

import pandas as pd
import requests

from .gdelt import GDELT_EPOCH, REQUIRED_ARTICLE_COLS, normalize_article_frame
from .http_cache import USER_AGENT

log = logging.getLogger(__name__)

GKG_BASE = "http://data.gdeltproject.org/gdeltv2"
SLOT_MINUTES = 15            # GKG files are published on a 15-minute grid
MIN_SLUG_WORDS = 4           # fewer words than this is an ID or a section name, not a headline

# GKG 2.1 column positions (tab-separated, no header row).
COL_DATE, COL_DOMAIN, COL_URL = 1, 3, 4
COL_V1THEMES, COL_V2THEMES = 7, 8
COL_V1LOCATIONS = 9
COL_TONE = 15
COL_EXTRAS = 26
N_COLS = 27

_PAGE_TITLE_OPEN, _PAGE_TITLE_CLOSE = "<PAGE_TITLE>", "</PAGE_TITLE>"
_URL_EXTENSIONS = (".html", ".htm", ".shtml", ".php", ".aspx", ".asp", ".cms")


# ---------------------------------------------------------------------------
# Slot schedule
# ---------------------------------------------------------------------------
def slot_stamps(day, files_per_day: int, offset_minutes: int = 0) -> List[datetime]:
    """Evenly spaced 15-minute file stamps within one UTC day."""
    n = max(1, int(files_per_day))
    step = (24 * 60) // n
    base = pd.Timestamp(day).normalize().to_pydatetime()
    stamps = []
    for i in range(n):
        minutes = (int(offset_minutes) + i * step) % (24 * 60)
        minutes -= minutes % SLOT_MINUTES
        stamps.append(base + timedelta(minutes=minutes))
    return sorted(set(stamps))


def gkg_url(stamp: datetime, base: str = GKG_BASE) -> str:
    return f"{base}/{stamp:%Y%m%d%H%M%S}.gkg.csv.zip"


# ---------------------------------------------------------------------------
# Parsing one GKG file
# ---------------------------------------------------------------------------
def _theme_tokens(v1: str, v2: str) -> Set[str]:
    """Exact theme codes: V1 is ``A;B``, V2 is ``A,offset;B,offset``.

    Exact tokens, never substrings — ``ENV_OIL`` must not match ``ENV_OILSPILL``.
    """
    tokens = {t for t in v1.split(";") if t}
    tokens.update(t.split(",", 1)[0] for t in v2.split(";") if t)
    return tokens


def title_from_slug(url: str) -> Optional[str]:
    """Recover a headline-like string from the longest word-slug in a URL path."""
    try:
        path = urlparse(url).path.lower()
    except ValueError:
        return None
    best: List[str] = []
    for seg in (s for s in path.split("/") if s):
        for ext in _URL_EXTENSIONS:
            if seg.endswith(ext):
                seg = seg[: -len(ext)]
                break
        words = [w for w in seg.replace("_", "-").replace("+", "-").split("-") if w.isalpha()]
        if len(words) > len(best):
            best = words
    return " ".join(best) if len(best) >= MIN_SLUG_WORDS else None


def _page_title(extras: str) -> Optional[str]:
    start = extras.find(_PAGE_TITLE_OPEN)
    if start < 0:
        return None
    end = extras.find(_PAGE_TITLE_CLOSE, start)
    if end < 0:
        return None
    title = html.unescape(extras[start + len(_PAGE_TITLE_OPEN):end]).strip()
    return title or None


def _primary_country(v1_locations: str) -> Optional[str]:
    """Most-mentioned FIPS country code (``type#name#CC#adm1#lat#lon#id;...``)."""
    counts: Dict[str, int] = defaultdict(int)
    for loc in v1_locations.split(";"):
        fields = loc.split("#")
        if len(fields) > 2 and fields[2]:
            counts[fields[2]] += 1
    if not counts:
        return None
    return max(sorted(counts), key=lambda cc: counts[cc])


def _tone(field: str) -> Optional[float]:
    try:
        return float(field.split(",", 1)[0])
    except ValueError:
        return None


def parse_gkg(text: str, families: Dict[str, List[str]]) -> List[dict]:
    """One row per article that matches at least one theme family."""
    fam_sets = {fam: set(codes) for fam, codes in families.items()}
    wanted: Set[str] = set().union(*fam_sets.values()) if fam_sets else set()
    rows = []
    for line in text.split("\n"):
        # Cheap pre-check on the first nine columns; the full split is only
        # paid for the minority of lines that carry a wanted theme. (The GCAM
        # column alone is ~10 KB per line.)
        head = line.split("\t", COL_V2THEMES + 1)
        if len(head) < COL_V2THEMES + 2:
            continue
        tokens = _theme_tokens(head[COL_V1THEMES], head[COL_V2THEMES])
        if not tokens & wanted:
            continue
        cols = line.split("\t")
        if len(cols) < N_COLS:
            continue
        title = _page_title(cols[COL_EXTRAS])
        title_source = "page_title"
        if title is None:
            title, title_source = title_from_slug(cols[COL_URL]), "url_slug"
        rows.append({
            "date": cols[COL_DATE],
            "domain": cols[COL_DOMAIN],
            "url": cols[COL_URL],
            "title": title,
            "title_source": title_source if title else None,
            "families": [fam for fam, codes in fam_sets.items() if tokens & codes],
            "tone": _tone(cols[COL_TONE]),
            "country": _primary_country(cols[COL_V1LOCATIONS]),
        })
    return rows


# ---------------------------------------------------------------------------
# Downloading (cached, resumable)
# ---------------------------------------------------------------------------
def _read_json_gz(fp: Path) -> dict:
    with gzip.open(fp, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json_gz(fp: Path, payload: dict) -> None:
    """Temp file + replace, so an interrupted run never leaves a half-written
    entry that later reads as a valid cache hit."""
    tmp = fp.with_suffix(fp.suffix + f".tmp{os.getpid()}-{threading.get_ident()}")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    os.replace(tmp, fp)


class GKGFetcher:
    """Downloads GKG slices, keeps only theme-matched rows, caches the result.

    The cache stores the *filtered* rows (tens of KB per slice), not the raw
    5-14 MB zip, so a full census fits on a laptop and reruns are instant.
    Like the DOC cache, it freezes the corpus: a cached slice is the exact
    input any later run will see.
    """

    def __init__(self, cache_dir: Path, families: Dict[str, List[str]], *,
                 base: str = GKG_BASE, timeout: int = 120,
                 max_retries: int = 4, backoff: float = 2.0) -> None:
        self.dir = Path(cache_dir) / "gkg"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.families = families
        self.base = base
        self.timeout = timeout
        self.max_retries = max(1, int(max_retries))
        self.backoff = backoff
        self._local = threading.local()

    def _session(self) -> requests.Session:
        sess = getattr(self._local, "session", None)
        if sess is None:
            sess = requests.Session()
            sess.headers.update({"User-Agent": USER_AGENT})
            self._local.session = sess
        return sess

    def fetch_slot(self, stamp: datetime) -> Tuple[datetime, Optional[dict], str]:
        """Return ``(stamp, payload, status)``; status is hit | miss | missing | fail.

        ``payload`` is ``{"status": "ok"|"missing", "rows": [...]}``, or None on
        failure. Failures are never cached, so a rerun retries them.
        """
        fp = self.dir / f"{stamp:%Y%m%d%H%M%S}.json.gz"
        if fp.exists():
            try:
                return stamp, _read_json_gz(fp), "hit"
            except (OSError, ValueError):
                log.warning("unreadable cache entry %s; refetching", fp.name)

        delay, last_err = 2.0, ""
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._session().get(gkg_url(stamp, self.base), timeout=self.timeout)
                if resp.status_code == 200:
                    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                        text = zf.read(zf.namelist()[0]).decode("utf-8", "replace")
                    payload = {"status": "ok", "rows": parse_gkg(text, self.families)}
                    _write_json_gz(fp, payload)
                    return stamp, payload, "miss"
                if resp.status_code == 404:
                    # GDELT's archive has genuine gaps; a missing slice stays missing.
                    payload = {"status": "missing", "rows": []}
                    _write_json_gz(fp, payload)
                    return stamp, payload, "missing"
                last_err = f"HTTP {resp.status_code}"
            except (requests.RequestException, zipfile.BadZipFile, IndexError) as exc:
                last_err = type(exc).__name__
            if attempt < self.max_retries:
                time.sleep(delay)
                delay *= self.backoff
        log.error("giving up on GKG slice %s (%s)", f"{stamp:%Y%m%d%H%M%S}", last_err)
        return stamp, None, "fail"


# ---------------------------------------------------------------------------
# Per-day assembly: cap sampling + timelines
# ---------------------------------------------------------------------------
def _gdelt_seendate(stamp14: str) -> str:
    """GKG ``20240115120000`` -> DOC-API style ``20240115T120000Z``."""
    return f"{stamp14[:8]}T{stamp14[8:14]}Z"


def finalize_day(day, payloads: List[Tuple[datetime, dict]], families: Iterable[str],
                 cap: int, seed) -> Tuple[List[dict], List[tuple]]:
    """Turn one day's slices into capped article rows and timeline points.

    Volume is articles per *retrieved* slice, counted before the cap, so a day
    with a missing file is not mistaken for a quiet news day. The cap draw is
    seeded by (seed, day, family) and runs over slices in stamp order, so it is
    identical on every rerun regardless of download order.
    """
    day = pd.Timestamp(day).normalize()
    ok = [p for _, p in sorted(payloads, key=lambda sp: sp[0]) if p.get("status") == "ok"]
    n_ok = len(ok)
    by_family: Dict[str, List[dict]] = {fam: [] for fam in families}
    for payload in ok:
        for row in payload["rows"]:
            for fam in row["families"]:
                if fam in by_family:
                    by_family[fam].append(row)

    articles: List[dict] = []
    timeline: List[tuple] = []
    for fam, rows in by_family.items():
        if n_ok:
            timeline.append((day, len(rows) / n_ok, fam, "volume"))
            tones = [r["tone"] for r in rows if r["tone"] is not None]
            if tones:
                timeline.append((day, sum(tones) / len(tones), fam, "tone"))
        titled = [r for r in rows if r["title"]]
        if cap and len(titled) > cap:
            rng = random.Random(f"{seed}|{day:%Y%m%d}|{fam}")
            titled = rng.sample(titled, cap)
        for r in titled:
            articles.append({
                "url": r["url"],
                "title": r["title"],
                "seendate": _gdelt_seendate(r["date"]),
                "domain": r["domain"],
                "language": "English",
                "sourcecountry": r["country"],
                "theme_family": fam,
                "query_day": day.date().isoformat(),
                "title_source": r["title_source"],
                "tone": r["tone"],
            })
    return articles, timeline


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def fetch_articles_gkg(cfg, *, families: Optional[List[str]] = None,
                       stride_days: int = 1, progress: bool = True
                       ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Collect articles + timelines for the study period from GKG slices.

    Returns ``(articles, timeline_long)``. ``articles`` has the same schema as
    ``gdelt.fetch_articles`` plus ``title_source`` and ``tone``;
    ``timeline_long`` has the columns ``gdelt.timelines_to_wide`` expects.
    """
    gk = cfg.dig("gkg", default={}) or {}
    fams: Dict[str, List[str]] = cfg.dig("theme_families", default={}) or {}
    if families:
        fams = {k: v for k, v in fams.items() if k in families}
    fetcher = GKGFetcher(
        Path(cfg.path("cache")), fams,
        base=gk.get("base_url", GKG_BASE),
        timeout=int(gk.get("timeout_seconds", 120)),
        max_retries=int(gk.get("max_retries", 4)),
        backoff=float(gk.get("backoff_factor", 2.0)),
    )
    files_per_day = int(gk.get("files_per_day", 4))
    offset = int(gk.get("slot_offset_minutes", 0))
    cap = int(gk.get("per_family_daily_cap", 250))
    workers = max(1, int(gk.get("workers", 4)))
    seed = cfg.seed

    start = pd.Timestamp(max(str(cfg.dig("period", "start")), GDELT_EPOCH))
    end = pd.Timestamp(str(cfg.dig("period", "end")))
    days = pd.date_range(start, end, freq=f"{max(1, int(stride_days))}D")
    schedule = {day: slot_stamps(day, files_per_day, offset) for day in days}
    total = sum(len(s) for s in schedule.values())
    log.info("GKG: %d days x %d slices = %d files, %d parallel downloads",
             len(days), files_per_day, total, workers)

    pending: Dict[pd.Timestamp, List[Tuple[datetime, dict]]] = defaultdict(list)
    remaining = {day: len(stamps) for day, stamps in schedule.items()}
    status_counts: Dict[str, int] = defaultdict(int)
    incomplete_days: Set[pd.Timestamp] = set()
    articles: List[dict] = []
    timeline: List[tuple] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetcher.fetch_slot, stamp): day
                   for day, stamps in schedule.items() for stamp in stamps}
        done = as_completed(futures)
        if progress:
            try:
                from tqdm.auto import tqdm
                done = tqdm(done, total=total, desc="GKG files", unit="file")
            except ImportError:
                pass
        for fut in done:
            day = futures[fut]
            stamp, payload, status = fut.result()
            status_counts[status] += 1
            if payload is None:
                incomplete_days.add(day)
            else:
                pending[day].append((stamp, payload))
            remaining[day] -= 1
            if remaining[day] == 0:
                # Finalise as soon as a day is complete, so memory holds only
                # the days still in flight rather than the whole census.
                day_articles, day_timeline = finalize_day(
                    day, pending.pop(day, []), fams.keys(), cap, seed)
                articles.extend(day_articles)
                timeline.extend(day_timeline)

    log.info("GKG files: %s", ", ".join(f"{k}={v}" for k, v in sorted(status_counts.items())))
    if incomplete_days:
        log.warning(
            "%d slices failed after every retry, leaving %d days partial "
            "(their volume is per retrieved slice, so it is not deflated). "
            "Re-run the same command to fetch them; failures are not cached.",
            status_counts["fail"], len(incomplete_days))

    timeline_long = pd.DataFrame(timeline, columns=["date", "value", "theme_family", "metric"])
    if not articles:
        log.warning("GKG returned no articles — check access to %s", fetcher.base)
        return pd.DataFrame(columns=REQUIRED_ARTICLE_COLS), timeline_long

    df = normalize_article_frame(pd.DataFrame(articles))
    share_slug = (df["title_source"] == "url_slug").mean()
    log.info("GKG: %d article rows, %.0f%% titled from the URL slug", len(df), 100 * share_slug)
    return df, timeline_long


__all__ = [
    "fetch_articles_gkg", "finalize_day", "parse_gkg", "slot_stamps",
    "title_from_slug", "gkg_url", "GKGFetcher", "GKG_BASE",
]
