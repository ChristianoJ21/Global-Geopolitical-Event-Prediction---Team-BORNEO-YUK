"""GDELT 2.0 acquisition: article headlines + daily theme volume/tone timelines.

Why GDELT is our backbone (report §3.2)
---------------------------------------
We evaluated five candidate corpora. GDELT won on the three axes that actually
constrain this project:

  * **History.** The DOC 2.0 full-text index starts 2017-01-01 and is free.
    NewsAPI's free tier gives one month; Kaggle dumps are frozen snapshots with
    no recency. Neither can span multiple macro regimes.
  * **Machine-coded structure.** GDELT ships a fixed *theme taxonomy* and a
    *tone* score per article. Querying by ``theme:ECON_SANCTIONS`` instead of
    the string ``"sanctions"`` protects us from vocabulary drift: our filter
    means the same thing in 2017 and 2026. Free-text keyword lists silently
    make a decade-long corpus non-stationary.
  * **Cost and reproducibility.** No key, no quota that expires mid-semester,
    no per-seat licence. Every team member can rerun the fetch.

What GDELT does *not* give us, and how we cope:
  * Only headline + URL, never full body text (copyright). We therefore design
    the whole NLP stack around **short text**: max 128 tokens, headline-tuned
    sentiment models, no long-document summarisation. This constraint is not a
    workaround, it is a scoping decision — and it keeps us legally clean,
    because we redistribute only metadata plus links.
  * An English-source skew and a documented crawler expansion around 2018 that
    inflates raw article counts. Both are handled downstream by using
    **relative** features (z-scores, shares) rather than raw volumes — see
    ``features.py`` and report §8.2.

Two layers are fetched here
---------------------------
``fetch_articles``   -> one row per article: the text the models will read.
``fetch_timelines``  -> daily volume & tone per theme family, one HTTP request
                        per family for the *entire* span. Cheap, dense, and it
                        is not subject to the 250-article-per-call ceiling, so
                        it gives us an unbiased volume measure to calibrate the
                        (capped) article-level counts against.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .http_cache import CachedSession

log = logging.getLogger(__name__)

DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
MAX_RECORDS = 250          # hard ceiling imposed by the API
GDELT_EPOCH = "2017-01-01"  # DOC 2.0 full-text index start


# ---------------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------------
def build_theme_query(themes: Iterable[str], languages: Iterable[str] = ("english",)) -> str:
    """Compose a GDELT DOC query from theme codes and a language restriction.

    Produces e.g.::

        (theme:ECON_SANCTIONS OR theme:ECON_TARIFF) sourcelang:english

    Note the deliberate absence of free-text terms: retrieval is theme-driven
    (Stage 1, recall-oriented). Free-text market terms are applied *later*, in
    the Stage-2 materiality gate, where we can tune precision without
    re-downloading anything.
    """
    themes = [t.strip() for t in themes if t and t.strip()]
    if not themes:
        raise ValueError("build_theme_query called with no themes")
    theme_clause = " OR ".join(f"theme:{t}" for t in themes)
    q = f"({theme_clause})"
    langs = list(languages)
    if len(langs) == 1:
        q += f" sourcelang:{langs[0]}"
    elif len(langs) > 1:
        q += " (" + " OR ".join(f"sourcelang:{l}" for l in langs) + ")"
    return q


def _stamp(d: datetime) -> str:
    return d.strftime("%Y%m%d%H%M%S")


# ---------------------------------------------------------------------------
# Layer 1 — article list
# ---------------------------------------------------------------------------
def fetch_articles_window(
    sess: CachedSession,
    query: str,
    start: datetime,
    end: datetime,
    *,
    maxrecords: int = MAX_RECORDS,
    sort: str = "datedesc",
) -> List[dict]:
    """One DOC-API artlist call for a [start, end) window."""
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": min(int(maxrecords), MAX_RECORDS),
        "format": "json",
        "sort": sort,
        "startdatetime": _stamp(start),
        "enddatetime": _stamp(end),
    }
    data = sess.get_json(DOC_API, params, namespace="gdelt_doc")
    if not data:
        return []
    return data.get("articles", []) or []


def fetch_articles(
    cfg,
    *,
    session: Optional[CachedSession] = None,
    families: Optional[List[str]] = None,
    stride_days: int = 1,
    sub_day_chunks: int = 1,
    progress: bool = True,
) -> pd.DataFrame:
    """Iterate the study period and collect article metadata.

    Parameters
    ----------
    families : restrict to a subset of theme families (default: all).
    stride_days : sample every N-th day. ``1`` = full census. Use 3 or 7 for a
        fast development run; the pipeline is identical, only denser/sparser.
    sub_day_chunks : split each day into N equal windows, each with its own
        250-record budget. **This is the lever that relaxes the API ceiling.**
        With ``1`` we get at most 250 articles per family per day, which on a
        very busy news day is a truncated (and therefore biased) sample.
        Raising it to 4 gives up to 1,000/day at 4x the request cost.

    Design note on *per-family* iteration
    -------------------------------------
    We deliberately issue one request per theme family rather than one big
    union query. A union query hits the 250 ceiling and returns whatever the
    dominant family of the day happens to be, so on any day with heavy conflict
    coverage our monetary-policy articles silently vanish. Per-family iteration
    gives each family its own quota, producing a *stratified* corpus whose
    family composition is a design choice rather than an artefact of the cap.
    The cost is ~6x the requests; the cache makes that a one-time expense.
    """
    sess = session or CachedSession(
        Path(cfg.path("cache")),
        sleep=cfg.dig("gdelt", "sleep_seconds", default=6.0),
        max_retries=cfg.dig("gdelt", "max_retries", default=6),
        backoff=cfg.dig("gdelt", "backoff_factor", default=3.0),
        timeout=cfg.dig("gdelt", "timeout_seconds", default=60),
    )
    fams: Dict[str, List[str]] = cfg.dig("theme_families", default={}) or {}
    if families:
        fams = {k: v for k, v in fams.items() if k in families}
    langs = cfg.dig("gdelt", "languages", default=["english"])

    start = pd.Timestamp(max(str(cfg.dig("period", "start")), GDELT_EPOCH))
    end = pd.Timestamp(str(cfg.dig("period", "end")))
    days = pd.date_range(start, end, freq=f"{max(1, int(stride_days))}D")

    iterator: Iterable = days
    if progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(days, desc="GDELT days", unit="d")
        except ImportError:
            pass

    rows: List[dict] = []
    chunk_hours = 24 / max(1, int(sub_day_chunks))

    for day in iterator:
        for fam, themes in fams.items():
            query = build_theme_query(themes, langs)
            for c in range(int(sub_day_chunks)):
                w0 = day.to_pydatetime() + timedelta(hours=c * chunk_hours)
                w1 = w0 + timedelta(hours=chunk_hours)
                for art in fetch_articles_window(sess, query, w0, w1):
                    art = dict(art)
                    art["theme_family"] = fam
                    art["query_day"] = day.date().isoformat()
                    rows.append(art)

    if not rows:
        log.warning("fetch_articles returned nothing — check network access to GDELT")
        return pd.DataFrame(columns=REQUIRED_ARTICLE_COLS)

    df = pd.DataFrame(rows)
    df = normalize_article_frame(df)
    log.info("fetched %d article rows (%s)", len(df), sess.report())
    return df


REQUIRED_ARTICLE_COLS = [
    "url", "title", "seendate", "domain", "language", "sourcecountry",
    "theme_family", "query_day",
]


def normalize_article_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce the raw API payload into our stable ``articles_raw`` schema.

    GDELT's field set has changed over the years (``socialimage`` appeared
    later, ``url_mobile`` is sometimes absent). Pinning the schema here means
    the rest of the pipeline never has to defend against a missing column.
    """
    df = df.copy()
    for col in REQUIRED_ARTICLE_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    # seendate arrives as '20250612T143000Z' — always UTC.
    df["seendate_utc"] = pd.to_datetime(
        df["seendate"], format="%Y%m%dT%H%M%SZ", errors="coerce", utc=True
    )
    # Fallback for any alternative formatting GDELT may emit.
    missing = df["seendate_utc"].isna()
    if missing.any():
        df.loc[missing, "seendate_utc"] = pd.to_datetime(
            df.loc[missing, "seendate"], errors="coerce", utc=True
        )

    df["domain"] = df["domain"].astype("string").str.lower().str.strip()
    df["title"] = df["title"].astype("string").str.strip()
    df["url"] = df["url"].astype("string").str.strip()
    df["sourcecountry"] = df["sourcecountry"].astype("string").str.upper().str.strip()

    keep = REQUIRED_ARTICLE_COLS + ["seendate_utc"]
    extra = [c for c in ("url_mobile", "socialimage", "title_source", "tone") if c in df.columns]
    df = df[keep + extra]

    # Drop rows we cannot place in time or cannot read — they are unusable.
    before = len(df)
    df = df.dropna(subset=["seendate_utc", "title", "url"])
    if before != len(df):
        log.info("dropped %d rows with missing title/url/timestamp", before - len(df))
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Layer 2 — daily timelines (volume + tone)
# ---------------------------------------------------------------------------
def fetch_timeline(
    sess: CachedSession,
    query: str,
    start: str,
    end: str,
    mode: str = "timelinevolraw",
) -> pd.DataFrame:
    """One request returns a full daily series for the whole study period.

    ``timelinevolraw`` -> raw article counts matching the query
    ``timelinetone``   -> average GDELT tone of matching articles

    This layer is the antidote to the 250-record ceiling in Layer 1: it gives
    an *uncapped* daily volume, so we can check whether a spike in our
    article-level counts is real news flow or just us hitting the cap.
    """
    params = {
        "query": query,
        "mode": mode,
        "format": "json",
        "timelinesmooth": 0,
        "startdatetime": _stamp(pd.Timestamp(start).to_pydatetime()),
        "enddatetime": _stamp(pd.Timestamp(end).to_pydatetime()),
    }
    data = sess.get_json(DOC_API, params, namespace="gdelt_timeline")
    if not data or "timeline" not in data:
        return pd.DataFrame(columns=["date", "value"])
    series = data["timeline"][0].get("data", [])
    if not series:
        return pd.DataFrame(columns=["date", "value"])
    out = pd.DataFrame(series)
    out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=True).dt.tz_localize(None)
    out = out[["date", "value"]].dropna()
    return out


def fetch_timelines(cfg, session: Optional[CachedSession] = None) -> pd.DataFrame:
    """Volume + tone timeline for every theme family, as a tidy long frame."""
    sess = session or CachedSession(
        Path(cfg.path("cache")),
        sleep=cfg.dig("gdelt", "sleep_seconds", default=6.0),
    )
    langs = cfg.dig("gdelt", "languages", default=["english"])
    start, end = str(cfg.dig("period", "start")), str(cfg.dig("period", "end"))
    start = max(start, GDELT_EPOCH)

    frames = []
    for fam, themes in (cfg.dig("theme_families", default={}) or {}).items():
        q = build_theme_query(themes, langs)
        for mode, label in (("timelinevolraw", "volume"), ("timelinetone", "tone")):
            tl = fetch_timeline(sess, q, start, end, mode=mode)
            if tl.empty:
                log.warning("empty %s timeline for family %s", label, fam)
                continue
            tl["theme_family"] = fam
            tl["metric"] = label
            frames.append(tl)

    if not frames:
        return pd.DataFrame(columns=["date", "value", "theme_family", "metric"])
    out = pd.concat(frames, ignore_index=True)
    log.info("timelines: %d rows across %d families",
             len(out), out["theme_family"].nunique())
    return out


def timelines_to_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot the tidy timeline frame into daily feature columns."""
    if long_df.empty:
        return pd.DataFrame()
    wide = long_df.pivot_table(
        index="date", columns=["metric", "theme_family"], values="value", aggfunc="mean"
    )
    wide.columns = [f"gdelt_{m}_{f}" for m, f in wide.columns]
    wide.index = pd.to_datetime(wide.index).normalize()
    wide.index.name = "date"
    return wide.sort_index()


__all__ = [
    "build_theme_query", "fetch_articles", "fetch_articles_window",
    "fetch_timeline", "fetch_timelines", "timelines_to_wide",
    "normalize_article_frame", "REQUIRED_ARTICLE_COLS", "DOC_API",
]
