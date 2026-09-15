"""Data-quality assurance (report §8).

Every check here exists because of a specific, named way this dataset can lie
to us. A QA suite that just prints ``df.describe()`` catches none of them.

    coverage_report        -> are there silent holes in the time series?
    cap_saturation_report  -> are we truncating busy days at the API ceiling?
    structural_break_report-> did GDELT's crawler change under us?
    fx_sanity_report       -> stale prints, impossible returns, fat tails
    corpus_quality_report  -> duplicates, empty text, language leakage
    validity_summary       -> one pass/fail table for the report
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
def coverage_report(feats: pd.DataFrame, trading_days) -> pd.DataFrame:
    """Per-year coverage: how many trading days actually carry news?

    A pipeline can fail silently for a whole quarter — a changed API parameter,
    an expired cache, a rate limit hit overnight — and the only symptom is a
    quietly thinner year. Yearly aggregation makes that visible immediately.
    """
    td = pd.DatetimeIndex(pd.to_datetime(trading_days)).normalize()
    have = feats.index.normalize() if len(feats) else pd.DatetimeIndex([])
    n_art = feats["n_articles"] if "n_articles" in feats.columns else pd.Series(dtype=float)

    rows = []
    for yr in sorted(td.year.unique()):
        y_td = td[td.year == yr]
        y_have = have[have.year == yr]
        y_art = n_art[n_art.index.year == yr] if len(n_art) else pd.Series(dtype=float)
        rows.append({
            "year": int(yr),
            "trading_days": len(y_td),
            "days_with_rows": len(y_have),
            "days_with_articles": int((y_art > 0).sum()) if len(y_art) else 0,
            "coverage_%": round(100 * (y_art > 0).sum() / max(len(y_td), 1), 1) if len(y_art) else 0.0,
            "median_articles": float(y_art.median()) if len(y_art) else 0.0,
        })
    return pd.DataFrame(rows)


def find_gaps(feats: pd.DataFrame, min_gap_days: int = 3) -> pd.DataFrame:
    """Consecutive runs of zero-article trading days — the fingerprint of an outage."""
    if "n_articles" not in feats.columns or feats.empty:
        return pd.DataFrame(columns=["start", "end", "length_days"])
    zero = feats["n_articles"].eq(0)
    runs, start = [], None
    for dt, is_zero in zero.items():
        if is_zero and start is None:
            start = dt
        elif not is_zero and start is not None:
            runs.append((start, prev, (prev - start).days + 1))
            start = None
        prev = dt
    if start is not None:
        runs.append((start, feats.index[-1], (feats.index[-1] - start).days + 1))
    out = pd.DataFrame(runs, columns=["start", "end", "length_days"])
    return out[out["length_days"] >= min_gap_days].reset_index(drop=True)


# ---------------------------------------------------------------------------
def cap_saturation_report(articles: pd.DataFrame, cap: int = 250) -> pd.DataFrame:
    """How often did a (day, family) request hit the 250-record API ceiling?

    Why this matters more than it looks
    -----------------------------------
    When a request saturates, we did not sample that day — we took the first
    250 the API chose to return. Saturation is *correlated with news intensity*,
    which is precisely our predictor. So the measurement error is not random
    noise; it is systematic censoring that compresses exactly the big-news days
    the hypothesis cares about.

    If saturation is common, the honest fixes are: raise ``sub_day_chunks``, or
    lean on the uncapped GDELT timeline volume instead of article counts. This
    table tells us which.
    """
    if articles.empty or "query_day" not in articles.columns:
        return pd.DataFrame()
    grp = articles.groupby(["query_day", "theme_family"]).size().rename("n").reset_index()
    grp["saturated"] = grp["n"] >= cap
    by_fam = grp.groupby("theme_family").agg(
        requests=("n", "size"),
        saturated=("saturated", "sum"),
        median_n=("n", "median"),
        max_n=("n", "max"),
    ).reset_index()
    by_fam["saturation_%"] = (100 * by_fam["saturated"] / by_fam["requests"]).round(1)
    return by_fam.sort_values("saturation_%", ascending=False)


def structural_break_report(feats: pd.DataFrame, col: str = "articles_per_24h") -> pd.DataFrame:
    """Yearly level shifts in news volume — the GDELT crawler-expansion check.

    GDELT's indexed source list expanded materially around 2018. If our yearly
    medians jump by a factor unrelated to world events, raw counts are
    contaminated and only relative features (z-scores, shares) may be used
    downstream. This table is the evidence for that decision.
    """
    if col not in feats.columns or feats.empty:
        return pd.DataFrame()
    y = feats[col].groupby(feats.index.year).agg(["median", "mean", "std", "count"])
    y["yoy_median_ratio"] = (y["median"] / y["median"].shift(1)).round(2)
    y.index.name = "year"
    return y.reset_index()


# ---------------------------------------------------------------------------
def fx_sanity_report(targets: pd.DataFrame, ret_col: str = "y_return") -> pd.DataFrame:
    """Distributional and integrity checks on the target series."""
    r = targets[ret_col].dropna()
    if r.empty:
        return pd.DataFrame([{"check": "target present", "value": 0, "status": "FAIL"}])

    sigma = r.std()
    rows = [
        ("observations", len(r), "OK" if len(r) > 1000 else "THIN"),
        ("exact-zero returns", int((r == 0).sum()),
         "OK" if (r == 0).mean() < 0.01 else "CHECK: stale quotes"),
        ("|return| > 5 sigma", int((r.abs() > 5 * sigma).sum()),
         "OK" if (r.abs() > 5 * sigma).mean() < 0.005 else "CHECK: outliers"),
        ("|return| > 3% in a day", int((r.abs() > 3).sum()),
         "OK" if (r.abs() > 3).sum() < 10 else "CHECK: implausible for a broad index"),
        ("excess kurtosis", round(float(r.kurt()), 2),
         "expected (FX is fat-tailed)" if r.kurt() > 1 else "unusually thin-tailed"),
        ("lag-1 autocorrelation", round(float(r.autocorr(1)), 4),
         "OK: near-zero as theory predicts" if abs(r.autocorr(1)) < 0.1
         else "CHECK: predictable from its own past"),
        ("lag-1 autocorr of |r|", round(float(r.abs().autocorr(1)), 4),
         "expected: volatility clusters" if r.abs().autocorr(1) > 0.1 else "unusual"),
    ]
    return pd.DataFrame(rows, columns=["check", "value", "status"])


def corpus_quality_report(articles: pd.DataFrame) -> pd.DataFrame:
    """Integrity of the article corpus itself."""
    n = len(articles)
    if n == 0:
        return pd.DataFrame([{"check": "articles", "value": 0, "pct": 0.0}])

    def pct(x) -> float:
        return round(100 * x / n, 2)

    rows = [
        ("total rows fetched", n, 100.0),
        ("unique URLs", articles["url"].nunique(), pct(articles["url"].nunique())),
        ("flagged duplicate", int(articles.get("is_duplicate", pd.Series(False)).sum()),
         pct(int(articles.get("is_duplicate", pd.Series(False)).sum()))),
        ("missing/blank title", int(articles["title"].fillna("").str.strip().eq("").sum()),
         pct(int(articles["title"].fillna("").str.strip().eq("").sum()))),
        ("non-English language tag",
         int((~articles.get("language", pd.Series("English", index=articles.index))
              .astype(str).str.lower().isin(["english", "eng", "en", "<na>", "nan"])).sum()),
         None),
        ("empty after Track-A preprocessing",
         int(articles.get("text_track_a", pd.Series("", index=articles.index)).eq("").sum()),
         pct(int(articles.get("text_track_a", pd.Series("", index=articles.index)).eq("").sum()))),
        ("survived the full funnel", int(articles.get("keep", pd.Series(True)).sum()),
         pct(int(articles.get("keep", pd.Series(True, index=articles.index)).sum()))),
        ("distinct domains", articles["domain"].nunique(), None),
        ("distinct source countries",
         articles.get("sourcecountry", pd.Series(dtype=str)).nunique(), None),
    ]
    out = pd.DataFrame(rows, columns=["check", "value", "pct_of_total"])
    for c in ("value",):
        out[c] = out[c].astype("Int64")
    return out


# ---------------------------------------------------------------------------
def validity_summary(
    articles: pd.DataFrame,
    feats: pd.DataFrame,
    targets: pd.DataFrame,
    leak_problems: Optional[List[str]] = None,
) -> pd.DataFrame:
    """One pass/fail table. This is the slide the lecturer will actually read."""
    checks = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    n_keep = int(articles.get("keep", pd.Series(True, index=articles.index)).sum()) if len(articles) else 0
    add("Corpus non-empty after filtering", n_keep > 0, f"{n_keep:,} articles kept")

    cov = (feats["n_articles"] > 0).mean() if "n_articles" in feats.columns and len(feats) else 0
    add("News coverage of trading days >= 90%", cov >= 0.90, f"{100*cov:.1f}% of trading days")

    n_t = int(targets["y_return"].notna().sum()) if "y_return" in targets.columns else 0
    add("Target series usable (>1,000 obs)", n_t > 1000, f"{n_t:,} daily returns")

    if "y_direction" in targets.columns:
        share = targets["y_direction"].value_counts(normalize=True)
        worst = float(share.min()) if len(share) else 0
        add("No degenerate class (<5%)", worst >= 0.05,
            ", ".join(f"{k} {100*v:.0f}%" for k, v in share.items()))

    dup = float(articles.get("is_duplicate", pd.Series(False, index=articles.index)).mean()) if len(articles) else 0
    add("Duplicate rate under control (<60%)", dup < 0.60, f"{100*dup:.1f}% flagged duplicate")

    lp = leak_problems or []
    add("No leakage assertion failures", len(lp) == 0,
        "all checks passed" if not lp else "; ".join(lp[:2]))

    return pd.DataFrame(checks)


def missingness_table(df: pd.DataFrame, top: int = 20) -> pd.DataFrame:
    """Columns ranked by missingness — the first thing to look at before modelling."""
    m = df.isna().mean().sort_values(ascending=False)
    out = (m * 100).round(2).rename("missing_%").reset_index()
    out.columns = ["column", "missing_%"]
    return out.head(top)


__all__ = [
    "coverage_report", "find_gaps", "cap_saturation_report",
    "structural_break_report", "fx_sanity_report", "corpus_quality_report",
    "validity_summary", "missingness_table",
]
