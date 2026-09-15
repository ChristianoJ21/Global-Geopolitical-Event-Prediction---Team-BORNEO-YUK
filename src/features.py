"""Aggregate the article corpus into the daily modelling panel (report §7).

Scope note
----------
Task 1 is acquisition and preprocessing, not modelling. What this module builds
is therefore the *baseline feature set* — the cheap, transparent aggregates that
Task 2's neural pipeline has to beat. Having them now is not scope creep: it is
the only way to answer "is this dataset actually usable?" before we commit to an
architecture in Task 2.

The design principle that governs every feature here
-----------------------------------------------------
**Never feed a raw count.**  GDELT's corpus grew substantially around 2018 when
its crawler was expanded, and our own 250-records-per-call ceiling caps busy
days. Both effects put non-stationary, non-economic trend into raw volume. A
model fed raw counts will learn "later years have more articles" and call it
signal. So every volume feature is expressed *relatively*:

    z-score vs a trailing 30-day window   -> "is today unusual for this era?"
    share of the day's total              -> composition, immune to level shifts
    log1p                                 -> tames the heavy right tail

All rolling statistics use a **trailing** window and are computed before the
alignment shift in ``align.make_prediction_frame``, so no future information
enters a feature. That is enforced, not assumed: ``align.leakage_assertions``
re-checks it on the joined frame.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _rolling_z(s: pd.Series, window: int = 30, min_periods: int = 10) -> pd.Series:
    """Trailing z-score. ``shift(1)`` keeps today out of its own baseline."""
    base = s.shift(1)
    mu = base.rolling(window, min_periods=min_periods).mean()
    sd = base.rolling(window, min_periods=min_periods).std()
    return (s - mu) / sd.replace(0, np.nan)


def _entropy(counts: pd.Series) -> float:
    """Shannon entropy of a distribution, in nats.

    Interpretation for this project: low entropy = the day's news is
    concentrated on one country or one theme (a single large event); high
    entropy = diffuse background chatter. Concentration is itself predictive —
    one big shock moves markets, a hundred small stories do not.
    """
    total = counts.sum()
    if total <= 0:
        return np.nan
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum())


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------
def build_daily_features(
    articles: pd.DataFrame,
    cfg,
    timeline_wide: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Collapse the filtered article frame into one row per trading day.

    Expects ``articles`` to already carry ``news_day`` (from ``align``),
    ``theme_family``, ``dup_count``, ``source_tier`` and the preprocessed text.
    """
    if articles.empty:
        return pd.DataFrame()

    df = articles.loc[articles.get("keep", True)].copy()
    if df.empty:
        log.warning("no articles survived filtering; feature frame will be empty")
        return pd.DataFrame()

    g = df.groupby("news_day")

    # --- volume ------------------------------------------------------------
    feats = pd.DataFrame(index=pd.DatetimeIndex(sorted(df["news_day"].unique())))
    feats.index.name = "date"
    feats["n_articles"] = g.size()
    feats["n_clusters"] = g["cluster_id"].nunique() if "cluster_id" in df.columns else feats["n_articles"]
    feats["log_n_articles"] = np.log1p(feats["n_articles"])

    # Window-length correction. The Friday bucket spans 72 hours (Fri+Sat+Sun), so it
    # mechanically holds ~2.7x the articles. Without normalising, the model
    # discovers "Mondays are newsworthy" — a calendar artefact, not geopolitics.
    if "window_hours" in df.columns:
        feats["window_hours"] = g["window_hours"].first()
        feats["articles_per_24h"] = feats["n_articles"] * 24.0 / feats["window_hours"].clip(lower=1)
        feats["is_post_weekend"] = (g["is_post_weekend"].first()).astype(int)
    else:
        feats["articles_per_24h"] = feats["n_articles"]
        feats["is_post_weekend"] = 0

    feats["vol_z30"] = _rolling_z(feats["articles_per_24h"], 30)
    feats["vol_z90"] = _rolling_z(feats["articles_per_24h"], 90)

    # --- syndication breadth ----------------------------------------------
    # Recovered from the dedup stage rather than discarded: how widely a story
    # was copied is the newsroom's own importance vote.
    if "dup_count" in df.columns:
        roots = df.loc[~df.get("is_duplicate", False)]
        rg = roots.groupby("news_day")["dup_count"]
        feats["syndication_mean"] = rg.mean()
        feats["syndication_max"] = rg.max()
        feats["syndication_p90"] = rg.quantile(0.9)

    # --- source composition ------------------------------------------------
    if "source_tier" in df.columns:
        tier = pd.crosstab(df["news_day"], df["source_tier"], normalize="index")
        for t in (1, 2, 3):
            feats[f"share_tier{t}"] = tier[t] if t in tier.columns else 0.0
    if "state_affiliated" in df.columns:
        feats["share_state_media"] = g["state_affiliated"].mean()

    # --- theme composition -------------------------------------------------
    fam_counts = pd.crosstab(df["news_day"], df["theme_family"])
    fam_share = fam_counts.div(fam_counts.sum(axis=1).replace(0, np.nan), axis=0)
    for fam in fam_share.columns:
        feats[f"share_{fam}"] = fam_share[fam]
        feats[f"z_{fam}"] = _rolling_z(fam_counts[fam].reindex(feats.index).fillna(0), 30)
    feats["theme_entropy"] = fam_counts.apply(_entropy, axis=1)

    # --- geographic concentration -----------------------------------------
    if "sourcecountry" in df.columns:
        geo = pd.crosstab(df["news_day"], df["sourcecountry"])
        feats["geo_entropy"] = geo.apply(_entropy, axis=1)
        feats["geo_hhi"] = geo.div(geo.sum(axis=1).replace(0, np.nan), axis=0).pow(2).sum(axis=1)
        feats["n_countries"] = (geo > 0).sum(axis=1)

    # --- materiality intensity --------------------------------------------
    for col, name in (("has_market_term", "share_market_term"),
                      ("has_institution", "share_institution"),
                      ("has_actor_country", "share_actor_country")):
        if col in df.columns:
            feats[name] = g[col].mean()

    # --- lexical --------------------------------------------------------------
    if "n_tokens_b" in df.columns:
        feats["mean_headline_len"] = g["n_tokens_b"].mean()

    # --- GDELT uncapped timelines -----------------------------------------
    # Joined last so the (capped) article counts can be validated against the
    # (uncapped) timeline volume — see qa.cap_saturation_report.
    if timeline_wide is not None and not timeline_wide.empty:
        tw = timeline_wide.copy()
        tw.index = pd.DatetimeIndex(tw.index).normalize()
        feats = feats.join(tw, how="left")
        for c in [c for c in tw.columns if c.startswith("gdelt_volume_")]:
            feats[f"{c}_z30"] = _rolling_z(feats[c], 30)

    feats = feats.sort_index()
    log.info("daily features: %d days x %d columns (%s..%s)",
             len(feats), feats.shape[1],
             feats.index.min().date(), feats.index.max().date())
    return feats


def reindex_to_trading_days(feats: pd.DataFrame, trading_days) -> pd.DataFrame:
    """Place features on the FX calendar, filling genuinely newsless days with 0.

    Zero, not forward-fill: a day with no matching articles really did have no
    matching articles. Forward-filling would invent yesterday's news flow and,
    worse, smear an event across days, blurring exactly the event-study timing
    the hypothesis depends on. Count columns get 0; ratio/z columns stay NaN so
    the model can mask them.
    """
    td = pd.DatetimeIndex(pd.to_datetime(trading_days)).normalize().unique().sort_values()
    out = feats.reindex(td)
    count_like = [c for c in out.columns
                  if c.startswith(("n_", "log_n_")) or c == "articles_per_24h"]
    out[count_like] = out[count_like].fillna(0)
    out.index.name = "date"
    missing = int(out["n_articles"].eq(0).sum()) if "n_articles" in out.columns else 0
    if missing:
        log.info("%d trading days (%.1f%%) have zero matching articles",
                 missing, 100 * missing / max(len(out), 1))
    return out


def feature_dictionary(feats: pd.DataFrame) -> pd.DataFrame:
    """Machine-generated data dictionary for the report appendix."""
    groups = {
        "n_": "volume", "log_n": "volume", "articles_per": "volume",
        "vol_z": "volume (relative)", "window_": "calendar", "is_post": "calendar",
        "syndication_": "syndication breadth", "share_tier": "source composition",
        "share_state": "source composition", "share_": "theme/materiality share",
        "z_": "theme intensity (relative)", "theme_entropy": "concentration",
        "geo_": "geographic concentration", "mean_headline": "lexical",
        "gdelt_volume": "GDELT uncapped volume", "gdelt_tone": "GDELT tone",
    }

    def grp(c: str) -> str:
        for pre, g in groups.items():
            if c.startswith(pre):
                return g
        return "other"

    rows = []
    for c in feats.columns:
        s = feats[c]
        rows.append({
            "column": c,
            "group": grp(c),
            "dtype": str(s.dtype),
            "non_null_%": round(100 * s.notna().mean(), 1),
            "mean": round(float(s.mean()), 4) if pd.api.types.is_numeric_dtype(s) else None,
            "std": round(float(s.std()), 4) if pd.api.types.is_numeric_dtype(s) else None,
        })
    return pd.DataFrame(rows)


__all__ = ["build_daily_features", "reindex_to_trading_days", "feature_dictionary"]
