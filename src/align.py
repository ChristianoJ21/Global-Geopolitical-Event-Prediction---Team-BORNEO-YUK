"""Temporal alignment of news to FX trading days (report §5).

This module is short and it is the most important file in the repository.
Almost every published failure of "news predicts markets" research is a
misalignment bug, not a modelling bug. There are exactly three ways to get it
wrong, and all three produce *better* backtest numbers, which is why they
survive code review:

  1. **Same-day pairing.**  Joining news from day D to the rate fixed on day D.
     A story published at 20:00 WIB sits next to that day's JISDOR, which BI
     had already fixed that afternoon, so the story "predicts" a number that
     existed before it did. Silent, devastating look-ahead. (Taking the date
     in UTC instead of WIB makes it worse: it moves late-UTC stories, much US
     news, onto the wrong day.)

  2. **Weekend and holiday leakage.**  JISDOR is not fixed on weekends or
     Indonesian holidays. Naive resampling either drops that news entirely
     (losing precisely the events governments time for a Friday night) or
     attaches it to a day with no rate.

  3. **Contemporaneous "prediction".**  Using news from day *t* to explain the
     return of day *t*. That is a *nowcast*, not a forecast, and it cannot
     distinguish "news moved the dollar" from "the dollar moved, so journalists
     wrote about it". We keep it as an explicitly labelled diagnostic column
     (``lag0``) but the headline claim in Task 4 must rest on ``lag>=1``.

Our convention: a day-level rule in WIB
---------------------------------------
JISDOR is fixed once per Indonesian business day, and its publication time has
not been constant: BI moved it from 10:00 to 16:15 WIB on 5 April 2021, and
under shortened market hours it is computed 09:00-15:00 and published at
15:15 WIB. Rather than reconstruct and defend a different intraday cutoff for
each period, we use a rule that never needs to know the fixing time:

  1. Convert GDELT's timestamp from UTC to WIB (UTC+7, no daylight saving)
     *first*, then take the calendar date D. The order matters: taking the
     UTC date first would file late-UTC-day stories — much US news — under
     the wrong WIB day.
  2. Bucket the article on the last trading day on or before D. News from a
     weekend or holiday therefore joins the bucket of the trading day before
     it (Friday + Saturday + Sunday share one bucket).
  3. ``make_prediction_frame`` shifts every bucket forward one trading day.

Net effect: news from WIB day D is paired with the first JISDOR fix strictly
after D, and everything that piled up while the market was closed is paired
with the same next available fix. No lookahead is possible: the fix a story is
paired with was always set after the story existed, whichever fixing-time
regime was in force.

We use GDELT's ``seendate`` (crawl time), not the article's self-declared
publication date. ``seendate`` is weakly later than true publication, so any
error it introduces is *conservative* — it can only ever delay a story into a
later bucket, never leak it into an earlier one.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def assign_news_day(
    seen_utc: pd.Series,
    trading_days: pd.DatetimeIndex,
    market_tz: str = "Asia/Jakarta",
) -> pd.Series:
    """Bucket each article on the last trading day on or before its WIB date.

    Steps 1 and 2 of the convention in the module docstring. The UTC -> WIB
    conversion happens *before* the date is taken (``tz_convert`` then
    ``normalize``). ``searchsorted(side="right") - 1`` finds the last trading
    day <= that date, so weekend and holiday news falls into the bucket of the
    trading day before it. Articles dated before the first trading day have no
    bucket and come back as NaT.
    """
    seen = pd.to_datetime(seen_utc, utc=True, errors="coerce")
    local_day = seen.dt.tz_convert(market_tz).dt.tz_localize(None).dt.normalize()

    td = pd.DatetimeIndex(trading_days).normalize()
    pos = np.searchsorted(td.values.astype("datetime64[ns]"),
                          local_day.values.astype("datetime64[ns]"), side="right") - 1

    valid = (pos >= 0) & local_day.notna().values
    out = pd.Series(pd.NaT, index=seen.index, dtype="datetime64[ns]")
    out.loc[valid] = td[pos[valid]]
    return out


def attach_news_day(articles: pd.DataFrame, trading_days, cfg) -> pd.DataFrame:
    """Add ``news_day`` plus the bucket-width diagnostic columns to the article frame."""
    tz = cfg.dig("project", "timezone_market", default="Asia/Jakarta")
    td = pd.DatetimeIndex(pd.to_datetime(trading_days)).normalize().unique().sort_values()

    out = articles.copy()
    out["news_day"] = assign_news_day(out["seendate_utc"], td, tz)

    n_drop = int(out["news_day"].isna().sum())
    if n_drop:
        log.info("%d articles fall before the first trading day and are dropped", n_drop)
    out = out.dropna(subset=["news_day"])

    # How many hours of news does this bucket hold? From its trading day to the
    # next one: 24h on a normal day, 72h for the Friday bucket (Fri+Sat+Sun),
    # more before holidays. The model must know, otherwise it reads that
    # mechanically larger article count as a genuine news spike every week.
    gaps = pd.Series(td).diff().shift(-1).dt.total_seconds().div(3600).fillna(24.0)
    gap_map = dict(zip(td, gaps))
    out["window_hours"] = out["news_day"].map(gap_map).astype(float)
    # Name kept for compatibility: True when the bucket spans a weekend/holiday.
    out["is_post_weekend"] = out["window_hours"] > 30.0
    return out


def trading_day_calendar(fx_panel: pd.DataFrame, cfg) -> pd.DatetimeIndex:
    """Derive the trading calendar from the target series' own observations.

    Deliberately *not* a hard-coded holiday list: the calendar that matters is
    the one on which our target actually prints. Deriving it from the data
    means US, UK and Japanese holidays are handled correctly for free, and the
    calendar can never drift out of sync with the price series.
    """
    primary = cfg.dig("target", "primary")
    if primary in fx_panel.columns:
        idx = fx_panel[primary].dropna().index
    else:
        idx = fx_panel.dropna(how="all").index
    return pd.DatetimeIndex(idx).normalize().unique().sort_values()


def make_prediction_frame(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    cfg,
    lag_days: Optional[int] = None,
) -> pd.DataFrame:
    """Join news features to FX targets with an explicit, auditable lag.

    ``lag_days = 1`` (default) shifts features forward one session, so row
    ``t`` holds *yesterday's* news next to *today's* return.
    ``lag_days = 0`` produces the contemporaneous nowcast frame, useful for
    diagnostics but never for the headline result.
    """
    lag = cfg.dig("alignment", "predict_lag_days", default=1) if lag_days is None else lag_days
    lag = int(lag)

    feats = features.sort_index().copy()
    feats = feats.shift(lag) if lag else feats

    tgt = targets.sort_index().copy()

    # Split the target frame into genuine LABELS and market columns that are
    # only legitimate as *lagged* predictors.
    #
    # This split is the fix for a leak the QA suite caught on the first run:
    # the target's raw return (``ret_USD_IDR``) is identical to ``y_return``, so leaving it
    # in the frame handed the model its own answer under a different name and
    # produced a flawless, worthless R^2. Market history is genuinely useful —
    # momentum and volatility are standard controls, and Task 4 needs them to
    # show that news adds information *beyond* price itself — but only from the
    # past. So every market column is lagged exactly like the news features and
    # renamed with a ``lag{n}_`` prefix that makes its timing visible in any
    # coefficient table.
    label_cols = [c for c in tgt.columns if c.startswith("y_")]
    market_cols = [c for c in tgt.columns if c not in label_cols]

    labels = tgt[label_cols]
    market_lagged = tgt[market_cols].shift(lag) if lag else tgt[market_cols]
    market_lagged = market_lagged.add_prefix(f"lag{lag}_")

    joined = feats.join(market_lagged, how="outer").join(labels, how="inner")
    joined["feature_lag_days"] = lag
    joined.index.name = "date"

    log.info(
        "prediction frame: %d rows x %d cols, lag=%d, %s..%s",
        len(joined), joined.shape[1], int(lag),
        joined.index.min().date() if len(joined) else "-",
        joined.index.max().date() if len(joined) else "-",
    )
    return joined


def chronological_split(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Label each row train / val / test by date. Never shuffle a time series.

    A random split on autocorrelated data lets the model interpolate between a
    Monday and a Wednesday it has already seen to "predict" the Tuesday between
    them. The resulting accuracy is real and completely meaningless.
    """
    train_end = pd.Timestamp(str(cfg.dig("period", "train_end")))
    val_end = pd.Timestamp(str(cfg.dig("period", "val_end")))
    out = df.copy()
    out["split"] = np.select(
        [out.index <= train_end, out.index <= val_end],
        ["train", "val"],
        default="test",
    )
    counts = out["split"].value_counts().to_dict()
    log.info("chronological split: %s", counts)
    return out


def leakage_assertions(df: pd.DataFrame, cfg) -> list[str]:
    """Self-checks that must pass before any model is trained.

    These are cheap, they run in the notebook, and they are the evidence we
    show when asked "how do you know you are not leaking?".
    """
    problems: list[str] = []

    lag = int(df["feature_lag_days"].iloc[0]) if "feature_lag_days" in df.columns else -1
    if lag < 1:
        problems.append(
            f"feature_lag_days={lag}: features are contemporaneous with the target. "
            "Valid only as a nowcast diagnostic, never as the headline result."
        )

    if not df.index.is_monotonic_increasing:
        problems.append("index is not sorted ascending — alignment cannot be trusted")
    if df.index.has_duplicates:
        problems.append("duplicate dates in the prediction frame")

    if "split" in df.columns:
        for a, b in (("train", "val"), ("val", "test")):
            if (df.index[df["split"] == a].max() if (df["split"] == a).any() else pd.NaT) is not pd.NaT:
                a_max = df.index[df["split"] == a].max()
                b_min = df.index[df["split"] == b].min() if (df["split"] == b).any() else None
                if b_min is not None and a_max >= b_min:
                    problems.append(f"{a} overlaps {b} in time ({a_max} >= {b_min})")

    # A feature that correlates ~perfectly with the target is almost always the
    # target itself, leaked in under another name.
    if "y_return" in df.columns:
        feat_cols = [c for c in df.columns if not c.startswith("y_") and
                     pd.api.types.is_numeric_dtype(df[c])]
        # Constant columns have zero variance, so corrwith divides by zero and
        # numpy warns. They cannot leak anything either way — drop them first.
        feat_cols = [c for c in feat_cols if df[c].nunique(dropna=True) > 1]
        if feat_cols:
            corr = df[feat_cols].corrwith(df["y_return"]).abs()
            suspicious = corr[corr > 0.95].index.tolist()
            if suspicious:
                problems.append(f"features with |corr| > 0.95 vs target (likely leak): {suspicious}")

    if not problems:
        log.info("leakage assertions: all passed")
    else:
        for p in problems:
            log.error("LEAKAGE CHECK FAILED: %s", p)
    return problems


__all__ = [
    "assign_news_day", "attach_news_day", "trading_day_calendar",
    "make_prediction_frame", "chronological_split", "leakage_assertions",
]
