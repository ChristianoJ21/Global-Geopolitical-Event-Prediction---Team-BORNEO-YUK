"""Acquisition of the TARGET variable: USD/IDR exchange-rate series.

Design decisions defended here (report §3 / Task 1 requirement 2)
-------------------------------------------------------------------
1. *Which* rate?  The assignment fixes this for us: "Retrieve daily exchange
   rate data for USD from 1st September 2021 - 1st September 2026. Use Bank
   Indonesia as the ultimate source." That is USD/IDR via BI's JISDOR
   reference rate, not a broad USD basket or DXY — JISDOR is Indonesia's own
   benchmark and moves on Indonesia-specific news (BI policy, domestic
   politics, commodity exports) that a global basket would dilute away.

2. *Acquisition method.*  USD_IDR is Bank Indonesia's own JISDOR series,
   taken from the official export on bi.go.id (the "Unduh" button on the
   JISDOR page, period 01/09/2021-01/09/2026) and kept unmodified as
   ``data/raw/Informasi Kurs Jisdor.xlsx`` — the raw source file.
   The export was downloaded in a browser, not by script, and that is a
   finding, not a shortcut: bi.go.id resets the connection on scripted
   clients (Python requests/urllib3 and PowerShell Invoke-WebRequest all fail
   identically), and while curl gets the page, its SharePoint form ignores a
   scripted "Unduh" postback and just returns the default 10-day table.
   ``load_jisdor_excel`` parses the export. Yahoo Finance's IDR=X is kept next
   to it as USD_IDR_YAHOO, purely as an independent cross-check; it is never a
   model input, and it is not a substitute: its daily returns correlate only
   0.53 with JISDOR's, because Yahoo's daily close and JISDOR's afternoon fix
   (15:15 or 16:15 WIB, depending on the period) are taken at different hours.

3. *Controls are part of the target acquisition, not an afterthought.* DXY and
   VIX are downloaded alongside USD_IDR so Task 4 can ask "did this news move
   the dollar broadly, or IDR specifically?" — the honest version of the
   hypothesis needs that distinction. Brent is included because Indonesia is
   a major commodity exporter/importer, a channel a pure "dollar" story misses.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .http_cache import CachedSession

log = logging.getLogger(__name__)

YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"


# ---------------------------------------------------------------------------
# Individual source loader
# ---------------------------------------------------------------------------
def parse_jisdor_frame(raw: pd.DataFrame) -> pd.Series:
    """Extract ``date -> JISDOR rate`` from BI's export sheet (read with header=None).

    The export has a title block above the table, so the header row is found
    by its ``Tanggal`` cell rather than a hard-coded offset that would break
    the day BI adds a line to the banner. Dates arrive as US-style text —
    ``9/1/2026 12:00:00 AM`` is 1 September 2026, NOT 9 January — so the format
    is pinned explicitly; letting pandas guess would silently swap day and
    month for every date whose day is 12 or less.
    """
    is_header = raw.apply(
        lambda row: row.astype(str).str.strip().str.lower().eq("tanggal").any(), axis=1)
    if not is_header.any():
        raise ValueError("no 'Tanggal' header row found in the JISDOR export")
    h = is_header[is_header].index[0]
    header = raw.loc[h].astype(str).str.strip().str.lower()
    date_col = header[header == "tanggal"].index[0]
    rate_col = header[header == "kurs"].index[0]
    body = raw.loc[h + 1:, [date_col, rate_col]]

    col = body[date_col]
    if pd.api.types.is_datetime64_any_dtype(col):
        dates = pd.to_datetime(col)
    else:
        dates = pd.to_datetime(col.astype(str).str.strip(),
                               format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    rates = pd.to_numeric(body[rate_col], errors="coerce")

    s = pd.Series(rates.to_numpy(), index=pd.DatetimeIndex(dates.to_numpy()))
    s = s[s.index.notna()].dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s.index.name = "date"
    s.name = "JISDOR"
    return s


def load_jisdor_excel(path) -> Optional[pd.Series]:
    """Read BI's JISDOR export (.xlsx) from disk. None if the file is missing."""
    p = Path(path)
    if not p.exists():
        log.warning(
            "JISDOR export not found at %s. Download it from "
            "https://www.bi.go.id/id/statistik/informasi-kurs/jisdor/default.aspx "
            "(set the period, click 'Unduh') and save it there.", p)
        return None
    return parse_jisdor_frame(pd.read_excel(p, sheet_name=0, header=None))


def fetch_yahoo_series(sess: CachedSession, ticker: str, *, start: str, end: str) -> Optional[pd.Series]:
    """Download one Yahoo Finance daily-close series as a pandas Series.

    No key or registration; the same public endpoint the Yahoo Finance
    website itself calls. ``start``/``end`` are ISO dates; Yahoo wants Unix
    seconds, and we pad both ends by a day so timezone rounding at the
    boundary never silently drops the first/last row.
    """
    p1 = int(pd.Timestamp(start, tz="UTC").timestamp()) - 86400
    p2 = int(pd.Timestamp(end, tz="UTC").timestamp()) + 86400
    url = YAHOO_CHART.format(ticker=ticker)
    params = {"period1": p1, "period2": p2, "interval": "1d"}
    data = sess.get_json(url, params, namespace="yahoo")
    if not data:
        log.warning("Yahoo ticker %s unavailable", ticker)
        return None
    try:
        result = data["chart"]["result"][0]
        ts = result["timestamp"]
        closes = result["indicators"]["quote"][0]["close"]
    except (KeyError, IndexError, TypeError):
        log.warning("Yahoo ticker %s returned an unexpected payload shape", ticker)
        return None
    idx = (pd.to_datetime(pd.Series(ts), unit="s", utc=True)
           .dt.tz_convert("Asia/Jakarta").dt.normalize().dt.tz_localize(None))
    s = pd.Series(closes, index=idx).dropna()
    s = s.groupby(s.index).last()  # a UTC day can straddle two WIB days; keep the later one
    s.name = ticker
    return s


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def build_fx_panel(cfg, session: Optional[CachedSession] = None) -> pd.DataFrame:
    """Fetch every configured FX/control series and return a wide daily frame.

    Columns are the *logical* names from the config (USD_IDR, DXY, ...), not
    the vendor codes, so downstream code never depends on which vendor we used.
    """
    sess = session or CachedSession(Path(cfg.path("cache")))
    frames: Dict[str, pd.Series] = {}
    start, end = str(cfg.dig("period", "start")), str(cfg.dig("period", "end"))

    for name, rel_path in (cfg.dig("fx_series", "bank_indonesia", default={}) or {}).items():
        s = load_jisdor_excel(cfg.root / rel_path)
        if s is not None:
            frames[name] = s
            log.info("BI    %-11s %-11s %d obs %s..%s",
                     name, "JISDOR", len(s), s.index.min().date(), s.index.max().date())

    for name, ticker in (cfg.dig("fx_series", "yahoo", default={}) or {}).items():
        s = fetch_yahoo_series(sess, ticker, start=start, end=end)
        if s is not None:
            frames[name] = s
            log.info("Yahoo %-11s %-11s %d obs %s..%s",
                     name, ticker, len(s), s.index.min().date(), s.index.max().date())

    if not frames:
        raise RuntimeError(
            "No FX series could be downloaded. Check network access to "
            "query1.finance.yahoo.com."
        )

    panel = pd.DataFrame(frames).sort_index()
    panel.index.name = "date"

    start, end = cfg.dig("period", "start"), cfg.dig("period", "end")
    panel = panel.loc[str(start): str(end)]
    return panel


# ---------------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------------
def make_targets(panel: pd.DataFrame, cfg) -> pd.DataFrame:
    """Turn raw levels into stationary, model-ready targets.

    Why log returns and not levels or simple percentage change:
      * FX index *levels* are (near) unit-root processes. A model fed levels
        will score a spectacular R^2 by predicting "tomorrow == today" while
        learning nothing about geopolitics. Differencing removes that trap.
      * Log returns are additive across time, so a 5-day cumulative return is
        simply the sum of five daily returns — no compounding bookkeeping.
      * They are scale-free, so DXY (~100) and USDIDR (~16,000) become directly
        comparable.
    """
    tgt_cfg = cfg["target"]
    primary = tgt_cfg["primary"]
    if primary not in panel.columns:
        raise KeyError(
            f"Primary target '{primary}' missing from the FX panel. "
            f"Available: {list(panel.columns)}"
        )

    # Work on the TARGET's calendar. JISDOR prints on Indonesian business days;
    # the Yahoo controls follow US/global calendars. Joined naively, a
    # log-return taken across a row where the target is missing is NaN — which
    # would silently delete the return for the day after every Indonesian-only
    # holiday (Idul Fitri, Nyepi, ...). So: carry each control's last value
    # forward (from the target's point of view it genuinely had not changed),
    # then keep only the days on which the target itself printed.
    panel = panel.copy()
    others = [c for c in panel.columns if c != primary]
    panel[others] = panel[others].ffill()
    panel = panel.loc[panel[primary].notna()]

    out = pd.DataFrame(index=panel.index)
    # *_YAHOO columns are cross-checks on the target, not predictors.
    price_cols = [c for c in panel.columns
                  if c not in ("VIX", "DGS10") and not c.endswith("_YAHOO")]

    for col in price_cols:
        out[f"ret_{col}"] = 100.0 * np.log(panel[col]).diff()

    # Controls enter as levels AND changes: the level captures the regime,
    # the change captures the shock.
    for ctrl in ("VIX", "DGS10"):
        if ctrl in panel.columns:
            out[f"lvl_{ctrl}"] = panel[ctrl]
            out[f"chg_{ctrl}"] = panel[ctrl].diff()

    r = out[f"ret_{primary}"]

    # Stale-quote guard: BI states that when interbank data is insufficient,
    # JISDOR "refers to the exchange rate of the previous day" — so an
    # *exactly* unchanged print can be a carry-forward, not a real no-change
    # day. Leaving those in inflates the FLAT class with fake days.
    if cfg.dig("alignment", "drop_zero_return_days", default=True):
        stale = r == 0.0
        if stale.sum():
            log.info("dropping %d stale (exactly-zero-return) days", int(stale.sum()))
        out.loc[stale, :] = np.nan
        r = out[f"ret_{primary}"]

    out["y_return"] = r
    out["y_abs_return"] = r.abs()
    out["y_realized_vol_5d"] = r.rolling(5).std()

    # Ternary direction with a dead-zone. Rationale: without a FLAT band the
    # label is sign(noise) on quiet days, and the model burns capacity on it.
    band = float(tgt_cfg.get("flat_band_bp", 10)) / 100.0   # bp -> percent
    out["y_direction"] = pd.Series(
        np.select([r > band, r < -band], ["UP", "DOWN"], default="FLAT"),
        index=out.index,
    ).where(r.notna())

    # Multi-horizon targets: the cumulative return over the h sessions STARTING
    # at t, i.e. r_t + r_{t+1} + ... + r_{t+h-1}.
    #
    # The horizon is defined forward from t (not t+1) because the one-session
    # prediction gap is created once, in ``align.make_prediction_frame``, by
    # lagging the *features*. Building the gap into the target as well would
    # apply it twice and quietly shift every horizon by a day — a bug that
    # shows up as mysteriously weak h=1 results and nothing else.
    for h in tgt_cfg.get("horizons", [1]):
        cum = r[::-1].rolling(h, min_periods=h).sum()[::-1]
        out[f"y_return_h{h}"] = cum
        out[f"y_direction_h{h}"] = pd.Series(
            np.select([cum > band * h**0.5, cum < -band * h**0.5],
                      ["UP", "DOWN"], default="FLAT"),
            index=out.index,
        ).where(cum.notna())

    out.index.name = "date"
    return out


def describe_targets(targets: pd.DataFrame, cfg) -> pd.DataFrame:
    """Sanity table for the report: is the target well-behaved and balanced?"""
    r = targets["y_return"].dropna()
    rows = [
        ("observations", len(r)),
        ("mean (%)", r.mean()),
        ("std (%)", r.std()),
        ("skew", r.skew()),
        ("excess kurtosis", r.kurt()),
        ("min (%)", r.min()),
        ("max (%)", r.max()),
        ("lag-1 autocorr", r.autocorr(1)),
        ("lag-1 autocorr of |r|", r.abs().autocorr(1)),
    ]
    dist = targets["y_direction"].value_counts(normalize=True)
    for k in ("UP", "DOWN", "FLAT"):
        rows.append((f"share {k}", float(dist.get(k, 0.0))))
    return pd.DataFrame(rows, columns=["statistic", "value"])


__all__ = [
    "build_fx_panel", "make_targets", "describe_targets", "fetch_yahoo_series",
    "parse_jisdor_frame", "load_jisdor_excel",
]
