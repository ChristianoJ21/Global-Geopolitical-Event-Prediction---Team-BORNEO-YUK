"""Synthetic fixtures for offline testing.

Purpose
-------
This module generates *fake but structurally faithful* data so that
``python -m src.build_dataset --offline`` exercises every line of the pipeline
with no network access at all.

Two reasons this is not busywork:

  1. **Fail fast, cheaply.**  A full GDELT census is ~1 hour of requests. Any
     schema bug, alignment bug or config typo should surface in 10 seconds on
     fixtures, not after an hour of polite crawling.
  2. **The pipeline is testable.**  Because we *control* the generating
     process, we can plant known effects and check that the pipeline recovers
     them. ``synthetic_articles`` deliberately injects a volume spike on a set
     of "shock days" and correlates the next day's FX return with it. If the
     pipeline is aligned correctly, ``vol_z30`` on day t correlates with
     ``y_return`` on day t+1 and NOT with day t. ``tests/test_pipeline.py``
     asserts exactly that — which turns temporal alignment from a claim in the
     report into a property the test suite verifies.

Nothing in here is ever used for the real study; ``manifest.json`` records
``offline_synthetic: true`` so a fixture run can never be mistaken for results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_HEADLINE_TEMPLATES = {
    "conflict": [
        "{A} strikes {B} positions near the border, officials say",
        "Clashes erupt in {B} as {A} moves troops - Reuters",
        "{A} says it will not de-escalate tensions with {B}",
    ],
    "sanctions_trade": [
        "{A} imposes new tariffs on {B} exports worth $12 billion",
        "{A} sanctions {B} energy firms over {B} policy | Al Jazeera",
        "{B} says it will never accept {A} trade restrictions",
    ],
    "diplomacy": [
        "{A} and {B} open talks on a new security treaty",
        "Summit between {A} and {B} ends without agreement - AP",
        "{A} envoy arrives in {B} for negotiations",
    ],
    "energy": [
        "Oil prices up 3.2% as {A} supply concerns mount",
        "OPEC weighs output cut after {A}-{B} tensions",
        "{A} halts gas exports to {B}, prices surge",
    ],
    "monetary_policy": [
        "Federal Reserve will not cut rates in December, officials signal",
        "{A} central bank raises interest rates by 25 basis points",
        "Dollar steady as {A} inflation data misses forecasts - Bloomberg",
    ],
    "political_risk": [
        "Protests grip {A} capital ahead of contested election",
        "{A} government faces no-confidence vote amid corruption probe",
        "Policy uncertainty in {A} weighs on investor sentiment",
    ],
}

_ACTORS = ["United States", "China", "Russia", "Iran", "Israel", "Ukraine",
           "Saudi Arabia", "Japan", "Germany", "India", "Turkey", "Venezuela"]
_COUNTRIES = ["US", "CH", "RS", "IR", "IS", "UP", "SA", "JA", "GM", "IN", "TU", "VE"]
_DOMAINS = ["reuters.com", "apnews.com", "bloomberg.com", "aljazeera.com",
            "bbc.co.uk", "cnbc.com", "scmp.com", "tass.com",
            "somecontentfarm.blogspot.com", "msn.com", "obscure-local-news.example"]

# Lexical jitter, so distinct stories look distinct to the dedup stage.
_QUALIFIERS = ["", "", "Exclusive: ", "Update: ", "Analysis: ", "Breaking: ",
               "Report: ", "Live: "]


def synthetic_fx_panel(cfg, n_days: int | None = None) -> pd.DataFrame:
    """Business-day FX panel with realistic volatility clustering."""
    rng = np.random.default_rng(cfg.seed)
    start = pd.Timestamp(str(cfg.dig("period", "start")))
    end = pd.Timestamp(str(cfg.dig("period", "end")))
    days = pd.bdate_range(start, end)
    if n_days:
        days = days[:n_days]
    n = len(days)

    # GARCH-ish: volatility persists, which is what real FX does.
    vol = np.empty(n)
    vol[0] = 0.30
    shocks = rng.standard_normal(n)
    for i in range(1, n):
        vol[i] = 0.02 + 0.90 * vol[i - 1] + 0.08 * abs(shocks[i - 1]) * 0.30
    rets = shocks * vol

    def series(level0: float, scale: float, extra_noise: float = 0.0) -> np.ndarray:
        r = rets * scale + rng.standard_normal(n) * extra_noise
        return level0 * np.exp(np.cumsum(r / 100.0))

    panel = pd.DataFrame(index=days)
    panel.index.name = "date"
    panel["USD_IDR"] = series(15500.0, 1.0)  # ~JISDOR-scale level, not the real rate
    panel["AFE_USD"] = series(105.0, 1.1, 0.05)
    panel["EME_USD"] = series(140.0, 0.9, 0.05)
    panel["DXY"] = series(97.0, 1.3, 0.08)
    panel["EURUSD"] = series(1.10, -1.4, 0.08)
    panel["USDJPY"] = series(130.0, 1.2, 0.10)
    panel["GBPUSD"] = series(1.27, -1.1, 0.09)
    panel["USDCNY"] = series(7.05, 0.5, 0.04)
    panel["USDIDR"] = series(15200.0, 0.8, 0.12)
    panel["BRENT"] = series(82.0, -0.6, 0.90)
    panel["GOLD"] = series(2050.0, -0.9, 0.50)
    panel["VIX"] = np.clip(14 + 40 * vol + rng.standard_normal(n) * 1.5, 9, 80)
    panel["DGS10"] = np.clip(4.0 + np.cumsum(rng.standard_normal(n) * 0.02), 0.5, 7.0)

    # Plant a few genuine holidays so the trading calendar has real gaps.
    holidays = rng.choice(n, size=max(1, n // 90), replace=False)
    panel.iloc[holidays] = np.nan
    return panel


def _shock_days(index: pd.DatetimeIndex, seed: int, frac: float = 0.06) -> pd.DatetimeIndex:
    rng = np.random.default_rng(seed + 1)
    k = max(1, int(len(index) * frac))
    return pd.DatetimeIndex(sorted(rng.choice(index, size=k, replace=False)))


def synthetic_articles(cfg, trading_days, base_per_family: int = 6) -> pd.DataFrame:
    """Article frame with a planted, recoverable volume spike on shock days.

    The spike is placed in the *news window ending at* a shock day's close, and
    ``tests/test_pipeline.py`` checks the correlation appears at lag 1 and not
    at lag 0. That is the alignment test.
    """
    rng = np.random.default_rng(cfg.seed)
    td = pd.DatetimeIndex(pd.to_datetime(trading_days)).normalize()
    fams = list((cfg.dig("theme_families", default={}) or {}).keys()) or ["conflict"]
    shocks = set(_shock_days(td, cfg.seed))

    rows = []
    for day in td:
        mult = 4.0 if day in shocks else 1.0
        for fam in fams:
            k = max(0, int(rng.poisson(base_per_family * mult)))
            tmpl = _HEADLINE_TEMPLATES.get(fam, _HEADLINE_TEMPLATES["conflict"])
            for _ in range(k):
                ai, bi = rng.choice(len(_ACTORS), size=2, replace=False)
                title = rng.choice(tmpl).format(A=_ACTORS[ai], B=_ACTORS[bi])
                # Lexical jitter so that base headlines are *distinct* stories.
                # Without it the fixture would be ~95% exact duplicates and the
                # dedup stage would look catastrophically aggressive for
                # reasons that have nothing to do with the code.
                title = f"{rng.choice(_QUALIFIERS)}{title}"
                if rng.random() < 0.5:
                    title += f", {int(rng.integers(2, 400))} reported"
                di = int(rng.integers(0, len(_DOMAINS)))

                # Place the timestamp INSIDE the window ending at this day's
                # 17:00 ET close. In UTC that close is ~21:00 (EDT) / 22:00
                # (EST), so sampling backwards from 21:00 UTC on `day` keeps
                # every article safely inside ((day-1) close, day close].
                seen = (pd.Timestamp(day).tz_localize("UTC")
                        + pd.Timedelta(hours=21)
                        - pd.Timedelta(hours=float(rng.uniform(0.5, 22.0))))
                rows.append({
                    "url": f"https://{_DOMAINS[di]}/story/{abs(hash(title)) % 10**9}-{len(rows)}",
                    "title": title,
                    "seendate": seen.strftime("%Y%m%dT%H%M%SZ"),
                    "domain": _DOMAINS[di],
                    "language": "English",
                    "sourcecountry": _COUNTRIES[int(rng.integers(0, len(_COUNTRIES)))],
                    "theme_family": fam,
                    "query_day": day.date().isoformat(),
                })

    # Syndication: republish ~17% of stories to another outlet within a few
    # hours, sometimes with a re-titled suffix. This is what the dedup stage is
    # supposed to find, so the fixture has to contain it.
    n0 = len(rows)
    for i in rng.choice(n0, size=max(1, n0 // 6), replace=False):
        src = dict(rows[int(i)])
        d = _DOMAINS[int(rng.integers(0, len(_DOMAINS)))]
        t0 = pd.Timestamp(src["seendate"])
        src["domain"] = d
        src["seendate"] = (t0 + pd.Timedelta(hours=float(rng.uniform(0.2, 6.0)))
                           ).strftime("%Y%m%dT%H%M%SZ")
        if rng.random() < 0.4:   # near-duplicate, not exact: SimHash must catch it
            src["title"] = src["title"] + f" - {d.split('.')[0].title()}"
        src["url"] = f"https://{d}/wire/{abs(hash(src['title'])) % 10**9}-copy{i}"
        rows.append(src)

    df = pd.DataFrame(rows)
    from .gdelt import normalize_article_frame
    return normalize_article_frame(df)


def synthetic_timelines(cfg, trading_days) -> pd.DataFrame:
    """Wide daily volume/tone timeline matching ``gdelt.timelines_to_wide`` output."""
    rng = np.random.default_rng(cfg.seed + 7)
    td = pd.DatetimeIndex(pd.to_datetime(trading_days)).normalize()
    fams = list((cfg.dig("theme_families", default={}) or {}).keys()) or ["conflict"]
    shocks = set(_shock_days(td, cfg.seed))
    out = pd.DataFrame(index=td)
    out.index.name = "date"
    for fam in fams:
        mult = np.where(td.isin(list(shocks)), 3.5, 1.0)
        out[f"gdelt_volume_{fam}"] = rng.poisson(400 * mult).astype(float)
        out[f"gdelt_tone_{fam}"] = rng.normal(-2.0, 1.5, len(td)) - 1.5 * (mult > 1)
    return out


__all__ = ["synthetic_fx_panel", "synthetic_articles", "synthetic_timelines"]
