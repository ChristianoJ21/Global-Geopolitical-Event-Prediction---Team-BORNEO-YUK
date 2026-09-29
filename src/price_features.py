"""Task 2 — features from exchange-rate history only (the baseline's inputs).

The brief (Task 2, 2b) asks for a baseline "using only historical exchange rate
data". Everything here is computed from JISDOR's own daily log return
``ret_USD_IDR``, so the baseline cannot borrow from news or from other markets.

Timing. Row ``t`` of the modelling table predicts ``y_return[t]``, the move
from fix ``t-1`` to fix ``t``. A feature on row ``t`` may therefore use returns
up to and including ``t-1`` and nothing later. Every column below is built from
``ret.shift(k)`` with ``k >= 1``; ``tests/test_task2.py`` checks it by changing
the return on day ``t`` and asserting no feature on day ``t`` moves.
"""
from __future__ import annotations

import pandas as pd


def build_price_features(targets: pd.DataFrame, cfg) -> pd.DataFrame:
    """Lagged returns, trailing mean (momentum) and trailing std (volatility).

    Why these three families:
      * lags 1-5   — a week of history; lets ARIMA-like autocorrelation show up
                     in the classifiers too.
      * mean 5/20d — momentum over a week and a month.
      * std 5/20d  — volatility regime; big moves cluster, so a volatile week
                     makes a non-FLAT day more likely.
    """
    pcfg = cfg.dig("task2", "price", default={}) or {}
    lags = [int(k) for k in pcfg.get("return_lags", [1, 2, 3, 4, 5])]
    windows = [int(w) for w in pcfg.get("rolling_windows", [5, 20])]

    primary = cfg.dig("target", "primary", default="USD_IDR")
    ret = targets[f"ret_{primary}"].sort_index()
    past = ret.shift(1)  # the most recent return known before today's fix

    out = pd.DataFrame(index=ret.index)
    for k in lags:
        out[f"px_ret_lag{k}"] = ret.shift(k)
    for w in windows:
        out[f"px_mean_{w}d"] = past.rolling(w, min_periods=w).mean()
        out[f"px_vol_{w}d"] = past.rolling(w, min_periods=w).std()
    out["px_abs_ret_lag1"] = past.abs()
    out.index.name = "date"
    return out


__all__ = ["build_price_features"]
