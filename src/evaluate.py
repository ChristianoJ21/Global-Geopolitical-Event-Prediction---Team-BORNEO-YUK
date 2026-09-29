"""Task 2 — evaluation metrics for both task formulations.

Primary task: 3-class direction (UP / FLAT / DOWN).
  * macro-F1          — the headline metric. The classes are 40/30/30, so plain
                        accuracy rewards always saying UP; macro-F1 weights the
                        three classes equally and does not.
  * accuracy          — reported because everyone asks for it.
  * directional hit rate + coverage — on the days the model commits to UP or
                        DOWN, how often was the move really in that direction,
                        and on what share of days did it commit? This is the
                        number a treasury desk would act on.

Secondary task: next-day log return (regression).
  * RMSE, MAE         — error size in percentage points of return.
  * directional accuracy — share of non-zero days where the predicted sign is
                        right.
  * R2 vs zero        — 1 - SSE(model) / SSE(predict 0). Positive only if the
                        model beats "no change", the standard FX benchmark.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

CLASSES = ["DOWN", "FLAT", "UP"]


def classification_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    per_class = f1_score(y_true, y_pred, labels=CLASSES, average=None, zero_division=0)

    called = y_pred != "FLAT"
    hit_rate = float((y_pred[called] == y_true[called]).mean()) if called.any() else np.nan
    return {
        "n": int(len(y_true)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "directional_hit_rate": hit_rate,
        "coverage": float(called.mean()),
        **{f"f1_{c}": float(v) for c, v in zip(CLASSES, per_class)},
    }


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    moved = y_true != 0
    sse_zero = float(np.sum(y_true ** 2))
    return {
        "n": int(len(y_true)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "directional_accuracy": float((np.sign(y_pred[moved]) == np.sign(y_true[moved])).mean())
        if moved.any() else np.nan,
        "r2_vs_zero": float(1 - np.sum(err ** 2) / sse_zero) if sse_zero > 0 else np.nan,
    }


def confusion_frame(y_true, y_pred) -> pd.DataFrame:
    """Rows = true class, columns = predicted class, in DOWN / FLAT / UP order."""
    cm = confusion_matrix(np.asarray(y_true, dtype=object), np.asarray(y_pred, dtype=object),
                          labels=CLASSES)
    return pd.DataFrame(cm, index=[f"true_{c}" for c in CLASSES],
                        columns=[f"pred_{c}" for c in CLASSES])


def block_bootstrap_gain(task: str, y_true, pred_base, pred_new, n_boot: int = 1000,
                         block: int = 10, seed: int = 42) -> dict:
    """Is ``pred_new`` really better than ``pred_base`` on these days, or is it luck?

    Resamples the evaluation days in blocks of ``block`` consecutive trading
    days (two weeks), because FX days are not independent: volatile days
    cluster. Both models are scored on the SAME resampled days each time, so
    the interval is for the difference. Gain = macro-F1 increase
    (classification) or RMSE reduction (regression); positive = better.
    ``p_no_gain`` is the share of resamples in which the new model did not win.
    """
    y, a, b = (np.asarray(v, dtype=object if task == "cls" else float)
               for v in (y_true, pred_base, pred_new))
    n = len(y)

    def score(yy, pp):
        if task == "cls":
            return f1_score(yy, pp, labels=CLASSES, average="macro", zero_division=0)
        return -np.sqrt(np.mean((pp.astype(float) - yy.astype(float)) ** 2))

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, n_blocks)
        idx = (starts[:, None] + np.arange(block)).ravel()[:n]
        diffs[i] = score(y[idx], b[idx]) - score(y[idx], a[idx])
    return {"ci_low": float(np.quantile(diffs, 0.025)),
            "ci_high": float(np.quantile(diffs, 0.975)),
            "p_no_gain": float((diffs <= 0).mean())}


def returns_to_direction(pred_returns, band_pct: float) -> np.ndarray:
    """Map a return forecast onto the same UP / FLAT / DOWN dead-zone as the label."""
    r = np.asarray(pred_returns, dtype=float)
    return np.select([r > band_pct, r < -band_pct], ["UP", "DOWN"], default="FLAT").astype(object)


__all__ = ["CLASSES", "classification_metrics", "regression_metrics",
           "confusion_frame", "block_bootstrap_gain", "returns_to_direction"]
