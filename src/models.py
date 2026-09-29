"""Task 2 — baseline and combined models.

Two tiers, as the brief asks (Task 2, 2b and 2c):

  Price-only baselines
    * majority      — always predict the training set's most common class.
                      Any model must beat this or it has learned nothing.
    * persistence   — tomorrow's direction = today's direction (classification);
                      tomorrow's return = today's return (regression).
    * zero return   — "no change", the standard FX forecasting benchmark.
    * ARIMA(p,0,q)  — on daily log returns (already stationary, so d = 0);
                      p and q chosen by AIC on the fitting data only.

  Learned models, each trained on several feature sets (price only, price +
  NLP, ...), so the baseline and the combined model differ ONLY in their inputs:
    * logistic regression / ridge — linear, heavily regularised; the honest
                      choice for ~700 training rows.
    * XGBoost       — shallow boosted trees; can pick up thresholds and
                      interactions ("negative news AND high volatility").

Class imbalance (UP 40% / FLAT 30% / DOWN 30%) is handled with balanced class
weights, so no model can win on accuracy by always saying UP.
"""
from __future__ import annotations

import itertools
import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from .evaluate import CLASSES

log = logging.getLogger(__name__)

LINEAR = {"cls": "logreg", "reg": "ridge"}


# ---------------------------------------------------------------------------
# Naive baselines
# ---------------------------------------------------------------------------
def majority_class(y_fit: pd.Series, n: int) -> np.ndarray:
    return np.full(n, y_fit.value_counts().idxmax(), dtype=object)


# ---------------------------------------------------------------------------
# ARIMA
# ---------------------------------------------------------------------------
def _fit_arima(y: np.ndarray, order):
    from statsmodels.tsa.arima.model import ARIMA
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # convergence chatter on near-white-noise returns
        return ARIMA(y, order=order, trend="c").fit()


def select_arima_order(y_fit: np.ndarray, cfg) -> tuple:
    """(p, 0, q) with the lowest AIC on the fitting data, over a small grid."""
    acfg = cfg.dig("task2", "arima", default={}) or {}
    best, best_aic = (0, 0, 0), np.inf
    for p, q in itertools.product(range(int(acfg.get("max_p", 2)) + 1),
                                  range(int(acfg.get("max_q", 2)) + 1)):
        try:
            aic = _fit_arima(y_fit, (p, 0, q)).aic
        except Exception as exc:  # a non-invertible start etc.; skip that order
            log.debug("ARIMA(%d,0,%d) failed: %s", p, q, exc)
            continue
        if aic < best_aic:
            best, best_aic = (p, 0, q), aic
    return best


def arima_one_step(y_fit: np.ndarray, y_eval: np.ndarray, order) -> np.ndarray:
    """One-step-ahead forecasts for every day of ``y_eval``.

    Parameters are estimated on ``y_fit`` only. The evaluation returns are then
    appended WITHOUT refitting, so the forecast for day t uses the observed
    returns up to t-1 and the frozen parameters — never day t itself.
    """
    res = _fit_arima(y_fit, order)
    extended = res.append(y_eval, refit=False)
    start = len(y_fit)
    return np.asarray(extended.predict(start=start, end=start + len(y_eval) - 1))


# ---------------------------------------------------------------------------
# Learned models
# ---------------------------------------------------------------------------
def param_grid(task: str, kind: str, cfg) -> list[dict]:
    g = cfg.dig("task2", "grid", default={}) or {}
    if kind == "logreg":
        return [{"C": float(c)} for c in g.get("logreg_C", [0.1])]
    if kind == "ridge":
        return [{"alpha": float(a)} for a in g.get("ridge_alpha", [10.0])]
    return [{"max_depth": int(d), "n_estimators": int(n)}
            for d, n in itertools.product(g.get("xgb_max_depth", [2]),
                                          g.get("xgb_n_estimators", [100]))]


def make_model(task: str, kind: str, params: dict, cfg):
    """A fresh, unfitted estimator. Linear models get median imputation + scaling;
    XGBoost handles missing values (e.g. GDELT outage days) natively."""
    seed = cfg.seed
    if kind in ("logreg", "ridge"):
        est = (LogisticRegression(C=params["C"], class_weight="balanced", max_iter=5000)
               if kind == "logreg" else Ridge(alpha=params["alpha"]))
        return Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                         ("scale", StandardScaler()),
                         ("model", est)])

    import xgboost as xgb
    g = cfg.dig("task2", "grid", default={}) or {}
    common = dict(max_depth=params["max_depth"], n_estimators=params["n_estimators"],
                  learning_rate=float(g.get("xgb_learning_rate", 0.05)),
                  subsample=float(g.get("xgb_subsample", 0.8)),
                  colsample_bytree=float(g.get("xgb_colsample_bytree", 0.8)),
                  random_state=seed, n_jobs=4, verbosity=0)
    if task == "cls":
        return xgb.XGBClassifier(objective="multi:softprob", num_class=len(CLASSES),
                                 eval_metric="mlogloss", **common)
    return xgb.XGBRegressor(objective="reg:squarederror", **common)


def fit_predict(task: str, kind: str, model, X_fit: pd.DataFrame, y_fit: pd.Series,
                X_pred: pd.DataFrame) -> np.ndarray:
    """Fit on (X_fit, y_fit), predict X_pred. Classes are DOWN / FLAT / UP strings."""
    if task == "reg":
        model.fit(X_fit, y_fit.to_numpy(dtype=float))
        return model.predict(X_pred)

    y_idx = pd.Categorical(y_fit, categories=CLASSES).codes
    if kind == "xgb":
        model.fit(X_fit, y_idx, sample_weight=compute_sample_weight("balanced", y_idx))
    else:
        model.fit(X_fit, y_idx)
    return np.asarray(CLASSES, dtype=object)[np.asarray(model.predict(X_pred)).astype(int)]


def feature_importance(model, columns) -> pd.Series:
    """Gain importance for XGBoost, |standardised coefficient| for linear models."""
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=columns)
    coef = np.atleast_2d(model.named_steps["model"].coef_)
    return pd.Series(np.abs(coef).mean(axis=0), index=columns)


__all__ = ["LINEAR", "majority_class", "select_arima_order", "arima_one_step",
           "param_grid", "make_model", "fit_predict", "feature_importance"]
