"""Tests for Task 2: the timing of every feature, the TF-IDF fit window, the metrics.

Run with:  py -m pytest tests/ -v

As in Task 1, we test the decisions that would silently invalidate the result
if they were wrong: a price feature that sees today's return, NLP features
paired with the wrong fix, a TF-IDF vocabulary learned from test-period news,
an ARIMA forecast that peeks at the day it forecasts.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import evaluate, models, nlp_features, price_features, run_task2  # noqa: E402
from src.config import Config, load_config  # noqa: E402

CFG = load_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")


def _cfg_with(**tfidf) -> Config:
    raw = copy.deepcopy(CFG.raw)
    raw["task2"]["tfidf"].update(tfidf)
    return Config(raw=raw, root=CFG.root)


def _targets(n: int = 40, seed: int = 0) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=n)
    r = pd.Series(np.random.default_rng(seed).normal(0, 0.3, n), index=idx)
    r.iloc[0] = np.nan
    direction = pd.Series(evaluate.returns_to_direction(r.fillna(0), 0.1), index=idx).where(r.notna())
    return pd.DataFrame({"ret_USD_IDR": r, "y_return": r, "y_direction": direction})


# ===========================================================================
# 1. Timing
# ===========================================================================
def test_price_features_never_see_the_return_they_predict():
    """Changing day t's return must not move any feature on row t, only later rows."""
    t = _targets()
    before = price_features.build_price_features(t, CFG)
    bumped = t.copy()
    bumped.iloc[30, 0] += 5.0
    after = price_features.build_price_features(bumped, CFG)

    pd.testing.assert_series_equal(before.iloc[30], after.iloc[30])
    assert not before.iloc[31].equals(after.iloc[31]), "day t's return is used from t+1 on"


def test_price_lag1_is_yesterdays_return():
    t = _targets()
    px = price_features.build_price_features(t, CFG)
    assert np.isclose(px["px_ret_lag1"].iloc[10], t["ret_USD_IDR"].iloc[9])


def test_news_bucket_meets_the_next_fix_not_its_own():
    """NLP features of bucket day D sit on the row of the first fix after D."""
    idx = pd.bdate_range("2024-03-04", periods=5)
    t = _targets(5).set_axis(idx)
    dataset = pd.DataFrame({"n_articles": 1.0, "y_return": t["y_return"],
                            "y_direction": t["y_direction"], "split": "train",
                            "feature_lag_days": 1}, index=idx)
    nlp_daily = pd.DataFrame({"nlp_x": [10.0, 20.0, 30.0, 40.0, 50.0]}, index=idx)

    frame, groups = run_task2.assemble_frame(dataset, t, nlp_daily, CFG)
    assert frame.loc[idx[2], "nlp_x"] == 20.0, "day-1 news must meet day-2's fix"
    assert frame.loc[idx[2], "persist_direction"] == t["y_direction"].iloc[1]
    assert groups["nlp"] == ["nlp_x"] and "n_articles" in groups["task1_news"]


def test_arima_forecast_for_day_t_ignores_day_t():
    y = np.random.default_rng(1).normal(0, 0.3, 230)
    fit, ev = y[:200], y[200:]
    p1 = models.arima_one_step(fit, ev, (1, 0, 0))
    ev2 = ev.copy()
    ev2[10] += 10.0
    p2 = models.arima_one_step(fit, ev2, (1, 0, 0))
    assert np.allclose(p1[:11], p2[:11]), "forecasts up to day 10 cannot know day 10"
    assert not np.isclose(p1[11], p2[11]), "day 10 is used from day 11 on"


# ===========================================================================
# 2. NLP features
# ===========================================================================
def test_tfidf_vocabulary_is_learned_from_training_news_only():
    days = pd.bdate_range("2024-01-01", periods=10)
    cutoff = days[6]
    rows = [{"news_day": d, "text_track_a": ("war oil price" if d < cutoff else "zzzfuture token"),
             "dup_count": 1} for d in days for _ in range(5)]
    arts = pd.DataFrame(rows)
    cfg = _cfg_with(min_df=1, max_df=1.0, svd_components=2)

    res = nlp_features.tfidf_svd_features(arts, pd.Series(1.0, index=arts.index), cutoff, cfg)
    assert "war" in res.vectorizer.vocabulary_
    assert "zzzfuture" not in res.vectorizer.vocabulary_, "test-period words leaked into the vocabulary"
    assert res.n_fit_articles == 30


def test_lm_counts_counts_each_category():
    lex = {"negative": frozenset({"crisis", "loss"}), "uncertainty": frozenset({"uncertain"})}
    out = nlp_features.lm_counts(["crisis deepen loss", "", "uncertain outlook"], lex)
    assert out["lm_negative"].tolist() == [2, 0, 0]
    assert out["lm_uncertainty"].tolist() == [0, 0, 1]
    assert out["n_tokens"].tolist() == [3, 0, 2]


def test_syndication_weight_is_damped():
    arts = pd.DataFrame({"dup_count": [1, 200]})
    w = nlp_features.article_weights(arts, CFG)
    assert w.iloc[1] / w.iloc[0] < 10, "200 copies should count more, but not 200x more"


# ===========================================================================
# 3. Metrics
# ===========================================================================
def test_directional_hit_rate_counts_only_committed_calls():
    m = evaluate.classification_metrics(["UP", "DOWN", "FLAT", "UP"],
                                        ["UP", "UP", "FLAT", "FLAT"])
    assert m["accuracy"] == 0.5
    assert m["coverage"] == 0.5          # two UP/DOWN calls out of four days
    assert m["directional_hit_rate"] == 0.5


def test_bootstrap_sees_no_gain_between_identical_models():
    y = np.array(["UP", "DOWN", "FLAT"] * 20, dtype=object)
    p = np.roll(y, 1)
    ci = evaluate.block_bootstrap_gain("cls", y, p, p, n_boot=200)
    assert ci["ci_low"] == ci["ci_high"] == 0.0 and ci["p_no_gain"] == 1.0

    better = evaluate.block_bootstrap_gain("cls", y, p, y, n_boot=200)
    assert better["ci_low"] > 0 and better["p_no_gain"] == 0.0


def test_zero_forecast_scores_zero_r2():
    m = evaluate.regression_metrics([0.2, -0.1, 0.3], [0.0, 0.0, 0.0])
    assert m["r2_vs_zero"] == 0.0
