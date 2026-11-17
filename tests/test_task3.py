"""Tests for Task 3: the decisions that would silently invalidate the ablation.

Run with:  py -m pytest tests/test_task3.py -v

* text features must be lagged one trading day (news t-1 -> fix t);
* PCA axes, PPMI word vectors and topic centroids may only see training news;
* temporal decay must be causal (no future news leaks backwards);
* salience attention must be a proper within-day softmax;
* late fusion with w = 0 must be exactly the time-series model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import fusion, run_task3, text_embeddings, text_features  # noqa: E402
from src.config import load_config  # noqa: E402

CFG = load_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")


def _articles(n_days=30, per_day=5, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2024-01-01", periods=n_days)
    nd = np.repeat(days, per_day)
    return pd.DataFrame({"news_day": nd, "dup_count": rng.integers(1, 5, len(nd)),
                         "theme_family": "conflict", "text_track_a": "war oil", "text_track_b": "War, oil"})


def test_lag_to_rows_pairs_previous_bucket():
    cal = pd.bdate_range("2024-01-01", periods=6)
    daily = pd.DataFrame({"x": np.arange(6.0)}, index=cal)
    out = run_task3.lag_to_rows(daily, cal, cal[1:])
    assert out["x"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]       # row t holds bucket t-1


def test_daily_mean_pooling_is_weighted_mean():
    arts = _articles(n_days=3, per_day=4)
    E = np.random.default_rng(1).normal(size=(len(arts), 5)).astype(np.float16)
    D = text_features.daily_vectors(E, arts, CFG, how="mean")
    w = np.log1p(arts["dup_count"].to_numpy())
    first = np.average(E[:4].astype(np.float64), axis=0, weights=w[:4])
    assert np.allclose(D.iloc[0].to_numpy(), first, atol=1e-3)


def test_decay_is_causal():
    idx = pd.bdate_range("2024-01-01", periods=10)
    a = pd.DataFrame({"x": np.r_[np.zeros(9), 100.0]}, index=idx)   # shock on the last day only
    sm = text_features.decay_smooth(a, halflife=3)
    assert (sm["x"].iloc[:9] == 0).all()                            # nothing leaks backwards


def test_salience_attention_is_within_day_softmax():
    arts = _articles(n_days=2, per_day=3)
    arts["dup_count"] = 2
    probs = pd.DataFrame({"finbert_neutral": [0.9, 0.1, 0.5, 0.2, 0.2, 0.2]})
    _, didx = text_features.day_index(arts)
    e = text_features.salience_weights(arts, probs, CFG, didx)
    P = text_features.pooling_matrix(didx, e, 2).toarray()
    assert np.allclose(P.sum(axis=1), 1.0)
    assert P[0, 1] > P[0, 2] > P[0, 0]       # less neutral headline -> more attention


def test_pca_fits_on_training_days_only():
    idx = pd.bdate_range("2024-01-01", periods=40)
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(40, 6)), index=idx)
    X.iloc[30:] += 1000.0                                           # wild "test period"
    fit = np.r_[np.ones(30, bool), np.zeros(10, bool)]
    pca = text_features.TrainPCA(2, 0)
    pca.fit_transform(X, fit, "t")
    assert np.allclose(pca.pca.mean_, X.iloc[:30].mean().to_numpy())


def test_ppmi_vocabulary_comes_from_training_headlines_only():
    toks = pd.Series(["war oil price"] * 30 + ["tariff trade deal"] * 30)
    fit = np.r_[np.ones(30, bool), np.zeros(30, bool)]
    vec, W = text_embeddings.train_ppmi_svd(toks, fit, dim=2, min_count=1)
    assert "tariff" not in vec.vocabulary_ and "war" in vec.vocabulary_
    E = text_embeddings.encode_ppmi(toks, vec, W)
    assert np.allclose(E[30:].astype(float), 0.0)                   # unseen words -> zero vector


def test_late_blend_with_zero_weight_is_time_model():
    rng = np.random.default_rng(0)
    P1, P2 = rng.dirichlet(np.ones(3), 20), rng.dirichlet(np.ones(3), 20)
    assert np.allclose(fusion.late_blend(P1, P2, 0.0), P1, atol=1e-5)
    assert np.allclose(fusion.late_blend(P1, P2, 0.3).sum(axis=1), 1.0)


def test_attention_net_masks_padding():
    import torch
    torch.set_num_threads(1)     # XGBoost (Task 2 tests) already loaded its OpenMP runtime; see run_task3
    net =fusion.build_attention_net(d_text=4, d_time=3, hidden=8, dropout=0.0)
    H = torch.randn(2, 5, 4)
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    _, attn = net(H, mask, torch.zeros(2, 5), torch.randn(2, 3))
    assert torch.allclose(attn.sum(-1), torch.ones(2))
    assert attn[0, 2:].abs().max() < 1e-6                          # padded slots get no attention
