"""Task 3 — from per-headline vectors to one row of text features per day.

A trading day has ~760 kept headlines, and the model needs one fixed-size row.
This module does that reduction in three steps, and each step is something the
Task 3 brief asks us to experiment with:

1. Temporal pooling: how do a day's headlines become one vector?
     mean       weighted mean, weight = log1p(dup_count) (same weight as Task 2)
     decay      the daily means, smoothed with an exponential decay over past
                days (half-life 3 days). News does not vanish at midnight: a war
                that started yesterday still matters today, just a bit less.
     attention  "salience attention": within a day, each headline gets
                softmax(log w_dup + (1 - p_neutral) / T). Headlines that are both
                widely syndicated and emotionally charged (FinBERT is confident
                they are not neutral) dominate the day. T = 0.25. This is a fixed
                (not learned) attention; the learned version is the attention
                network in ``fusion.py``.
2. Compression: PCA fitted on TRAINING days only, 16 components by default.
   726 training days cannot support 768 raw columns.
3. Extra views of the same headlines:
     finbert_sent  daily FinBERT tone: mean P(pos), P(neg), net = P(pos) - P(neg),
                   share of headlines FinBERT labels negative, spread, a 30-day
                   z-score, and the net tone per Task 1 theme family.
     topics        BERTopic-style topics: MiniLM vectors -> k-means (k = 20,
                   centroids fitted on training headlines) -> labels from
                   class-based TF-IDF. Per day: share of each topic and the
                   FinBERT net tone inside each topic.

Every daily frame returned here is UNLAGGED (row = news bucket day). The driver
applies the same one-trading-day shift as Task 1 and Task 2.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA

from .features import _rolling_z
from .nlp_features import article_weights

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def day_index(articles: pd.DataFrame) -> tuple[pd.DatetimeIndex, np.ndarray]:
    days = pd.DatetimeIndex(sorted(articles["news_day"].unique()))
    return days, days.get_indexer(articles["news_day"])


def pooling_matrix(day_idx: np.ndarray, weights: np.ndarray, n_days: int) -> sparse.csr_matrix:
    """Row-normalised (days x articles) matrix: D = P @ E is the weighted daily mean."""
    W = sparse.csr_matrix((weights, (day_idx, np.arange(len(day_idx)))),
                          shape=(n_days, len(day_idx)))
    tot = np.asarray(W.sum(axis=1)).ravel()
    tot[tot == 0] = 1.0
    return (sparse.diags(1.0 / tot) @ W).tocsr()


def chunked_pool(P: sparse.csr_matrix, E: np.ndarray, chunk: int = 100_000) -> np.ndarray:
    """P @ E without converting the whole float16 memmap to float32 at once."""
    P = P.tocsc()
    out = np.zeros((P.shape[0], E.shape[1]), dtype=np.float64)
    for s in range(0, E.shape[0], chunk):
        e = np.asarray(E[s:s + chunk], dtype=np.float32)
        out += P[:, s:s + chunk] @ e
    return out


def salience_weights(articles: pd.DataFrame, probs: pd.DataFrame, cfg,
                     day_idx: np.ndarray) -> np.ndarray:
    """Within-day softmax over log(dup weight) + (1 - p_neutral) / T."""
    T = float(cfg.dig("task3", "pooling", "attention_temperature", default=0.25))
    w = article_weights(articles, cfg).to_numpy(dtype=float)
    score = np.log(np.clip(w, 1e-6, None)) + (1.0 - probs["finbert_neutral"].to_numpy()) / T
    # Softmax within each day: subtract the day's max for numerical stability.
    day_max = pd.Series(score).groupby(day_idx).transform("max").to_numpy()
    return np.exp(score - day_max)   # pooling_matrix row-normalises = the softmax denominator


def decay_smooth(daily: pd.DataFrame, halflife: float) -> pd.DataFrame:
    """Causal exponential decay over past news days (row t uses rows <= t only)."""
    return daily.ewm(halflife=halflife, adjust=True, ignore_na=True).mean()


class TrainPCA:
    """PCA whose centring and axes come from training days only."""

    def __init__(self, k: int, seed: int):
        self.pca = PCA(n_components=k, random_state=seed)
        self.k = k

    def fit_transform(self, X: pd.DataFrame, fit_mask: np.ndarray, prefix: str) -> pd.DataFrame:
        ok = X.notna().all(axis=1).to_numpy()
        self.pca.fit(X.to_numpy()[fit_mask & ok])
        Z = np.full((len(X), self.k), np.nan)
        Z[ok] = self.pca.transform(X.to_numpy()[ok])
        cols = [f"{prefix}_pc{j + 1:02d}" for j in range(self.k)]
        return pd.DataFrame(Z, index=X.index, columns=cols)


# ---------------------------------------------------------------------------
# Embedding blocks
# ---------------------------------------------------------------------------
def daily_vectors(E: np.ndarray, articles: pd.DataFrame, cfg, how: str = "mean",
                  probs: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per news day: the pooled headline vector (no PCA yet)."""
    days, didx = day_index(articles)
    if how in ("mean", "decay"):
        w = article_weights(articles, cfg).to_numpy(dtype=float)
    elif how == "attention":
        w = salience_weights(articles, probs, cfg, didx)
    else:
        raise ValueError(how)
    D = pd.DataFrame(chunked_pool(pooling_matrix(didx, w, len(days)), E), index=days)
    if how == "decay":
        hl = float(cfg.dig("task3", "pooling", "decay_halflife_days", default=3))
        D = decay_smooth(D, hl)
    D.index.name = "date"
    return D


def embedding_block(E, articles, cfg, name: str, fit_before: pd.Timestamp,
                    how: str = "mean", k: int | None = None,
                    probs: pd.DataFrame | None = None, daily: pd.DataFrame | None = None):
    """Pooled daily vectors -> PCA (train days) -> ``t3_{name}_{how}_pcXX`` columns.

    Returns (features, fitted TrainPCA, the un-reduced daily vectors). Pass
    ``daily`` to reuse already-pooled vectors (e.g. when only k changes).
    """
    k = int(k or cfg.dig("task3", "pca_components", default=16))
    D = daily if daily is not None else daily_vectors(E, articles, cfg, how=how, probs=probs)
    pca = TrainPCA(k, cfg.seed)
    Z = pca.fit_transform(D, np.asarray(D.index < fit_before), f"t3_{name}_{how}")
    log.info("%s/%s: PCA %d comps explain %.1f%% of training-day variance",
             name, how, k, 100 * pca.pca.explained_variance_ratio_.sum())
    return Z, pca, D


# ---------------------------------------------------------------------------
# FinBERT sentiment block
# ---------------------------------------------------------------------------
def finbert_sentiment_block(articles: pd.DataFrame, probs: pd.DataFrame, cfg) -> pd.DataFrame:
    """Daily FinBERT tone, overall and per Task 1 theme family."""
    w = article_weights(articles, cfg)
    day = articles["news_day"]
    p = probs.set_index(articles.index)
    net = p["finbert_positive"] - p["finbert_negative"]
    is_neg = (p.idxmax(axis=1) == "finbert_negative").astype(float)

    def wmean(v, keys):
        return (v * w).groupby(keys).sum() / w.groupby(keys).sum()

    out = pd.DataFrame({
        "t3_finbert_pos": wmean(p["finbert_positive"], day),
        "t3_finbert_neg": wmean(p["finbert_negative"], day),
        "t3_finbert_neu": wmean(p["finbert_neutral"], day),
        "t3_finbert_net": wmean(net, day),
        "t3_finbert_neg_share": wmean(is_neg, day),
        "t3_finbert_net_std": net.groupby(day).std(),
    })
    out["t3_finbert_net_z30"] = _rolling_z(out["t3_finbert_net"], 30)
    fam = wmean(net, [day, articles["theme_family"]]).unstack()
    for f in sorted(articles["theme_family"].dropna().unique()):
        out[f"t3_finbert_net_{f}"] = fam.get(f)
    out.index.name = "date"
    return out.sort_index()


# ---------------------------------------------------------------------------
# Topic block (BERTopic-style, without the UMAP/HDBSCAN dependencies)
# ---------------------------------------------------------------------------
def fit_topics(E: np.ndarray, articles: pd.DataFrame, fit_before: pd.Timestamp, cfg):
    """k-means on L2-normalised MiniLM vectors of a sample of training headlines.

    BERTopic = embed -> reduce -> cluster -> c-TF-IDF labels. We keep the embed,
    cluster and label steps and use k-means instead of HDBSCAN so that every
    headline gets a topic (HDBSCAN puts a large share into an "outlier" bin)
    and k is fixed in advance.
    """
    from sklearn.cluster import MiniBatchKMeans
    tcfg = cfg.dig("task3", "topics", default={}) or {}
    k = int(tcfg.get("n_topics", 20))
    train_idx = np.flatnonzero((articles["news_day"] < fit_before).to_numpy())
    rng = np.random.default_rng(cfg.seed)
    n_fit = min(len(train_idx), int(tcfg.get("fit_sample", 200_000)))
    sample = np.sort(rng.choice(train_idx, n_fit, replace=False))
    km = MiniBatchKMeans(n_clusters=k, random_state=cfg.seed, batch_size=4096, n_init=5)
    km.fit(np.asarray(E[sample], dtype=np.float32))
    labels = np.empty(len(articles), dtype=np.int16)
    for s in range(0, len(articles), 100_000):
        labels[s:s + 100_000] = km.predict(np.asarray(E[s:s + 100_000], dtype=np.float32))
    return km, labels, train_idx


def topic_labels(articles: pd.DataFrame, labels: np.ndarray, train_idx: np.ndarray,
                 n_words: int = 8) -> pd.DataFrame:
    """Class-based TF-IDF (BERTopic's labelling step) on training headlines.

    All headlines of a topic are treated as one document; a word scores high if
    it is frequent in this topic and rare across topics.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    k = int(labels.max()) + 1
    docs = (pd.Series(articles["text_track_a"].to_numpy()[train_idx])
            .groupby(labels[train_idx]).apply(" ".join).reindex(range(k), fill_value=""))
    cv = CountVectorizer(min_df=1, token_pattern=r"(?u)\b[a-z][a-z_]{2,}\b")
    X = cv.fit_transform(docs).astype(float)
    tf = X.multiply(1 / np.asarray(X.sum(axis=1)).clip(min=1)).tocsr()
    avg = X.sum() / k
    idf = np.log(1 + avg / np.asarray(X.sum(axis=0)).ravel().clip(min=1))
    ctfidf = np.asarray(tf.multiply(idf).todense())
    vocab = cv.get_feature_names_out()
    sizes = np.bincount(labels[train_idx], minlength=k)
    rows = [{"topic": t, "train_headlines": int(sizes[t]),
             "share": float(sizes[t] / sizes.sum()),
             "top_words": ", ".join(vocab[np.argsort(ctfidf[t])[::-1][:n_words]])}
            for t in range(k)]
    return pd.DataFrame(rows)


def topic_block(articles: pd.DataFrame, labels: np.ndarray, probs: pd.DataFrame, cfg) -> pd.DataFrame:
    """Per day: weighted share of each topic, and FinBERT net tone inside each topic."""
    w = article_weights(articles, cfg)
    day = articles["news_day"]
    lab = pd.Series(labels.astype(int), index=articles.index)
    wsum = w.groupby([day, lab]).sum().unstack(fill_value=0.0)
    share = wsum.div(wsum.sum(axis=1), axis=0)
    net = pd.Series(probs["finbert_positive"].to_numpy() - probs["finbert_negative"].to_numpy(),
                    index=articles.index)
    tone = (net * w).groupby([day, lab]).sum().unstack() / wsum.replace(0, np.nan)
    share.columns = [f"t3_topic_share_{int(t):02d}" for t in share.columns]
    tone.columns = [f"t3_topic_tone_{int(t):02d}" for t in tone.columns]
    out = pd.concat([share, tone], axis=1)
    out.index.name = "date"
    return out.sort_index()


__all__ = ["day_index", "pooling_matrix", "chunked_pool", "salience_weights", "decay_smooth",
           "TrainPCA", "daily_vectors", "embedding_block", "finbert_sentiment_block",
           "fit_topics", "topic_labels", "topic_block"]
