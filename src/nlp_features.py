"""Task 2 — turn geopolitical headlines into daily numeric features.

Three extractors, all classical. The Task 2 brief forbids pre-trained
embeddings and transformer models, so nothing here downloads a model.

  VADER              general-purpose sentiment lexicon plus rules (negation,
                     intensifiers, capitals, "!"). Run on Track B, the near-raw
                     headline, because those rules read exactly the case and
                     punctuation that Track A strips.
  Loughran-McDonald  finance-domain word lists (negative, positive,
                     uncertainty, litigious, constraining). Run on Track A,
                     whose lowercased lemmas match dictionary entries. Chosen
                     because general lexicons misread finance words: "tax",
                     "liability" and "crude" are not negative in a market story.
  TF-IDF -> SVD      a vocabulary learned from OUR headlines, so it can pick up
                     geopolitical terms neither lexicon lists ("invade",
                     "missile", "tariff"). Reduced to a few dense dimensions
                     (latent semantic analysis) because 727 training days cannot
                     support 20,000 sparse columns.

Why headlines. GDELT publishes no article bodies (Task 1, cleaning section),
and a headline is the editor's own one-line summary of the event. Entities
enter through Track A, which already merges US / China / Russia / EU / Fed / ECB
aliases into single tokens that TF-IDF can weight.

Unit of analysis. Articles are scored one by one, then averaged within the
trading-day bucket built in Task 1 (``news_day``). Each article is weighted by
log1p(dup_count): a story copied 200 times counts more than one copied twice,
but not 100x more.

Leakage. Everything *fitted* here (TF-IDF vocabulary and IDF weights, SVD axes)
is fitted on training-period news only. The lexicons are fixed lists and fit on
nothing. Rolling z-scores use trailing windows. The daily frame returned here is
NOT yet lagged; ``run_task2.assemble_frame`` applies the same one-trading-day
shift as Task 1's ``align.make_prediction_frame``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .features import _rolling_z

log = logging.getLogger(__name__)

ARTICLE_COLS = ["news_day", "theme_family", "dup_count", "text_track_a", "text_track_b"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_kept_articles(cfg) -> pd.DataFrame:
    """Articles that passed every Task 1 filter (``keep``), text columns only."""
    path = cfg.path("interim") / "articles_flagged.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. It is not on GitHub (> 100 MB); rebuild it with "
            "`py -m src.build_dataset` (about 30 min from the GDELT cache)."
        )
    df = pd.read_parquet(path, columns=ARTICLE_COLS + ["keep"])
    df = df.loc[df["keep"]].drop(columns="keep").reset_index(drop=True)
    df["news_day"] = pd.to_datetime(df["news_day"])
    for col in ("text_track_a", "text_track_b"):
        df[col] = df[col].fillna("").astype(str)
    log.info("loaded %d kept articles, %s..%s", len(df),
             df["news_day"].min().date(), df["news_day"].max().date())
    return df


def load_lm_lexicon(path) -> Dict[str, frozenset]:
    """Category -> word set, from ``scripts/make_lm_lexicon.py``'s CSV."""
    lex = pd.read_csv(path)
    lex["word"] = lex["word"].astype(str).str.lower()
    cats = [c for c in lex.columns if c != "word"]
    return {c: frozenset(lex.loc[lex[c] > 0, "word"]) for c in cats}


# ---------------------------------------------------------------------------
# Article-level scoring
# ---------------------------------------------------------------------------
def lm_counts(texts: Iterable[str], lexicon: Dict[str, frozenset]) -> pd.DataFrame:
    """Per headline: how many tokens fall in each LM category, and the token count."""
    rows = []
    for text in texts:
        toks = text.split() if isinstance(text, str) else []
        row = {f"lm_{cat}": sum(tok in words for tok in toks) for cat, words in lexicon.items()}
        row["n_tokens"] = len(toks)
        rows.append(row)
    return pd.DataFrame(rows)


def vader_scores(texts: Iterable[str]) -> pd.DataFrame:
    """VADER compound / pos / neg per headline; empty headline -> NaN."""
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    try:
        sia = SentimentIntensityAnalyzer()
    except LookupError as exc:  # lexicon not downloaded yet
        raise RuntimeError(
            "VADER lexicon missing. Run once: py -m nltk.downloader vader_lexicon"
        ) from exc

    empty = {"compound": np.nan, "pos": np.nan, "neg": np.nan}
    recs = [sia.polarity_scores(t) if t else empty for t in texts]
    out = pd.DataFrame(recs)[["compound", "pos", "neg"]]
    return out.add_prefix("vader_")


def score_articles(articles: pd.DataFrame, cfg, use_cache: bool = True) -> pd.DataFrame:
    """VADER + LM scores for every kept article, cached (VADER takes minutes).

    The cache stores ``news_day`` alongside the scores and is reused only if it
    lines up row for row with the current corpus, so a rebuilt corpus is never
    matched to stale scores.
    """
    t2 = cfg.dig("task2", default={}) or {}
    cache = cfg.root / t2["paths"]["article_scores"]
    if use_cache and cache.exists():
        cached = pd.read_parquet(cache)
        if len(cached) == len(articles) and cached["news_day"].equals(articles["news_day"]):
            log.info("article scores: reusing cache %s", cache)
            return cached.drop(columns="news_day")
        log.warning("article score cache does not match the corpus; rescoring")

    vcol = t2["vader"]["text_col"]
    lcfg = t2["loughran_mcdonald"]
    lexicon = load_lm_lexicon(cfg.root / lcfg["path"])
    lexicon = {c: w for c, w in lexicon.items() if c in lcfg["categories"]}

    log.info("scoring %d headlines with VADER (%s) and Loughran-McDonald (%s)",
             len(articles), vcol, lcfg["text_col"])
    scores = pd.concat([vader_scores(articles[vcol]),
                        lm_counts(articles[lcfg["text_col"]], lexicon)], axis=1)
    scores.index = articles.index

    cache.parent.mkdir(parents=True, exist_ok=True)
    scores.assign(news_day=articles["news_day"].values).to_parquet(cache)
    return scores


def article_weights(articles: pd.DataFrame, cfg) -> pd.Series:
    """log1p(dup_count) if syndication weighting is on, else 1."""
    if cfg.dig("task2", "weight_by_syndication", default=True):
        return np.log1p(articles["dup_count"].clip(lower=1).astype(float))
    return pd.Series(1.0, index=articles.index)


# ---------------------------------------------------------------------------
# Daily aggregation of lexicon scores
# ---------------------------------------------------------------------------
def _weighted_mean(values: pd.Series, weights: pd.Series, keys) -> pd.Series:
    """Weighted mean per group, ignoring NaN values (and their weights)."""
    ok = values.notna()
    num = (values[ok] * weights[ok]).groupby([k[ok] for k in keys]).sum()
    den = weights[ok].groupby([k[ok] for k in keys]).sum()
    return num / den.replace(0, np.nan)


def _weighted_ratio(num_col: pd.Series, den_col: pd.Series, weights: pd.Series, keys) -> pd.Series:
    """sum(w * num) / sum(w * den) per group — a token-weighted rate."""
    num = (num_col * weights).groupby(keys).sum()
    den = (den_col * weights).groupby(keys).sum()
    return num / den.replace(0, np.nan)


def aggregate_sentiment(articles: pd.DataFrame, scores: pd.DataFrame, cfg) -> pd.DataFrame:
    """One row per ``news_day``: overall and per-theme lexicon features."""
    vcfg = cfg.dig("task2", "vader", default={}) or {}
    cats = cfg.dig("task2", "loughran_mcdonald", "categories", default=[]) or []
    w = article_weights(articles, cfg)
    day = articles["news_day"]
    fam = articles["theme_family"]

    compound = scores["vader_compound"]
    is_neg = (compound <= float(vcfg.get("neg_threshold", -0.05))).where(compound.notna())
    is_pos = (compound >= float(vcfg.get("pos_threshold", 0.05))).where(compound.notna())

    out = pd.DataFrame({
        "nlp_vader_mean": _weighted_mean(compound, w, [day]),
        "nlp_vader_std": compound.groupby(day).std(),
        "nlp_vader_neg_share": _weighted_mean(is_neg.astype(float), w, [day]),
        "nlp_vader_pos_share": _weighted_mean(is_pos.astype(float), w, [day]),
    })
    for cat in cats:
        out[f"nlp_lm_{cat}_rate"] = _weighted_ratio(scores[f"lm_{cat}"], scores["n_tokens"], w, day)

    pos = (scores["lm_positive"] * w).groupby(day).sum()
    neg = (scores["lm_negative"] * w).groupby(day).sum()
    out["nlp_lm_net_tone"] = (pos - neg) / (pos + neg).replace(0, np.nan)
    out["nlp_lm_neg_share"] = _weighted_mean((scores["lm_negative"] > 0).astype(float), w, [day])

    # Per theme family: is conflict news getting darker while monetary news is calm?
    fam_vader = _weighted_mean(compound, w, [day, fam]).unstack()
    fam_lmneg = _weighted_ratio(scores["lm_negative"], scores["n_tokens"], w, [day, fam]).unstack()
    for f in sorted(fam.dropna().unique()):
        out[f"nlp_vader_mean_{f}"] = fam_vader.get(f)
        out[f"nlp_lm_negative_rate_{f}"] = fam_lmneg.get(f)

    out.index.name = "date"
    return out.sort_index()


def lexicon_coverage(articles: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """How many headlines does each lexicon actually say anything about?

    A lexicon that scores 70% of headlines as exactly neutral is mostly silent,
    whatever its daily average looks like. This table is the evidence for that
    judgement in the report.
    """
    lm_cols = [c for c in scores.columns if c.startswith("lm_")]
    rows = [{"measure": "headlines", "value": float(len(scores))},
            {"measure": "vader_nonzero_share", "value": float((scores["vader_compound"].fillna(0) != 0).mean())},
            {"measure": "vader_negative_share", "value": float((scores["vader_compound"] <= -0.05).mean())},
            {"measure": "vader_positive_share", "value": float((scores["vader_compound"] >= 0.05).mean())},
            {"measure": "lm_any_hit_share", "value": float((scores[lm_cols].sum(axis=1) > 0).mean())}]
    rows += [{"measure": f"{c}_hit_share", "value": float((scores[c] > 0).mean())} for c in lm_cols]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TF-IDF -> SVD daily topic vectors
# ---------------------------------------------------------------------------
@dataclass
class TfidfResult:
    features: pd.DataFrame        # one row per news_day, nlp_tfidf_svd_XX columns
    vectorizer: TfidfVectorizer
    svd: TruncatedSVD
    n_fit_articles: int
    fit_before: pd.Timestamp


def _daily_mean_matrix(X: sparse.csr_matrix, day_idx: np.ndarray, weights: np.ndarray,
                       n_days: int) -> sparse.csr_matrix:
    """Weighted mean of the article rows of ``X`` within each day."""
    W = sparse.csr_matrix((weights, (day_idx, np.arange(len(day_idx)))),
                          shape=(n_days, len(day_idx)))
    totals = np.asarray(W.sum(axis=1)).ravel()
    totals[totals == 0] = 1.0
    return sparse.diags(1.0 / totals) @ W @ X


def tfidf_svd_features(articles: pd.DataFrame, weights: pd.Series,
                       fit_before: pd.Timestamp, cfg) -> TfidfResult:
    """Daily TF-IDF centroids projected onto SVD axes learned from training news.

    ``fit_before``: only articles bucketed strictly before this trading day
    (the last training-row date) are used to learn the vocabulary, the IDF
    weights and the SVD axes. News bucketed on that day is paired with the
    first validation fix, so it stays out.
    """
    tcfg = cfg.dig("task2", "tfidf", default={}) or {}
    texts = articles[tcfg.get("text_col", "text_track_a")]
    has_text = (texts.str.len() > 0).to_numpy()
    fit_mask = (articles["news_day"] < fit_before).to_numpy() & has_text

    vec = TfidfVectorizer(
        ngram_range=tuple(tcfg.get("ngram_range", [1, 2])),
        min_df=int(tcfg.get("min_df", 50)),
        max_df=float(tcfg.get("max_df", 0.5)),
        max_features=int(tcfg.get("max_features", 20000)),
        sublinear_tf=bool(tcfg.get("sublinear_tf", True)),
        lowercase=False,                 # Track A is already lowercased
        token_pattern=r"(?u)[^\s]+",     # Track A is already tokenised; keep <num>, federal_reserve
    )
    vec.fit(texts[fit_mask])
    X = vec.transform(texts)
    log.info("TF-IDF: vocabulary %d terms, fitted on %d headlines before %s",
             len(vec.vocabulary_), int(fit_mask.sum()), fit_before.date())

    days = pd.DatetimeIndex(sorted(articles["news_day"].unique()))
    day_idx = days.get_indexer(articles["news_day"])
    w = weights.to_numpy(dtype=float) * has_text     # empty headlines add nothing
    D = _daily_mean_matrix(X.tocsr(), day_idx, w, len(days))

    fit_days = np.asarray(days < fit_before)
    svd = TruncatedSVD(n_components=int(tcfg.get("svd_components", 20)), random_state=cfg.seed)
    svd.fit(D[fit_days])
    Z = svd.transform(D)
    log.info("SVD: %d components explain %.1f%% of training-day TF-IDF variance",
             svd.n_components, 100 * svd.explained_variance_ratio_.sum())

    feats = pd.DataFrame(Z, index=days,
                         columns=[f"nlp_tfidf_svd_{k + 1:02d}" for k in range(Z.shape[1])])
    no_text = np.bincount(day_idx, weights=w, minlength=len(days)) == 0
    feats.loc[no_text] = np.nan
    feats.index.name = "date"
    return TfidfResult(feats, vec, svd, int(fit_mask.sum()), fit_before)


def svd_top_terms(result: TfidfResult, n_terms: int = 12) -> pd.DataFrame:
    """The terms that load most (+ and -) on each SVD axis: what each axis 'means'."""
    terms = np.asarray(result.vectorizer.get_feature_names_out())
    rows = []
    for k, comp in enumerate(result.svd.components_):
        order = np.argsort(comp)
        rows.append({
            "component": f"nlp_tfidf_svd_{k + 1:02d}",
            "explained_variance_ratio": float(result.svd.explained_variance_ratio_[k]),
            "top_positive": ", ".join(terms[order[::-1][:n_terms]]),
            "top_negative": ", ".join(terms[order[:n_terms]]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def build_nlp_features(cfg, trading_days: pd.DatetimeIndex, fit_before: pd.Timestamp,
                       articles: pd.DataFrame | None = None, use_cache: bool = True):
    """All NLP features on the trading calendar, UNLAGGED (row = news bucket day).

    Returns ``(daily, extras)`` where ``extras`` holds the diagnostics tables
    for the report (lexicon coverage, SVD top terms).
    """
    if articles is None:
        articles = load_kept_articles(cfg)
    scores = score_articles(articles, cfg, use_cache=use_cache)
    weights = article_weights(articles, cfg)

    sentiment = aggregate_sentiment(articles, scores, cfg)
    tfidf = tfidf_svd_features(articles, weights, fit_before, cfg)

    td = pd.DatetimeIndex(trading_days).normalize()
    daily = sentiment.join(tfidf.features, how="outer").reindex(td)

    window = int(cfg.dig("task2", "zscore_window", default=30))
    for col in cfg.dig("task2", "zscore_columns", default=[]) or []:
        if col in daily.columns:
            daily[f"{col}_z{window}"] = _rolling_z(daily[col], window)

    daily.index.name = "date"
    extras = {
        "lexicon_coverage": lexicon_coverage(articles, scores),
        "tfidf_svd_top_terms": svd_top_terms(tfidf),
        "tfidf_fit": {"fit_before": str(fit_before.date()),
                      "n_fit_articles": tfidf.n_fit_articles,
                      "vocabulary_size": len(tfidf.vectorizer.vocabulary_),
                      "svd_explained_variance": float(tfidf.svd.explained_variance_ratio_.sum())},
    }
    log.info("NLP features: %d days x %d columns", len(daily), daily.shape[1])
    return daily, extras


__all__ = [
    "load_kept_articles", "load_lm_lexicon", "lm_counts", "vader_scores",
    "score_articles", "article_weights", "aggregate_sentiment", "lexicon_coverage",
    "TfidfResult", "tfidf_svd_features", "svd_top_terms", "build_nlp_features",
]
