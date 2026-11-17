"""Task 3 — headline representations: static and contextual embeddings.

Task 2 was limited to lexicons and a TF-IDF we fitted ourselves. Task 3 lifts
that limit, so this module turns headlines into dense vectors with four
encoders. The static encoders and MiniLM cover all 905,713 kept headlines;
FinBERT covers the Task 3 corpus (the 100 most-syndicated headlines per news
day, 119,100), and every Task 3 feature is computed on that corpus. They are deliberately
paired so that each comparison isolates one design choice:

                        general-purpose                  domain-specific
  static (1 vector     GloVe 6B-300d                    PPMI-SVD word vectors trained
  per word type)       (Wikipedia + Gigaword)           on OUR training-period headlines
  contextual           all-MiniLM-L6-v2                 FinBERT (ProsusAI/finbert)
  (vector depends      (sentence-similarity model,      (BERT further pre-trained on
  on the sentence)     1B sentence pairs)               financial text, fine-tuned on
                                                         Financial PhraseBank sentiment)

Why these four
  * GloVe vs PPMI-SVD: same family (count-based static vectors), but one was
    learned on general English and one on our geopolitical news. Levy & Goldberg
    (2014) showed that SVD on a shifted PPMI matrix optimises essentially the
    same objective as Word2Vec skip-gram with negative sampling, so PPMI-SVD is
    our "Word2Vec trained on our corpus" without a gensim dependency (gensim
    has no wheel for Python 3.14).
  * MiniLM vs FinBERT: both are transformers, but MiniLM was trained to make
    sentences with similar meaning sit close together (good for topics), while
    FinBERT was trained to read financial tone (good for sentiment). This is the
    "domain-specific vs general-purpose contextual embedding" comparison the
    Task 3 brief asks for.

Tokenisation and pooling (the report explains each choice)
  * Contextual models read Track B (cased, punctuation kept), because their
    WordPiece tokenisers were trained on raw text. Track A's lemmas would feed
    them text they never saw in pre-training.
  * max_length = 64 WordPiece tokens. 99% of headlines have <= 22 words, so
    nothing real is truncated, and short padding keeps the GPU batches fast.
  * MiniLM: mean pooling over non-padding tokens, then L2-normalised, which is
    exactly how the model was trained (sentence-transformers recipe).
  * FinBERT: two outputs per headline. (1) The sentiment head's softmax
    (positive / negative / neutral), which is what the [CLS] token was
    fine-tuned for. (2) A mean-pooled last-layer embedding, which keeps the
    topical content that the [CLS] vector discards when it specialises in tone.
  * Static models: the headline vector is the mean of its word vectors.

Leakage. The pre-trained encoders are frozen and were trained on text that has
nothing to do with our labels. The only thing *fitted* here, the PPMI-SVD word
vectors, uses training-period headlines only (news_day < fit_before).

Outputs are cached as float16 .npy files under data/interim/embeddings/, one
row per kept article in ``nlp_features.load_kept_articles`` order, plus an
index file used to check that a cache still lines up with the corpus.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Hugging Face model ids. Pinned here so the report and the code agree.
MODELS = {
    "glove": "sentence-transformers/average_word_embeddings_glove.6B.300d",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "finbert": "ProsusAI/finbert",
    "ner": "dslim/bert-base-NER",
}
ENCODERS = ("glove", "ppmi", "minilm", "finbert")
MAX_LEN = 64


def device() -> str:
    """Apple-silicon GPU if available, else CPU. CUDA is used if present.

    ``TASK3_DEVICE=cpu`` overrides it: on an 8 GB Mac under memory pressure the
    MPS GPU stalls on long runs (~20 headlines/s), while the CPU stays at ~200/s.
    """
    import os

    import torch
    if os.environ.get("TASK3_DEVICE"):
        return os.environ["TASK3_DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Cache handling
# ---------------------------------------------------------------------------
def cache_dir(cfg) -> Path:
    d = cfg.path("interim") / "embeddings"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _index_frame(articles: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"news_day": articles["news_day"].to_numpy(),
                         "dup_count": articles["dup_count"].to_numpy()})


def cache_matches(cfg, articles: pd.DataFrame) -> bool:
    """True if the cached embeddings were built for exactly this corpus."""
    path = cache_dir(cfg) / "index.parquet"
    if not path.exists():
        return False
    idx = pd.read_parquet(path)
    cur = _index_frame(articles)
    return len(idx) == len(cur) and idx["news_day"].equals(cur["news_day"]) \
        and idx["dup_count"].equals(cur["dup_count"])


def write_index(cfg, articles: pd.DataFrame) -> None:
    _index_frame(articles).to_parquet(cache_dir(cfg) / "index.parquet")


# ---------------------------------------------------------------------------
# The Task 3 corpus: the 100 most-syndicated headlines of each news day
# ---------------------------------------------------------------------------
# FinBERT reads ~400-1000 headlines/s on an M3 laptop GPU but stalls badly under
# memory pressure, so encoding all 905k headlines was not reliable on an 8 GB
# machine. Every encoder therefore reads the same 119,100 headlines: the 100
# most-copied stories of each day (22.6% of the syndication-weighted news).
# Using one set for all four encoders keeps the comparison fair.
_ROWS: np.ndarray | None = None


def top_k_per_day(articles: pd.DataFrame, k: int) -> np.ndarray:
    """Row positions of the k most-syndicated headlines of each news day.

    Ties on dup_count keep the corpus order (stable sort), so the choice is
    deterministic.
    """
    order = (articles.reset_index(drop=True)
             .assign(_row=np.arange(len(articles)))
             .sort_values(["news_day", "dup_count"], ascending=[True, False], kind="stable"))
    return np.sort(order.groupby("news_day").head(k)["_row"].to_numpy())


def corpus_rows(cfg, articles: pd.DataFrame) -> np.ndarray:
    return top_k_per_day(articles, int(cfg.dig("task3", "corpus_top_k_per_day", default=100)))


def use_rows(rows: np.ndarray | None) -> None:
    """From now on, loaders return only these rows (positions in the full corpus)."""
    global _ROWS
    _ROWS = None if rows is None else np.asarray(rows)


def load_embedding(cfg, name: str, mmap: bool = True) -> np.ndarray:
    """Per-headline matrix (n x dim), float16; restricted to ``use_rows`` if set."""
    E = np.load(cache_dir(cfg) / f"{name}.npy", mmap_mode="r" if mmap else None)
    return np.asarray(E[_ROWS]) if _ROWS is not None else E


def load_finbert_probs(cfg) -> pd.DataFrame:
    p = pd.read_parquet(cache_dir(cfg) / "finbert_probs.parquet")
    return p.iloc[_ROWS].reset_index(drop=True) if _ROWS is not None else p


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------
def _batched(items, size: int):
    for i in range(0, len(items), size):
        yield i, items[i:i + size]


def _length_sorted(texts: list[str]) -> np.ndarray:
    """Encode in length order so each batch pads to a similar length (2-3x faster)."""
    return np.argsort([len(t) for t in texts], kind="stable")


def encode_sentence_transformer(texts: list[str], model_id: str, batch_size: int = 256,
                                normalize: bool = True) -> np.ndarray:
    """MiniLM and GloVe both load through sentence-transformers.

    MiniLM = WordPiece -> 6-layer transformer -> mean pooling -> L2 norm.
    GloVe  = whitespace tokeniser (lower-cased) -> word lookup -> mean pooling.
    """
    from sentence_transformers import SentenceTransformer
    dev = "cpu" if "glove" in model_id else device()
    model = SentenceTransformer(model_id, device=dev)
    if getattr(model, "max_seq_length", None):
        model.max_seq_length = MAX_LEN
    order = _length_sorted(texts)
    out = model.encode([texts[i] for i in order], batch_size=batch_size,
                       normalize_embeddings=normalize, show_progress_bar=False,
                       convert_to_numpy=True)
    emb = np.empty_like(out)
    emb[order] = out
    return emb.astype(np.float16)


def encode_finbert(texts: list[str], part_dir: Path, batch_size: int = 128, part_size: int = 50_000,
                   max_new_parts: int | None = None):
    """FinBERT: sentiment probabilities from the head + mean-pooled embedding.

    Resumable: headlines are encoded in length order, in parts of 50k, and each
    part is saved to ``part_dir`` before the next starts, so a crash or a
    memory stall on an 8 GB laptop loses at most one part. Only the last
    hidden layer is kept (asking for all 13 layers costs far more memory).
    ``max_new_parts``: stop after encoding that many parts and return
    (None, None). On an 8 GB Mac, MPS throughput decays inside one long-lived
    process (700/s -> 26/s), so the driver runs one part per fresh process
    (see ``python -m src.text_embeddings --finbert-part``).

    Returns (embeddings float16 [n x 768], probs DataFrame [positive, negative, neutral]).
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = device()
    tok = AutoTokenizer.from_pretrained(MODELS["finbert"])
    model = AutoModelForSequenceClassification.from_pretrained(MODELS["finbert"]).to(dev).eval()
    labels = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]
    part_dir.mkdir(parents=True, exist_ok=True)

    order = _length_sorted(texts)
    n_parts = int(np.ceil(len(order) / part_size))
    done_now = 0
    for p in range(n_parts):
        path = part_dir / f"part_{p:03d}.npz"
        if path.exists():
            continue
        if max_new_parts is not None and done_now >= max_new_parts:
            return None, None
        done_now += 1
        rows = order[p * part_size:(p + 1) * part_size]
        emb = np.empty((len(rows), model.config.hidden_size), dtype=np.float16)
        probs = np.empty((len(rows), len(labels)), dtype=np.float32)
        t0 = time.time()
        with torch.inference_mode():
            for _, idx in _batched(np.arange(len(rows)), batch_size):
                enc = tok([texts[i] for i in rows[idx]], padding=True, truncation=True,
                          max_length=MAX_LEN, pad_to_multiple_of=8, return_tensors="pt").to(dev)
                out = model.bert(**enc)
                last = out.last_hidden_state                   # [b, seq, 768]
                mask = enc["attention_mask"].unsqueeze(-1).to(last.dtype)
                pooled = (last * mask).sum(1) / mask.sum(1).clamp(min=1)
                logits = model.classifier(out.pooler_output)   # same path as model.forward in eval mode
                emb[idx] = pooled.float().cpu().numpy().astype(np.float16)
                probs[idx] = torch.softmax(logits, -1).float().cpu().numpy()
        np.savez(path, rows=rows, emb=emb, probs=probs)
        if dev == "mps":
            torch.mps.empty_cache()
        log.info("finbert: part %d/%d (%d headlines, %.0f/s)", p + 1, n_parts, len(rows),
                 len(rows) / (time.time() - t0))

    emb = np.empty((len(texts), model.config.hidden_size), dtype=np.float16)
    probs = np.empty((len(texts), len(labels)), dtype=np.float32)
    for p in range(n_parts):
        z = np.load(part_dir / f"part_{p:03d}.npz")
        emb[z["rows"]], probs[z["rows"]] = z["emb"], z["probs"]
    return emb, pd.DataFrame(probs, columns=[f"finbert_{l}" for l in labels])


def train_ppmi_svd(token_lists: pd.Series, fit_mask: np.ndarray, dim: int = 100,
                   min_count: int = 20, alpha: float = 0.75, seed: int = 42):
    """Static word vectors learned from training-period headlines only.

    Co-occurrence window = the whole headline (headlines are ~10 tokens, so a
    +-5 window would cover almost all of it anyway). Steps:
      counts C[w, c]  ->  PPMI with context-distribution smoothing (alpha = 0.75,
      as in word2vec's negative sampling)  ->  truncated SVD  ->  W = U * sqrt(S).
    Returns (vectorizer, word_vectors [vocab x dim]).
    """
    from scipy import sparse
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import CountVectorizer

    vec = CountVectorizer(min_df=min_count, binary=True, lowercase=False,
                          token_pattern=r"(?u)[^\s]+")
    X = vec.fit_transform(token_lists[fit_mask])               # headlines x vocab
    C = (X.T @ X).tocsr().astype(np.float64)
    C.setdiag(0)
    C.eliminate_zeros()

    total = C.sum()
    w_count = np.asarray(C.sum(axis=1)).ravel()
    c_count = np.asarray(C.sum(axis=0)).ravel() ** alpha
    p_w = w_count / total
    p_c = c_count / c_count.sum()
    C = C.tocoo()
    pmi = np.log((C.data / total) / (p_w[C.row] * p_c[C.col]))
    keep = pmi > 0
    ppmi = sparse.csr_matrix((pmi[keep], (C.row[keep], C.col[keep])), shape=C.shape)

    svd = TruncatedSVD(n_components=dim, random_state=seed)
    U = svd.fit_transform(ppmi)                                # = U * S
    W = U / np.sqrt(svd.singular_values_)                      # = U * sqrt(S)
    W /= np.linalg.norm(W, axis=1, keepdims=True).clip(min=1e-9)
    log.info("PPMI-SVD: %d words, %d dims, fitted on %d training headlines",
             W.shape[0], dim, int(fit_mask.sum()))
    return vec, W.astype(np.float32)


def encode_ppmi(token_lists: pd.Series, vec, W: np.ndarray) -> np.ndarray:
    """Headline vector = mean of in-vocabulary word vectors, L2-normalised."""
    from sklearn.preprocessing import normalize
    X = vec.transform(token_lists)                             # binary counts
    X = normalize(X.astype(np.float32), norm="l1")             # mean, not sum
    E = np.asarray(X @ W)
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    E = np.where(norms > 0, E / np.clip(norms, 1e-9, None), 0.0)
    return E.astype(np.float16)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def build_all(cfg, articles: pd.DataFrame, fit_before: pd.Timestamp,
              which=ENCODERS, force: bool = False) -> None:
    """Encode every headline with every encoder in ``which``; skip cached ones."""
    d = cache_dir(cfg)
    if not cache_matches(cfg, articles):
        if any((d / f"{n}.npy").exists() for n in ENCODERS):
            log.warning("embedding cache does not match the corpus; re-encoding everything")
        force = True
        write_index(cfg, articles)

    text_b = articles["text_track_b"].tolist()
    for name in which:
        path = d / f"{name}.npy"
        if path.exists() and not force:
            log.info("%s: cached", name)
            continue
        t0 = time.time()
        if name == "glove":
            emb = encode_sentence_transformer(text_b, MODELS["glove"], batch_size=4096)
        elif name == "minilm":
            emb = encode_sentence_transformer(text_b, MODELS["minilm"], batch_size=256)
        elif name == "ppmi":
            fit_mask = (articles["news_day"] < fit_before).to_numpy()
            vec, W = train_ppmi_svd(articles["text_track_a"], fit_mask, seed=cfg.seed)
            emb = encode_ppmi(articles["text_track_a"], vec, W)
            pd.DataFrame(W, index=vec.get_feature_names_out()).to_parquet(d / "ppmi_word_vectors.parquet")
        elif name == "finbert":
            emb, probs = encode_finbert(text_b, d / "finbert_parts")
            probs.to_parquet(d / "finbert_probs.parquet")
        else:
            raise ValueError(name)
        np.save(path, emb)
        log.info("%s: %s in %.1f min", name, emb.shape, (time.time() - t0) / 60)


def _finbert_model():
    import torch  # noqa: F401
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dev = device()
    tok = AutoTokenizer.from_pretrained(MODELS["finbert"])
    model = AutoModelForSequenceClassification.from_pretrained(MODELS["finbert"]).to(dev).eval()
    return tok, model, dev


def encode_finbert_rows(texts: list[str], rows: np.ndarray, part_dir: Path, chunk: int = 20_000,
                        max_new_parts: int | None = 1, batch_size: int = 128) -> bool:
    """Encode the given corpus rows (skipping rows already in any saved part).

    Saves ``sub_NNN.npz`` parts. Returns True when nothing is left to encode.
    One part per call by default: run it in a fresh process each time (see
    ``python -m src.text_embeddings``).
    """
    import torch
    part_dir.mkdir(parents=True, exist_ok=True)
    done = [np.load(f)["rows"] for f in sorted(part_dir.glob("*.npz"))]
    need = np.setdiff1d(rows, np.concatenate(done) if done else np.array([], int))
    if not len(need):
        return True
    need = need[np.argsort([len(texts[i]) for i in need], kind="stable")]
    tok, model, dev = _finbert_model()
    k = len(list(part_dir.glob("sub_*.npz")))
    for c in range(0, len(need), chunk):
        if max_new_parts is not None and c // chunk >= max_new_parts:
            return False
        part = need[c:c + chunk]
        emb = np.empty((len(part), model.config.hidden_size), dtype=np.float16)
        probs = np.empty((len(part), model.config.num_labels), dtype=np.float32)
        t0 = time.time()
        with torch.inference_mode():
            for _, idx in _batched(np.arange(len(part)), batch_size):
                enc = tok([texts[i] for i in part[idx]], padding=True, truncation=True, max_length=MAX_LEN,
                          pad_to_multiple_of=8, return_tensors="pt").to(dev)
                out = model.bert(**enc)
                mask = enc["attention_mask"].unsqueeze(-1).to(out.last_hidden_state.dtype)
                pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
                emb[idx] = pooled.float().cpu().numpy().astype(np.float16)
                probs[idx] = torch.softmax(model.classifier(out.pooler_output), -1).float().cpu().numpy()
        np.savez(part_dir / f"sub_{k:03d}.npz", rows=part, emb=emb, probs=probs)
        k += 1
        log.info("finbert: part sub_%03d (%d headlines, %.0f/s), %d left", k - 1, len(part),
                 len(part) / (time.time() - t0), len(need) - c - len(part))
    return True


def assemble_finbert(cfg, n_articles: int, labels=("positive", "negative", "neutral")) -> None:
    """Merge all saved parts into full-size finbert.npy / finbert_probs.parquet.

    Rows that were never encoded (outside the Task 3 corpus) get a zero vector
    and NaN probabilities; the Task 3 code only ever reads corpus rows.
    """
    d = cache_dir(cfg)
    emb = np.lib.format.open_memmap(d / "finbert.npy", mode="w+", dtype=np.float16, shape=(n_articles, 768))
    probs = np.full((n_articles, 3), np.nan, dtype=np.float32)
    for f in sorted((d / "finbert_parts").glob("*.npz")):
        z = np.load(f)
        emb[z["rows"]], probs[z["rows"]] = z["emb"], z["probs"]
    emb.flush()
    del emb
    pd.DataFrame(probs, columns=[f"finbert_{l}" for l in labels]).to_parquet(d / "finbert_probs.parquet")
    log.info("finbert: assembled; %d rows encoded", int(np.isfinite(probs[:, 0]).sum()))


def _finbert_one_part() -> int:
    """Encode the next missing FinBERT part in this (fresh) process; assemble when all exist.

    Exit code 0 = everything assembled, 3 = one more part written, run again.
    """
    import sys
    from .config import load_config
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = load_config("config/config.yaml")
    a = pd.read_parquet(cfg.path("interim") / "articles_flagged.parquet",
                        columns=["keep", "news_day", "dup_count", "text_track_b"])
    a = a.loc[a["keep"]].drop(columns="keep").reset_index(drop=True)
    a["news_day"] = pd.to_datetime(a["news_day"])
    a["text_track_b"] = a["text_track_b"].fillna("").astype(str)
    if not cache_matches(cfg, a):
        write_index(cfg, a)
    d = cache_dir(cfg)
    finished = encode_finbert_rows(a["text_track_b"].tolist(), corpus_rows(cfg, a), d / "finbert_parts")
    if not finished:
        return 3
    assemble_finbert(cfg, len(a))
    sys.stdout.write("FINBERT_DONE\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_finbert_one_part())


__all__ = ["MODELS", "ENCODERS", "device", "cache_dir", "cache_matches", "load_embedding",
           "load_finbert_probs", "encode_sentence_transformer", "encode_finbert",
           "train_ppmi_svd", "encode_ppmi", "build_all"]
