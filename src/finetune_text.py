"""Task 3 — fine-tuning a transformer on our own labels.

Frozen encoders were trained for other jobs (sentence similarity, financial
PhraseBank tone). Fine-tuning asks the transformer itself to learn what in a
headline points to tomorrow's rupiah move.

Setup
  * Model: all-MiniLM-L6-v2 (22M parameters; FinBERT at 110M is ~5x slower to
    train on a laptop GPU) + a linear head with 3 outputs (DOWN / FLAT / UP).
  * Training examples: the 64 most-syndicated headlines of every TRAINING news
    day. Each headline inherits the label of the fix its news bucket feeds
    (bucket t-1 -> fix t, the same one-day shift as every other feature).
    This is weak supervision: most headlines on an UP day have nothing to do
    with the rupiah, so the labels are very noisy.
  * 2 epochs, AdamW, lr 2e-5, class-balanced cross-entropy, max 48 tokens.
  * Validation news days are used only to report how well it generalises.

How it is used (and why that is leakage-safe)
  The fine-tuned model scores every headline of every day, and a day's text
  probability is the syndication-weighted mean over its headlines. These
  probabilities are NOT fed back into a model trained on the same training
  days (they would be over-confident there, since the network saw those
  labels). They are only used in LATE fusion, where the blend weight is chosen
  on validation days and scored on test days, neither of which the network
  was trained on.
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd

from .entity_filter import top_k_per_day
from .evaluate import CLASSES
from .nlp_features import article_weights
from .text_embeddings import MODELS, cache_dir, device

log = logging.getLogger(__name__)


def bucket_labels(trading_days: pd.DatetimeIndex, y_direction: pd.Series) -> pd.Series:
    """Label of the fix each news bucket feeds: bucket of trading day t-1 -> y at t."""
    td = pd.DatetimeIndex(trading_days)
    y = y_direction.reindex(td)
    return pd.Series(y.shift(-1).to_numpy(), index=td)


def finetune_and_score(cfg, articles: pd.DataFrame, labels_by_bucket: pd.Series,
                       train_end: pd.Timestamp, val_end: pd.Timestamp,
                       use_cache: bool = True) -> pd.DataFrame:
    """Fine-tune on training buckets, score all buckets; returns daily mean probabilities.

    ``train_end``: buckets strictly before it are training news (= fit_before).
    ``val_end``: buckets in [train_end, val_end) are validation news.
    """
    path = cache_dir(cfg) / "finetune_daily_probs.parquet"
    if use_cache and path.exists():
        log.info("fine-tune: reusing cache %s", path)
        return pd.read_parquet(path)

    import torch
    from torch import nn
    from transformers import AutoModel, AutoTokenizer

    fcfg = cfg.dig("task3", "finetune", default={}) or {}
    k = int(fcfg.get("top_k_per_day", 64))
    sub = articles.iloc[top_k_per_day(articles, k)].reset_index(drop=True)
    sub["label"] = labels_by_bucket.reindex(sub["news_day"]).to_numpy()
    texts = sub["text_track_b"].tolist()
    has_y = sub["label"].notna().to_numpy()
    tr = np.flatnonzero((sub["news_day"] < train_end).to_numpy() & has_y)
    va = np.flatnonzero(((sub["news_day"] >= train_end) & (sub["news_day"] < val_end)).to_numpy() & has_y)
    y = pd.Categorical(sub["label"], categories=CLASSES).codes

    dev = device()
    torch.manual_seed(cfg.seed)
    tok = AutoTokenizer.from_pretrained(MODELS["minilm"])
    enc = AutoModel.from_pretrained(MODELS["minilm"])

    class Clf(nn.Module):
        """MiniLM -> mean pooling -> dropout -> linear(3)."""

        def __init__(self):
            super().__init__()
            self.enc, self.drop, self.head = enc, nn.Dropout(0.1), nn.Linear(enc.config.hidden_size, 3)

        def forward(self, **batch):
            h = self.enc(**batch).last_hidden_state
            m = batch["attention_mask"].unsqueeze(-1).float()
            return self.head(self.drop((h * m).sum(1) / m.sum(1).clamp(min=1)))

    model = Clf().to(dev)
    counts = np.bincount(y[tr], minlength=3).astype(float)
    class_w = torch.tensor(len(tr) / (3 * counts), dtype=torch.float32).to(dev)
    loss_fn = nn.CrossEntropyLoss(weight=class_w)
    opt = torch.optim.AdamW(model.parameters(), lr=float(fcfg.get("learning_rate", 2e-5)),
                            weight_decay=0.01)
    bs = int(fcfg.get("batch_size", 64))

    def batch_of(idx):
        b = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=48,
                return_tensors="pt")
        return {kk: v.to(dev) for kk, v in b.items()}

    def predict(idx):
        model.eval()
        res = []
        with torch.inference_mode():
            for s in range(0, len(idx), 512):
                res.append(torch.softmax(model(**batch_of(idx[s:s + 512])), -1).float().cpu().numpy())
        return np.concatenate(res)

    rng = np.random.default_rng(cfg.seed)
    t0 = time.time()
    for epoch in range(int(fcfg.get("epochs", 2))):
        model.train()
        perm = rng.permutation(tr)
        for s in range(0, len(perm), bs):
            idx = perm[s:s + bs]
            opt.zero_grad()
            loss = loss_fn(model(**batch_of(idx)), torch.as_tensor(y[idx]).long().to(dev))
            loss.backward()
            opt.step()
        acc = float((predict(va).argmax(1) == y[va]).mean())
        log.info("fine-tune epoch %d: headline-level val accuracy %.3f (%.1f min)",
                 epoch + 1, acc, (time.time() - t0) / 60)

    P = predict(np.arange(len(sub)))
    w = article_weights(sub, cfg).to_numpy()
    probs = pd.DataFrame(P * w[:, None], columns=["ft_p_down", "ft_p_flat", "ft_p_up"])
    day = sub["news_day"].to_numpy()
    daily = probs.groupby(day).sum().div(pd.Series(w).groupby(day).sum(), axis=0)
    daily.index = pd.DatetimeIndex(daily.index, name="date")
    acc = {"n_train_headlines": int(len(tr)), "n_val_headlines": int(len(va)),
           "train_accuracy": float((P[tr].argmax(1) == y[tr]).mean()),
           "val_accuracy": float((P[va].argmax(1) == y[va]).mean()),
           "val_majority_share": float(np.bincount(y[va], minlength=3).max() / len(va))}
    log.info("fine-tune headline accuracy: %s", acc)
    daily.to_parquet(path)
    (cache_dir(cfg) / "finetune_headline_accuracy.json").write_text(json.dumps(acc, indent=2))
    return daily


def main() -> int:
    """Run the fine-tuning in its own process (torch only, no XGBoost loaded).

    On macOS, torch and XGBoost each bring an OpenMP runtime; multi-threaded
    torch training in a process that already ran XGBoost can deadlock, so
    ``run_task3`` calls this as a subprocess and then reads the cache.
    """
    from .build_dataset import _setup_logging
    from .config import load_config
    from .nlp_features import load_kept_articles
    from .text_embeddings import corpus_rows

    _setup_logging(True)
    cfg = load_config("config/config.yaml")
    sdir = cfg.root / cfg.dig("task2", "paths", "splits")
    frame = pd.concat([pd.read_csv(sdir / f"{s}.csv", index_col=0, parse_dates=True)
                       for s in ("train", "val", "test")]).sort_index()
    calendar = pd.DatetimeIndex(pd.read_parquet(cfg.path("processed") / "dataset.parquet").index).sort_values()
    articles = load_kept_articles(cfg)
    articles = articles.iloc[corpus_rows(cfg, articles)].reset_index(drop=True)
    fit_before = frame.index[frame["split"] == "train"].max()
    val_end = frame.index[frame["split"] == "val"].max()
    finetune_and_score(cfg, articles, bucket_labels(calendar, frame["y_direction"]), fit_before, val_end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["bucket_labels", "finetune_and_score", "main"]
