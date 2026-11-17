"""Task 3 — multimodal fusion of text features X_text and time-series features X_time.

Three fusion designs, from simplest to most flexible:

  Early fusion   concatenate [X_time | X_text] and train ONE model on it
                 (logistic regression or XGBoost, the Task 2 models). Cheap and
                 lets trees learn interactions ("negative Fed news AND a strong
                 overnight dollar"), but the text columns compete with the
                 strong market columns for the same small sample.

  Late fusion    train two unimodal models, one on X_time and one on X_text,
                 and combine their class probabilities with a log-linear blend:
                     log p(c) = (1 - w) log p_time(c) + w log p_text(c) + const
                 w is chosen on validation. w = 0 is Model A exactly, so late
                 fusion can only hurt on validation if text is pure noise. It
                 also tells us directly how much weight the data gives to news.

  Attention      a small neural network (PyTorch). The text branch looks at the
  fusion net     day's top-64 headline embeddings (frozen FinBERT/MiniLM),
                 projects them to 32 dims and pools them with a LEARNED attention
                 query, so the network decides which headlines matter. The time
                 branch is a one-layer MLP on X_time. The two 32-d vectors are
                 concatenated and a linear layer gives the 3 class logits
                 ("intermediate fusion"). It is trained with class-balanced
                 cross-entropy, dropout, weight decay and early stopping on the
                 validation macro-F1, because 726 training days is very little
                 for a neural network.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight

from .evaluate import CLASSES
from .models import make_model

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Early fusion (concatenation) — probability version of models.fit_predict
# ---------------------------------------------------------------------------
def fit_proba(kind: str, params: dict, cfg, X_fit: pd.DataFrame, y_fit: pd.Series,
              X_pred: pd.DataFrame, seed: int | None = None):
    """Fit one classifier and return (model, P[n x 3]) with columns in CLASSES order."""
    model = make_model("cls", kind, params, cfg)
    if kind == "xgb":
        extra = {k: v for k, v in params.items() if k in ("learning_rate", "min_child_weight")}
        if seed is not None:
            extra["random_state"] = seed
        if extra:
            model.set_params(**extra)
    y_idx = pd.Categorical(y_fit, categories=CLASSES).codes
    if kind == "xgb":
        model.fit(X_fit, y_idx, sample_weight=compute_sample_weight("balanced", y_idx))
    else:
        model.fit(X_fit, y_idx)
    return model, np.asarray(model.predict_proba(X_pred))


def proba_to_label(P: np.ndarray) -> np.ndarray:
    return np.asarray(CLASSES, dtype=object)[np.argmax(P, axis=1)]


def macro_f1(y_true, P_or_labels) -> float:
    pred = proba_to_label(P_or_labels) if np.ndim(P_or_labels) == 2 else P_or_labels
    return float(f1_score(np.asarray(y_true, dtype=object), pred, labels=CLASSES,
                          average="macro", zero_division=0))


# ---------------------------------------------------------------------------
# Late fusion
# ---------------------------------------------------------------------------
def late_blend(P_time: np.ndarray, P_text: np.ndarray, w: float) -> np.ndarray:
    """Log-linear (product-of-experts) blend of two probability matrices."""
    eps = 1e-6
    z = (1 - w) * np.log(P_time + eps) + w * np.log(P_text + eps)
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def choose_blend_weight(y_val, P_time_val, P_text_val, weights) -> tuple[float, pd.DataFrame]:
    """Pick w on validation macro-F1 (ties -> smaller w, i.e. trust prices)."""
    rows = [{"w": w, "val_macro_f1": macro_f1(y_val, late_blend(P_time_val, P_text_val, w))}
            for w in weights]
    table = pd.DataFrame(rows)
    best = table.sort_values(["val_macro_f1", "w"], ascending=[False, True]).iloc[0]
    return float(best["w"]), table


# ---------------------------------------------------------------------------
# Attention fusion network (PyTorch)
# ---------------------------------------------------------------------------
def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


def build_attention_net(d_text: int, d_time: int, hidden: int = 32, dropout: float = 0.3):
    torch, nn = _torch()

    class AttentionFusionNet(nn.Module):
        """Text branch: headline vectors -> tanh projection -> learned-query attention.
        Time branch: MLP. Fusion: concatenate -> dropout -> linear -> 3 logits."""

        def __init__(self):
            super().__init__()
            self.text_proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_text, hidden), nn.Tanh())
            self.query = nn.Parameter(torch.randn(hidden) * 0.1)
            self.time_mlp = nn.Sequential(nn.Linear(d_time, hidden), nn.ReLU())
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, len(CLASSES)))

        def forward(self, H, mask, prior, x_time):
            h = self.text_proj(H)                                   # [B, K, hidden]
            score = h @ self.query + prior                          # [B, K]; prior = log dup weight
            score = score.masked_fill(~mask, -1e4)
            attn = torch.softmax(score, dim=-1)
            text_vec = (attn.unsqueeze(-1) * h).sum(dim=1)          # [B, hidden]
            z = torch.cat([text_vec, self.time_mlp(x_time)], dim=-1)
            return self.head(z), attn

    return AttentionFusionNet()


def train_attention_net(H, mask, prior, X_time, y_idx, fit_idx, val_idx=None, *, cfg,
                        seed: int, n_epochs: int | None = None):
    """Train on ``fit_idx`` rows. With ``val_idx``: early stopping on val macro-F1,
    returns (net, best_epoch). Without: trains exactly ``n_epochs`` (the refit)."""
    torch, nn = _torch()
    acfg = cfg.dig("task3", "attention_net", default={}) or {}
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = build_attention_net(H.shape[-1], X_time.shape[-1], int(acfg.get("hidden", 32)),
                              float(acfg.get("dropout", 0.3)))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=float(acfg.get("weight_decay", 1e-3)))
    yf = y_idx[fit_idx]
    counts = np.bincount(yf, minlength=len(CLASSES)).astype(float)
    class_w = torch.tensor(len(yf) / (len(CLASSES) * np.clip(counts, 1, None)), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=class_w)
    Ht, Mt, Pt, Xt = (torch.as_tensor(a) for a in (H, mask, prior, X_time))
    Yt = torch.as_tensor(y_idx).long()
    max_epochs = int(n_epochs or acfg.get("max_epochs", 60))
    patience = int(acfg.get("patience", 10))
    rng = np.random.default_rng(seed)
    best = (-1.0, 0, None)
    for epoch in range(1, max_epochs + 1):
        net.train()
        for b in np.array_split(rng.permutation(fit_idx), max(1, len(fit_idx) // 32)):
            b = torch.as_tensor(b, dtype=torch.long)
            opt.zero_grad()
            logits, _ = net(Ht[b], Mt[b], Pt[b], Xt[b])
            loss_fn(logits, Yt[b]).backward()
            opt.step()
        if val_idx is None:
            continue
        P = predict_attention_net(net, H, mask, prior, X_time, val_idx)[0]
        score = macro_f1(np.asarray(CLASSES, dtype=object)[y_idx[val_idx]], P)
        if score > best[0]:
            best = (score, epoch, {k: v.clone() for k, v in net.state_dict().items()})
        elif epoch - best[1] >= patience:
            break
    if val_idx is not None:
        net.load_state_dict(best[2])
        return net, best[1]
    return net, max_epochs


def predict_attention_net(net, H, mask, prior, X_time, idx):
    torch, _ = _torch()
    net.eval()
    with torch.no_grad():
        logits, attn = net(torch.as_tensor(H[idx]), torch.as_tensor(mask[idx]),
                           torch.as_tensor(prior[idx]), torch.as_tensor(X_time[idx]))
        return torch.softmax(logits, -1).numpy(), attn.numpy()


__all__ = ["fit_proba", "proba_to_label", "macro_f1", "late_blend", "choose_blend_weight",
           "build_attention_net", "train_attention_net", "predict_attention_net"]
