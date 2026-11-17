#!/usr/bin/env python3
"""Draw docs/task3_architecture.png — how X_text and X_time are merged in Task 3.

    py scripts/make_task3_diagram.py

Same palette as the Task 1/2 figures: green = news, red = FX, blue = shared,
yellow = models, light blue = evaluation.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "task3_architecture.png"

NEWS, FX, SHARED, MODEL, EVAL = "#e3f4ea", "#fbe6e1", "#e6e8fa", "#fff4d6", "#eef6fb"
EDGE = {NEWS: "#5aa77a", FX: "#d0806c", SHARED: "#7b82d6", MODEL: "#d9a93a", EVAL: "#5b9bc4"}


def box(ax, x, y, w, h, title, body="", color=SHARED, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc=color, ec=EDGE[color], lw=1.2))
    ax.text(x + w / 2, y + h - 0.16, title, ha="center", va="top", fontsize=fs, weight="bold")
    if body:
        ax.text(x + w / 2, y + h - 0.45, body, ha="center", va="top", fontsize=fs - 1.6, linespacing=1.3)


def arrow(ax, x1, y1, x2, y2, text="", rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10,
                                 color="#555555", lw=1.0, connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((x1 + x2) / 2 + 0.06, (y1 + y2) / 2, text, fontsize=6.8, color="#555555", va="center")


def main() -> int:
    fig, ax = plt.subplots(figsize=(13, 7.4))
    ax.set_xlim(0, 13), ax.set_ylim(0, 7.4), ax.axis("off")

    # ---- X_text branch -------------------------------------------------------
    ax.text(0.2, 7.2, "X_text  (unstructured news)", fontsize=9.5, weight="bold", color=EDGE[NEWS])
    box(ax, 0.2, 6.15, 3.3, 0.9, "119,100 headlines (top 100 / day)",
        "most-copied of Task 1's 905,713 · Track B (raw)\nfor transformers · Track A (lemmas) for PPMI-SVD", NEWS)
    encs = [("GloVe 300d", "static · general\nmean of word vectors"),
            ("PPMI-SVD 100d", "static · our corpus\nmean of word vectors"),
            ("MiniLM 384d", "contextual · general\nWordPiece<=64, mean pool"),
            ("FinBERT 768d", "contextual · finance\nmean pool + P(pos/neg/neu)")]
    for i, (t, b) in enumerate(encs):
        box(ax, 0.2 + i * 1.95, 4.65, 1.85, 0.95, t, b, NEWS, fs=8.2)
        arrow(ax, 1.85, 6.15, 1.1 + i * 1.95, 5.6)
    box(ax, 4.1, 6.15, 3.7, 0.9, "Extra views of the same headlines",
        "BERTopic-style topics (MiniLM -> k-means 20)\nNER entity filter (BERT-NER: US, Fed, China, ...)", NEWS, fs=8.4)
    box(ax, 0.2, 3.3, 3.6, 1.05, "Temporal pooling per news day",
        "mean (log1p dup weight) · exp. decay (half-life 3d)\nsalience attention: softmax(log w + (1-p_neu)/T)", NEWS, fs=8.4)
    box(ax, 4.2, 3.3, 3.6, 1.05, "Compress + align",
        "PCA fitted on TRAIN days only (16 comps)\nshift 1 trading day: news t-1 -> fix t", NEWS, fs=8.4)
    for i in range(4):
        arrow(ax, 1.1 + i * 1.95, 4.65, 1.6 + i * 0.4, 4.35)
    arrow(ax, 3.8, 3.82, 4.2, 3.82)
    arrow(ax, 5.95, 6.15, 6.0, 4.35)

    # ---- X_time branch -------------------------------------------------------
    ax.text(9.0, 7.2, "X_time  (structured time series)", fontsize=9.5, weight="bold", color=EDGE[FX])
    box(ax, 9.0, 5.1, 1.85, 1.95, "JISDOR history (10)",
        "returns lag 1-5\n5/20-day mean\n& volatility\n|ret| lag 1", FX, fs=8.4)
    box(ax, 10.95, 5.1, 1.85, 1.95, "Overnight markets (4)",
        "DXY return\nBrent return\nVIX level & change\n(all lag 1)", FX, fs=8.4)
    box(ax, 9.0, 3.3, 3.8, 1.05, "X_time = 14 features",
        "all known before the fix at t\n(same columns as Task 2 Model A)", FX)
    arrow(ax, 9.9, 5.1, 10.3, 4.35)
    arrow(ax, 11.9, 5.1, 11.5, 4.35)

    # ---- Fusion layer (each design is chosen on validation) -------------------
    arrow(ax, 7.95, 5.0, 8.9, 2.8, rad=0.0)
    ax.text(8.5, 4.05, "top-64\nheadline\nvectors", fontsize=6.8, color="#555555")
    box(ax, 0.2, 1.3, 3.95, 1.5, "Fusion 1 · Early",
        "concat [X_time | X_text]\n-> LogReg (balanced) or XGBoost\none model sees everything;\n"
        "trees can learn interactions", MODEL, fs=8.8)
    box(ax, 4.45, 1.3, 3.95, 1.5, "Fusion 2 · Late",
        "p_time = LogReg(X_time)\np_text = LogReg(X_text) or fine-tuned MiniLM\n"
        "log p = (1-w) log p_time + w log p_text\nw in {0,...,0.6} picked on validation", MODEL, fs=8.8)
    box(ax, 8.7, 1.3, 4.1, 1.5, "Fusion 3 · Attention network",
        "top-64 headline vectors -> Linear+tanh (32)\n-> learned-query attention -> text vec (32)\n"
        "X_time -> MLP (32);  concat -> dropout\n-> Linear -> softmax(DOWN, FLAT, UP)", MODEL, fs=8.8)
    arrow(ax, 5.6, 3.3, 2.2, 2.8)
    arrow(ax, 6.0, 3.3, 6.2, 2.8)
    arrow(ax, 10.2, 3.3, 3.0, 2.8)
    arrow(ax, 10.6, 3.3, 7.0, 2.8)
    arrow(ax, 11.4, 3.3, 11.6, 2.8)

    # ---- Evaluation ----------------------------------------------------------
    box(ax, 0.2, 0.1, 12.6, 0.9, "P(DOWN), P(FLAT), P(UP) for fix t   ->   fit on train · choose on validation · "
        "refit train+val · score test once",
        "Ablation: A = X_time  ·  B = A + Task 2 lexicon/TF-IDF NLP  ·  C = A + Task 3 embeddings  |  "
        "macro-F1 (main), accuracy, hit rate; 10-day block-bootstrap 95% intervals; XGBoost/NN over 3 seeds",
        EVAL, fs=9)
    for x in (2.2, 6.4, 10.75):
        arrow(ax, x, 1.3, x, 1.0)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
