#!/usr/bin/env python3
"""Draw docs/task2_pipeline.png — the detailed Task 2 architecture diagram.

    py scripts/make_task2_diagram.py      (after `py -m src.run_task2`)

It continues the Task 1 diagram (GDELT -> Filter -> Clean text; JISDOR ->
Clean rates; Align by date -> Aligned dataset) and adds everything Task 2
builds on top. Split sizes are read from reports/task2/manifest.json so the
figure always matches the last pipeline run.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "task2_pipeline.png"

# Same palette family as the Task 1 figure: green = news, red = FX, blue = shared
NEWS, FX, SHARED, MODEL, EVAL, TASK1 = "#e3f4ea", "#fbe6e1", "#e6e8fa", "#fff4d6", "#eef6fb", "#f3f3f3"
EDGE = {NEWS: "#5aa77a", FX: "#d0806c", SHARED: "#7b82d6", MODEL: "#d9a93a", EVAL: "#5b9bc4", TASK1: "#aaaaaa"}


def box(ax, x, y, w, h, title, body="", color=SHARED, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=color, ec=EDGE[color], lw=1.2))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=fs, weight="bold")
    if body:
        ax.text(x + w / 2, y + h - 0.55, body, ha="center", va="top", fontsize=fs - 1.5,
                linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, text=""):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                                 color="#555555", lw=1.0, connectionstyle="arc3,rad=0"))
    if text:
        ax.text((x1 + x2) / 2 + 0.08, (y1 + y2) / 2, text, fontsize=7, color="#555555", va="center")


def main() -> int:
    man = json.loads((ROOT / "reports" / "task2" / "manifest.json").read_text())
    rows, tf = man["rows"], man["tfidf"]

    fig, ax = plt.subplots(figsize=(14, 11))
    ax.set_xlim(0, 14); ax.set_ylim(0, 11); ax.axis("off")

    # ---- Task 1 band -------------------------------------------------------
    ax.add_patch(FancyBboxPatch((0.2, 8.55), 13.6, 2.25, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc="white", ec="#bbbbbb", lw=1, ls="--"))
    ax.text(0.4, 10.62, "TASK 1 (done)", fontsize=8, color="#888888", weight="bold")
    box(ax, 0.5, 9.55, 2.4, 0.95, "GDELT news", "GKG bulk files, 2021-26", NEWS)
    box(ax, 3.4, 9.55, 2.4, 0.95, "Filter", "source, topic, relevance, dedup", NEWS)
    box(ax, 6.3, 9.55, 2.6, 0.95, "Clean text", "Track A (lemmas) · Track B (raw)", NEWS)
    box(ax, 0.5, 8.7, 2.4, 0.75, "JISDOR rate", "official BI export", FX, fs=8.5)
    box(ax, 3.4, 8.7, 2.4, 0.75, "Clean rates", "log returns, UP/FLAT/DOWN", FX, fs=8.5)
    box(ax, 9.4, 8.7, 4.1, 1.8, "Align by date (WIB)",
        "news of day D -> first fix after D\nweekends/holidays roll forward\n"
        "one row per trading day (1,202)\n+ lagged Task 1 news aggregates", SHARED)
    arrow(ax, 2.9, 10.02, 3.4, 10.02); arrow(ax, 5.8, 10.02, 6.3, 10.02)
    arrow(ax, 8.9, 10.02, 9.4, 10.02); arrow(ax, 2.9, 9.07, 3.4, 9.07); arrow(ax, 5.8, 9.07, 9.4, 9.07)

    # ---- NLP feature extraction -------------------------------------------
    ax.text(0.4, 8.2, "TASK 2 · NLP feature extraction (headlines only, no pre-trained embeddings)",
            fontsize=9, weight="bold", color="#2f6b47")
    box(ax, 0.5, 6.55, 2.9, 1.45, "VADER", "on Track B\ncompound, neg/pos share\n(reads 'not', caps, '!')", NEWS)
    box(ax, 3.7, 6.55, 3.0, 1.45, "Loughran-McDonald", "on Track A\nnegative, positive, uncertainty,\nlitigious, constraining rates", NEWS)
    box(ax, 7.0, 6.55, 3.2, 1.45, "TF-IDF -> SVD",
        f"on Track A, uni+bigrams\nvocab {tf['vocabulary_size']:,} terms -> 20 axes\nfitted on train news only", NEWS)
    for x in (1.95, 5.2, 8.6):
        arrow(ax, 7.6, 9.55, x, 8.0)
    box(ax, 0.5, 4.95, 9.7, 1.25, "Daily aggregation per trading-day bucket",
        "weighted by log1p(dup_count) · overall + per theme family (conflict, energy, monetary, ...)\n"
        "trailing 30-day z-scores · shifted 1 trading day (news of bucket t-1 -> fix t)", NEWS)
    for x in (1.95, 5.2, 8.6):
        arrow(ax, x, 6.55, x, 6.2)

    # ---- Price + market ----------------------------------------------------
    box(ax, 10.6, 6.55, 3.0, 1.45, "Price features", "JISDOR return lags 1-5\n5/20-day mean & volatility\n(all from t-1 or earlier)", FX)
    box(ax, 10.6, 4.95, 3.0, 1.25, "Market controls", "DXY, Brent returns,\nVIX level & change (lag 1)", FX)
    arrow(ax, 12.1, 8.7, 12.1, 8.0)

    # ---- Modelling table + split ---------------------------------------------
    box(ax, 0.5, 3.35, 13.1, 1.2, "Modelling table -> strict chronological split (never shuffled)",
        f"TRAIN  2021-09 .. 2024-08-31  ({rows['train']} days)     |     "
        f"VALIDATION  .. 2025-08-31  ({rows['val']} days)     |     TEST  2025-09 .. 2026-09  ({rows['test']} days)", SHARED)
    arrow(ax, 5.35, 4.95, 5.35, 4.55); arrow(ax, 12.1, 4.95, 12.1, 4.55)
    # price features go straight to the table, beside (not through) the market box
    arrow(ax, 13.75, 7.25, 13.75, 4.55)
    ax.plot([13.6, 13.75], [7.25, 7.25], color="#555555", lw=1.0)

    # ---- Models ------------------------------------------------------------
    box(ax, 0.5, 1.55, 4.2, 1.45, "Baseline: price history only",
        "majority · persistence · zero-return\nARIMA(p,0,q), order by AIC on train\nLogReg/Ridge · XGBoost on price features", MODEL)
    box(ax, 5.0, 1.55, 4.2, 1.45, "Combined: price + NLP",
        "same LogReg/Ridge and XGBoost,\nsame tuning grid -> the only\ndifference is the news features", MODEL)
    box(ax, 9.5, 1.55, 4.1, 1.45, "Ablations",
        "+ market controls, with/without NLP\n+ Task 1 volume/theme aggregates\n(does news add beyond markets?)", MODEL)
    for x in (2.6, 7.1, 11.55):
        arrow(ax, x, 3.35, x, 3.0)

    # ---- Evaluation ----------------------------------------------------------
    box(ax, 0.5, 0.1, 13.1, 1.15, "Tune on validation -> refit on train+validation -> score test once",
        "Primary (UP/FLAT/DOWN): macro-F1 · accuracy · directional hit rate & coverage    |    "
        "Secondary (next-day return): RMSE · MAE · directional accuracy · R² vs zero", EVAL)
    for x in (2.6, 7.1, 11.55):
        arrow(ax, x, 1.55, x, 1.25)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
