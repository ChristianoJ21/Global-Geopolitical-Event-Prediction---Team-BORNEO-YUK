#!/usr/bin/env python3
"""Generate notebooks/task3/task3_ablation_and_analysis.ipynb.

    py -m src.run_task3 && py -m src.analysis_task3
    py scripts/make_task3_notebook.py
    py -m jupyter nbconvert --to notebook --execute --inplace notebooks/task3/task3_ablation_and_analysis.ipynb

Generated rather than hand-edited (same reason as the Task 2 notebook): review
happens on this readable file, and the notebook only READS what the pipeline
wrote (plus a few light computations on its predictions), so it cannot drift
from the numbers in the report.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "task3" / "task3_ablation_and_analysis.ipynb"

nb = nbf.v4.new_notebook()
C: list = []
def md(s: str) -> None: C.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s: str) -> None: C.append(nbf.v4.new_code_cell(s.strip()))


md(r"""
# Task 3: Model Refinement & Multimodal Integration

**Question:** do pre-trained / transformer representations of the news make our USD/IDR (JISDOR) direction forecast measurably better than the Task 2 pipeline?

This notebook is the reproducible record of every Task 3 experiment. It reads what `python -m src.run_task3` and `python -m src.analysis_task3` wrote to `reports/task3/`, so every number is the pipeline's own.

| § | Content |
|---|---|
| 1 | What Task 3 adds: four encoders, feature strategies, fusion designs |
| 2 | Feature extraction sanity checks (word vectors, FinBERT tone, topics, NER) |
| 3 | Representation comparison (static vs contextual, general vs domain) |
| 4 | Feature strategies: pooling, topics, entities |
| 5 | Fusion: early vs late vs attention network, fine-tuning |
| 6 | Tuning and the A / B / C ablation (with seeds and bootstrap intervals) |
| 7 | Embedding space: t-SNE of high- vs low-volatility days |
| 8 | Interpretability: SHAP, topics, biggest prediction shifts |
| 9 | Error analysis and distributions; drift; the magnitude target |
| 10 | Findings |
""")

code(r"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "config" / "config.yaml").exists())
REP, FIG, SPL = ROOT / "reports" / "task3", ROOT / "reports" / "task3" / "figures", ROOT / "data" / "splits" / "task3"
pd.set_option("display.width", 170); pd.set_option("display.max_colwidth", 110)
res = pd.read_csv(REP / "results.csv")
man = json.loads((REP / "manifest.json").read_text())
groups = json.loads((SPL / "feature_groups.json").read_text())
frame = pd.concat([pd.read_csv(SPL / f"{s}.csv", index_col=0, parse_dates=True) for s in ("train", "val", "test")]).sort_index()

def table(df, idx=("experiment", "model")):
    return (df.pivot_table(index=list(idx), columns="split", values="macro_f1")
              .reindex(columns=["val", "test"]).round(3).sort_values("val", ascending=False))

print("rows:", man["rows"], "| headlines:", f"{man['n_headlines']:,}", "| fit_before:", man["fit_before"])
print("encoders:", json.dumps(man["encoders"], indent=1))
print({g: len(c) for g, c in groups.items()})
""")

md(r"""
## 1. What Task 3 adds

Task 2 was limited to lexicons (VADER, Loughran-McDonald) and a TF-IDF we fitted ourselves. Task 3 adds four encoders, chosen as a 2 x 2 so each comparison isolates one design choice:

| | general-purpose | domain-specific |
|---|---|---|
| **static** (one vector per word) | GloVe 6B-300d | PPMI-SVD word vectors trained on our 2021-24 headlines (= Word2Vec's objective, Levy & Goldberg 2014) |
| **contextual** (vector depends on the sentence) | all-MiniLM-L6-v2 (sentence similarity) | FinBERT (ProsusAI, financial tone) |

Each headline becomes a vector; a day's vectors are pooled (mean, exponential decay, or salience attention), compressed with PCA fitted on training days, and shifted one trading day (news of day t-1 -> fix t), exactly like every Task 2 feature.
""")

md("## 2. Feature extraction sanity checks")
code(r"""
wv = ROOT / "data" / "interim" / "embeddings" / "ppmi_word_vectors.parquet"
if wv.exists():
    W = pd.read_parquet(wv)
    for w in ["war", "fed", "sanction", "oil", "tariff", "inflation"]:
        if w in W.index:
            s = W.values @ W.loc[w].values
            print(f"{w:>10} -> {', '.join(W.index[np.argsort(-s)[1:8]])}")
else:
    print("word vectors are in data/interim (not on GitHub); run src.run_task3 to rebuild")
""")
code(r"""
print("PCA variance kept by 16 components (training days):")
display(pd.read_csv(REP / "pca_variance.csv").round(3))
print("Topics (k-means on MiniLM, labelled by class-based TF-IDF on training headlines):")
display(pd.read_csv(REP / "topic_labels.csv"))
print("NER: share of top-100 daily headlines that mention each watchlist group")
print(json.dumps(man["ner_coverage"], indent=1))
""")
code(r"""
fig, ax = plt.subplots(figsize=(11, 3))
fb = frame["t3_finbert_net"].rolling(20).mean()
vd = frame["nlp_vader_mean"].rolling(20).mean()
ax.plot(fb.index, (fb - fb.mean()) / fb.std(), label="FinBERT net tone (z, 20-day mean)")
ax.plot(vd.index, (vd - vd.mean()) / vd.std(), label="VADER mean (Task 2, z, 20-day mean)", alpha=0.7)
for s, c in (("val", "#ddd"), ("test", "#eee")):
    d = frame.index[frame["split"] == s]; ax.axvspan(d.min(), d.max(), color=c, zorder=0)
ax.legend(fontsize=8); ax.set_title("Daily news tone: FinBERT vs Task 2's VADER")
plt.show()
print("correlation of daily FinBERT net tone with VADER mean:",
      round(frame[["t3_finbert_net", "nlp_vader_mean"]].corr().iloc[0, 1], 3))
""")

md(r"""
## 3. Representation comparison

Each encoder's daily-mean PCA block is added to Model A (price + overnight markets) with the Task 2 grid (early fusion), and also used **alone** (text only). Macro-F1, chosen on validation, scored once on test. Chance for three balanced classes is about 0.33.
""")
code(r"""
display(table(res.loc[res["stage"] == "1_representation"]))
display(table(res.loc[res["stage"] == "0_task2"]))
""")

md("## 4. Feature strategies: temporal pooling, FinBERT tone, topics, entities")
code(r"""
display(table(res.loc[res["stage"] == "2_strategy"]))
""")

md(r"""
## 5. Fusion: early vs late vs attention network

*Late fusion* blends the probabilities of the time-series model and a text-only model, `log p = (1-w) log p_time + w log p_text`, with `w` picked on validation. `w = 0` means "ignore the news". The *attention network* reads the top-64 headline vectors of the day directly and learns which to attend to. *Fine-tuned MiniLM* is a transformer trained on our own labels, used only through late fusion so it is never scored on days it was trained on.
""")
code(r"""
display(table(res.loc[(res["stage"] == "3_fusion") & ~res["experiment"].str.contains("seed")]))
lw = pd.read_csv(REP / "late_fusion_weights.csv")
chosen = lw.loc[lw.groupby("experiment")["val_macro_f1"].idxmax(), ["experiment", "w", "val_macro_f1"]]
print("weight given to text by late fusion (chosen on validation):")
display(chosen.sort_values("w", ascending=False).round(3))
ft = ROOT / "data" / "interim" / "embeddings" / "finetune_headline_accuracy.json"
if ft.exists():
    print("fine-tuned MiniLM, headline-level accuracy:", ft.read_text())
seeds = res.loc[res["experiment"].str.contains("attention_net_.*_seed", regex=True)]
print("attention network, per seed:")
display(seeds.pivot_table(index="experiment", columns="split", values="macro_f1").round(3))
""")

md(r"""
## 6. Tuning and the ablation matrix

Model C is the Task 3 text configuration with the best **validation** macro-F1, re-tuned on a wider grid (PCA size, regularisation, tree depth / count / learning rate). The ablation then compares:

* **A0**: price history only  · **A**: time series only (price + overnight markets, the Task 2 baseline)
* **B**: A + Task 2 basic NLP (46 lexicon / TF-IDF features)
* **C**: A + Task 3 refined text features · **B+C**: both

XGBoost is run with 3 seeds (mean ± std); logistic regression is deterministic.
""")
code(r"""
print("best Task 3 config on validation:", man["best_on_validation"])
print("Model C per model family:", json.dumps(man["model_C"], indent=1))
display(pd.read_csv(REP / "tuning.csv").head(10).round(3))
abl = res.loc[res["stage"] == "5_ablation"].copy()
abl["set"] = abl["experiment"].str.split("|").str[0]
summary = (abl.groupby(["set", "model", "split"])[["macro_f1", "accuracy", "directional_hit_rate"]]
              .agg(["mean", "std"]).round(3))
display(summary)
display(Image(FIG / "ablation.png"))
print("Block-bootstrap (10-day blocks) 95% intervals on TEST macro-F1 gains:")
display(pd.read_csv(REP / "ablation_gain.csv").round(3))
print("Secondary task, next-day return (regression):")
display(pd.read_csv(REP / "regression.csv").pivot_table(index=["experiment", "model"], columns="split",
        values=["rmse", "directional_accuracy"]).round(3))
""")

md("## 7. Embedding space: headlines before high- vs low-volatility fixes")
code(r"""
display(Image(FIG / "tsne_volatility.png"))
display(pd.read_csv(REP / "analysis_vol_separability.csv").round(3))
""")

md("## 8. Interpretability: what drives Model C?")
code(r"""
if (FIG / "shap_model_c.png").exists():
    display(Image(FIG / "shap_model_c.png"))
tc = pd.read_csv(REP / "analysis_topic_correlation.csv")
print("Topics most related to the SIZE of the next move (training days, Spearman):")
display(tc.reindex(tc["rho_abs_return"].abs().sort_values(ascending=False).index).head(8)
          [["feature", "top_words", "rho_abs_return", "p_abs_return", "rho_return", "p_return"]].round(3))
display(pd.read_csv(REP / "analysis_feature_correlations.csv"))
print("Test days where Model C moved furthest from Model A (XGBoost):")
display(pd.read_csv(REP / "analysis_prediction_shifts_xgb.csv").head(6))
if (REP / "attention_top_headlines.csv").exists():
    at = pd.read_csv(REP / "attention_top_headlines.csv")
    print("Headlines the attention network focused on most (test days, highest weight):")
    display(at.sort_values("attention", ascending=False).head(8))
""")

md("## 9. Error analysis, error distributions, drift")
code(r"""
from sklearn.metrics import confusion_matrix
pred = pd.read_csv(REP / "predictions.csv", index_col=0, parse_dates=True)
te = pred.loc[pred["split"] == "test"]
cls = ["DOWN", "FLAT", "UP"]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
for ax, key, title in ((axes[0], "A|seed42|logreg", "A (time series)"), (axes[1], "C|seed42|logreg", "C (+ Task 3 text)")):
    p = np.asarray(cls)[te[[f"{key}|{c}" for c in cls]].to_numpy().argmax(1)]
    cm = confusion_matrix(te["y_direction"], p, labels=cls)
    ax.imshow(cm, cmap="Blues"); ax.set_xticks(range(3), cls); ax.set_yticks(range(3), cls)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{title}, test")
for key, lab in (("A|seed42|logreg", "A"), ("C|seed42|logreg", "C")):
    P = te[[f"{key}|{c}" for c in cls]].to_numpy()
    true_p = P[np.arange(len(te)), pd.Categorical(te["y_direction"], categories=cls).codes]
    axes[2].hist(true_p, bins=20, alpha=0.6, label=lab)
axes[2].set_title("probability given to the TRUE class (test)"); axes[2].legend()
plt.tight_layout(); plt.show()
print(json.loads((REP / "analysis_error_summary.json").read_text()))
display(Image(FIG / "error_by_regime.png"))
display(pd.read_csv(REP / "analysis_error_examples.csv"))
""")
code(r"""
print("Adversarial validation: ROC-AUC of telling a TRAIN day from a TEST day from each block alone")
display(pd.read_csv(REP / "analysis_drift.csv").round(3))
print("Does text predict the SIZE of tomorrow's move? |return| regression (Ridge), test")
display(pd.read_csv(REP / "analysis_magnitude.csv").round(4))
""")

md(r"""
## 10. Findings

1. **Transformer representations did not measurably improve direction forecasting.** Model C (A + FinBERT tone, the best Task 3 configuration on *validation*) scores 0.445 ± 0.009 test macro-F1 with XGBoost (3 seeds) vs 0.440 for Model A, and 0.422 vs 0.458 with logistic regression. Every block-bootstrap interval for C vs A includes zero.
2. **They are better than Task 2's basic NLP.** C beats B by about +0.03 with both models. B overfits the validation year the most (0.509 val -> 0.388 test).
3. **Domain-specific > general and contextual > static, but by little:** with XGBoost, FinBERT 0.487 vs MiniLM 0.463, and corpus-trained PPMI 0.498 vs GloVe 0.457. These variants were not the validation winners, so they are reported, not selected. Text alone is near chance (0.25-0.36).
4. **Fusion:** late fusion gives the news a weight of 0.1-0.5 on validation, but test stays within about ±0.02 of A. The attention network overfits (test 0.39-0.42). Fine-tuned MiniLM reaches 37% headline accuracy, below the 42% majority share: the labels are too noisy.
5. **Why:** the time-series features look alike in train and test (adversarial AUC 0.51-0.61), while every text block identifies the period (0.83-0.99). News semantics drift from era to era (COVID, Ukraine, Gaza -> tariffs, Iran). In t-SNE, headlines before high- and low-volatility days are indistinguishable (probe AUC about 0.5).
6. **Errors:** the text makes the model more cautious. It fixes false UP/DOWN calls on truly FLAT days (5 cases), but hedges to FLAT on big-move days (error 0.54 vs 0.40 in the top move tercile). McNemar p = 0.09.
7. **SHAP:** Model C splits its weight evenly across markets 34%, price 33% and FinBERT tone 33%. The most-used text features are the tone of sanctions/trade, energy and diplomacy news. The attention network focuses on Russia-Ukraine strikes on energy sites.

**Next (Task 4):** rolling re-training against drift; use news for the *size* of the move or as a confidence gate; add Indonesia-specific sources (only 0.1% of top headlines mention Indonesia).
""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print("wrote", OUT)
