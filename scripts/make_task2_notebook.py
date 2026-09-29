#!/usr/bin/env python3
"""Generate notebook/task2_eda_and_baselines.ipynb.

    py scripts/make_task2_notebook.py
    py -m jupyter nbconvert --to notebook --execute --inplace notebook/task2_eda_and_baselines.ipynb

Generated rather than hand-edited (same reason as scripts/make_notebook.py):
review happens on this readable file, and the notebook only READS what
``py -m src.run_task2`` wrote, so it cannot drift from the pipeline's numbers.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebook" / "task2_eda_and_baselines.ipynb"

nb = nbf.v4.new_notebook()
C: list = []
def md(s: str) -> None: C.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s: str) -> None: C.append(nbf.v4.new_code_cell(s.strip()))


md(r"""
# Task 2: Pipeline Proposal & Baseline Experimentation

**Hypothesis:** global geopolitical news helps predict the USD/IDR exchange rate (Bank Indonesia's JISDOR fix).

This notebook is the exploratory analysis and the first experimental results. It reads only what `py -m src.run_task2` produced, so every number here is the pipeline's own.

| § | Content |
|---|---|
| 1 | Task formulation: the target, the class balance, the split |
| 2 | NLP features: what each lexicon sees, how sentiment moves over time |
| 3 | TF-IDF topics: what the SVD axes mean |
| 4 | Do news features relate to the next day's move? (training data only) |
| 5 | Results: price-only baselines vs price + NLP |
| 6 | Findings |
""")

code(r"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == "notebook" else Path.cwd()
SPLITS, REP = ROOT / "data" / "splits", ROOT / "reports" / "task2"
pd.set_option("display.width", 160); pd.set_option("display.max_colwidth", 90)
plt.rcParams.update({"figure.figsize": (11, 3.6), "axes.grid": True, "grid.alpha": 0.3})

parts = {s: pd.read_csv(SPLITS / f"{s}.csv", index_col="date", parse_dates=True) for s in ("train", "val", "test")}
frame = pd.concat(parts.values()).sort_index()
groups = json.loads((SPLITS / "feature_groups.json").read_text())
results = pd.read_csv(REP / "results.csv")
manifest = json.loads((REP / "manifest.json").read_text())
print({g: len(c) for g, c in groups.items()}, manifest["rows"])
""")

# ---------------------------------------------------------------------------
md(r"""
## 1. Task formulation

**Primary target: `y_direction`**, the direction of the next JISDOR fix: UP / FLAT / DOWN, where FLAT means the log return is within ±0.10%. The dead-zone was set in Task 1 so the model is not asked to predict the sign of noise on quiet days.

**Secondary target: `y_return`**, the next-day log return in percent (regression).

**Split:** by date, never shuffled. Train to 31 Aug 2024, validation to 31 Aug 2025, test after. Hyperparameters are chosen on validation; test is scored once.
""")

code(r"""
balance = frame.groupby("split")["y_direction"].value_counts(normalize=True).unstack().reindex(["train", "val", "test"])
display((balance * 100).round(1).assign(days=frame["split"].value_counts()))
balance[["DOWN", "FLAT", "UP"]].plot.bar(stacked=True, figsize=(6, 3), color=["#c0504d", "#a6a6a6", "#4f81bd"], rot=0,
                                         title="Class balance per split")
plt.ylabel("share of days"); plt.legend(loc="center left", bbox_to_anchor=(1, 0.5)); plt.show()

print(frame.groupby("split")["y_return"].describe()[["mean", "std", "min", "max"]].round(3))
""")

code(r"""
fx = pd.read_parquet(ROOT / "data" / "raw" / "fx_panel.parquet")["USD_IDR"].dropna()
ax = fx.plot(color="black", lw=1, title="USD/IDR (JISDOR) with the chronological split")
for s, c in (("train", "#4f81bd"), ("val", "#f2c14e"), ("test", "#c0504d")):
    idx = parts[s].index
    ax.axvspan(idx.min(), idx.max(), color=c, alpha=0.15, label=s)
ax.set_ylabel("IDR per USD"); ax.legend(); plt.show()
""")

md(r"""
The three periods are different regimes: the rupiah weakened through 2022 to 2024 and again in 2025–2026. A model that learns a level or a trend from training data would fail on test, which is one more reason the target is a *return*, not a level.
""")

# ---------------------------------------------------------------------------
md(r"""
## 2. NLP features

Three classical extractors (the brief forbids pre-trained embeddings and transformers), all on **headlines**: GDELT gives no article bodies.

| Extractor | Text | Why |
|---|---|---|
| VADER | Track B (near-raw) | rules read capitals, "!" and negation, which Track A removes |
| Loughran-McDonald | Track A (lemmas) | finance-domain word lists: negative, positive, uncertainty, litigious, constraining |
| TF-IDF → SVD | Track A | learns the vocabulary of *our* corpus, including geopolitical words neither lexicon lists |

Each headline is scored, then averaged per trading-day bucket, weighted by `log1p(dup_count)`.

**First question: how often does each lexicon say anything at all?**
""")

code(r"""
cov = pd.read_csv(REP / "lexicon_coverage.csv").set_index("measure")["value"]
display(cov.to_frame().style.format("{:.3f}"))
""")

code(r"""
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import sys; sys.path.insert(0, str(ROOT))
from src.config import load_config
from src.nlp_features import load_lm_lexicon, lm_counts

cfg = load_config(ROOT / "config" / "config.yaml")
lex = load_lm_lexicon(ROOT / cfg.dig("task2", "loughran_mcdonald", "path"))
sia = SentimentIntensityAnalyzer()

sample = pd.read_csv(ROOT / "data" / "samples" / "articles_flagged_sample.csv", usecols=["keep", "text_track_a", "text_track_b"])
sample = sample.loc[sample["keep"]].dropna().sample(12, random_state=7)
lm = lm_counts(sample["text_track_a"], {c: lex[c] for c in ("negative", "positive", "uncertainty")})
show = pd.DataFrame({"headline (Track B)": sample["text_track_b"].str[:85].values,
                     "VADER": [sia.polarity_scores(t)["compound"] for t in sample["text_track_b"]],
                     "LM neg": lm["lm_negative"].values, "LM pos": lm["lm_positive"].values,
                     "LM unc": lm["lm_uncertainty"].values})
show
""")

code(r"""
examples = ["Russia invades Ukraine, markets plunge", "Fed will not cut rates", "Fed will cut rates",
            "Rupiah weakens as uncertainty over tariffs grows", "Ceasefire agreed after peace talks succeed"]
pd.DataFrame({"headline": examples, "VADER compound": [sia.polarity_scores(e)["compound"] for e in examples]})
""")

md(r"""
The hand-picked examples show each lexicon's blind spot. VADER was built for social media: it reads "not" correctly, but has no entry for *invade* or *plunge*, so a war headline can score as neutral. Loughran-McDonald was built from company 10-K filings: it knows *uncertainty*, *loss* and *crisis*, but not *missile* or *sanction*. Neither was built for geopolitics. That is the reason TF-IDF is in the feature set: it learns its vocabulary from our own headlines.
""")

code(r"""
nlp_daily = pd.read_parquet(ROOT / "data" / "processed" / "nlp_daily_features.parquet")
events = {"2022-02-24": "Russia invades Ukraine", "2023-10-07": "Hamas attack on Israel", "2025-04-02": "US 'Liberation Day' tariffs"}
fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
for ax, col, title in zip(axes, ["nlp_vader_mean", "nlp_lm_negative_rate", "nlp_lm_uncertainty_rate"],
                          ["VADER mean compound", "LM negative words per token", "LM uncertainty words per token"]):
    s = nlp_daily[col]
    ax.plot(s.index, s, color="#bbbbbb", lw=0.6)
    ax.plot(s.index, s.rolling(20, min_periods=5).mean(), color="#1f4e79", lw=1.4)
    ax.set_title(title, loc="left", fontsize=10)
    for d, label in events.items():
        ax.axvline(pd.Timestamp(d), color="#c0504d", ls="--", lw=0.8)
for d, label in events.items():
    axes[0].annotate(label, (pd.Timestamp(d), axes[0].get_ylim()[1]), fontsize=8, color="#c0504d",
                     ha="left", va="top", rotation=0, xytext=(3, -2), textcoords="offset points")
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------------------
md(r"""
## 3. TF-IDF topics

The TF-IDF vocabulary, the IDF weights and the SVD axes are fitted on **training-period news only**. Each axis is a direction in word space; the terms that load most on it say what it measures.
""")

code(r"""
print(manifest["tfidf"])
pd.read_csv(REP / "tfidf_svd_top_terms.csv").head(10)
""")

# ---------------------------------------------------------------------------
md(r"""
## 4. Do news features relate to the next day's move?

Spearman correlation between each (lagged) NLP feature and the next JISDOR return, and its absolute size (a volatility proxy), **on training days only**, so this look at the data cannot influence anything scored on validation or test.
""")

code(r"""
tr = parts["train"]
nlp_cols = groups["nlp"]
corr = pd.DataFrame({
    "corr_with_return": tr[nlp_cols].corrwith(tr["y_return"], method="spearman"),
    "corr_with_abs_return": tr[nlp_cols].corrwith(tr["y_return"].abs(), method="spearman"),
})
corr.reindex(corr["corr_with_abs_return"].abs().sort_values(ascending=False).index).head(15).round(3)
""")

code(r"""
n = len(tr)
print(f"With n = {n} training days, |rho| above ~{1.96 / np.sqrt(n):.3f} is distinguishable from zero at 5% "
      f"(before correcting for testing {len(nlp_cols)} features at once).")
""")

# ---------------------------------------------------------------------------
md(r"""
## 5. Results

Every learned model is run on several feature sets with the **same** model and tuning grid, so the difference between `price` (the brief's baseline) and `price+nlp` (the combined model) is the news and nothing else. The ablations answer two further questions: does news add anything beyond market controls (`price+market` vs `price+market+nlp`), and do Task 1's volume/theme aggregates carry signal (`price+task1_news`)?
""")

code(r"""
cls = results.query("task == 'cls'").pivot_table(index=["model", "feature_set"], columns="split",
                                                 values=["macro_f1", "accuracy", "directional_hit_rate"])
cls = cls.reindex(columns=["val", "test"], level=1)
cls.round(3)
""")

code(r"""
gain = pd.read_csv(REP / "news_gain.csv")
gain.pivot_table(index=["task", "model", "without_news", "with_news"], columns="split", values="gain").round(4)
""")

code(r"""
test_cls = results.query("task == 'cls' and split == 'test'").copy()
test_cls["label"] = test_cls["model"] + " | " + test_cls["feature_set"]
test_cls = test_cls.sort_values("macro_f1")
ax = test_cls.plot.barh(x="label", y="macro_f1", legend=False, figsize=(8, 7), color="#4f81bd",
                        title="Test macro-F1 (3 classes; majority-class baseline marked)")
maj = test_cls.loc[test_cls["model"] == "majority", "macro_f1"].iloc[0]
ax.axvline(maj, color="#c0504d", ls="--"); ax.axvline(1 / 3, color="grey", ls=":")
ax.set_xlabel("macro-F1"); plt.tight_layout(); plt.show()
""")

code(r"""
reg = results.query("task == 'reg'").pivot_table(index=["model", "feature_set"], columns="split",
                                                 values=["rmse", "mae", "directional_accuracy", "r2_vs_zero"])
reg.reindex(columns=["val", "test"], level=1).round(4)
""")

code(r"""
cm = pd.read_csv(REP / "confusion_test.csv")
keys = ["cls|logreg|price", "cls|logreg|price+nlp", "cls|xgb|price", "cls|xgb|price+nlp"]
fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
for ax, k in zip(axes, keys):
    m = cm.query("model_key == @k").pivot(index="true", columns="pred", values="count")
    ax.imshow(m.values, cmap="Blues")
    ax.set_xticks(range(3), [c.replace("pred_", "") for c in m.columns]); ax.set_yticks(range(3), [c.replace("true_", "") for c in m.index])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, m.values[i, j], ha="center", va="center")
    ax.set_title(k.replace("cls|", ""), fontsize=9); ax.grid(False)
plt.suptitle("Test confusion matrices (rows = true, columns = predicted)", y=1.03); plt.show()
""")

code(r"""
imp = pd.read_csv(REP / "feature_importance.csv")
top = imp.query("model_key == 'cls|xgb|price+market+nlp' and rank <= 15")
top.plot.barh(x="feature", y="importance", legend=False, figsize=(7, 5), color="#1f4e79",
              title="XGBoost (price + market + NLP): top-15 features by gain").invert_yaxis()
plt.tight_layout(); plt.show()
""")

md(r"""
## 6. Findings

Numbers from the run of 27 Sep 2026 (seed 42; a rerun reproduces them). Intervals are 95% block-bootstrap intervals over 10-day blocks (`reports/task2/news_gain.csv`).

**1. Price history alone predicts almost nothing.** On test, macro-F1 is 0.190 for the majority class, 0.208 for ARIMA(2,0,2) (AIC-selected), 0.321 for logistic regression and 0.331 for XGBoost on price features. The simplest rule, "tomorrow moves like today" (persistence), scores 0.370 and beats both learned price models. JISDOR's own past says little about its next move.

**2. The overnight dollar is the strongest predictor.** Adding the lagged market controls (DXY, VIX, Brent) raises test macro-F1 to 0.458 (logistic) and 0.436 (XGBoost), gains of +0.14 [+0.07, +0.21] and +0.11 [+0.06, +0.17]. `lag1_ret_DXY` alone has a Spearman correlation of 0.45 with the next JISDOR return on training days. The mechanism: JISDOR is fixed in the Jakarta afternoon, and the dollar's move during the US session that follows reaches the rupiah at the next fix. (Timing checked: Yahoo stamps DXY bars at 00:00 New York time, so the bar used for fix *t* closes around 05:00 WIB on day *t*, before the fix.)

**3. The NLP features did not add reliable predictive power for direction in this baseline.**
- Price → price + NLP (the brief's baseline vs combined model), test macro-F1: logistic +0.021 [−0.056, +0.125], XGBoost −0.018 [−0.104, +0.077]. Both intervals include zero.
- Price + market → price + market + NLP: on **validation** news appeared to help (logistic +0.069, bootstrap p = 0.04; XGBoost +0.051), but the gain **did not carry over to test** (−0.070 and −0.006). This is the pattern of a feature set that fits one period and not the next, which is why the test set is scored only once.
- Regression tells the same story. One of the 16 news comparisons favours news with an interval above zero (XGBoost, price → price + NLP, test RMSE −0.013). With 16 comparisons, about one such result is expected by chance.
- The best test model overall, XGBoost on all features (macro-F1 0.456), does beat persistence (+0.086 [+0.018, +0.159]), but it owes that mainly to the market controls (finding 2).

**4. News relates more to how much the rupiah moves than to which way.** On training days, 10 of the 46 NLP features correlate with the *absolute* next-day return beyond the 5% threshold (|ρ| > 0.073), against 1 of 46 for the *signed* return, about what chance alone would give (≈ 2.3). The strongest are VADER sentiment of energy and monetary-policy headlines and several TF-IDF topic axes. Geopolitical news looks like a volatility signal more than a direction signal.

**5. Why the lexicons struggle.** VADER scores 36.5% of headlines as exactly neutral, and Loughran-McDonald finds no word at all in 53.9% of them (negative words in 31.3%, uncertainty in 5.8%). Neither lexicon was built for geopolitics: VADER has no entry for *invade*; Loughran-McDonald, built from 10-K filings, has none for *missile* or *sanction*. The TF-IDF axes are interpretable (Ukraine war, COVID, Israel–Gaza, US elections, the debt ceiling), but they describe *which events* happened in the training years, and those events do not recur in the test year.

### What this means for Task 3
- **Target:** add a volatility formulation (|return| or large-move vs normal day), where the news signal is actually visible (finding 4).
- **Features:** a geopolitical-risk dictionary (for example the word lists behind Caldara & Iacoviello's GPR index) in place of general-purpose sentiment; news *surprise* (change against the trailing window) rather than level; Indonesia-specific subsets (headlines mentioning Indonesia, Bank Indonesia or commodity exports).
- **Evaluation:** rolling-origin (walk-forward) evaluation, so a result does not rest on one validation year and one test year.
- **Baseline to beat:** price + market controls (test macro-F1 ≈ 0.44–0.46), not price alone. Any news claim must add to what the overnight dollar already tells us.
""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT}")
