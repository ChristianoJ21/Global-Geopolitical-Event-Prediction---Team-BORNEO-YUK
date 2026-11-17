"""Task 3 — analysis of the features and the models (brief section 3c).

    py -m src.analysis_task3          # after src.run_task3; writes reports/task3/analysis_* + figures/

1. Embedding space: t-SNE of headlines from high- vs low-volatility days, and a
   number to go with the picture (how well can a simple classifier tell the
   two groups apart from the embedding alone?).
2. Interpretability: TreeSHAP for the XGBoost Model C (which feature blocks and
   which features move the prediction), and the days with the largest
   prediction shift between Model A and Model C, with their top headlines.
3. Correlation & error analysis: which text features correlate with the size
   vs the direction of the next move, and on which test days the news helped
   (A gave a false directional signal, C did not) or added noise.

"Volatility" of a news bucket = |return| of the fix that bucket feeds, so the
question is "does the news BEFORE a big move look different?", which is the
question a forecaster cares about.
"""
from __future__ import annotations

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost  # noqa: F401  must load before torch (macOS OpenMP clash, see run_task3)
from scipy import stats

from . import nlp_features, text_embeddings
from .build_dataset import _setup_logging
from .config import load_config
from .evaluate import CLASSES

log = logging.getLogger("analysis_task3")
COLORS = {"high": "#c0392b", "low": "#2e86c1", "A": "#7f8c8d", "C": "#16a085"}


def _paths(cfg):
    t3 = cfg.dig("task3", "paths")
    rep, fig = cfg.root / t3["reports"], cfg.root / t3["figures"]
    rep.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)
    return rep, fig, cfg.root / t3["splits"]


def load_task3_frame(cfg):
    _, _, sdir = _paths(cfg)
    frame = pd.concat([pd.read_csv(sdir / f"{s}.csv", index_col=0, parse_dates=True)
                       for s in ("train", "val", "test")]).sort_index()
    groups = json.loads((sdir / "feature_groups.json").read_text())
    return frame, groups


def bucket_volatility(frame: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.Series:
    """|next return| that each news bucket feeds (bucket of day t-1 -> |y_return| at t)."""
    absret = frame["y_return"].abs().reindex(calendar)
    return pd.Series(absret.shift(-1).to_numpy(), index=calendar)


# ---------------------------------------------------------------------------
# 1. Embedding space
# ---------------------------------------------------------------------------
def tsne_volatility(cfg, articles, frame, calendar, encoders=("finbert", "minilm", "glove"),
                    n_per_group: int = 1500, q: float = 0.15):
    """t-SNE of headlines from the most / least volatile TRAINING buckets."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.manifold import TSNE
    from sklearn.model_selection import cross_val_score

    rep, fig_dir, _ = _paths(cfg)
    vol = bucket_volatility(frame, calendar)
    train_buckets = vol.loc[vol.index < frame.index[frame["split"] == "train"].max()].dropna()
    hi_days = train_buckets[train_buckets >= train_buckets.quantile(1 - q)].index
    lo_days = train_buckets[train_buckets <= train_buckets.quantile(q)].index
    rng = np.random.default_rng(cfg.seed)
    nd = articles["news_day"]
    hi = rng.choice(np.flatnonzero(nd.isin(hi_days).to_numpy()), n_per_group, replace=False)
    lo = rng.choice(np.flatnonzero(nd.isin(lo_days).to_numpy()), n_per_group, replace=False)
    rows = np.r_[hi, lo]
    group = np.r_[np.ones(len(hi)), np.zeros(len(lo))]

    quant, coords = [], {}
    w = nlp_features.article_weights(articles, cfg).to_numpy()
    for enc in encoders:
        E = text_embeddings.load_embedding(cfg, enc)
        X = np.asarray(E[rows], dtype=np.float32)
        # Headline level: can a linear probe tell which kind of day a headline came from?
        head_auc = cross_val_score(LogisticRegression(C=0.1, max_iter=2000), X, group, cv=5,
                                   scoring="roc_auc").mean()
        # Day level: daily mean vector -> high vs low day (5-fold CV AUC).
        dmat, ylab = [], []
        for d, lab in zip(list(hi_days) + list(lo_days), [1] * len(hi_days) + [0] * len(lo_days)):
            idx = np.flatnonzero((nd == d).to_numpy())
            if len(idx):
                dmat.append(np.average(np.asarray(E[idx], dtype=np.float32), axis=0, weights=w[idx]))
                ylab.append(lab)
        day_auc = cross_val_score(LogisticRegression(C=0.1, max_iter=2000), np.vstack(dmat), ylab,
                                  cv=5, scoring="roc_auc").mean()
        quant.append({"encoder": enc, "headline_auc": float(head_auc), "day_auc": float(day_auc),
                      "n_high_days": len(hi_days), "n_low_days": len(lo_days)})
        coords[enc] = TSNE(n_components=2, perplexity=30, init="pca", random_state=cfg.seed).fit_transform(X)
        log.info("t-SNE %s: headline AUC %.3f, day AUC %.3f", enc, head_auc, day_auc)

    quant = pd.DataFrame(quant)
    quant.to_csv(rep / "analysis_vol_separability.csv", index=False)
    fig, axes = plt.subplots(1, len(encoders), figsize=(4.2 * len(encoders), 4))
    for ax, enc in zip(np.atleast_1d(axes), encoders):
        Z = coords[enc]
        for g, name in ((0, "low"), (1, "high")):
            sel = group == g
            ax.scatter(Z[sel, 0], Z[sel, 1], s=3, alpha=0.45, c=COLORS[name],
                       label=f"{name}-volatility days")
        r = quant.set_index("encoder").loc[enc]
        ax.set_title(f"{enc}  (headline AUC {r.headline_auc:.2f}, day AUC {r.day_auc:.2f})", fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
    np.atleast_1d(axes)[0].legend(markerscale=4, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(fig_dir / "tsne_volatility.png", dpi=170)
    plt.close(fig)
    return quant


# ---------------------------------------------------------------------------
# 2. Interpretability
# ---------------------------------------------------------------------------
def shap_model_c(cfg, frame, groups, manifest):
    """TreeSHAP of the XGBoost Model C refitted on train+val, explained on test days."""
    import xgboost as xgb

    from .fusion import fit_proba
    rep, fig_dir, _ = _paths(cfg)
    mc = next(r for r in manifest["model_C"] if r["model"] == "xgb")
    blk = mc["C_block"]
    cols = groups["price"] + groups["market"] + groups[blk]
    tv = frame["split"].isin(["train", "val"]).to_numpy()
    te = (frame["split"] == "test").to_numpy()
    model, _ = fit_proba("xgb", mc["C_params"], cfg, frame.loc[tv, cols], frame.loc[tv, "y_direction"],
                         frame.loc[te, cols], seed=cfg.seed)
    contrib = model.get_booster().predict(xgb.DMatrix(frame.loc[te, cols]), pred_contribs=True)
    # contrib: [n_days, n_classes, n_features + 1 bias] -> mean |SHAP| over days and classes
    imp = pd.Series(np.abs(contrib[:, :, :-1]).mean(axis=(0, 1)), index=cols).sort_values(ascending=False)
    block_of = {c: g for g in ("price", "market") for c in groups[g]}
    block_of.update({c: "text" for c in groups[blk]})
    out = imp.rename("mean_abs_shap").to_frame()
    out["block"] = [block_of[c] for c in out.index]
    out.to_csv(rep / "analysis_shap_model_c.csv")
    share = out.groupby("block")["mean_abs_shap"].sum()
    share = (share / share.sum()).reindex(["price", "market", "text"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1, 1.5]})
    axes[0].barh(share.index, share.values, color=["#95a5a6", "#34495e", COLORS["C"]])
    for i, v in enumerate(share.values):
        axes[0].text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=8)
    axes[0].set_xlim(0, max(share.values) * 1.25)
    axes[0].set_title("Share of |SHAP| by block (test days)", fontsize=10)
    top = out.head(12).iloc[::-1]
    axes[1].barh(top.index, top["mean_abs_shap"],
                 color=[COLORS["C"] if b == "text" else "#95a5a6" if b == "price" else "#34495e"
                        for b in top["block"]])
    axes[1].set_title(f"Top 12 features, XGBoost Model C ({blk})", fontsize=10)
    axes[1].tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "shap_model_c.png", dpi=170)
    plt.close(fig)
    return out


def topic_importance(cfg, frame, groups):
    """Which topics' share/tone correlate with the next move (train days, Spearman)?"""
    rep, _, _ = _paths(cfg)
    labels = pd.read_csv(rep / "topic_labels.csv").set_index("topic")
    tr = frame.loc[frame["split"] == "train"]
    rows = []
    for c in groups["topics"]:
        kind, t = c.split("_")[2], int(c.rsplit("_", 1)[1])
        ok = tr[c].notna()
        r_dir, p_dir = stats.spearmanr(tr.loc[ok, c], tr.loc[ok, "y_return"])
        r_abs, p_abs = stats.spearmanr(tr.loc[ok, c], tr.loc[ok, "y_return"].abs())
        rows.append({"feature": c, "kind": kind, "topic": t, "top_words": labels.loc[t, "top_words"],
                     "rho_return": r_dir, "p_return": p_dir, "rho_abs_return": r_abs, "p_abs_return": p_abs})
    out = pd.DataFrame(rows)
    out.to_csv(rep / "analysis_topic_correlation.csv", index=False)
    return out


def feature_correlations(frame, groups, blocks):
    """Per block: how many features correlate (p<0.05) with direction vs size of the next move."""
    tr = frame.loc[frame["split"] == "train"]
    rows = []
    for b in blocks:
        n_dir = n_abs = 0
        for c in groups[b]:
            ok = tr[c].notna()
            n_dir += stats.spearmanr(tr.loc[ok, c], tr.loc[ok, "y_return"])[1] < 0.05
            n_abs += stats.spearmanr(tr.loc[ok, c], tr.loc[ok, "y_return"].abs())[1] < 0.05
        rows.append({"block": b, "n_features": len(groups[b]), "sig_direction": int(n_dir),
                     "sig_magnitude": int(n_abs), "expected_by_chance": 0.05 * len(groups[b])})
    return pd.DataFrame(rows)


def headlines_for(articles, day, n=3, probs=None):
    """The n most-syndicated headlines of one news bucket (with FinBERT tone if given)."""
    sel = articles.loc[articles["news_day"] == day].sort_values("dup_count", ascending=False, kind="stable")
    out = []
    for i, r in sel.head(n).iterrows():
        tone = ""
        if probs is not None:
            p = probs.iloc[i]
            tone = f" [FinBERT net {p['finbert_positive'] - p['finbert_negative']:+.2f}]"
        out.append(f"{r['text_track_b']} (x{int(r['dup_count'])}){tone}")
    return out


def prediction_shifts(cfg, articles, calendar, key_a: str, key_c: str, n: int = 8):
    """Test days where Model C's probabilities moved furthest from Model A's."""
    rep, _, _ = _paths(cfg)
    pred = pd.read_csv(rep / "predictions.csv", index_col=0, parse_dates=True)
    te = pred.loc[pred["split"] == "test"]
    PA = te[[f"{key_a}|{c}" for c in CLASSES]].to_numpy()
    PC = te[[f"{key_c}|{c}" for c in CLASSES]].to_numpy()
    shift = np.abs(PC - PA).sum(axis=1) / 2               # total-variation distance, 0..1
    prev = pd.Series(calendar, index=calendar).shift(1)
    probs = text_embeddings.load_finbert_probs(cfg)
    rows = []
    for i in np.argsort(-shift)[:n]:
        d = te.index[i]
        rows.append({"date": d.date(), "news_bucket": prev.loc[d].date(), "shift": float(shift[i]),
                     "true": te["y_direction"].iloc[i],
                     "pred_A": CLASSES[int(PA[i].argmax())], "pred_C": CLASSES[int(PC[i].argmax())],
                     "top_headlines": " || ".join(headlines_for(articles, prev.loc[d], 3, probs))})
    return pd.DataFrame(rows), pd.Series(shift, index=te.index)


# ---------------------------------------------------------------------------
# 3. Error analysis
# ---------------------------------------------------------------------------
def error_analysis(cfg, articles, frame, calendar, key_a: str, key_c: str):
    """Where did text help (A made a false directional call, C did not) and where did it hurt?"""
    rep, fig_dir, _ = _paths(cfg)
    pred = pd.read_csv(rep / "predictions.csv", index_col=0, parse_dates=True)
    te = pred.loc[pred["split"] == "test"].copy()
    lab = lambda k: np.asarray(CLASSES, dtype=object)[te[[f"{k}|{c}" for c in CLASSES]].to_numpy().argmax(1)]
    te["pred_A"], te["pred_C"] = lab(key_a), lab(key_c)
    te["ok_A"], te["ok_C"] = te["pred_A"] == te["y_direction"], te["pred_C"] == te["y_direction"]
    te["false_signal_A"] = (te["pred_A"] != "FLAT") & ~te["ok_A"]
    te["false_signal_C"] = (te["pred_C"] != "FLAT") & ~te["ok_C"]
    table = pd.crosstab(te["ok_A"], te["ok_C"]).rename(index={True: "A right", False: "A wrong"},
                                                       columns={True: "C right", False: "C wrong"})
    b, c = int(((~te["ok_A"]) & te["ok_C"]).sum()), int((te["ok_A"] & ~te["ok_C"]).sum())
    mcnemar_p = float(stats.binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0

    f = frame.reindex(te.index)
    te["abs_return"] = te["y_return"].abs()
    te["news_volume_z"] = f["vol_z30"]
    te["finbert_net"] = f["t3_finbert_net"]
    regimes = []
    for col, name in (("abs_return", "size of the move"), ("news_volume_z", "news volume (z30)"),
                      ("finbert_net", "FinBERT net tone")):
        bins = pd.qcut(te[col], 3, labels=["low", "mid", "high"])
        for b_, g in te.groupby(bins, observed=True):
            regimes.append({"regime": name, "tercile": b_, "n": len(g),
                            "error_A": float(1 - g["ok_A"].mean()), "error_C": float(1 - g["ok_C"].mean())})
    regimes = pd.DataFrame(regimes)
    regimes.to_csv(rep / "analysis_error_by_regime.csv", index=False)

    prev = pd.Series(calendar, index=calendar).shift(1)
    probs = text_embeddings.load_finbert_probs(cfg)
    helped = te.loc[te["false_signal_A"] & te["ok_C"]].sort_values("abs_return", ascending=False)
    hurt = te.loc[te["ok_A"] & ~te["ok_C"]].sort_values("abs_return", ascending=False)
    ex = []
    for kind, df in (("helped", helped), ("hurt", hurt)):
        for d, r in df.head(5).iterrows():
            ex.append({"kind": kind, "date": d.date(), "true": r["y_direction"], "return": r["y_return"],
                       "pred_A": r["pred_A"], "pred_C": r["pred_C"],
                       "top_headlines": " || ".join(headlines_for(articles, prev.loc[d], 2, probs))})
    ex = pd.DataFrame(ex)
    ex.to_csv(rep / "analysis_error_examples.csv", index=False)
    summary = {"n_test": int(len(te)), "both_right": int((te.ok_A & te.ok_C).sum()),
               "only_C_right": b, "only_A_right": c, "both_wrong": int((~te.ok_A & ~te.ok_C).sum()),
               "A_false_signals": int(te["false_signal_A"].sum()),
               "C_false_signals": int(te["false_signal_C"].sum()),
               "false_signals_fixed_by_C": int((te["false_signal_A"] & te["ok_C"]).sum()),
               "mcnemar_p": mcnemar_p}
    (rep / "analysis_error_summary.json").write_text(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    x = np.arange(len(regimes))
    ax.bar(x - 0.2, regimes["error_A"], 0.4, label="A: time series only", color=COLORS["A"])
    ax.bar(x + 0.2, regimes["error_C"], 0.4, label="C: + Task 3 text", color=COLORS["C"])
    ax.set_xticks(x, list(regimes["tercile"]), fontsize=8)
    for i, name in enumerate(regimes["regime"].unique()):
        ax.text(3 * i + 1, 0.97, name, ha="center", fontsize=8, transform=ax.get_xaxis_transform())
        if i:
            ax.axvline(3 * i - 0.5, color="#ccc", lw=0.8)
    ax.set_ylabel("error rate (test)")
    ax.set_ylim(0, 0.95)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_dir / "error_by_regime.png", dpi=170)
    plt.close(fig)
    return table, summary, regimes, ex


def drift_analysis(cfg, frame, groups, blocks: list[str]):
    """Adversarial validation: can each feature block tell a training day from a test day?

    A logistic regression is asked to classify rows as train-period vs
    test-period (5-fold CV ROC-AUC). AUC near 0.5 = the block looks the same in
    both periods, so a mapping learned on train can transfer. AUC near 1.0 =
    the block mostly encodes WHICH ERA it is (COVID, Ukraine, Gaza, tariffs),
    and a model fitted on 2021-24 news has to extrapolate on 2025-26 news.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rep, _, _ = _paths(cfg)
    sel = frame["split"].isin(["train", "test"])
    y = (frame.loc[sel, "split"] == "test").astype(int)
    cv = StratifiedKFold(5, shuffle=True, random_state=cfg.seed)
    rows = []
    for b in blocks:
        X = frame.loc[sel, groups[b] if b in groups else sum((groups[g] for g in b.split("+")), [])]
        pipe = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                             LogisticRegression(C=0.1, max_iter=3000))
        rows.append({"block": b, "n_features": X.shape[1],
                     "train_vs_test_auc": float(cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean())})
    out = pd.DataFrame(rows)
    out.to_csv(rep / "analysis_drift.csv", index=False)
    return out


def magnitude_ablation(cfg, frame, groups, blocks_c: list[str]):
    """Does text predict HOW BIG tomorrow's move is? Target = |next-day return|.

    Task 2 found news correlates with the size of the move, not its sign. Here
    the same A / B / C feature sets predict |y_return| with Ridge (alpha chosen
    on validation, refit on train+val, test once). Scored by RMSE, R^2 against
    the training mean, and Spearman rank correlation, with a block-bootstrap
    interval for the RMSE reduction of each set vs A.
    """
    from .evaluate import block_bootstrap_gain
    from .models import fit_predict, make_model
    rep, _, _ = _paths(cfg)
    y = frame["y_return"].abs()
    m = {s: (frame["split"] == s).to_numpy() for s in ("train", "val", "test")}
    m["trainval"] = m["train"] | m["val"]
    sets = {"A": ["price", "market"], "B": ["price", "market", "nlp"]}
    for b in blocks_c:
        sets[f"C: A + {b}"] = ["price", "market", b]
    preds, rows = {}, []
    for name, blocks in sets.items():
        X = frame[[c for b in blocks for c in groups[b]]]
        best = (None, np.inf)
        for alpha in (1.0, 10.0, 100.0, 1000.0):
            pv = fit_predict("reg", "ridge", make_model("reg", "ridge", {"alpha": alpha}, cfg),
                             X[m["train"]], y[m["train"]], X[m["val"]])
            rmse = float(np.sqrt(np.mean((pv - y[m["val"]]) ** 2)))
            if rmse < best[1]:
                best = (alpha, rmse)
        pt = fit_predict("reg", "ridge", make_model("reg", "ridge", {"alpha": best[0]}, cfg),
                         X[m["trainval"]], y[m["trainval"]], X[m["test"]])
        yt = y[m["test"]].to_numpy()
        preds[name] = pt
        rows.append({"set": name, "alpha": best[0], "val_rmse": best[1],
                     "test_rmse": float(np.sqrt(np.mean((pt - yt) ** 2))),
                     "test_r2_vs_mean": float(1 - np.sum((pt - yt) ** 2) /
                                              np.sum((yt - y[m["trainval"]].mean()) ** 2)),
                     "test_spearman": float(stats.spearmanr(pt, yt)[0])})
    out = pd.DataFrame(rows)
    yt = y[m["test"]]
    for i, name in enumerate(out["set"]):
        if name == "A":
            continue
        ci = block_bootstrap_gain("reg", yt, preds["A"], preds[name], seed=cfg.seed)
        out.loc[i, ["rmse_gain_ci_low", "rmse_gain_ci_high", "p_no_gain"]] = \
            [ci["ci_low"], ci["ci_high"], ci["p_no_gain"]]
    out.to_csv(rep / "analysis_magnitude.csv", index=False)
    return out


def ablation_figure(cfg):
    """Bar chart of macro-F1 (mean +- std over seeds) for A0 / A / B / C / B+C."""
    rep, fig_dir, _ = _paths(cfg)
    res = pd.read_csv(rep / "results.csv")
    abl = res.loc[(res["stage"] == "5_ablation")].copy()
    abl["set"] = abl["experiment"].str.split("|").str[0]
    s = abl.groupby(["set", "model", "split"])["macro_f1"].agg(["mean", "std"]).reset_index()
    order = ["A0", "A", "B", "C", "B+C"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.1), sharey=True)
    for ax, split in zip(axes, ("val", "test")):
        for j, (kind, col) in enumerate((("logreg", "#34495e"), ("xgb", "#16a085"))):
            d = s.loc[(s["model"] == kind) & (s["split"] == split)].set_index("set").reindex(order)
            ax.bar(np.arange(5) + (j - 0.5) * 0.38, d["mean"], 0.38, yerr=d["std"].fillna(0),
                   color=col, label="LogReg" if kind == "logreg" else "XGBoost (mean of 3 seeds)", capsize=2)
        ax.set_xticks(range(5), ["A0\nprice", "A\n+markets", "B\nA + T2 NLP", "C\nA + T3 text", "B+C"],
                      fontsize=8)
        ax.axhline(0.370, ls="--", lw=0.8, color="#999")
        ax.text(-0.4, 0.373, "persistence (test)", fontsize=7, color="#777", va="bottom")
        ax.set_title(f"{'Validation' if split == 'val' else 'Test'} macro-F1", fontsize=10)
        ax.set_ylim(0.25, 0.6)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(fig_dir / "ablation.png", dpi=170)
    plt.close(fig)
    return s


def main(argv=None) -> int:
    _setup_logging(True)
    cfg = load_config("config/config.yaml")
    rep, _, _ = _paths(cfg)
    manifest = json.loads((rep / "manifest.json").read_text())
    frame, groups = load_task3_frame(cfg)
    calendar = pd.DatetimeIndex(pd.read_parquet(cfg.path("processed") / "dataset.parquet").index).sort_values()
    articles = nlp_features.load_kept_articles(cfg)
    rows = text_embeddings.corpus_rows(cfg, articles)      # same Task 3 corpus as run_task3
    text_embeddings.use_rows(rows)
    articles = articles.iloc[rows].reset_index(drop=True)

    ablation_figure(cfg)
    tsne_volatility(cfg, articles, frame, calendar)
    shap_model_c(cfg, frame, groups, manifest)
    topic_importance(cfg, frame, groups)
    text_blocks = [g for g in groups if g not in ("price", "market", "task1_news")]
    feature_correlations(frame, groups, text_blocks).to_csv(rep / "analysis_feature_correlations.csv",
                                                              index=False)
    for kind in ("logreg", "xgb"):
        prediction_shifts(cfg, articles, calendar, f"A|seed42|{kind}", f"C|seed42|{kind}")[0].to_csv(
            rep / f"analysis_prediction_shifts_{kind}.csv", index=False)
    error_analysis(cfg, articles, frame, calendar, "A|seed42|logreg", "C|seed42|logreg")
    emb_mean = [g for g in groups if g.startswith("emb_") and g.endswith("_mean")]
    magnitude_ablation(cfg, frame, groups, emb_mean + ["finbert_sent", "topics", "entity",
                                                       "emb_finbert_decay", "emb_finbert_attention"])
    drift_analysis(cfg, frame, groups, ["price", "market", "price+market", "nlp"] + emb_mean
                   + ["emb_finbert_decay", "emb_finbert_attention", "finbert_sent", "topics", "entity"])
    log.info("analysis written to %s", rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
