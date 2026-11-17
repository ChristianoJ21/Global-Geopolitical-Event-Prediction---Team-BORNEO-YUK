"""Task 3 pipeline driver: embeddings -> daily text blocks -> fusion -> ablation.

    py -m src.run_task3              # full run (first run encodes 905k headlines, ~1 h on an M3)
    py -m src.run_task3 --skip-nn    # skip the attention network and fine-tuning

Inputs
    data/splits/{train,val,test}.csv          Task 2's modelling table (unchanged)
    data/processed/dataset.parquet            the full trading calendar (for the 1-day shift)
    data/interim/articles_flagged.parquet     kept headlines, both text tracks

Outputs
    data/interim/embeddings/*                 per-headline vectors, FinBERT probs, NER, fine-tune cache
    data/processed/task3_text_daily.parquet   all Task 3 text features per news day (unlagged)
    data/splits/task3/{train,val,test}.csv    Task 2 table + Task 3 text features (lagged)
    data/splits/task3/feature_groups.json     which columns belong to which block
    reports/task3/*.csv, manifest.json        every table used in the Task 3 report

Experiment plan (the order is the argument)
    Stage 1  representation comparison: A + each encoder's daily-mean PCA block,
             and each block alone (text only). Fixed settings, Task 2 grid.
    Stage 2  feature strategies: FinBERT tone, topics, NER entity filter, and
             three temporal poolings (mean / decay / salience attention).
    Stage 3  fusion: early (concat), late (probability blend), attention network,
             and late fusion with the fine-tuned transformer.
    Stage 4  Model C = the early-fusion text block with the best VALIDATION
             macro-F1, tuned on a wider grid. Then the A / B / C ablation on
             test, with block-bootstrap intervals and 3 seeds for XGBoost.
    The Task 2 protocol holds throughout: fit on train, choose on validation,
    refit on train+validation, score test once. Nothing is chosen on test.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
# XGBoost must be imported BEFORE torch: on macOS each ships its own OpenMP
# runtime, and loading torch's first makes the first XGBoost fit segfault.
import xgboost  # noqa: F401  isort: skip

from . import entity_filter, fusion, nlp_features, text_embeddings, text_features
from .build_dataset import _setup_logging
from .config import load_config
from .evaluate import CLASSES, block_bootstrap_gain, classification_metrics, regression_metrics
from .models import fit_predict, make_model, param_grid

log = logging.getLogger("run_task3")
TIME_KEY = "A: time series (price+market)|logreg"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_task2_frame(cfg) -> tuple[pd.DataFrame, dict, pd.DatetimeIndex]:
    sdir = cfg.root / cfg.dig("task2", "paths", "splits")
    frame = pd.concat([pd.read_csv(sdir / f"{s}.csv", index_col=0, parse_dates=True)
                       for s in ("train", "val", "test")]).sort_index()
    groups = json.loads((sdir / "feature_groups.json").read_text())
    calendar = pd.read_parquet(cfg.path("processed") / "dataset.parquet").index.sort_values()
    return frame, groups, pd.DatetimeIndex(calendar).normalize()


def ensure_finbert(cfg) -> None:
    """FinBERT is encoded by ``python -m src.text_embeddings``, one part per fresh process.

    (On an 8 GB Mac a single long-lived process slows to a crawl; see
    text_embeddings.device.) Exit code 3 = one more part done, 0 = assembled.
    """
    import os
    import subprocess
    import sys
    if (text_embeddings.cache_dir(cfg) / "finbert_probs.parquet").exists():
        return
    for _ in range(100):
        rc = subprocess.run([sys.executable, "-m", "src.text_embeddings"], cwd=cfg.root,
                            env={**os.environ}).returncode
        if rc == 0:
            return
        if rc != 3:
            raise RuntimeError(f"FinBERT encoding failed (exit {rc})")
    raise RuntimeError("FinBERT encoding did not finish")


def lag_to_rows(daily: pd.DataFrame, calendar: pd.DatetimeIndex, rows: pd.DatetimeIndex) -> pd.DataFrame:
    """Same shift as Task 1/2: news bucket of trading day t-1 sits on row t."""
    return daily.reindex(calendar).shift(1).reindex(rows)


def build_text_daily(cfg, articles, fit_before, use_cache=True):
    """Every Task 3 text block, per news day (unlagged). Returns (daily, groups, extras)."""
    probs = text_embeddings.load_finbert_probs(cfg)
    blocks, groups, extras = [], {}, {"pca_variance": []}

    def add(name, df):
        blocks.append(df)
        groups[name] = list(df.columns)

    for enc in cfg.dig("task3", "encoders"):
        E = text_embeddings.load_embedding(cfg, enc)
        hows = ("mean", "decay", "attention") if enc in ("minilm", "finbert") else ("mean",)
        for how in hows:
            Z, pca, _ = text_features.embedding_block(E, articles, cfg, enc, fit_before, how=how, probs=probs)
            add(f"emb_{enc}_{how}", Z)
            extras["pca_variance"].append({"encoder": enc, "pooling": how, "k": pca.k,
                                           "explained": float(pca.pca.explained_variance_ratio_.sum())})

    add("finbert_sent", text_features.finbert_sentiment_block(articles, probs, cfg))

    E = text_embeddings.load_embedding(cfg, "minilm")
    _, labels, train_idx = text_features.fit_topics(E, articles, fit_before, cfg)
    extras["topic_labels"] = text_features.topic_labels(articles, labels, train_idx)
    np.save(text_embeddings.cache_dir(cfg) / "topic_labels.npy", labels)
    add("topics", text_features.topic_block(articles, labels, probs, cfg))

    ner = entity_filter.run_ner(cfg, articles, use_cache=use_cache)
    flags = ner[[f"ent_{g}" for g in entity_filter.GROUPS]]
    extras["ner_coverage"] = {**{g: float(ner[f"ent_{g}"].mean()) for g in entity_filter.GROUPS},
                              "any": float(flags.any(axis=1).mean())}
    add("entity", entity_filter.entity_block(articles, ner, probs, cfg))

    daily = pd.concat(blocks, axis=1).sort_index()
    daily.index.name = "date"
    return daily, groups, extras


# ---------------------------------------------------------------------------
# Experiment helpers
# ---------------------------------------------------------------------------
class Lab:
    """Holds the frame, masks and every result, so each experiment is one call."""

    def __init__(self, frame, groups, cfg):
        self.f, self.g, self.cfg = frame, groups, cfg
        s = frame["split"]
        self.m = {"train": (s == "train").to_numpy(), "val": (s == "val").to_numpy(),
                  "test": (s == "test").to_numpy()}
        self.m["trainval"] = self.m["train"] | self.m["val"]
        self.y = frame["y_direction"]
        self.records, self.grid_rows, self.probs = [], [], {}

    def cols(self, blocks):
        return [c for b in blocks for c in self.g[b]]

    def record(self, stage, name, model, params, pv, pt, n_features, extra=None):
        for split, P in (("val", pv), ("test", pt)):
            met = classification_metrics(self.y[self.m[split]], fusion.proba_to_label(P))
            self.records.append({"stage": stage, "experiment": name, "model": model, "split": split,
                                 "n_features": n_features, "params": json.dumps(params or {}, default=str),
                                 **met, **(extra or {})})
        self.probs[f"{name}|{model}"] = (pv, pt)

    def early(self, stage, name, blocks, kinds=("logreg", "xgb"), grid=None, seed=None):
        """Early fusion: one model on the concatenated blocks. Returns {kind: (params, val_f1)}."""
        X = self.f[self.cols(blocks)]
        out = {}
        for kind in kinds:
            best = (None, -1.0, None)
            for params in (grid[kind] if grid else param_grid("cls", kind, self.cfg)):
                _, pv = fusion.fit_proba(kind, params, self.cfg, X[self.m["train"]], self.y[self.m["train"]],
                                         X[self.m["val"]], seed=seed)
                score = fusion.macro_f1(self.y[self.m["val"]], pv)
                self.grid_rows.append({"stage": stage, "experiment": name, "model": kind,
                                       "params": json.dumps(params), "val_macro_f1": score})
                if score > best[1]:
                    best = (params, score, pv)
            params, score, pv = best
            _, pt = fusion.fit_proba(kind, params, self.cfg, X[self.m["trainval"]], self.y[self.m["trainval"]],
                                     X[self.m["test"]], seed=seed)
            self.record(stage, name, kind, params, pv, pt, X.shape[1], {"blocks": "+".join(blocks)})
            out[kind] = (params, score)
            log.info("[%s] %-40s %-6s val=%.3f test=%.3f", stage, name, kind, score,
                     fusion.macro_f1(self.y[self.m["test"]], pt))
        return out

    def late(self, stage, name, time_key, text_blocks=None, text_probs=None):
        """Late fusion: blend the time model's probabilities with a text-only model's."""
        pv_time, pt_time = self.probs[time_key]
        if text_probs is None:
            X = self.f[self.cols(text_blocks)]
            best = (None, -1.0, None)
            for params in param_grid("cls", "logreg", self.cfg):
                _, pv = fusion.fit_proba("logreg", params, self.cfg, X[self.m["train"]],
                                         self.y[self.m["train"]], X[self.m["val"]])
                sc = fusion.macro_f1(self.y[self.m["val"]], pv)
                if sc > best[1]:
                    best = (params, sc, pv)
            _, pt = fusion.fit_proba("logreg", best[0], self.cfg, X[self.m["trainval"]],
                                     self.y[self.m["trainval"]], X[self.m["test"]])
            pv_text, pt_text = best[2], pt
        else:
            pv_text, pt_text = text_probs
        weights = self.cfg.dig("task3", "late_fusion_weights")
        w, table = fusion.choose_blend_weight(self.y[self.m["val"]], pv_time, pv_text, weights)
        self.record(stage, name, "late", {"w": w, "time_model": time_key},
                    fusion.late_blend(pv_time, pv_text, w), fusion.late_blend(pt_time, pt_text, w),
                    None, {"blocks": "+".join(text_blocks or ["finetuned"])})
        log.info("[%s] %-40s late   w=%.1f val=%.3f", stage, name, w, table["val_macro_f1"].max())
        return w, table.assign(experiment=name)


def attention_net_experiment(lab: Lab, cfg, articles, calendar, enc: str, seeds):
    """Attention fusion network on the top-k headline vectors of each row's news bucket."""
    acfg = cfg.dig("task3", "attention_net", default={}) or {}
    k = int(acfg.get("top_k_per_day", 64))
    rows = entity_filter.top_k_per_day(articles, k)
    sub = articles.iloc[rows]
    E = np.asarray(text_embeddings.load_embedding(cfg, enc)[rows], dtype=np.float32)
    w = np.log(nlp_features.article_weights(sub, cfg).to_numpy(dtype=np.float32))
    days = pd.DatetimeIndex(sorted(sub["news_day"].unique()))
    didx = days.get_indexer(sub["news_day"])          # rows are sorted, so days are contiguous
    counts = np.bincount(didx, minlength=len(days))
    pos = np.concatenate([np.arange(c) for c in counts])
    H = np.zeros((len(days), k, E.shape[1]), dtype=np.float32)
    M = np.zeros((len(days), k), dtype=bool)
    Pr = np.zeros((len(days), k), dtype=np.float32)
    H[didx, pos], M[didx, pos], Pr[didx, pos] = E, True, w

    # Row t reads the bucket of trading day t-1 (the same shift as every feature).
    bucket = pd.Series(calendar, index=calendar).shift(1).reindex(lab.f.index)
    bidx = days.get_indexer(pd.DatetimeIndex(bucket.fillna(pd.Timestamp("1900-01-01"))))
    safe = np.clip(bidx, 0, None)
    Hr, Mr, Pr_r = H[safe], M[safe].copy(), Pr[safe]
    Mr[bidx < 0] = False

    Xt = lab.f[lab.cols(["price", "market"])]
    mu, sd = Xt[lab.m["train"]].mean(), Xt[lab.m["train"]].std().replace(0, 1)
    Xt = ((Xt - mu) / sd).fillna(0.0).to_numpy(dtype=np.float32)
    y_idx = pd.Categorical(lab.y, categories=CLASSES).codes
    idx = {s: np.flatnonzero(lab.m[s]) for s in lab.m}

    pvs, pts, epochs, attn_test = [], [], [], []
    for seed in seeds:
        net, ep = fusion.train_attention_net(Hr, Mr, Pr_r, Xt, y_idx, idx["train"], idx["val"],
                                             cfg=cfg, seed=seed)
        pv, _ = fusion.predict_attention_net(net, Hr, Mr, Pr_r, Xt, idx["val"])
        net2, _ = fusion.train_attention_net(Hr, Mr, Pr_r, Xt, y_idx, idx["trainval"], None,
                                             cfg=cfg, seed=seed, n_epochs=ep)
        pt, at = fusion.predict_attention_net(net2, Hr, Mr, Pr_r, Xt, idx["test"])
        pvs.append(pv), pts.append(pt), epochs.append(ep), attn_test.append(at)
        lab.record("3_fusion", f"attention_net_{enc}_seed{seed}", "attn_net",
                   {"epochs": ep, "seed": seed}, pv, pt, None, {"blocks": f"price+market+headlines_{enc}"})
    lab.record("3_fusion", f"attention_net_{enc}", "attn_net", {"epochs": epochs, "seeds": list(seeds)},
               np.mean(pvs, 0), np.mean(pts, 0), None, {"blocks": f"price+market+headlines_{enc}"})
    log.info("[3_fusion] attention_net_%s epochs=%s", enc, epochs)

    # Which headline did the network attend to most on each test day? (for the report)
    att = np.mean(attn_test, 0)
    starts = np.r_[0, np.cumsum(counts)]
    text = sub["text_track_b"].to_numpy()
    top = []
    for i, r in enumerate(idx["test"]):
        b = bidx[r]
        if b < 0:
            continue
        j = int(np.argmax(att[i]))
        top.append({"date": lab.f.index[r], "news_bucket": days[b], "attention": float(att[i, j]),
                    "headline": text[starts[b] + j]})
    return pd.DataFrame(top)


def regression_suite(lab: Lab, cfg, fsets: dict) -> list[dict]:
    """Next-day return (secondary task) for the A / B / C feature sets."""
    out = []
    y = lab.f["y_return"]
    for name, blocks in fsets.items():
        X = lab.f[lab.cols(blocks)]
        for kind in ("ridge", "xgb"):
            best = (None, np.inf, None)
            for params in param_grid("reg", kind, cfg):
                pv = fit_predict("reg", kind, make_model("reg", kind, params, cfg),
                                 X[lab.m["train"]], y[lab.m["train"]], X[lab.m["val"]])
                rmse = regression_metrics(y[lab.m["val"]], pv)["rmse"]
                if rmse < best[1]:
                    best = (params, rmse, pv)
            pt = fit_predict("reg", kind, make_model("reg", kind, best[0], cfg),
                             X[lab.m["trainval"]], y[lab.m["trainval"]], X[lab.m["test"]])
            for split, p in (("val", best[2]), ("test", pt)):
                out.append({"experiment": name, "model": kind, "split": split,
                            "params": json.dumps(best[0]), **regression_metrics(y[lab.m[split]], p)})
            lab.probs[f"reg|{name}|{kind}"] = (best[2], pt)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Task 3 — embeddings, fusion, ablation")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--skip-nn", action="store_true", help="skip attention net and fine-tuning")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)
    _setup_logging(not a.quiet)
    cfg = load_config(a.config)
    t3 = cfg.dig("task3")
    rep = cfg.root / t3["paths"]["reports"]
    rep.mkdir(parents=True, exist_ok=True)
    seeds = list(t3.get("seeds", [42]))

    frame, groups, calendar = load_task2_frame(cfg)
    fit_before = frame.index[frame["split"] == "train"].max()
    val_end = frame.index[frame["split"] == "val"].max()

    log.info("STEP 1/5  Headline embeddings (fit_before %s)", fit_before.date())
    articles = nlp_features.load_kept_articles(cfg)
    n_total = len(articles)
    text_embeddings.build_all(cfg, articles, fit_before, force=a.no_cache,
                              which=tuple(e for e in t3["encoders"] if e != "finbert"))
    ensure_finbert(cfg)
    # The Task 3 corpus: the 100 most-syndicated headlines of each news day, for every encoder.
    rows = text_embeddings.corpus_rows(cfg, articles)
    text_embeddings.use_rows(rows)
    articles = articles.iloc[rows].reset_index(drop=True)
    log.info("Task 3 corpus: %d of %d headlines (top %d per news day)", len(articles), n_total,
             int(t3.get("corpus_top_k_per_day", 100)))

    log.info("STEP 2/5  Daily text blocks (pooling, PCA, FinBERT tone, topics, NER)")
    daily, tgroups, extras = build_text_daily(cfg, articles, fit_before, use_cache=not a.no_cache)
    daily.to_parquet(cfg.path("processed") / "task3_text_daily.parquet")
    frame = frame.join(lag_to_rows(daily, calendar, frame.index))
    groups = {**groups, **tgroups}

    # Leakage guard: row t must carry the news bucket of the previous trading day.
    probe = tgroups["finbert_sent"][0]
    prev = pd.Series(calendar, index=calendar).shift(1).reindex(frame.index)
    expected = daily[probe].reindex(pd.DatetimeIndex(prev.to_numpy())).to_numpy()
    assert np.allclose(frame[probe].to_numpy(), expected, equal_nan=True), "text features are not lagged"

    sdir = cfg.root / t3["paths"]["splits"]
    sdir.mkdir(parents=True, exist_ok=True)
    keep_cols = [c for g in groups.values() for c in g] + \
        [c for c in frame.columns if c.startswith("y_")] + ["persist_direction", "split"]
    keep_cols = list(dict.fromkeys(c for c in keep_cols if c in frame.columns))
    for s in ("train", "val", "test"):
        frame.loc[frame["split"] == s, keep_cols].to_csv(sdir / f"{s}.csv")
    (sdir / "feature_groups.json").write_text(json.dumps(groups, indent=2))
    extras["topic_labels"].to_csv(rep / "topic_labels.csv", index=False)
    pd.DataFrame(extras["pca_variance"]).to_csv(rep / "pca_variance.csv", index=False)

    lab = Lab(frame, groups, cfg)
    A, B = ["price", "market"], ["price", "market", "nlp"]

    log.info("STEP 3/5  Stage 1-2: representations and feature strategies (early fusion)")
    lab.early("0_task2", "A0: price only", ["price"])
    lab.early("0_task2", "A: time series (price+market)", A)
    lab.early("0_task2", "B: A + Task 2 NLP", B)
    for blk in tgroups:
        stage = "1_representation" if blk.endswith("_mean") else "2_strategy"
        lab.early(stage, f"A + {blk}", A + [blk])
        lab.early(stage, f"text only: {blk}", [blk], kinds=("logreg",))

    log.info("STEP 4/5  Stage 3: late fusion, fine-tuning, attention network")
    late_tables = []
    for blk in tgroups:
        _, tbl = lab.late("3_fusion", f"late: A(logreg) + {blk}", TIME_KEY, text_blocks=[blk])
        late_tables.append(tbl)
    attn_top = None
    if not a.skip_nn:
        import subprocess
        import sys

        import torch
        torch.set_num_threads(1)     # macOS: multi-threaded torch after XGBoost can deadlock (OpenMP clash)
        from .finetune_text import bucket_labels, finetune_and_score
        if a.no_cache or not (text_embeddings.cache_dir(cfg) / "finetune_daily_probs.parquet").exists():
            # Fine-tuning gets its own process, with full threads and no XGBoost loaded.
            subprocess.run([sys.executable, "-m", "src.finetune_text"], cwd=cfg.root, check=True)
        ft = finetune_and_score(cfg, articles, bucket_labels(calendar, frame["y_direction"]),
                                fit_before, val_end, use_cache=True)
        ftl = lag_to_rows(ft, calendar, frame.index)[["ft_p_down", "ft_p_flat", "ft_p_up"]]
        ftl = ftl.fillna(1 / 3).to_numpy()
        _, tbl = lab.late("3_fusion", "late: A(logreg) + fine-tuned MiniLM", TIME_KEY,
                          text_probs=(ftl[lab.m["val"]], ftl[lab.m["test"]]))
        late_tables.append(tbl)
        for enc in ("finbert", "minilm"):
            top = attention_net_experiment(lab, cfg, articles, calendar, enc, seeds)
            if enc == "finbert":
                attn_top = top
    pd.concat(late_tables).to_csv(rep / "late_fusion_weights.csv", index=False)
    if attn_top is not None:
        attn_top.to_csv(rep / "attention_top_headlines.csv", index=False)

    log.info("STEP 5/5  Model C: choose on validation, tune, ablation, seeds")
    res = pd.DataFrame(lab.records)
    cand = res.loc[(res["split"] == "val") & (res["stage"] != "0_task2")
                   & ~res["experiment"].str.startswith("text only")
                   & ~res["experiment"].str.contains("seed")]
    best = cand.sort_values("macro_f1", ascending=False).iloc[0]
    log.info("best Task 3 config on validation: %s / %s (val %.3f)", best["experiment"], best["model"],
             best["macro_f1"])

    # Tuning (3a): best early-fusion text block, wider grid incl. PCA size.
    early_cand = cand.loc[cand["model"].isin(["logreg", "xgb"])].sort_values("macro_f1", ascending=False)
    best_block = early_cand.iloc[0]["blocks"].split("+")[-1]
    tg = t3["tuning"]
    grid = {"logreg": [{"C": c} for c in tg["logreg_C"]],
            "xgb": [{"max_depth": d, "n_estimators": n, "learning_rate": lr, "min_child_weight": mcw}
                    for d, n, lr, mcw in itertools.product(tg["xgb_max_depth"], tg["xgb_n_estimators"],
                                                           tg["xgb_learning_rate"], tg["xgb_min_child_weight"])]}
    tuning = []
    probs = text_embeddings.load_finbert_probs(cfg)
    for kpc in (tg["pca_components"] if best_block.startswith("emb_") else [None]):
        name = best_block
        if kpc is not None:
            _, enc, how = best_block.split("_", 2)
            E = text_embeddings.load_embedding(cfg, enc)
            Z, _, _ = text_features.embedding_block(E, articles, cfg, enc, fit_before, how=how, k=kpc, probs=probs)
            name = f"{best_block}_k{kpc}"
            Z = Z.add_suffix(f"_k{kpc}")
            lab.g[name] = list(Z.columns)
            lab.f = lab.f.join(lag_to_rows(Z, calendar, lab.f.index))
        r = lab.early("4_tuning", f"tuned: A + {name}", A + [name], grid=grid)
        tuning += [{"block": name, "k": kpc, "model": kk, "params": json.dumps(v[0]), "val_macro_f1": v[1]}
                   for kk, v in r.items()]
    tuning = pd.DataFrame(tuning).sort_values("val_macro_f1", ascending=False)
    tuning.to_csv(rep / "tuning.csv", index=False)
    # Re-write the Task 3 splits so they also hold the re-tuned PCA blocks.
    keep_cols = list(dict.fromkeys(keep_cols + [c for g in lab.g.values() for c in g if c in lab.f.columns]))
    for s in ("train", "val", "test"):
        lab.f.loc[lab.f["split"] == s, keep_cols].to_csv(sdir / f"{s}.csv")
    (sdir / "feature_groups.json").write_text(json.dumps(lab.g, indent=2))

    # Final ablation: A0, A, B, C, B+C with both model families. C uses, per model
    # family, the block (incl. PCA size) and settings that won tuning on validation.
    seed_runs = []
    for kind in ("logreg", "xgb"):
        tk = tuning.loc[tuning["model"] == kind].iloc[0]
        c_block, c_params = tk["block"], json.loads(tk["params"])
        final_sets = {"A0": ["price"], "A": A, "B": B, "C": A + [c_block], "B+C": B + [c_block]}
        for nm, blocks in final_sets.items():
            g = {kind: [c_params]} if nm in ("C", "B+C") else None
            for seed in (seeds if kind == "xgb" else [seeds[0]]):
                lab.early("5_ablation", f"{nm}|seed{seed}", blocks, kinds=(kind,), grid=g, seed=seed)
        seed_runs.append({"model": kind, "C_block": c_block, "C_params": c_params,
                          "val_macro_f1": float(tk["val_macro_f1"])})
    res = pd.DataFrame(lab.records)
    res.to_csv(rep / "results.csv", index=False)
    pd.DataFrame(lab.grid_rows).to_csv(rep / "grid_search.csv", index=False)

    # Block bootstrap on test: B vs A, C vs A, C vs B (seed-42 models).
    yt = lab.y[lab.m["test"]]
    gains = []

    def gain_row(model, base, new, kb, kn):
        pb, pn = (fusion.proba_to_label(lab.probs[k][1]) for k in (kb, kn))
        fb, fn = fusion.macro_f1(yt, pb), fusion.macro_f1(yt, pn)
        return {"model": model, "base": base, "new": new, "f1_base": fb, "f1_new": fn,
                "gain": fn - fb, **block_bootstrap_gain("cls", yt, pb, pn, seed=cfg.seed)}

    for kind in ("logreg", "xgb"):
        key = lambda nm: f"{nm}|seed{seeds[0]}|{kind}"
        for base, new in (("A", "B"), ("A", "C"), ("B", "C"), ("A", "B+C")):
            gains.append(gain_row(kind, base, new, key(base), key(new)))
    best_key = f"{best['experiment']}|{best['model']}"
    gains.append(gain_row(best["model"], "A", f"best-on-val: {best['experiment']}",
                          f"A|seed{seeds[0]}|logreg", best_key))
    pd.DataFrame(gains).to_csv(rep / "ablation_gain.csv", index=False)

    c_lr = seed_runs[0]["C_block"]
    reg = regression_suite(lab, cfg, {"A": A, "B": B, "C": A + [c_lr]})
    pd.DataFrame(reg).to_csv(rep / "regression.csv", index=False)

    # Probabilities for the analysis notebook (validation + test days).
    idx = lab.f.index[lab.m["val"] | lab.m["test"]]
    pf = []
    for k, (pv, pt) in lab.probs.items():
        if k.startswith("reg|"):
            pf.append(pd.DataFrame({k: np.r_[pv, pt]}, index=idx))
        elif "|seed" in k or k == best_key or k.startswith(("A + ", "late: ", "attention_net")):
            pf.append(pd.DataFrame(np.vstack([pv, pt]), index=idx, columns=[f"{k}|{c}" for c in CLASSES]))
    pred = pd.concat(pf, axis=1)
    pred.insert(0, "split", lab.f["split"].reindex(pred.index))
    pred.insert(1, "y_direction", lab.y.reindex(pred.index))
    pred.insert(2, "y_return", lab.f["y_return"].reindex(pred.index))
    pred.to_csv(rep / "predictions.csv")

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": {s: int(lab.m[s].sum()) for s in ("train", "val", "test")},
        "n_headlines": int(len(articles)), "n_headlines_total": int(n_total),
        "fit_before": str(fit_before.date()),
        "encoders": {k: text_embeddings.MODELS.get(k, "PPMI-SVD trained on our headlines")
                     for k in t3["encoders"]},
        "feature_groups": {g: len(c) for g, c in groups.items()},
        "best_on_validation": {"experiment": best["experiment"], "model": best["model"],
                               "val_macro_f1": float(best["macro_f1"])},
        "model_C": seed_runs,
        "ner_coverage": extras["ner_coverage"], "seeds": seeds,
    }
    (rep / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    abl = res.loc[(res["stage"] == "5_ablation") & (res["split"] == "test")]
    summ = abl.groupby([abl["experiment"].str.split("|").str[0], "model"])["macro_f1"].agg(["mean", "std"])
    log.info("\nABLATION (test macro-F1)\n%s", summ.to_string(float_format="%.3f"))
    log.info("Task 3 pipeline complete -> %s", rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
