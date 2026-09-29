"""Task 2 pipeline driver: NLP features -> splits -> baselines -> combined models.

    py -m src.run_task2               # full run (first run scores ~0.9M headlines, a few minutes)
    py -m src.run_task2 --no-cache    # rescore every headline

Inputs (from Task 1)
    data/processed/dataset.parquet     lagged Task 1 features + targets + split
    data/processed/targets.parquet     JISDOR returns (for the price features)
    data/interim/articles_flagged.parquet  kept headlines, both text tracks

Outputs
    data/processed/nlp_daily_features.parquet  NLP features per news day (unlagged)
    data/splits/{train,val,test}.csv           the modelling table, split by date
    data/splits/feature_groups.json            which columns belong to which group
    reports/task2/*.csv, manifest.json         every result table used in the report

Protocol (the order matters, and it is the leakage argument):
    1. Anything fitted on text (TF-IDF, SVD) sees training-period news only.
    2. Hyperparameters are chosen on VALIDATION, with models fitted on TRAIN.
    3. The chosen setting is refitted on TRAIN+VALIDATION and scored on TEST
       exactly once. No decision is ever taken by looking at a test score.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import align, nlp_features
from .build_dataset import _setup_logging
from .config import load_config
from .evaluate import (block_bootstrap_gain, classification_metrics, confusion_frame,
                       regression_metrics, returns_to_direction)
from .models import (LINEAR, arima_one_step, feature_importance, fit_predict, make_model,
                     majority_class, param_grid, select_arima_order)
from .price_features import build_price_features

log = logging.getLogger("run_task2")

TASKS = {"cls": "y_direction", "reg": "y_return"}


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def load_task1(cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    proc = cfg.path("processed")
    dataset = pd.read_parquet(proc / "dataset.parquet").sort_index()
    targets = pd.read_parquet(proc / "targets.parquet").sort_index()
    return dataset, targets


def assemble_frame(dataset: pd.DataFrame, targets: pd.DataFrame, nlp_daily: pd.DataFrame, cfg):
    """Join price and NLP features onto Task 1's modelling table.

    The NLP frame is indexed by news bucket day. It gets the same one-trading-
    day shift Task 1 applied to its own news features (``feature_lag_days``),
    so row t holds news from bucket t-1 next to the return of fix t.
    """
    lag = int(dataset["feature_lag_days"].iloc[0])
    px = build_price_features(targets, cfg).reindex(dataset.index)
    nlp = nlp_daily.reindex(dataset.index).shift(lag)

    frame = dataset.join(px).join(nlp)
    # Yesterday's label: known once fix t-1 is published, so a legal input.
    frame["persist_direction"] = targets["y_direction"].reindex(dataset.index).shift(1)
    frame = frame.loc[frame["y_return"].notna()]

    market = [c for c in cfg.dig("task2", "market_controls", default=[]) if c in frame.columns]
    task1 = [c for c in dataset.columns
             if not c.startswith(("y_", "lag")) and c not in ("split", "feature_lag_days")]
    groups = {"price": list(px.columns), "market": market,
              "task1_news": task1, "nlp": list(nlp.columns)}
    log.info("frame: %d rows; groups: %s", len(frame),
             {g: len(c) for g, c in groups.items()})
    return frame, groups


def write_splits(frame: pd.DataFrame, groups: dict, cfg) -> None:
    out_dir = cfg.root / cfg.dig("task2", "paths", "splits")
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_cols = [c for g in groups.values() for c in g]
    label_cols = [c for c in frame.columns if c.startswith("y_")]
    cols = feat_cols + ["persist_direction"] + label_cols + ["feature_lag_days", "split"]
    for name in ("train", "val", "test"):
        part = frame.loc[frame["split"] == name, cols]
        part.to_csv(out_dir / f"{name}.csv")
        log.info("wrote %-5s split: %d rows %s..%s", name, len(part),
                 part.index.min().date(), part.index.max().date())
    (out_dir / "feature_groups.json").write_text(json.dumps(groups, indent=2), encoding="utf-8")


def _masks(frame: pd.DataFrame) -> dict:
    s = frame["split"]
    return {"train": (s == "train").to_numpy(), "val": (s == "val").to_numpy(),
            "test": (s == "test").to_numpy(), "trainval": s.isin(["train", "val"]).to_numpy()}


def _record(task, model, feature_set, split, y_true, y_pred, params=None) -> dict:
    metrics = classification_metrics(y_true, y_pred) if task == "cls" else regression_metrics(y_true, y_pred)
    return {"task": task, "model": model, "feature_set": feature_set, "split": split,
            "params": json.dumps(params or {}), **metrics}


# ---------------------------------------------------------------------------
# Baselines (price history only)
# ---------------------------------------------------------------------------
def run_baselines(frame: pd.DataFrame, cfg):
    m = _masks(frame)
    band = float(cfg.dig("target", "flat_band_bp", default=10)) / 100.0
    y_cls, y_reg = frame["y_direction"], frame["y_return"]
    records, preds = [], {}

    order = select_arima_order(y_reg[m["train"]].to_numpy(), cfg)
    log.info("ARIMA order chosen by AIC on train: %s", order)

    for split, fit in (("val", "train"), ("test", "trainval")):
        ev, ft = m[split], m[fit]
        arima = arima_one_step(y_reg[ft].to_numpy(), y_reg[ev].to_numpy(), order)
        cls_preds = {
            "majority": majority_class(y_cls[ft], int(ev.sum())),
            "persistence": frame.loc[ev, "persist_direction"].fillna("FLAT").to_numpy(dtype=object),
            "arima": returns_to_direction(arima, band),
        }
        reg_preds = {
            "zero": np.zeros(int(ev.sum())),
            "persistence": frame.loc[ev, "px_ret_lag1"].fillna(0.0).to_numpy(),
            "train_mean": np.full(int(ev.sum()), y_reg[ft].mean()),
            "arima": arima,
        }
        for task, table, y in (("cls", cls_preds, y_cls), ("reg", reg_preds, y_reg)):
            for name, p in table.items():
                params = {"order": list(order)} if name == "arima" else None
                records.append(_record(task, name, "price", split, y[ev], p, params))
                key = f"{task}|{name}|price"
                preds.setdefault(key, []).append(pd.Series(p, index=frame.index[ev]))

    preds = {k: pd.concat(v) for k, v in preds.items()}
    return records, preds, order


# ---------------------------------------------------------------------------
# Learned models (baseline = price-only feature set; combined = + NLP)
# ---------------------------------------------------------------------------
def run_learned(frame: pd.DataFrame, groups: dict, cfg):
    m = _masks(frame)
    fsets = cfg.dig("task2", "feature_sets", default={"price": ["price"]})
    records, grid_rows, preds, importances = [], [], {}, []

    for task, ycol in TASKS.items():
        y = frame[ycol]
        for kind in (LINEAR[task], "xgb"):
            for fs_name, group_names in fsets.items():
                cols = [c for g in group_names for c in groups[g]]
                X = frame[cols]

                best = (None, -np.inf, None)          # params, score, val predictions
                for params in param_grid(task, kind, cfg):
                    model = make_model(task, kind, params, cfg)
                    pv = fit_predict(task, kind, model, X[m["train"]], y[m["train"]], X[m["val"]])
                    met = (classification_metrics(y[m["val"]], pv) if task == "cls"
                           else regression_metrics(y[m["val"]], pv))
                    score = met["macro_f1"] if task == "cls" else -met["rmse"]
                    grid_rows.append({"task": task, "model": kind, "feature_set": fs_name,
                                      "params": json.dumps(params), "val_score": score})
                    if score > best[1]:
                        best = (params, score, pv)
                params, _, pv = best

                final = make_model(task, kind, params, cfg)
                pt = fit_predict(task, kind, final, X[m["trainval"]], y[m["trainval"]], X[m["test"]])

                records.append(_record(task, kind, fs_name, "val", y[m["val"]], pv, params))
                records.append(_record(task, kind, fs_name, "test", y[m["test"]], pt, params))
                key = f"{task}|{kind}|{fs_name}"
                preds[key] = pd.Series(np.concatenate([pv, pt]),
                                       index=frame.index[m["val"] | m["test"]])
                imp = feature_importance(final, cols).sort_values(ascending=False)
                importances.append(pd.DataFrame({"model_key": key, "feature": imp.index,
                                                 "importance": imp.to_numpy(),
                                                 "rank": np.arange(1, len(imp) + 1)}))
                log.info("%-4s %-7s %-18s val=%.3f  params=%s", task, kind, fs_name,
                         records[-2]["macro_f1" if task == "cls" else "rmse"], params)

    return records, pd.DataFrame(grid_rows), preds, pd.concat(importances, ignore_index=True)


def news_gain_table(results: pd.DataFrame, preds: dict, frame: pd.DataFrame, cfg) -> pd.DataFrame:
    """Does adding NLP features help the SAME model? Positive = news helped.

    Compared pairs: price -> price+nlp (the brief's baseline vs combined) and
    price+market -> price+market+nlp (does news add anything beyond markets?).
    Metric: macro-F1 for classification, RMSE reduction for regression. Each
    gain carries a 95% block-bootstrap interval: with ~240 days, a gain whose
    interval spans zero is indistinguishable from luck.
    """
    pairs = [("price", "price+nlp"), ("price+market", "price+market+nlp")]
    rows = []
    learned = results.loc[results["model"].isin(["logreg", "ridge", "xgb"])]
    for (task, model, split), grp in learned.groupby(["task", "model", "split"]):
        by_fs = grp.set_index("feature_set")
        metric = "macro_f1" if task == "cls" else "rmse"
        for base, combo in pairs:
            if base not in by_fs.index or combo not in by_fs.index:
                continue
            b, c = by_fs.at[base, metric], by_fs.at[combo, metric]
            days = frame.index[frame["split"] == split]
            ci = block_bootstrap_gain(task, frame.loc[days, TASKS[task]],
                                      preds[f"{task}|{model}|{base}"].reindex(days),
                                      preds[f"{task}|{model}|{combo}"].reindex(days), seed=cfg.seed)
            rows.append({"task": task, "model": model, "split": split, "metric": metric,
                         "without_news": base, "with_news": combo,
                         "score_without": b, "score_with": c,
                         "gain": (c - b) if task == "cls" else (b - c), **ci})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Task 2 — NLP features, baselines, evaluation")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--no-cache", action="store_true", help="rescore every headline")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)

    _setup_logging(not a.quiet)
    cfg = load_config(a.config)
    rep = cfg.root / cfg.dig("task2", "paths", "reports")
    rep.mkdir(parents=True, exist_ok=True)

    dataset, targets = load_task1(cfg)
    fit_before = dataset.index[dataset["split"] == "train"].max()

    log.info("STEP 1/4  NLP features (TF-IDF/SVD fitted on news before %s)", fit_before.date())
    nlp_daily, extras = nlp_features.build_nlp_features(cfg, dataset.index, fit_before,
                                                        use_cache=not a.no_cache)
    nlp_daily.to_parquet(cfg.path("processed") / "nlp_daily_features.parquet")

    log.info("STEP 2/4  Assemble, leakage checks, write splits")
    frame, groups = assemble_frame(dataset, targets, nlp_daily, cfg)
    leaks = align.leakage_assertions(frame, cfg)
    if leaks:
        raise RuntimeError(f"leakage checks failed: {leaks}")
    write_splits(frame, groups, cfg)

    log.info("STEP 3/4  Price-only baselines")
    base_records, base_preds, arima_order = run_baselines(frame, cfg)

    log.info("STEP 4/4  Learned models: price-only vs price+NLP (and ablations)")
    records, grid, learned_preds, importances = run_learned(frame, groups, cfg)

    results = pd.DataFrame(base_records + records)
    results.to_csv(rep / "results.csv", index=False)
    all_preds = {**base_preds, **learned_preds}
    news_gain_table(results, all_preds, frame, cfg).to_csv(rep / "news_gain.csv", index=False)
    grid.to_csv(rep / "grid_search.csv", index=False)
    importances.to_csv(rep / "feature_importance.csv", index=False)
    extras["lexicon_coverage"].to_csv(rep / "lexicon_coverage.csv", index=False)
    extras["tfidf_svd_top_terms"].to_csv(rep / "tfidf_svd_top_terms.csv", index=False)

    pred_frame = pd.DataFrame(all_preds)
    pred_frame.insert(0, "split", frame["split"].reindex(pred_frame.index))
    pred_frame.insert(1, "y_direction", frame["y_direction"].reindex(pred_frame.index))
    pred_frame.insert(2, "y_return", frame["y_return"].reindex(pred_frame.index))
    pred_frame.to_csv(rep / "predictions.csv")

    test = pred_frame["split"] == "test"
    cms = [confusion_frame(pred_frame.loc[test, "y_direction"], pred_frame.loc[test, k])
           .stack().rename("count").reset_index().assign(model_key=k)
           for k in all_preds if k.startswith("cls|")]
    pd.concat(cms, ignore_index=True).rename(columns={"level_0": "true", "level_1": "pred"}) \
        .to_csv(rep / "confusion_test.csv", index=False)

    balance = (frame.groupby("split")["y_direction"].value_counts(normalize=True)
               .unstack().reindex(["train", "val", "test"]))
    balance.to_csv(rep / "class_balance.csv")

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": {s: int((frame["split"] == s).sum()) for s in ("train", "val", "test")},
        "feature_groups": {g: len(c) for g, c in groups.items()},
        "tfidf": extras["tfidf_fit"],
        "arima_order": list(arima_order),
        "leakage_problems": leaks,
        "seed": cfg.seed,
    }
    (rep / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    cls_test = results.query("task == 'cls' and split == 'test'")[
        ["model", "feature_set", "macro_f1", "accuracy", "directional_hit_rate", "coverage"]]
    log.info("\nTEST (classification)\n%s", cls_test.to_string(index=False, float_format="%.3f"))
    log.info("Task 2 pipeline complete -> %s", rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
