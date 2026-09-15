"""End-to-end Task 1 pipeline driver.

    python -m src.build_dataset --config config/config.yaml
    python -m src.build_dataset --stride 7 --families conflict,sanctions_trade  # dev run
    python -m src.build_dataset --offline                                       # synthetic smoke test

Outputs (under ``data/``)
-------------------------
    raw/fx_panel.parquet            raw FX / control levels
    raw/articles_raw.parquet        everything GDELT returned
    interim/articles_flagged.parquet articles + every filter/dedup/text column
    processed/daily_features.parquet news features on the FX calendar
    processed/targets.parquet        FX targets
    processed/dataset.parquet        THE modelling table (features + targets + split)
    reports/*.csv                    every QA and funnel table from the report
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import align, dedup, features, filtering, fx_data, gdelt, gkg, preprocess, qa
from .config import load_config
from .http_cache import CachedSession, single_fetch_lock

log = logging.getLogger("build_dataset")


def _setup_logging(verbose: bool = True) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)-7s | %(name)-14s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def _save(df: pd.DataFrame, path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path)
    except Exception as exc:  # pyarrow missing, or an object column it dislikes
        path = path.with_suffix(".csv")
        df.to_csv(path)
        log.warning("parquet unavailable (%s); wrote CSV instead", type(exc).__name__)
    log.info("wrote %-22s %6d rows x %3d cols -> %s", label, len(df), df.shape[1], path)


def run(cfg, *, stride: int = 1, families=None, sub_day_chunks: int = 1,
        offline: bool = False, skip_timelines: bool = False,
        source: str = "doc") -> dict:
    """Execute the full acquisition + preprocessing pipeline."""
    raw_dir, interim_dir = cfg.path("raw"), cfg.path("interim")
    proc_dir, rep_dir = cfg.path("processed"), cfg.path("reports")
    out: dict = {}

    # ---- 1. TARGET -------------------------------------------------------
    log.info("=" * 78)
    log.info("STEP 1/7  Target acquisition (FX + controls)")
    log.info("=" * 78)
    if offline:
        from .synthetic import synthetic_fx_panel
        fx_panel = synthetic_fx_panel(cfg)
    else:
        fx_panel = fx_data.build_fx_panel(cfg)
    _save(fx_panel, raw_dir / "fx_panel.parquet", "fx_panel")

    targets = fx_data.make_targets(fx_panel, cfg)
    trading_days = align.trading_day_calendar(fx_panel, cfg)
    log.info("trading calendar: %d sessions %s..%s",
             len(trading_days), trading_days.min().date(), trading_days.max().date())

    # ---- 2. NEWS ---------------------------------------------------------
    log.info("=" * 78)
    log.info("STEP 2/7  News acquisition (GDELT)")
    log.info("=" * 78)
    if offline:
        from .synthetic import synthetic_articles, synthetic_timelines
        articles = synthetic_articles(cfg, trading_days)
        tl_wide = synthetic_timelines(cfg, trading_days)
    else:
        # One fetching run at a time: GDELT throttles per IP, and two runs
        # would also race on the shared cache.
        with single_fetch_lock(Path(cfg.path("cache"))):
            tl_wide = pd.DataFrame()
            if source == "gkg":
                articles, tl_long = gkg.fetch_articles_gkg(
                    cfg, families=families, stride_days=stride,
                )
                if not skip_timelines:
                    tl_wide = gdelt.timelines_to_wide(tl_long)
            else:
                sess = CachedSession(
                    Path(cfg.path("cache")),
                    sleep=cfg.dig("gdelt", "sleep_seconds", default=6.0),
                    max_retries=cfg.dig("gdelt", "max_retries", default=6),
                    backoff=cfg.dig("gdelt", "backoff_factor", default=3.0),
                    timeout=cfg.dig("gdelt", "timeout_seconds", default=60),
                )
                articles = gdelt.fetch_articles(
                    cfg, session=sess, families=families,
                    stride_days=stride, sub_day_chunks=sub_day_chunks,
                )
                if not skip_timelines:
                    tl_wide = gdelt.timelines_to_wide(gdelt.fetch_timelines(cfg, session=sess))
                log.info("GDELT fetch finished — %s", sess.report())
    _save(articles, raw_dir / "articles_raw.parquet", "articles_raw")

    if articles.empty:
        raise RuntimeError(
            "No articles retrieved. Check network access to api.gdeltproject.org, "
            "or run with --offline for a synthetic smoke test."
        )

    # ---- 3. FILTER + DEDUP ----------------------------------------------
    log.info("=" * 78)
    log.info("STEP 3/7  Filtering funnel and deduplication")
    log.info("=" * 78)
    articles = dedup.deduplicate(articles, cfg)
    articles = filtering.run_funnel(articles, cfg)

    funnel = filtering.funnel_report(articles)
    fam_bal = filtering.family_balance(articles)
    funnel.to_csv(rep_dir / "funnel_attrition.csv", index=False)
    fam_bal.to_csv(rep_dir / "family_balance.csv")
    log.info("\n%s", funnel.to_string(index=False))

    # ---- 4. ALIGN --------------------------------------------------------
    log.info("=" * 78)
    log.info("STEP 4/7  Temporal alignment to the FX calendar")
    log.info("=" * 78)
    articles = align.attach_news_day(articles, trading_days, cfg)

    # ---- 5. PREPROCESS ---------------------------------------------------
    log.info("=" * 78)
    log.info("STEP 5/7  Text preprocessing (dual track)")
    log.info("=" * 78)
    articles = preprocess.preprocess_frame(articles, cfg)
    _save(articles, interim_dir / "articles_flagged.parquet", "articles_flagged")

    # ---- 6. FEATURES -----------------------------------------------------
    log.info("=" * 78)
    log.info("STEP 6/7  Daily feature aggregation")
    log.info("=" * 78)
    feats = features.build_daily_features(articles, cfg, timeline_wide=tl_wide)
    feats = features.reindex_to_trading_days(feats, trading_days)
    _save(feats, proc_dir / "daily_features.parquet", "daily_features")
    _save(targets, proc_dir / "targets.parquet", "targets")

    dataset = align.make_prediction_frame(feats, targets, cfg)
    dataset = align.chronological_split(dataset, cfg)
    _save(dataset, proc_dir / "dataset.parquet", "dataset")

    # ---- 7. QA -----------------------------------------------------------
    log.info("=" * 78)
    log.info("STEP 7/7  Quality assurance")
    log.info("=" * 78)
    leaks = align.leakage_assertions(dataset, cfg)

    reports = {
        "funnel_attrition": funnel,
        "family_balance": fam_bal.reset_index(),
        "coverage_by_year": qa.coverage_report(feats, trading_days),
        "coverage_gaps": qa.find_gaps(feats),
        "cap_saturation": qa.cap_saturation_report(articles),
        "structural_break": qa.structural_break_report(feats),
        "fx_sanity": qa.fx_sanity_report(targets),
        "corpus_quality": qa.corpus_quality_report(articles),
        "target_stats": fx_data.describe_targets(targets, cfg),
        "feature_dictionary": features.feature_dictionary(feats),
        "missingness": qa.missingness_table(dataset),
        "validity_summary": qa.validity_summary(articles, feats, targets, leaks),
        "top_syndicated": dedup.dedup_report(articles),
        "preprocessing_examples": preprocess.preprocessing_examples(articles),
        "negation_demo": preprocess.demo_negation_failure(),
    }
    for name, tbl in reports.items():
        if isinstance(tbl, pd.DataFrame) and not tbl.empty:
            tbl.to_csv(rep_dir / f"{name}.csv", index=False)

    log.info("\nVALIDITY SUMMARY\n%s", reports["validity_summary"].to_string(index=False))

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "offline_synthetic": bool(offline),
        "period": {"start": str(cfg.dig("period", "start")), "end": str(cfg.dig("period", "end"))},
        "stride_days": stride,
        "sub_day_chunks": sub_day_chunks,
        "news_source": "synthetic" if offline else source,
        "families": families or list((cfg.dig("theme_families", default={}) or {}).keys()),
        "n_articles_raw": int(len(articles)),
        "n_articles_kept": int(articles["keep"].sum()),
        "n_trading_days": int(len(trading_days)),
        "n_dataset_rows": int(len(dataset)),
        "n_features": int(feats.shape[1]),
        "leakage_problems": leaks,
        "seed": cfg.seed,
    }
    (proc_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("manifest -> %s", proc_dir / "manifest.json")

    out.update(dict(fx_panel=fx_panel, targets=targets, articles=articles,
                    features=feats, dataset=dataset, reports=reports,
                    manifest=manifest))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Task 1 — data acquisition & preprocessing")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--stride", type=int, default=1,
                   help="sample every N-th day (1 = full census; 7 for a fast dev run)")
    p.add_argument("--families", default=None,
                   help="comma-separated theme families to restrict to")
    p.add_argument("--sub-day-chunks", type=int, default=1,
                   help="split each day into N windows to relax the 250-record cap")
    p.add_argument("--offline", action="store_true",
                   help="run on synthetic fixtures; no network access required")
    p.add_argument("--skip-timelines", action="store_true")
    p.add_argument("--source", choices=["gkg", "doc"], default=None,
                   help="news source: gkg = bulk GKG files (no API quota), "
                        "doc = DOC 2.0 API. Default: news_source in the config")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)

    _setup_logging(not a.quiet)
    cfg = load_config(a.config)
    fams = [f.strip() for f in a.families.split(",")] if a.families else None
    source = a.source or cfg.dig("news_source", default="doc")
    run(cfg, stride=a.stride, families=fams, sub_day_chunks=a.sub_day_chunks,
        offline=a.offline, skip_timelines=a.skip_timelines, source=source)
    log.info("Task 1 pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
