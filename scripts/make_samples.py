#!/usr/bin/env python3
"""Write GitHub-sized copies of the pipeline outputs to data/samples/.

    py scripts/make_samples.py            # 5,000-article sample
    py scripts/make_samples.py --n 3000

Why this exists
---------------
articles_flagged.parquet (~390 MB) is over GitHub's 100 MB limit, so the repo
ships a stratified sample of it instead. The aligned dataset is small, so it
ships whole — as CSV, so a reader can open it in Excel without pyarrow.

Run it only after `py -m src.build_dataset` has finished: the script refuses
to sample a half-written or stale build.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402

# Same 70/30 split as the audit sample: mostly what we model, but enough
# dropped rows that a reader can see what each filter removed.
KEPT_SHARE = 0.7


def _check_build_finished(raw: Path, flagged: Path, manifest: Path) -> dict:
    """One build writes raw -> flagged -> manifest, in that order. If the
    timestamps are out of that order, a build is still running (or died)."""
    for p in (raw, flagged, manifest):
        if not p.exists():
            raise SystemExit(f"{p} not found — run `py -m src.build_dataset` first.")
    if not (raw.stat().st_mtime <= flagged.stat().st_mtime <= manifest.stat().st_mtime):
        raise SystemExit("A build is still running (or stopped midway). "
                         "Wait for 'Task 1 pipeline complete.' and rerun.")
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    if meta.get("offline_synthetic"):
        raise SystemExit("data/processed/ holds SYNTHETIC data (--offline run). "
                         "Rebuild with `py -m src.build_dataset` first.")
    return meta


def _stratified(pool: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Equal draws per (year, theme_family), so quiet years and small
    families are visible instead of being swamped by the busiest ones."""
    if pool.empty or n <= 0:
        return pool.head(0)
    strata = pool.groupby(["year", "theme_family"], observed=True)
    per = max(1, n // strata.ngroups)
    picks = pd.concat([g.sample(min(per, len(g)), random_state=seed) for _, g in strata])
    return picks.sample(min(n, len(picks)), random_state=seed)


def main() -> int:
    p = argparse.ArgumentParser(description="GitHub-sized samples of the outputs")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--n", type=int, default=5000, help="articles in the sample")
    args = p.parse_args()

    cfg = load_config(args.config)
    flagged = cfg.path("interim") / "articles_flagged.parquet"
    proc = cfg.path("processed")
    meta = _check_build_finished(cfg.path("raw") / "articles_raw.parquet",
                                 flagged, proc / "manifest.json")

    out_dir = Path(cfg.path("raw")).parent / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    arts = pd.read_parquet(flagged)
    arts = arts.assign(year=pd.to_datetime(arts["seendate_utc"], utc=True).dt.year)
    n_kept = round(args.n * KEPT_SHARE)
    sample = pd.concat([
        _stratified(arts[arts["keep"]], n_kept, cfg.seed),
        _stratified(arts[~arts["keep"]], args.n - n_kept, cfg.seed),
    ]).sort_values("seendate_utc").drop(columns="year")
    art_path = out_dir / "articles_flagged_sample.csv"
    sample.to_csv(art_path, index=False, encoding="utf-8")

    ds_path = out_dir / "dataset_aligned.csv"
    pd.read_parquet(proc / "dataset.parquet").to_csv(ds_path, index=False)

    print(f"build: {meta['generated_utc']}  source={meta['news_source']}")
    for path, rows in ((art_path, len(sample)), (ds_path, meta["n_dataset_rows"])):
        print(f"  {path}  {rows:,} rows  {path.stat().st_size / 1e6:.1f} MB")
    print(f"  sample: {int(sample['keep'].sum()):,} kept / "
          f"{int((~sample['keep']).sum()):,} dropped of {len(arts):,} articles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
