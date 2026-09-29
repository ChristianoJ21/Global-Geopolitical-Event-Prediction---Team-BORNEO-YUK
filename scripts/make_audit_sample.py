#!/usr/bin/env python3
"""Stage 4 — generate the stratified human-audit sample, and score it.

    # 1. produce the blank labelling sheet (two copies, one per annotator)
    python scripts/make_audit_sample.py build

    # 2. two team members fill in the `label` column independently:
    #    1 = geopolitically material to the USD, 0 = not, ? = unsure
    # 3. score agreement and filter precision
    python scripts/make_audit_sample.py score \
        --a reports/audit_annotator_A.csv --b reports/audit_annotator_B.csv

Why this exists
---------------
Stages 0-3 are *our opinion* about what a dollar-relevant geopolitical article
is, expressed as code. The audit is where that opinion meets evidence. Two
annotators labelling the same stratified sample gives us two numbers we can put
in the report and defend:

    precision  — of the articles the filter KEPT, what share are genuinely
                 material? (Is the corpus clean?)
    recall     — of the articles the filter DISCARDED, what share should we
                 have kept? (Is the corpus complete?)
    Cohen's k  — do two humans even agree on what "material" means? If they do
                 not, the label is ill-defined and no precision number from it
                 means anything. This check comes first.

Stratification is by (year, theme_family). A simple random sample would be
dominated by the largest family in the busiest years, and would tell us nothing
about whether the filter behaves consistently across the decade — which is
exactly the failure mode we are worried about.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402

GUIDELINE = """\
LABELLING GUIDELINES — read before starting, do not skip.

Question for every row:
  "Could a well-informed FX trader reading ONLY this headline reasonably
   update their view on the US dollar over the next 1-5 trading days?"

Label 1 (MATERIAL) if the headline involves any of:
  - armed conflict, military escalation or de-escalation between states
  - sanctions, tariffs, export controls, trade agreements or trade disputes
  - central bank policy, interest rates, inflation, currency or reserves
  - energy supply disruption, OPEC decisions, major commodity shocks
  - political instability in a G20 / major-trade economy (coup, snap election,
    government collapse, debt crisis, capital controls)
  - explicit currency-market commentary

Label 0 (NOT MATERIAL) if it is:
  - sport, entertainment, celebrity, lifestyle, weather (unless it is a supply
    shock of national scale)
  - local crime or local politics with no national economic consequence
  - corporate news with no macro or geopolitical dimension
  - opinion pieces with no new fact
  - a listicle, a horoscope, a product review, or obvious SEO filler

Label ? if you genuinely cannot tell from the headline alone.

IMPORTANT: judge ONLY the headline text. Do not open the URL, and do not use
hindsight about what the dollar actually did. Both would contaminate the audit.
Do not discuss rows with the other annotator until both sheets are finished.
"""


def build(cfg, args) -> None:
    interim = cfg.path("interim")
    src = interim / "articles_flagged.parquet"
    if not src.exists():
        src = src.with_suffix(".csv")
    if not src.exists():
        raise SystemExit(
            f"{src} not found — run `python -m src.build_dataset` first."
        )
    df = pd.read_parquet(src) if src.suffix == ".parquet" else pd.read_csv(src)

    a = cfg.dig("audit", default={}) or {}
    n_total = int(args.n or a.get("sample_size", 300))
    rng = np.random.default_rng(cfg.seed)

    df = df.copy()
    df["year"] = pd.to_datetime(df["seendate_utc"], errors="coerce", utc=True).dt.year
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    # We sample from BOTH sides of the filter. Auditing only what we kept
    # measures precision and is blind to recall — and a filter that throws away
    # half the real signal is exactly as broken as one that keeps garbage.
    kept = df[df["keep"]]
    dropped = df[~df["keep"]]
    n_kept = int(round(n_total * 0.7))          # weighted toward precision:
    n_drop = n_total - n_kept                   # the kept corpus is what we model

    def stratified(pool: pd.DataFrame, n: int) -> pd.DataFrame:
        if pool.empty or n <= 0:
            return pool.head(0)
        strata = pool.groupby(["year", "theme_family"], observed=True)
        per = max(1, n // max(len(strata), 1))
        picks = [
            g.sample(min(per, len(g)), random_state=int(rng.integers(0, 10**6)))
            for _, g in strata
        ]
        out = pd.concat(picks) if picks else pool.head(0)
        if len(out) > n:
            out = out.sample(n, random_state=cfg.seed)
        return out

    sample = pd.concat([stratified(kept, n_kept), stratified(dropped, n_drop)])
    sample = sample.sample(frac=1.0, random_state=cfg.seed)  # shuffle

    # Columns the annotator sees. `keep` is deliberately EXCLUDED: showing the
    # filter's own verdict would anchor the annotator and make the audit
    # worthless. It is preserved in a separate key file.
    sheet = pd.DataFrame({
        "audit_id": range(1, len(sample) + 1),
        "title": sample["title"].values,
        "date": pd.to_datetime(sample["seendate_utc"]).dt.date.values,
        "domain": sample["domain"].values,
        "label": "",
        "notes": "",
    })

    key = pd.DataFrame({
        "audit_id": sheet["audit_id"].values,
        "filter_kept": sample["keep"].astype(int).values,
        "theme_family": sample["theme_family"].values,
        "source_tier": sample.get("source_tier", pd.Series(np.nan)).values,
        "url": sample["url"].values,
    })

    rep = cfg.path("reports")
    (rep / "audit_GUIDELINES.txt").write_text(GUIDELINE, encoding="utf-8")
    key.to_csv(rep / "audit_key.csv", index=False)
    for who in ("A", "B"):
        sheet.to_csv(rep / f"audit_annotator_{who}.csv", index=False)

    print(f"Wrote {len(sheet)} rows to:")
    print(f"  {rep/'audit_annotator_A.csv'}")
    print(f"  {rep/'audit_annotator_B.csv'}")
    print(f"  {rep/'audit_key.csv'}        <- do NOT open before labelling")
    print(f"  {rep/'audit_GUIDELINES.txt'}")
    print(f"\nComposition: {int(key['filter_kept'].sum())} kept / "
          f"{int((1-key['filter_kept']).sum())} dropped by the filter")


def cohens_kappa(a: pd.Series, b: pd.Series) -> float:
    """Chance-corrected agreement. Raw agreement is meaningless when one class
    dominates: two annotators who both say '1' 90% of the time agree 82% of the
    time by luck alone."""
    cats = sorted(set(a.dropna()) | set(b.dropna()))
    if len(cats) < 2:
        return float("nan")
    m = pd.crosstab(a, b).reindex(index=cats, columns=cats, fill_value=0).values
    n = m.sum()
    if n == 0:
        return float("nan")
    po = np.trace(m) / n
    pe = (m.sum(0) * m.sum(1)).sum() / (n * n)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def score(cfg, args) -> None:
    rep = cfg.path("reports")
    a = pd.read_csv(args.a).set_index("audit_id")
    b = pd.read_csv(args.b).set_index("audit_id")
    key = pd.read_csv(rep / "audit_key.csv").set_index("audit_id")

    def clean(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s.astype(str).str.strip().replace({"?": np.nan, "": np.nan}),
                             errors="coerce")

    la, lb = clean(a["label"]), clean(b["label"])
    both = pd.DataFrame({"a": la, "b": lb}).join(key).dropna(subset=["a", "b"])
    if both.empty:
        raise SystemExit("No usable labels found — fill in the `label` column first.")

    kappa = cohens_kappa(both["a"], both["b"])
    raw_agree = float((both["a"] == both["b"]).mean())

    # Only rows the two annotators agree on become ground truth. Disagreements
    # are genuinely ambiguous and should not be forced into a verdict.
    agreed = both[both["a"] == both["b"]].copy()
    agreed["truth"] = agreed["a"].astype(int)

    kept = agreed[agreed["filter_kept"] == 1]
    dropped = agreed[agreed["filter_kept"] == 0]

    precision = float(kept["truth"].mean()) if len(kept) else float("nan")
    false_neg_rate = float(dropped["truth"].mean()) if len(dropped) else float("nan")

    # Recall over the audited sample, reweighted by how many articles each side
    # of the filter actually contains — an unweighted figure would be wrong,
    # because we oversampled the kept side 70/30 on purpose.
    tp, fn = precision * len(kept), false_neg_rate * len(dropped)
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

    thresh_p = float((cfg.dig("audit", "accept_precision", default=0.85)))
    thresh_k = float((cfg.dig("audit", "accept_kappa", default=0.60)))

    rows = [
        ("rows labelled by both", len(both), ""),
        ("raw agreement", round(raw_agree, 3), ""),
        ("Cohen's kappa", round(kappa, 3),
         "PASS" if kappa >= thresh_k else f"FAIL (< {thresh_k})"),
        ("filter precision (kept -> material)", round(precision, 3),
         "PASS" if precision >= thresh_p else f"FAIL (< {thresh_p})"),
        ("false-negative rate (dropped -> material)", round(false_neg_rate, 3), ""),
        ("estimated recall", round(recall, 3), ""),
        ("unresolved disagreements", int((both["a"] != both["b"]).sum()), ""),
    ]
    res = pd.DataFrame(rows, columns=["metric", "value", "verdict"])
    res.to_csv(rep / "audit_results.csv", index=False)
    print(res.to_string(index=False))

    by_fam = agreed.groupby("theme_family").apply(
        lambda g: pd.Series({
            "n": len(g),
            "precision": g.loc[g["filter_kept"] == 1, "truth"].mean(),
            "miss_rate": g.loc[g["filter_kept"] == 0, "truth"].mean(),
        }), include_groups=False,
    ).round(3)
    by_fam.to_csv(rep / "audit_by_family.csv")
    print("\nBy theme family:\n", by_fam.to_string())

    if kappa < thresh_k:
        print("\n>> Kappa below threshold: the LABEL is ill-defined, not the filter. "
              "Revise the guidelines and re-annotate before reading precision.")
    elif precision < thresh_p:
        print("\n>> Precision below threshold: tighten the Stage-2 materiality rules "
              "(see config/config.yaml) and re-run. Do not tune on the test period.")
    else:
        print("\n>> Filter accepted. Record these numbers in the report.")


def main() -> int:
    p = argparse.ArgumentParser(description="Stage 4 human audit")
    p.add_argument("--config", default="config/config.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("build")
    pb.add_argument("--n", type=int, default=None)
    pb.add_argument("--config", default="config/config.yaml")

    ps = sub.add_parser("score")
    ps.add_argument("--a", required=True)
    ps.add_argument("--b", required=True)
    ps.add_argument("--config", default="config/config.yaml")

    args = p.parse_args()

    cfg = load_config(args.config)
    (build if args.cmd == "build" else score)(cfg, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
