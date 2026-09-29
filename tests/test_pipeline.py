"""Property tests for the Task 1 pipeline.

Run with:  python -m pytest tests/ -v        (or: python tests/test_pipeline.py)

Scope
-----
These tests target the decisions that would *silently* invalidate the study if
they were wrong. We do not test that pandas can group by a column; we test that
an article published one minute after the FX close lands in tomorrow's bucket,
because that single off-by-one is the difference between a forecast and a leak.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import align, dedup, filtering, fx_data, preprocess  # noqa: E402
from src.config import load_config  # noqa: E402
from src.synthetic import synthetic_articles, synthetic_fx_panel  # noqa: E402

CFG = load_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")


# ===========================================================================
# 1. TEMPORAL ALIGNMENT — the tests that matter most
# ===========================================================================
def test_utc_is_converted_to_wib_before_taking_the_date():
    """18:30 UTC on Tue 5 Mar is 01:30 WIB on Wed 6 Mar: it is WEDNESDAY's news."""
    days = pd.DatetimeIndex(["2024-03-05", "2024-03-06", "2024-03-07"])  # Tue-Thu
    late_utc = pd.Timestamp("2024-03-05 18:30:00", tz="UTC")
    early_utc = pd.Timestamp("2024-03-05 16:59:00", tz="UTC")  # 23:59 WIB Tue

    got = align.assign_news_day(pd.Series([late_utc, early_utc]), days)
    assert got.iloc[0] == pd.Timestamp("2024-03-06"), "converted first -> Wednesday"
    assert got.iloc[1] == pd.Timestamp("2024-03-05"), "still Tuesday in WIB"


def test_weekend_news_shares_fridays_bucket_and_mondays_fix():
    """Fri + Sat + Sun news -> one bucket -> paired with Monday's JISDOR fix."""
    days = pd.DatetimeIndex(["2024-03-08", "2024-03-11"])  # Fri, Mon
    fri = pd.Timestamp("2024-03-08 03:00:00", tz="UTC")     # 10:00 WIB Fri
    sat = pd.Timestamp("2024-03-09 05:00:00", tz="UTC")
    sun = pd.Timestamp("2024-03-10 16:00:00", tz="UTC")     # 23:00 WIB Sun
    got = align.assign_news_day(pd.Series([fri, sat, sun]), days)
    assert (got == pd.Timestamp("2024-03-08")).all(), "one weekend bucket, on Friday"

    # The 1-day shift in make_prediction_frame then pairs that bucket with Monday.
    feats = pd.DataFrame({"n": [3.0, 0.0]}, index=days)
    targets = pd.DataFrame({"y_return": [0.1, 0.2]}, index=days)
    frame = align.make_prediction_frame(feats, targets, CFG)
    assert frame.loc["2024-03-11", "n"] == 3.0, "weekend news must meet Monday's fix"


def test_holiday_news_rolls_to_the_next_fix():
    """Idul Fitri 2024: no JISDOR 8-12 Apr. News on 10 Apr meets the 15 Apr fix."""
    days = pd.DatetimeIndex(["2024-04-05", "2024-04-15"])
    got = align.assign_news_day(pd.Series([pd.Timestamp("2024-04-10 06:00", tz="UTC")]), days)
    assert got.iloc[0] == pd.Timestamp("2024-04-05"), "holiday news joins the prior bucket"


def test_prediction_frame_has_no_contemporaneous_market_column():
    """The leak the QA suite caught on day one must not be able to come back."""
    fx = synthetic_fx_panel(CFG, n_days=300)
    targets = fx_data.make_targets(fx, CFG)
    feats = pd.DataFrame(
        {"dummy_news": np.arange(len(targets), dtype=float)}, index=targets.index
    )
    frame = align.make_prediction_frame(feats, targets, CFG)

    assert "ret_USD_IDR" not in frame.columns, "raw contemporaneous return leaked in"
    assert "lag1_ret_USD_IDR" in frame.columns, "lagged market control is missing"

    both = frame[["lag1_ret_USD_IDR", "y_return"]].dropna()
    assert abs(both.corr().iloc[0, 1]) < 0.5, "lagged control still tracks the target"
    assert not align.leakage_assertions(frame, CFG), "leakage assertions must pass"


def test_horizon_targets_are_cumulative_forward_sums():
    fx = synthetic_fx_panel(CFG, n_days=200)
    t = fx_data.make_targets(fx, CFG).dropna(subset=["y_return"])
    r = t["y_return"]
    expected = r.iloc[0] + r.iloc[1] + r.iloc[2]
    assert np.isclose(t["y_return_h3"].iloc[0], expected), "h=3 must sum r_t..r_{t+2}"
    assert np.isclose(t["y_return_h1"].iloc[5], r.iloc[5]), "h=1 must equal r_t"


def test_chronological_split_does_not_overlap():
    fx = synthetic_fx_panel(CFG, n_days=1500)
    targets = fx_data.make_targets(fx, CFG)
    feats = pd.DataFrame({"x": 1.0}, index=targets.index)
    frame = align.chronological_split(
        align.make_prediction_frame(feats, targets, CFG), CFG
    )
    tr, va, te = (frame.index[frame["split"] == s] for s in ("train", "val", "test"))
    if len(tr) and len(va):
        assert tr.max() < va.min()
    if len(va) and len(te):
        assert va.max() < te.min()


# ===========================================================================
# 2. PREPROCESSING
# ===========================================================================
def test_negation_survives_track_a():
    """The keep-list is the whole justification for our stopword config."""
    a_cfg = CFG.dig("preprocess", "track_a")
    out = preprocess.preprocess_track_a("Fed will not cut rates in December", a_cfg)
    assert "not" in out.split(), f"negation destroyed by preprocessing: {out!r}"


def test_track_b_preserves_case_and_function_words():
    b_cfg = CFG.dig("preprocess", "track_b")
    out = preprocess.preprocess_track_b("The Fed will not cut rates - Reuters", b_cfg)
    assert "Fed" in out, "casing lost; transformers need it"
    assert "not" in out and "will" in out, "function words stripped from Track B"
    assert "Reuters" not in out, "outlet suffix not stripped"


def test_numbers_become_typed_placeholders():
    a_cfg = CFG.dig("preprocess", "track_a")
    out = preprocess.preprocess_track_a("Inflation rose 4.5% to a record 12,000 level", a_cfg)
    assert "<pct>" in out.lower(), f"percentage not masked: {out!r}"
    assert "<num>" in out.lower(), f"number not masked: {out!r}"


def test_entity_aliases_collapse_to_one_symbol():
    pats = preprocess.build_entity_patterns(CFG.dig("entity_aliases"))
    for variant in ("U.S. imposes tariffs", "Washington imposes tariffs",
                    "United States imposes tariffs"):
        got = preprocess.canonicalize_entities(variant, pats)
        assert "UNITED_STATES" in got, f"{variant!r} -> {got!r}"


# ===========================================================================
# 3. DEDUPLICATION
# ===========================================================================
def test_simhash_near_duplicates_are_close_and_opposites_are_not():
    a = dedup.simhash(dedup.normalize_title("Fed raises interest rates by 25 basis points"))
    b = dedup.simhash(dedup.normalize_title("Fed raises interest rates by 25 basis points - Reuters"))
    c = dedup.simhash(dedup.normalize_title("Brazil wins the football world cup final"))
    assert dedup.hamming(a, b) <= 6, "syndicated copy not detected as near-duplicate"
    assert dedup.hamming(a, c) > 6, "unrelated headlines collapsed together"


def test_dedup_keeps_the_earliest_member_and_records_cluster_size():
    base = "Oil prices surge after new sanctions announced"
    rows = [
        {"url": "https://a.com/1", "title": base,
         "seendate_utc": pd.Timestamp("2024-01-02 08:00", tz="UTC"), "domain": "a.com"},
        {"url": "https://b.com/2", "title": base + " - B News",
         "seendate_utc": pd.Timestamp("2024-01-02 11:00", tz="UTC"), "domain": "b.com"},
        {"url": "https://c.com/3", "title": base,
         "seendate_utc": pd.Timestamp("2024-01-02 14:00", tz="UTC"), "domain": "c.com"},
    ]
    out = dedup.deduplicate(pd.DataFrame(rows), CFG)
    survivors = out.loc[~out["is_duplicate"]]
    assert len(survivors) == 1, "cluster not collapsed to a single survivor"
    assert survivors["domain"].iloc[0] == "a.com", "earliest publisher not kept"
    assert int(survivors["dup_count"].iloc[0]) == 3, "syndication breadth not recorded"


def test_identical_headline_far_apart_is_not_deduplicated():
    """A recurring headline is a repeated event, not a duplicate story."""
    base = "Oil prices rise as Middle East tensions mount"
    rows = [
        {"url": "https://a.com/1", "title": base,
         "seendate_utc": pd.Timestamp("2024-01-02 08:00", tz="UTC"), "domain": "a.com"},
        {"url": "https://a.com/2", "title": base,
         "seendate_utc": pd.Timestamp("2024-07-02 08:00", tz="UTC"), "domain": "a.com"},
    ]
    out = dedup.deduplicate(pd.DataFrame(rows), CFG)
    assert int(out["is_duplicate"].sum()) == 0, "six months apart must not merge"


# ===========================================================================
# 4. FILTERING
# ===========================================================================
def test_source_gate_blocks_farms_and_tiers_wires():
    df = pd.DataFrame({
        "domain": ["reuters.com", "bbc.co.uk", "random-local.example",
                   "spam.blogspot.com", "msn.com"],
        "title": ["x"] * 5, "sourcecountry": ["US"] * 5,
    })
    out = filtering.apply_source_gate(df, CFG)
    assert out["source_tier"].tolist()[:3] == [1, 2, 3]
    assert out.loc[3, "is_blocked_domain"] and out.loc[4, "is_blocked_domain"]
    assert not out.loc[3, "pass_source"]


def test_materiality_gate_is_or_not_and():
    """Early, currency-agnostic coverage must survive — that is where the
    predictive value lives, before the move has happened."""
    df = pd.DataFrame({
        "title": ["Russia masses troops near the border",       # country only
                  "Federal Reserve signals a pause",            # institution
                  "Local council debates parking permits",      # nothing
                  "Tariff package announced"],                  # market term
        "sourcecountry": ["RS", "US", "ZZ", "ZZ"],
        "domain": ["reuters.com"] * 4,
    })
    out = filtering.apply_materiality_gate(df, CFG)
    assert out["pass_materiality"].tolist() == [True, True, False, True]


# ===========================================================================
# 5. END-TO-END
# ===========================================================================
def test_full_offline_pipeline_produces_a_clean_frame():
    fx = synthetic_fx_panel(CFG, n_days=260)
    targets = fx_data.make_targets(fx, CFG)
    td = align.trading_day_calendar(fx, CFG)
    arts = synthetic_articles(CFG, td, base_per_family=2)

    arts = dedup.deduplicate(arts, CFG)
    arts = filtering.run_funnel(arts, CFG)
    arts = align.attach_news_day(arts, td, CFG)
    arts = preprocess.preprocess_frame(arts, CFG)

    from src.features import build_daily_features, reindex_to_trading_days
    feats = reindex_to_trading_days(build_daily_features(arts, CFG), td)
    frame = align.chronological_split(
        align.make_prediction_frame(feats, targets, CFG), CFG
    )

    assert len(frame) > 100
    assert frame.index.is_monotonic_increasing and not frame.index.has_duplicates
    assert "y_return" in frame.columns and "vol_z30" in frame.columns
    assert not align.leakage_assertions(frame, CFG)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
