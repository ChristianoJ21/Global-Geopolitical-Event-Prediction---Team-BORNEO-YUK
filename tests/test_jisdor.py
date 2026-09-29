"""Tests for the Bank Indonesia JISDOR target (src/fx_data.py).

Both guard mistakes that would corrupt the target without raising an error:
reading 9/1/2026 as 9 January, and losing returns around Indonesian holidays.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import fx_data  # noqa: E402
from src.config import load_config  # noqa: E402

CFG = load_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")


def test_jisdor_export_parses_us_style_dates_under_a_title_block():
    raw = pd.DataFrame([
        [None, None, None, None],
        ["Informasi Kurs Jisdor"] * 4,
        [None, None, None, None],
        ["NO", "Tanggal", "Kurs", None],
        [1, "9/1/2026 12:00:00 AM", 17727, None],
        [2, "8/31/2026 12:00:00 AM", 17746, None],
        [3, "9/2/2021 12:00:00 AM", 14281, None],
    ])
    s = fx_data.parse_jisdor_frame(raw)

    assert list(s.index) == [pd.Timestamp("2021-09-02"),
                             pd.Timestamp("2026-08-31"),
                             pd.Timestamp("2026-09-01")]
    assert s.loc["2026-09-01"] == 17727, "9/1 must be 1 September, not 9 January"


def test_day_after_indonesian_holiday_keeps_its_return():
    # Idul Fitri 2024: JISDOR did not print on 9-10 April; the US controls did.
    idx = pd.DatetimeIndex(["2024-04-08", "2024-04-09", "2024-04-10", "2024-04-11"])
    panel = pd.DataFrame({"USD_IDR": [15800.0, None, None, 15900.0],
                          "DXY": [104.0, 104.2, 104.5, 104.1]}, index=idx)
    t = fx_data.make_targets(panel, CFG)

    assert pd.notna(t.loc["2024-04-11", "y_return"]), "return after the holiday was lost"
    assert pd.Timestamp("2024-04-09") not in t.index, "non-JISDOR days must not become rows"
