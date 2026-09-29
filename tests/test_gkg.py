"""Tests for the GKG bulk-file news source (src/gkg.py).

Like test_pipeline.py, these target decisions that would silently corrupt the
corpus if wrong: a substring theme match that drags in the wrong family, a
volume count taken after the cap, a sample that changes between reruns.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import gkg  # noqa: E402

FAMILIES = {
    "energy": ["ENV_OIL"],
    "conflict": ["ARMEDCONFLICT", "MILITARY"],
}


def _line(url="https://example.com/a", v1="", v2="", locs="", tone="1.5,2,3,4,5,6,7",
          extras="", date="20240115120000", domain="example.com") -> str:
    cols = [""] * gkg.N_COLS
    cols[0] = f"{date}-1"
    cols[gkg.COL_DATE] = date
    cols[gkg.COL_DOMAIN] = domain
    cols[gkg.COL_URL] = url
    cols[gkg.COL_V1THEMES] = v1
    cols[gkg.COL_V2THEMES] = v2
    cols[gkg.COL_V1LOCATIONS] = locs
    cols[gkg.COL_TONE] = tone
    cols[gkg.COL_EXTRAS] = extras
    return "\t".join(cols)


def test_theme_match_is_exact_token_not_substring():
    rows = gkg.parse_gkg(_line(v1="ENV_OILSPILL;TAX_FNCACT"), FAMILIES)
    assert rows == []

    rows = gkg.parse_gkg(_line(v2="ENV_OIL,120;MILITARY,40"), FAMILIES)
    assert sorted(rows[0]["families"]) == ["conflict", "energy"]


def test_page_title_is_preferred_and_unescaped():
    extras = "<PAGE_LINKS></PAGE_LINKS><PAGE_TITLE>Fed &amp; ECB hold rates</PAGE_TITLE>"
    row = gkg.parse_gkg(_line(v1="ENV_OIL", extras=extras,
                              url="https://x.com/some-other-slug-words-here"), FAMILIES)[0]
    assert row["title"] == "Fed & ECB hold rates"
    assert row["title_source"] == "page_title"


def test_slug_fallback_when_no_page_title():
    url = "https://x.com/2017/01/03/china-slaps-new-tariffs-on-us-goods.html"
    row = gkg.parse_gkg(_line(v1="ENV_OIL", url=url), FAMILIES)[0]
    assert row["title"] == "china slaps new tariffs on us goods"
    assert row["title_source"] == "url_slug"

    row = gkg.parse_gkg(_line(v1="ENV_OIL", url="https://x.com/news/article?id=12345"), FAMILIES)[0]
    assert row["title"] is None and row["title_source"] is None


def test_country_is_most_mentioned_fips_code():
    locs = ("1#Russia#RS#RS##60#100#RS;4#Moscow, Moscow City, Russia#RS#RS48#55.75#37.6#-2960561;"
            "1#United States#US#US##39.8#-98.5#US")
    row = gkg.parse_gkg(_line(v1="MILITARY", locs=locs), FAMILIES)[0]
    assert row["country"] == "RS"


def _payload(n, fam="conflict", tone=-2.0, status="ok"):
    rows = [{"date": "20240115120000", "domain": "d.com", "url": f"https://d.com/{i}",
             "title": f"headline {i}", "title_source": "page_title",
             "families": [fam], "tone": tone, "country": "US"} for i in range(n)]
    return {"status": status, "rows": rows}


def test_volume_is_counted_before_the_cap_and_sample_is_deterministic():
    day = datetime(2024, 1, 15)
    payloads = [(datetime(2024, 1, 15, 6), _payload(6)), (datetime(2024, 1, 15, 0), _payload(4))]

    arts, tl = gkg.finalize_day(day, payloads, FAMILIES.keys(), cap=3, seed=42)
    again, _ = gkg.finalize_day(day, list(reversed(payloads)), FAMILIES.keys(), cap=3, seed=42)

    assert len(arts) == 3
    assert [a["url"] for a in arts] == [a["url"] for a in again]
    volume = {(f, m): v for _, v, f, m in tl}
    assert volume[("conflict", "volume")] == 5.0      # 10 articles over 2 slices, pre-cap
    assert volume[("conflict", "tone")] == -2.0
    assert volume[("energy", "volume")] == 0.0
    assert arts[0]["seendate"] == "20240115T120000Z"


def test_missing_slice_does_not_deflate_volume():
    day = datetime(2024, 1, 15)
    payloads = [(datetime(2024, 1, 15, 0), _payload(4)),
                (datetime(2024, 1, 15, 6), {"status": "missing", "rows": []})]
    _, tl = gkg.finalize_day(day, payloads, ["conflict"], cap=250, seed=42)
    assert {(f, m): v for _, v, f, m in tl}[("conflict", "volume")] == 4.0


def test_slot_stamps_sit_on_the_15_minute_grid():
    stamps = gkg.slot_stamps("2024-01-15", 4)
    assert [s.strftime("%H:%M") for s in stamps] == ["00:00", "06:00", "12:00", "18:00"]
    assert all(s.minute % 15 == 0 for s in gkg.slot_stamps("2024-01-15", 7, offset_minutes=10))
