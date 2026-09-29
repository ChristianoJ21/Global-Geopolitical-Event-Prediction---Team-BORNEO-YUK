#!/usr/bin/env python3
"""Build data/lexicons/loughran_mcdonald_2014.csv from the LM Master Dictionary.

    py scripts/make_lm_lexicon.py <path to LM master dictionary CSV>

Source: Loughran, T. and McDonald, B. (2011), "When Is a Liability Not a
Liability? Textual Analysis, Dictionaries, and 10-Ks", Journal of Finance 66(1).
The master dictionary lists ~86,000 words; a word belongs to a sentiment
category when that category's column is non-zero (the value is the year the
word was added). We keep only words in at least one of the five categories we
use, so the repository carries a few thousand rows instead of the full list.

The copy used by the team is the 2014 release bundled in the MIT-licensed
``pysentiment2`` package (``pysentiment2/static/LM.csv``). The official file
from https://sraf.nd.edu/loughranmcdonald-master-dictionary/ has the same
columns and works with this script unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "lexicons" / "loughran_mcdonald_2014.csv"
CATEGORIES = ["Negative", "Positive", "Uncertainty", "Litigious", "Constraining"]


def main(src: str) -> int:
    master = pd.read_csv(src)
    missing = [c for c in ["Word", *CATEGORIES] if c not in master.columns]
    if missing:
        raise SystemExit(f"{src} is not an LM master dictionary (missing {missing})")

    flags = (master[CATEGORIES] > 0).astype(int)
    lex = pd.concat([master["Word"].str.lower().rename("word"), flags], axis=1)
    lex.columns = ["word", *[c.lower() for c in CATEGORIES]]
    lex = lex.loc[lex[[c.lower() for c in CATEGORIES]].any(axis=1)]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    lex.to_csv(OUT, index=False)
    print(f"wrote {len(lex)} words -> {OUT}")
    print(lex.drop(columns="word").sum().to_string())
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    raise SystemExit(main(sys.argv[1]))
