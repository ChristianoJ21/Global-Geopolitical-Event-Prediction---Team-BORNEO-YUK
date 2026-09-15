"""Stage 3 — three-level deduplication with SimHash near-duplicate detection.

Why this stage carries so much weight (report §4.4)
---------------------------------------------------
A single Reuters story about an oil embargo appears in GDELT hundreds of times:
mirrored by syndication partners, re-titled by aggregators, re-crawled with a
different URL parameter. If we do not deduplicate, then:

  * "news volume", our headline feature, measures **syndication reach**, not
    event importance;
  * TF-IDF is poisoned — a copied phrase looks like a strong repeated pattern
    rather than one observation;
  * and any train/test split leaks, because the same story lands on both sides.

But naive deduplication throws away real information. How *widely* a story is
copied is a genuine proxy for how important editors judged it to be. So we do
not delete and forget: we collapse each cluster to its earliest member and
carry the cluster size forward as ``dup_count``, a feature in its own right.

Three levels, cheapest first:
  1. exact URL
  2. normalised-title exact hash   (strips outlet suffixes, punctuation, case)
  3. SimHash Hamming distance <= threshold within a +/-48h window

Why SimHash and not embeddings
------------------------------
Near-duplicate detection here is a *lexical* problem: the same sentence with a
different outlet tag. SimHash over character 4-grams solves it in O(n) with a
64-bit integer per document, runs on a laptop over ~1M headlines, and is fully
deterministic — so a teammate re-running the pipeline gets byte-identical
clusters. A sentence-transformer would be slower, need a GPU, introduce a model
dependency and a random seed, and would *over*-merge: "Fed raises rates" and
"Fed cuts rates" are semantically close but are opposite events. Cheap and
deterministic is the right engineering call here.
"""
from __future__ import annotations

import hashlib
import logging
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# " ... - Reuters", " ... | Al Jazeera", " ... — BBC News"
_OUTLET_SUFFIX = re.compile(r"\s*[\-–—|·:]\s*[A-Z][\w .&'’]{2,30}\s*$")
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")


def normalize_title(title: str) -> str:
    """Canonical form used for both exact-hash and SimHash comparison."""
    if not isinstance(title, str):
        return ""
    t = _OUTLET_SUFFIX.sub("", title.strip())
    t = t.lower()
    t = _NON_ALNUM.sub(" ", t)
    return _WS.sub(" ", t).strip()


def canonical_url(url: str) -> str:
    """Strip tracking parameters and fragments so mirrors collapse."""
    if not isinstance(url, str):
        return ""
    u = url.split("#", 1)[0]
    u = re.sub(r"[?&](utm_[^=&]+|fbclid|gclid|ref|src|amp)=[^&]*", "", u)
    return u.rstrip("?&/").lower()


# ---------------------------------------------------------------------------
# SimHash
# ---------------------------------------------------------------------------
def _shingles(text: str, k: int) -> List[str]:
    if len(text) < k:
        return [text] if text else []
    return [text[i: i + k] for i in range(len(text) - k + 1)]


def simhash(text: str, bits: int = 64, k: int = 4) -> int:
    """Charikar SimHash over character k-grams.

    Character shingles (not word tokens) are deliberate: they are robust to the
    small edits syndication introduces — a swapped preposition, a dropped
    article, a localised spelling ("labour"/"labor").
    """
    v = np.zeros(bits, dtype=np.int64)
    grams = _shingles(text, k)
    if not grams:
        return 0
    for g in grams:
        h = int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest(), "big")
        for i in range(bits):
            v[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i in range(bits):
        if v[i] > 0:
            out |= 1 << i
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _bands(h: int, bits: int, n_bands: int) -> List[Tuple[int, int]]:
    """LSH banding: split the hash into bands so only plausible pairs are compared.

    Two hashes within d bits of each other must agree exactly on at least one
    band when n_bands > d. Comparing only within-band candidates turns an
    O(n^2) all-pairs scan into something that finishes on a laptop.
    """
    w = bits // n_bands
    return [(i, (h >> (i * w)) & ((1 << w) - 1)) for i in range(n_bands)]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def deduplicate(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Flag duplicates and attach ``dup_count`` / ``cluster_id``.

    Returns the frame with four new columns:
      ``norm_title``  canonical title used for matching
      ``cluster_id``  id of the near-duplicate cluster (earliest member's index)
      ``is_duplicate`` True for every member except the earliest
      ``dup_count``   cluster size, carried on the surviving row as a FEATURE
    """
    d = cfg.dig("dedup", default={}) or {}
    bits = int(d.get("simhash_bits", 64))
    k = int(d.get("shingle_size", 4))
    thresh = int(d.get("hamming_threshold", 6))
    window_h = int(d.get("window_hours", 48))

    out = df.copy().reset_index(drop=True)
    if out.empty:
        for c in ("norm_title", "cluster_id", "is_duplicate", "dup_count"):
            out[c] = pd.Series(dtype="object")
        return out

    out["norm_title"] = out["title"].map(normalize_title)
    out["canon_url"] = out["url"].map(canonical_url)
    out = out.sort_values("seendate_utc", kind="mergesort").reset_index(drop=True)

    n = len(out)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # Lower index wins => earliest timestamp becomes the cluster root,
            # because the frame is sorted by time.
            parent[max(ra, rb)] = min(ra, rb)

    # --- Level 1: exact canonical URL ---------------------------------------
    for _, idx in out.groupby("canon_url", sort=False).groups.items():
        idx = list(idx)
        for j in idx[1:]:
            union(idx[0], j)
    lvl1 = n - len({find(i) for i in range(n)})

    # --- Level 2: exact normalised title, WITHIN the time window ------------
    # The time window matters here as much as at level 3. "Oil prices rise as
    # tensions mount" is a headline that recurs verbatim every few months for a
    # decade. Merging all of its occurrences into one cluster would delete a
    # genuine, repeated signal and silently thin out the corpus. Two articles
    # are the same *story* only if they are also close in time.
    ts_all = out["seendate_utc"].values.astype("datetime64[s]").astype(np.int64)
    window_s_l2 = window_h * 3600
    for _, idx in out.groupby("norm_title", sort=False).groups.items():
        idx = list(idx)
        if len(idx) < 2:
            continue
        anchor = idx[0]
        for j in idx[1:]:
            if abs(int(ts_all[j]) - int(ts_all[anchor])) <= window_s_l2:
                union(anchor, j)
            else:
                anchor = j  # start a new temporal cluster for the same headline
    lvl2 = n - len({find(i) for i in range(n)}) - lvl1

    # --- Level 3: SimHash near-duplicates within a time window --------------
    hashes = np.array([simhash(t, bits=bits, k=k) for t in out["norm_title"]], dtype=object)
    ts = out["seendate_utc"].values.astype("datetime64[s]").astype(np.int64)
    window_s = window_h * 3600
    n_bands = max(4, thresh + 1)

    buckets: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for i, h in enumerate(hashes):
        for band in _bands(int(h), bits, n_bands):
            buckets[band].append(i)

    compared = 0
    for cand in buckets.values():
        if len(cand) < 2 or len(cand) > 2000:
            # Huge buckets are degenerate (e.g. near-empty titles); skipping
            # them avoids a quadratic blow-up for no recall gain.
            continue
        for a_pos in range(len(cand)):
            i = cand[a_pos]
            for j in cand[a_pos + 1:]:
                if abs(int(ts[j]) - int(ts[i])) > window_s:
                    continue
                compared += 1
                if hamming(int(hashes[i]), int(hashes[j])) <= thresh:
                    union(i, j)

    roots = [find(i) for i in range(n)]
    out["cluster_id"] = roots
    out["is_duplicate"] = [i != r for i, r in enumerate(roots)]

    sizes = pd.Series(roots).value_counts()
    out["dup_count"] = out["cluster_id"].map(sizes).astype(int)

    lvl3 = int(out["is_duplicate"].sum()) - lvl1 - lvl2
    log.info(
        "Stage 3 dedup: %d/%d flagged (%.1f%%) | url=%d title=%d simhash=%d | %d pairs compared | %d clusters",
        int(out["is_duplicate"].sum()), n, 100 * out["is_duplicate"].mean(),
        lvl1, lvl2, max(lvl3, 0), compared, out["cluster_id"].nunique(),
    )
    return out.drop(columns=["canon_url"])


def dedup_report(df: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """The most-syndicated stories — a useful qualitative sanity check.

    If the top clusters are recognisable major geopolitical events, the
    dedup + filter stack is behaving. If they are horoscopes, it is not.
    """
    if "cluster_id" not in df.columns:
        return pd.DataFrame()
    roots = df.loc[~df["is_duplicate"], ["cluster_id", "title", "seendate_utc", "dup_count"]]
    return roots.nlargest(top, "dup_count").reset_index(drop=True)


__all__ = [
    "deduplicate", "dedup_report", "simhash", "hamming",
    "normalize_title", "canonical_url",
]
