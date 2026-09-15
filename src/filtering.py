"""The four-stage filtering funnel (report §4).

The problem
-----------
"Geopolitical news that moves the dollar" is not a keyword and not a single
GDELT theme. Any single-stage filter fails in one of two directions:

  * a broad one (``theme:MILITARY``) floods the corpus with military-parade
    coverage and video-game reviews, and
  * a narrow one (``"dollar" AND "sanctions"``) returns only articles that
    *already mention* the outcome we are trying to predict — which would make
    our "prediction" a tautology.

So we use a funnel of four cheap, independently auditable stages. Each stage is
a *human* decision with a stated cost, and each stage writes a boolean column
rather than deleting rows. Nothing is thrown away irreversibly: Task 2 can
ablate any stage by flipping a flag, which is exactly the kind of evidence we
want when we are asked to justify the design.

    Stage 0  source gate        precision, cheap   -> `pass_source`
    Stage 1  thematic gate      recall             -> done at query time
    Stage 2  materiality gate   precision          -> `pass_materiality`
    Stage 3  deduplication      noise              -> `is_duplicate`
    Stage 4  human audit        validation         -> scripts/make_audit_sample.py
"""
from __future__ import annotations

import fnmatch
import logging
import re
from typing import Dict, Iterable, List, Set

import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 0 — source gate
# ---------------------------------------------------------------------------
def apply_source_gate(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Tag each article with a source tier and a blocklist decision.

    Rationale
    ---------
    GDELT indexes a long tail of content farms that republish one wire story
    hundreds of times. If we count raw articles, our "news volume" feature is
    really "how many SEO sites copied Reuters today" — a quantity with no
    macroeconomic meaning that nonetheless trends upward over the decade and
    would be happily mistaken for signal by any model.

    We use ``soft`` mode by default: non-listed domains are kept but tagged
    ``tier=3``. Hard-deleting them would destroy our ability to *test* whether
    the gate helped, and it would bias coverage against non-Western outlets
    that simply are not on our (inevitably Anglophone) allowlist.

    State-affiliated outlets are kept on purpose and flagged. They publish
    propaganda, but propaganda *is* a geopolitical signal — a TASS framing
    shift is information. Excluding them would leave us with only the Western
    narrative of every event, which is a bias, not a cleaning step.
    """
    gate = cfg.dig("source_gate", default={}) or {}
    tier1: Set[str] = {d.lower() for d in gate.get("tier1_wires", [])}
    tier2: Set[str] = {d.lower() for d in gate.get("tier2_quality", [])}
    state: Set[str] = {d.lower() for d in gate.get("state_affiliated", [])}
    blocks: List[str] = [p.lower() for p in gate.get("blocklist_patterns", [])]
    mode = str(gate.get("allowlist_mode", "soft")).lower()

    dom = df["domain"].fillna("").astype(str).str.lower()

    def tier_of(d: str) -> int:
        if any(d == t or d.endswith("." + t) for t in tier1):
            return 1
        if any(d == t or d.endswith("." + t) for t in tier2):
            return 2
        return 3

    def blocked(d: str) -> bool:
        return any(fnmatch.fnmatch(d, p) if "*" in p else (p in d) for p in blocks)

    out = df.copy()
    out["source_tier"] = [tier_of(d) for d in dom]
    out["state_affiliated"] = [
        any(d == s or d.endswith("." + s) for s in state) for d in dom
    ]
    out["is_blocked_domain"] = [blocked(d) for d in dom]

    if mode == "hard":
        out["pass_source"] = (~out["is_blocked_domain"]) & (out["source_tier"] <= 2)
    else:
        out["pass_source"] = ~out["is_blocked_domain"]

    log.info(
        "Stage 0 source gate (%s): pass %d/%d (%.1f%%) | tier1 %d tier2 %d tier3 %d | blocked %d",
        mode, int(out["pass_source"].sum()), len(out),
        100 * out["pass_source"].mean(),
        int((out["source_tier"] == 1).sum()),
        int((out["source_tier"] == 2).sum()),
        int((out["source_tier"] == 3).sum()),
        int(out["is_blocked_domain"].sum()),
    )
    return out


# ---------------------------------------------------------------------------
# Stage 2 — materiality gate
# ---------------------------------------------------------------------------
def _compile_terms(terms: Iterable[str]) -> re.Pattern:
    """Word-boundary alternation, longest-first so 'rate hike' wins over 'rate'."""
    escaped = sorted((re.escape(t.strip().lower()) for t in terms if t.strip()),
                     key=len, reverse=True)
    if not escaped:
        return re.compile(r"(?!x)x")  # never matches
    # Non-capturing group: a capturing group makes pandas' .str.contains warn
    # and changes its semantics.
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b", flags=re.IGNORECASE)


def apply_materiality_gate(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Keep only articles that plausibly touch a USD transmission channel.

    An article passes if **any** of three independent conditions holds:

      (a) ``actor_country``  — it comes from (or is about) a G20 /
          reserve-currency / geopolitically pivotal state;
      (b) ``institution``    — its headline names a monetary, trade or security
          institution (Fed, ECB, OPEC, NATO, WTO, IMF, BRICS, ...);
      (c) ``market_term``    — its headline contains an explicitly
          market-relevant term (tariff, sanction, rate cut, oil price, ...).

    Why OR and not AND
    ------------------
    AND would be far more precise and badly wrong. The articles with the most
    predictive value are often the ones that do *not* yet mention the dollar:
    by the time a headline says "dollar rises on sanctions news", the move has
    already happened and there is nothing left to predict. Requiring a market
    term would therefore select for *post hoc* explanation articles and induce
    exactly the reverse-causality problem we must guard against (report §9.3).
    Condition (a) deliberately admits early, currency-agnostic coverage.

    Why a country list at all
    -------------------------
    Coverage is not importance. A cabinet crisis in a state with 0.05% of world
    trade generates headlines but no dollar flow. Excluding it is not a value
    judgement about the country; it is a statement about the transmission
    mechanism we hypothesised. We record the decision as a flag so the
    assumption is testable, not baked in.
    """
    mat = cfg.dig("materiality", default={}) or {}
    countries: Set[str] = {c.upper() for c in mat.get("actor_countries", [])}
    inst_re = _compile_terms(mat.get("institutions", []))
    term_re = _compile_terms(cfg.dig("materiality_terms", default=[]) or [])

    title = df["title"].fillna("").astype(str)
    src = df["sourcecountry"].fillna("").astype(str).str.upper()

    out = df.copy()
    out["has_actor_country"] = src.isin(countries)
    out["has_institution"] = title.str.contains(inst_re, na=False)
    out["has_market_term"] = title.str.contains(term_re, na=False)

    required = mat.get("require_any_of", ["actor_country", "institution", "market_term"])
    colmap = {
        "actor_country": "has_actor_country",
        "institution": "has_institution",
        "market_term": "has_market_term",
    }
    cols = [colmap[r] for r in required if r in colmap]
    out["pass_materiality"] = out[cols].any(axis=1) if cols else True

    log.info(
        "Stage 2 materiality gate: pass %d/%d (%.1f%%) | country %.1f%% institution %.1f%% market-term %.1f%%",
        int(out["pass_materiality"].sum()), len(out),
        100 * out["pass_materiality"].mean(),
        100 * out["has_actor_country"].mean(),
        100 * out["has_institution"].mean(),
        100 * out["has_market_term"].mean(),
    )
    return out


# ---------------------------------------------------------------------------
# Funnel driver + accounting
# ---------------------------------------------------------------------------
def run_funnel(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Apply Stage 0 and Stage 2 and compute the final ``keep`` column.

    Stage 1 (thematic) already happened at query time; Stage 3 (dedup) runs in
    ``dedup.py`` and is folded in here if its column is present.
    """
    out = apply_source_gate(df, cfg)
    out = apply_materiality_gate(out, cfg)
    if "is_duplicate" not in out.columns:
        out["is_duplicate"] = False
    out["keep"] = out["pass_source"] & out["pass_materiality"] & (~out["is_duplicate"])
    return out


def funnel_report(df: pd.DataFrame) -> pd.DataFrame:
    """The attrition table that goes straight into the report.

    Reviewers always ask "how much did you throw away, and why". This answers
    it in one table, stage by stage, with the survivors at each step.
    """
    n0 = len(df)
    stages = []
    surv = pd.Series(True, index=df.index)

    stages.append(("0. Retrieved from GDELT (post-thematic query)", n0, n0, 100.0))

    surv = surv & ~df.get("is_blocked_domain", pd.Series(False, index=df.index))
    stages.append(("0a. minus blocklisted domains", int((~surv).sum()), int(surv.sum()),
                   100 * surv.mean()))

    surv = surv & df.get("pass_source", pd.Series(True, index=df.index))
    stages.append(("0b. minus source-gate rejects", n0 - int(surv.sum()), int(surv.sum()),
                   100 * surv.mean()))

    prev = int(surv.sum())
    surv = surv & df.get("pass_materiality", pd.Series(True, index=df.index))
    stages.append(("2. minus non-material", prev - int(surv.sum()), int(surv.sum()),
                   100 * surv.mean()))

    prev = int(surv.sum())
    surv = surv & ~df.get("is_duplicate", pd.Series(False, index=df.index))
    stages.append(("3. minus duplicates", prev - int(surv.sum()), int(surv.sum()),
                   100 * surv.mean()))

    return pd.DataFrame(stages, columns=["stage", "removed", "surviving", "pct_of_raw"])


def family_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Corpus composition before vs after filtering, per theme family.

    A filter that silently wipes out one family (say, monetary_policy) has
    changed the research question without anyone noticing. This table is the
    check for that.
    """
    if "theme_family" not in df.columns:
        return pd.DataFrame()
    raw = df["theme_family"].value_counts()
    kept = df.loc[df.get("keep", True), "theme_family"].value_counts()
    out = pd.DataFrame({"raw": raw, "kept": kept}).fillna(0).astype(int)
    out["retention_%"] = (100 * out["kept"] / out["raw"].replace(0, pd.NA)).round(1)
    out["share_raw_%"] = (100 * out["raw"] / out["raw"].sum()).round(1)
    out["share_kept_%"] = (100 * out["kept"] / max(out["kept"].sum(), 1)).round(1)
    return out.sort_values("kept", ascending=False)


__all__ = [
    "apply_source_gate", "apply_materiality_gate", "run_funnel",
    "funnel_report", "family_balance",
]
