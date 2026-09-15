"""Text preprocessing — two tracks over one corpus (report §6).

The central argument
--------------------
The reflex answer to "preprocess the text" is: lowercase, strip punctuation,
remove stopwords, lemmatise. Applied to a transformer pipeline that reflex is
actively harmful, and applied to a sentiment task it is sometimes *wrong*.

Two concrete failures we are protecting against:

  * **Negation destruction.**  "Fed will not cut rates" with standard NLTK
    stopword removal becomes "fed cut rates". The sentiment has inverted and
    the economic meaning is now the opposite of the source. Our Track-A
    stopword list therefore carries an explicit *keep-list* of negators and
    directional words (``not``, ``no``, ``never``, ``against``, ``up``,
    ``down``, ...). This is the single highest-value line in the config.

  * **Casing as information.**  "US" (the country) vs "us" (the pronoun);
    "Fed" vs "fed". Lowercasing merges them. A subword transformer uses that
    casing, so Track B does not lowercase at all.

So: one cleaned corpus, two *views* of it.

    Track A  ->  TF-IDF, LDA, VADER, classical ML
                 aggressive: lowercase, stopwords, lemmatise, number masking
    Track B  ->  FinBERT / DeBERTa / mBERT
                 minimal: unicode repair, boilerplate strip, whitespace only

Both tracks share one *pre*-step — entity canonicalisation — because
"U.S.", "US", "United States" and "Washington" must be one symbol under every
model, otherwise the country with the most aliases looks like four rare
entities instead of one dominant one.

NLTK is optional. If it is unavailable (or its download is blocked on a lab
machine) the module degrades to a bundled stopword list and a suffix-stripping
lemmatiser, so the pipeline still runs end-to-end. Graceful degradation is
deliberate: a preprocessing step that only works on one team member's laptop
is a reproducibility bug.
"""
from __future__ import annotations

import html
import logging
import re
import unicodedata
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Optional dependencies
# --------------------------------------------------------------------------
try:
    import ftfy  # repairs mojibake ("â€™" -> "'"), common in scraped feeds
    _HAS_FTFY = True
except ImportError:  # pragma: no cover
    _HAS_FTFY = False

try:
    from nltk.corpus import stopwords as _nltk_sw
    from nltk.stem import WordNetLemmatizer as _WNL
    _HAS_NLTK = True
except ImportError:  # pragma: no cover
    _HAS_NLTK = False


# Minimal fallback list, used only if NLTK is unavailable.
_FALLBACK_STOPWORDS = set("""
a an the and or but if while of to in on at by for with from as is are was were
be been being do does did doing have has had having this that these those it its
i you he she they we them his her their our your my me him us s t will would can
could should may might must shall there here what which who whom whose when where
why how all any both each other some such only own same so than too very just
""".split())

_URL = re.compile(r"https?://\S+|www\.\S+")
_HTML_TAG = re.compile(r"<[^>]+>")
_OUTLET_SUFFIX = re.compile(r"\s*[\-–—|·]\s*[A-Z][\w .&'’]{2,30}\s*$")
# NB: the trailing boundary must be inside the alternation. A plain `\b` after
# the group never matches the `%` form, because `%` is not a word character and
# the following space gives no word boundary — so "4.5%" silently fell through
# to the generic <NUM> rule. Caught by tests/test_pipeline.py.
_PCT = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:%|(?:percent|per cent|pct)\b)", re.IGNORECASE
)
_MONEY = re.compile(r"[$€£¥₹]\s?\d[\d.,]*\s?(?:bn|billion|mn|million|trillion|tn|k)?", re.IGNORECASE)
_NUM = re.compile(r"\b\d[\d.,]*\b")
_MULTISPACE = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s<>]")

CONTRACTIONS: Dict[str, str] = {
    "won't": "will not", "can't": "cannot", "n't": " not", "'re": " are",
    "'s": " is", "'d": " would", "'ll": " will", "'ve": " have", "'m": " am",
    "shan't": "shall not", "let's": "let us",
}


# --------------------------------------------------------------------------
# Shared pre-step
# --------------------------------------------------------------------------
def fix_unicode(text: str) -> str:
    """Repair mojibake, unescape HTML entities, NFKC-normalise."""
    if not isinstance(text, str):
        return ""
    t = html.unescape(text)
    if _HAS_FTFY:
        t = ftfy.fix_text(t)
    t = unicodedata.normalize("NFKC", t)
    # Curly quotes and dashes -> ASCII, so "don't" and "don’t" tokenise alike.
    for a, b in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'), ("–", "-"), ("—", "-")):
        t = t.replace(a, b)
    return t


def strip_boilerplate(text: str) -> str:
    """Remove URLs, HTML tags and the trailing outlet name."""
    t = _HTML_TAG.sub(" ", text)
    t = _URL.sub(" ", t)
    t = _OUTLET_SUFFIX.sub("", t.strip())
    return _MULTISPACE.sub(" ", t).strip()


@lru_cache(maxsize=1)
def _alias_patterns_cache_key() -> None:  # pragma: no cover - cache sentinel
    return None


def build_entity_patterns(aliases: Dict[str, Sequence[str]]) -> List[tuple]:
    """Compile alias -> canonical replacements, longest alias first.

    Longest-first matters: without it, "united states" would be partially
    consumed by the shorter alias "us" and produce "UNITED_STATES tates".
    """
    pairs = []
    for canon, alias_list in (aliases or {}).items():
        for a in sorted(alias_list, key=len, reverse=True):
            pat = re.compile(r"\b" + re.escape(a).replace(r"\ ", r"\s+") + r"\b", re.IGNORECASE)
            pairs.append((pat, canon))
    pairs.sort(key=lambda p: -len(p[0].pattern))
    return pairs


def canonicalize_entities(text: str, patterns: List[tuple]) -> str:
    for pat, canon in patterns:
        text = pat.sub(canon, text)
    return text


# --------------------------------------------------------------------------
# Track A — aggressive, for sparse / count-based models
# --------------------------------------------------------------------------
@lru_cache(maxsize=4)
def _stopword_set(keeplist: tuple) -> set:
    if _HAS_NLTK:
        try:
            base = set(_nltk_sw.words("english"))
        except LookupError:
            log.warning("NLTK stopwords corpus not downloaded; using fallback list. "
                        "Run: python -m nltk.downloader stopwords wordnet omw-1.4")
            base = set(_FALLBACK_STOPWORDS)
    else:
        base = set(_FALLBACK_STOPWORDS)
    # The keep-list is the whole point: negators and directional words survive.
    return base - set(keeplist)


class _SimpleLemmatizer:
    """Suffix-stripping fallback when WordNet is unavailable."""

    _RULES = (("ies", "y"), ("sses", "ss"), ("ches", "ch"), ("shes", "sh"),
              ("xes", "x"), ("ing", ""), ("ed", ""), ("s", ""))

    def lemmatize(self, w: str) -> str:  # noqa: D102
        for suf, rep in self._RULES:
            if len(w) > len(suf) + 2 and w.endswith(suf):
                return w[: -len(suf)] + rep
        return w


@lru_cache(maxsize=1)
def _lemmatizer():
    if _HAS_NLTK:
        try:
            wnl = _WNL()
            wnl.lemmatize("tests")  # forces the WordNet load; raises if missing
            return wnl
        except Exception:  # LookupError or similar
            log.warning("WordNet unavailable; using the suffix-rule lemmatizer fallback")
    return _SimpleLemmatizer()


def expand_contractions(text: str) -> str:
    for k, v in CONTRACTIONS.items():
        text = re.sub(re.escape(k), v, text, flags=re.IGNORECASE)
    return text


def preprocess_track_a(
    text: str,
    cfg_a: dict,
    entity_patterns: Optional[List[tuple]] = None,
) -> str:
    """Aggressive cleaning for TF-IDF / LDA / VADER / classical ML."""
    t = fix_unicode(text) if cfg_a.get("unicode_fix", True) else str(text or "")
    if cfg_a.get("strip_outlet_suffix", True):
        t = strip_boilerplate(t)
    if entity_patterns:
        t = canonicalize_entities(t, entity_patterns)
    if cfg_a.get("expand_contractions", True):
        t = expand_contractions(t)

    # Numbers become typed placeholders BEFORE lowercasing/punctuation removal.
    # Rationale: the exact figure ("4.25%") is a hapax that TF-IDF cannot use,
    # but the *presence of a percentage* in a headline is a strong signal that
    # it is a rate/inflation story. Masking keeps the signal, drops the noise,
    # and shrinks the vocabulary by thousands of useless types.
    if cfg_a.get("number_placeholder", True):
        t = _PCT.sub(" <PCT> ", t)
        t = _MONEY.sub(" <MONEY> ", t)
        t = _NUM.sub(" <NUM> ", t)

    if cfg_a.get("lowercase", True):
        t = t.lower()
    t = _PUNCT.sub(" ", t)
    t = _MULTISPACE.sub(" ", t).strip()

    tokens = t.split()
    if cfg_a.get("remove_stopwords", True):
        keep = tuple(cfg_a.get("stopword_keeplist", []) or [])
        sw = _stopword_set(keep)
        tokens = [w for w in tokens if w not in sw or w.startswith("<")]
    if cfg_a.get("lemmatize", True):
        lem = _lemmatizer()
        tokens = [w if w.startswith("<") else lem.lemmatize(w) for w in tokens]

    tokens = [w for w in tokens if len(w) > 1 or w in {"<"}]
    if len(tokens) < int(cfg_a.get("min_tokens", 3)):
        return ""
    return " ".join(tokens)


# --------------------------------------------------------------------------
# Track B — minimal, for transformers
# --------------------------------------------------------------------------
def preprocess_track_b(
    text: str,
    cfg_b: dict,
    entity_patterns: Optional[List[tuple]] = None,
) -> str:
    """Minimal cleaning for FinBERT / DeBERTa / mBERT.

    We repair encoding and strip the outlet tag — both are noise no model can
    use — and stop there. No lowercasing, no stopword removal, no lemmatising.
    A subword tokeniser was pre-trained on ordinary cased prose with function
    words intact; feeding it a stemmed bag of content words moves the input
    off-distribution and reliably *costs* several points of F1.

    Entity canonicalisation is optional here and off by default in spirit: it
    helps count-based models but can fragment a transformer's tokenisation of
    a familiar name. We expose it so Task 2 can ablate the choice.
    """
    t = fix_unicode(text) if cfg_b.get("unicode_fix", True) else str(text or "")
    if cfg_b.get("strip_outlet_suffix", True):
        t = strip_boilerplate(t)
    if entity_patterns and cfg_b.get("canonicalize_entities", False):
        t = canonicalize_entities(t, entity_patterns)
    t = _MULTISPACE.sub(" ", t).strip()

    # Whitespace-word truncation is a safe over-estimate of the subword budget
    # (subwords >= words), so the real tokeniser never has to truncate mid-entity.
    max_tok = int(cfg_b.get("max_tokens", 128))
    words = t.split()
    if len(words) > max_tok:
        t = " ".join(words[:max_tok])
    return t


# --------------------------------------------------------------------------
# Frame-level driver
# --------------------------------------------------------------------------
def preprocess_frame(df: pd.DataFrame, cfg, text_col: str = "title") -> pd.DataFrame:
    """Add ``text_track_a`` and ``text_track_b`` columns to an article frame."""
    pcfg = cfg.dig("preprocess", default={}) or {}
    a_cfg = pcfg.get("track_a", {}) or {}
    b_cfg = pcfg.get("track_b", {}) or {}
    patterns = build_entity_patterns(cfg.dig("entity_aliases", default={}) or {})

    out = df.copy()
    src = out[text_col].fillna("").astype(str)

    out["text_track_a"] = [preprocess_track_a(t, a_cfg, patterns) for t in src]
    out["text_track_b"] = [preprocess_track_b(t, b_cfg, patterns) for t in src]
    out["n_tokens_a"] = out["text_track_a"].str.split().str.len().fillna(0).astype(int)
    out["n_tokens_b"] = out["text_track_b"].str.split().str.len().fillna(0).astype(int)

    empty_a = int((out["text_track_a"] == "").sum())
    log.info(
        "preprocess: %d rows | track A median %d tokens (%d emptied by min_tokens) | track B median %d tokens",
        len(out), int(out["n_tokens_a"].median() or 0), empty_a,
        int(out["n_tokens_b"].median() or 0),
    )
    return out


def preprocessing_examples(df: pd.DataFrame, n: int = 8, seed: int = 42) -> pd.DataFrame:
    """Side-by-side before/after table for the report.

    Showing real examples is how we demonstrate that the negation keep-list is
    doing its job — an assertion nobody has to take on trust.
    """
    cols = [c for c in ("title", "text_track_a", "text_track_b") if c in df.columns]
    if not cols or df.empty:
        return pd.DataFrame()
    return df.sample(min(n, len(df)), random_state=seed)[cols].reset_index(drop=True)


def demo_negation_failure() -> pd.DataFrame:
    """The worked example that justifies ``stopword_keeplist`` (report §6.2)."""
    naive_sw = _FALLBACK_STOPWORDS | {"not", "no", "never", "against", "up", "down"}
    rows = []
    for s in [
        "Fed will not cut rates in December",
        "Sanctions are not expected to be lifted",
        "China says it will never devalue the yuan",
        "Oil prices down as OPEC talks collapse",
    ]:
        toks = s.lower().split()
        rows.append({
            "original": s,
            "naive_stopword_removal": " ".join(w for w in toks if w not in naive_sw),
            "ours_with_keeplist": " ".join(
                w for w in toks if w not in (naive_sw - {"not", "no", "never", "against", "up", "down"})
            ),
        })
    return pd.DataFrame(rows)


__all__ = [
    "preprocess_frame", "preprocess_track_a", "preprocess_track_b",
    "fix_unicode", "strip_boilerplate", "build_entity_patterns",
    "canonicalize_entities", "preprocessing_examples", "demo_negation_failure",
]
