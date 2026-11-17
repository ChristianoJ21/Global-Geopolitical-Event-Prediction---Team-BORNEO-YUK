"""Task 3 — entity-level filtering with a transformer NER model.

Task 1 already flags headlines that mention a country or an institution, but
with keyword lists. Here a real named-entity recogniser (dslim/bert-base-NER,
BERT-base fine-tuned on CoNLL-2003: PER / ORG / LOC / MISC) reads each headline
in context, so "Turkey prices" (food) and "Turkey sanctions" (country) are not
treated alike, and leaders ("Powell", "Putin") count for their institution.

The recognised entities are mapped onto a small watchlist of the actors that
can plausibly move USD/IDR: the US and the Fed, Indonesia and Bank Indonesia,
China, the Russia-Ukraine theatre, the Middle East, and central banks in
general. Per day we then compute, for each group, the share of headlines that
mention it and FinBERT's net tone inside those headlines ("is Fed news getting
more negative?"). That is the entity-filtered view of the news the brief asks
for (1b-iii).

Cost. NER runs at ~200 headlines/s on an M3 GPU, so it is applied to the 100
most-syndicated headlines of each news day (119k headlines, ~10 min) rather
than all 905k. Those are the stories most readers saw.
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from .nlp_features import article_weights
from .text_embeddings import MODELS, cache_dir, device

log = logging.getLogger(__name__)

# Group -> aliases. Matched case-insensitively against recognised entity spans.
WATCHLIST = {
    "us": ["united states", "u.s.", "us", "usa", "america", "washington", "white house",
           "biden", "trump", "congress", "pentagon", "wall street", "senate"],
    "fed": ["fed", "federal reserve", "powell", "fomc", "yellen", "treasury"],
    "indonesia": ["indonesia", "jakarta", "bank indonesia", "jokowi", "widodo", "prabowo",
                  "sri mulyani", "ojk", "bali"],
    "china": ["china", "beijing", "chinese", "xi", "xi jinping", "pboc", "people's bank of china",
              "hong kong", "taiwan"],
    "russia_ukraine": ["russia", "russian", "moscow", "kremlin", "putin", "ukraine", "ukrainian",
                       "kyiv", "kiev", "zelensky", "zelenskiy", "nato"],
    "middle_east": ["israel", "israeli", "gaza", "hamas", "iran", "iranian", "tehran", "hezbollah",
                    "saudi", "saudi arabia", "opec", "yemen", "houthi", "houthis", "lebanon", "syria"],
    "central_banks": ["fed", "federal reserve", "fomc", "powell", "ecb", "european central bank",
                      "lagarde", "bank of japan", "boj", "bank of england", "boe", "pboc",
                      "people's bank of china", "bank indonesia", "imf", "world bank"],
}
GROUPS = list(WATCHLIST)
_PATTERNS = {g: re.compile(r"\b(" + "|".join(re.escape(a) for a in sorted(al, key=len, reverse=True))
                           + r")\b", re.I) for g, al in WATCHLIST.items()}


def top_k_per_day(articles: pd.DataFrame, k: int) -> np.ndarray:
    """Row positions of the k most-syndicated headlines of each news day.

    Ties on dup_count keep the corpus order (stable sort), so the choice is
    deterministic.
    """
    order = (articles.reset_index(drop=True)
             .assign(_row=np.arange(len(articles)))
             .sort_values(["news_day", "dup_count"], ascending=[True, False], kind="stable"))
    return np.sort(order.groupby("news_day").head(k)["_row"].to_numpy())


def match_groups(entity_texts: list[str]) -> dict:
    joined = " | ".join(entity_texts)
    return {f"ent_{g}": bool(p.search(joined)) for g, p in _PATTERNS.items()}


def run_ner(cfg, articles: pd.DataFrame, use_cache: bool = True,
            batch_size: int = 128, min_score: float = 0.6) -> pd.DataFrame:
    """NER on the top-k headlines per day; returns one row per processed headline."""
    k = int(cfg.dig("task3", "ner", "top_k_per_day", default=100))
    path = cache_dir(cfg) / f"ner_top{k}.parquet"
    rows = top_k_per_day(articles, k)
    if use_cache and path.exists():
        cached = pd.read_parquet(path)
        if np.array_equal(cached["row"].to_numpy(), rows):
            log.info("NER: reusing cache %s", path)
            return cached
        log.warning("NER cache does not match the corpus; re-running")

    from transformers import pipeline
    ner = pipeline("ner", model=MODELS["ner"], aggregation_strategy="simple", device=device())
    texts = articles["text_track_b"].to_numpy()[rows].tolist()
    log.info("NER: %d headlines (top %d per day)", len(texts), k)
    out = []
    for s in range(0, len(texts), 4096):
        for spans in ner(texts[s:s + 4096], batch_size=batch_size):
            ents = [f"{e['entity_group']}:{e['word']}" for e in spans
                    if e["score"] >= min_score and not e["word"].startswith("##")]
            out.append({"entities": "|".join(ents),
                        **match_groups([e.split(":", 1)[1] for e in ents])})
        log.info("NER: %d / %d", min(s + 4096, len(texts)), len(texts))
    res = pd.DataFrame(out)
    res.insert(0, "row", rows)
    res.to_parquet(path)
    return res


def entity_block(articles: pd.DataFrame, ner: pd.DataFrame, probs: pd.DataFrame, cfg) -> pd.DataFrame:
    """Per day and watchlist group: share of top headlines mentioning it, and their FinBERT tone."""
    sub = articles.iloc[ner["row"].to_numpy()]
    w = article_weights(sub, cfg).to_numpy()
    day = sub["news_day"].to_numpy()
    p = probs.iloc[ner["row"].to_numpy()]
    net = p["finbert_positive"].to_numpy() - p["finbert_negative"].to_numpy()
    frame = pd.DataFrame({"day": day, "w": w, "net": net})
    tot = frame.groupby("day")["w"].sum()
    out = {}
    any_flag = np.zeros(len(ner), dtype=bool)
    for g in GROUPS:
        flag = ner[f"ent_{g}"].to_numpy()
        any_flag |= flag
        f = frame.loc[flag]
        out[f"t3_ent_share_{g}"] = (f.groupby("day")["w"].sum() / tot).reindex(tot.index).fillna(0.0)
        out[f"t3_ent_tone_{g}"] = (f["w"] * f["net"]).groupby(f["day"]).sum() / f.groupby("day")["w"].sum()
    f = frame.loc[any_flag]
    out["t3_ent_share_any"] = (f.groupby("day")["w"].sum() / tot).reindex(tot.index).fillna(0.0)
    out["t3_ent_tone_any"] = (f["w"] * f["net"]).groupby(f["day"]).sum() / f.groupby("day")["w"].sum()
    res = pd.DataFrame(out)
    res.index = pd.DatetimeIndex(res.index, name="date")
    return res.sort_index()


__all__ = ["WATCHLIST", "GROUPS", "top_k_per_day", "match_groups", "run_ner", "entity_block"]
