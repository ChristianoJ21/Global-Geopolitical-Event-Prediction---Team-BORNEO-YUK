#!/usr/bin/env python3
"""Generate notebook/task1/task1_data_acquisition.ipynb.

The notebook is generated rather than hand-edited so that its narrative stays
in sync with the modules, and so that code review happens on a readable .py
file instead of a JSON blob full of output.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebook" / "task1" / "task1_data_acquisition.ipynb"

nb = nbf.v4.new_notebook()
C: list = []
def md(s: str) -> None: C.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s: str) -> None: C.append(nbf.v4.new_code_cell(s.strip()))


# ===========================================================================
md(r"""
# Task 1 — Strategic Data Acquisition & Preprocessing

**Project:** Predicting Global Geopolitical Events — Impact on the USD Exchange Rate
**Hypothesis:** global geopolitical news carries significant, *predictive* information about USD exchange-rate fluctuations.

---

This notebook does not test the hypothesis. It builds the dataset that makes testing it possible — and, deliberately, makes a **false positive hard to reach**.

That second goal shapes almost everything below. It is easy to produce a pipeline that reports impressive accuracy: bin news by calendar date, shuffle the split, feed raw article counts. Each of those is a bug, each one *improves* the reported metric, and each one is invisible in the final number. So this notebook spends as much effort on guards as on acquisition.

### Route

| § | Step | Key decision |
|---|---|---|
| 1 | Configuration | every tunable is in YAML, nothing hard-coded |
| 2 | Target acquisition | USD/IDR from Bank Indonesia's JISDOR, not DXY |
| 3 | Target construction | log returns + ternary label with a dead-zone |
| 4 | News acquisition | GDELT theme codes, not keywords |
| 5 | Deduplication | collapse syndication, keep breadth as a feature |
| 6 | Filtering funnel | four stages, none of them destructive |
| 7 | Temporal alignment | WIB calendar day, paired with the next JISDOR fix |
| 8 | Preprocessing | two tracks, because preprocessing is model-dependent |
| 9 | Daily features | never a raw count |
| 10 | Assembly | the modelling table |
| 11 | Quality assurance | the checks that could still fail |
| 12 | Stage-4 human audit | the filter meets two human annotators |
| 13 | Handover to Task 2 | what is ready, what is not |
""")

md(r"""
> **Run modes.** Set `OFFLINE = True` to run the whole notebook on synthetic fixtures with no network access — useful for checking the code path in ~60 seconds before committing to a live fetch. Set `STRIDE = 7` for a fast live run that samples every 7th day.
""")

code(r"""
import sys, logging, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "config" / "config.yaml").exists())
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src import fx_data, gdelt, dedup, filtering, align, preprocess, features, qa
from src.http_cache import CachedSession

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(name)-12s | %(message)s",
                    stream=sys.stdout, force=True)

pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 60)
plt.rcParams.update({"figure.figsize": (11, 3.6), "axes.grid": True,
                     "grid.alpha": 0.25, "font.size": 9})

# ---- run mode -------------------------------------------------------------
OFFLINE = True     # True  -> synthetic fixtures, no network
STRIDE  = 7        # live runs: sample every N-th day (1 = full census)
SUB_DAY_CHUNKS = 1 # raise to 4 to relax the 250-record-per-call ceiling

# Offline runs are a SMOKE TEST of the code path, not a study. 400 sessions is
# enough to exercise every stage (including post-weekend windows and the
# rolling 90-day baselines) while keeping the notebook to about a minute.
# Set to None for the full synthetic span.
OFFLINE_DAYS = 400

cfg = load_config(ROOT / "config" / "config.yaml")
print(f"period      : {cfg.dig('period','start')} .. {cfg.dig('period','end')}")
print(f"target      : {cfg.dig('target','primary')}  ({cfg.dig('target','transform')})")
print(f"families    : {list(cfg.dig('theme_families').keys())}")
print(f"themes      : {len(cfg.all_themes)} GDELT theme codes")
print(f"seed        : {cfg.seed}   |   offline: {OFFLINE}")
""")


# ===========================================================================
md(r"""
---
## 1. Configuration as an artefact

Every threshold, country list, theme code and window length lives in `config/config.yaml` with its rationale written beside it as a comment. There are no magic numbers in the source.

This is not tidiness for its own sake. The project rules require us to justify every architectural decision under evaluation, and **a decision we cannot point at is a decision we cannot defend**. The YAML file is the thing we point at.
""")

code(r"""
print(open(ROOT / "config" / "config.yaml").read()[:2600], "\n... (truncated)")
""")


# ===========================================================================
md(r"""
---
## 2. The target — *which* dollar?

"The USD exchange rate" is under-specified, so the brief pins it down: daily USD rates for 1 September 2021 to 1 September 2026, with **Bank Indonesia as the ultimate source** (requirement 2b).

**Why not DXY or a broad basket.** The ICE Dollar Index is **57.6% euro**, so a model trained on it is largely a EUR/USD model. A broad basket has the same problem at a smaller scale. JISDOR is Indonesia's own reference rate, and it moves on Indonesia-specific news (BI policy, domestic politics, commodity exports) that a basket would dilute away.

**Acquisition.** `USD_IDR` is BI's official JISDOR export from bi.go.id, saved unmodified in `data/raw/`. It is downloaded in a browser because bi.go.id refuses scripted clients (see `src/fx_data.py`). Yahoo's `IDR=X` is fetched alongside it as `USD_IDR_YAHOO`, **only as an independent cross-check**, never as a model input.

**Controls are acquired here, not bolted on later.** DXY, VIX and Brent come from Yahoo's keyless chart endpoint in the same step. DXY asks whether a move is USD/IDR-specific or dollar-wide, VIX separates risk-off flight to the dollar from dollar-specific news, and Brent matters because Indonesia trades heavily in commodities. The honest version of our hypothesis is *"geopolitical news adds information beyond generic risk appetite and the dollar's own move"*, and without these columns we could not make that claim.

> In `OFFLINE` mode the panel below is synthetic: the `USD_IDR` level is JISDOR-scale, not the real rate.
""")

code(r"""
if OFFLINE:
    from src.synthetic import synthetic_fx_panel
    fx_panel = synthetic_fx_panel(cfg, n_days=OFFLINE_DAYS)
else:
    fx_panel = fx_data.build_fx_panel(cfg)

trading_days = align.trading_day_calendar(fx_panel, cfg)
print(f"\nFX panel     : {fx_panel.shape[0]} rows x {fx_panel.shape[1]} series")
print(f"trading days : {len(trading_days)}  ({trading_days.min().date()} .. {trading_days.max().date()})")
fx_panel.tail(3)
""")

md(r"""
The trading calendar is **derived from the target series' own observations**, not hard-coded. Indonesian holidays, when no JISDOR is fixed, are then handled correctly for free, and the calendar can never drift out of sync with the price series.
""")

code(r"""
fig, ax = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
for c in ["USD_IDR", "DXY"]:
    if c in fx_panel:
        ax[0].plot(fx_panel.index, fx_panel[c] / fx_panel[c].dropna().iloc[0] * 100, label=c, lw=1)
ax[0].set_ylabel("index (start = 100)"); ax[0].legend(loc="upper left")
ax[0].set_title("Dollar measures — rebased")

if "VIX" in fx_panel:
    ax[1].plot(fx_panel.index, fx_panel["VIX"], lw=0.8, color="crimson")
    ax[1].set_ylabel("VIX"); ax[1].set_title("Risk appetite control")
plt.tight_layout(); plt.show()
""")


# ===========================================================================
md(r"""
---
## 3. From levels to a learnable target

**Log returns, not levels.** An FX index *level* is a near unit-root process. A model fed levels scores a spectacular $R^2$ by predicting "tomorrow ≈ today" while learning nothing whatsoever about geopolitics. Differencing removes that trap. Log returns are also additive across time (a 5-day return is the sum of five daily returns) and scale-free, so DXY (~100) and USDIDR (~16,000) become directly comparable.

**A ternary label with a dead-zone.** Moves smaller than 10 bp are labelled `FLAT`. Without the dead-zone the label on a quiet day is `sign(noise)`, and the model burns its capacity learning microstructure it cannot possibly predict from news.

**Exactly-unchanged fixes are kept.** A stale-quote guard makes sense for a many-decimal index, where an identical value really does mean a repeated print. JISDOR is quoted in whole rupiah, so two identical consecutive fixes happen by chance (13 of 1,202 in 2021–2026). Dropping them would punch holes in the target for no gain (`alignment.drop_zero_return_days: false`).
""")

code(r"""
targets = fx_data.make_targets(fx_panel, cfg)
display(fx_data.describe_targets(targets, cfg).round(4))
""")

md(r"""
Two properties to check in that table, both of which are *reassuring when they look boring*:

- **lag-1 autocorrelation of $r_t$ near zero** — the return is not predictable from its own past. If it were, we would have a momentum strategy, not a news study, and news features would be competing against a trivial baseline.
- **lag-1 autocorrelation of $|r_t|$ clearly positive** — volatility clusters, exactly as FX theory predicts. This is why `y_abs_return` is a genuinely easier auxiliary target than direction, and why Task 2 should report both.
""")

code(r"""
r = targets["y_return"].dropna()
fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
ax[0].plot(r.index, r, lw=0.5); ax[0].set_title("Daily log return (%)")
ax[1].hist(r, bins=80, color="steelblue"); ax[1].set_title("Distribution (fat-tailed)")
vc = targets["y_direction"].value_counts()
ax[2].bar(vc.index.astype(str), vc.values, color=["seagreen", "indianred", "grey"])
ax[2].set_title("Direction label balance")
plt.tight_layout(); plt.show()
""")


# ===========================================================================
md(r"""
---
## 4. News acquisition — theme codes, not keywords

We tried the two realistic sources on the approved list, GDELT and CNBC, before choosing:

| Source | Coverage | Access | Verdict |
|---|---|---|---|
| **GDELT 2.0 GKG bulk files** | every English-language article GDELT processed, in 15-minute files | static files, keyless, no request quota | **primary** (`news_source: gkg`) |
| GDELT DOC 2.0 API | the same index, 250 records per call | per-IP quota; it blocked this project's IP for over a day in September 2026 | kept so earlier runs reproduce |
| CNBC | one newsroom | live search, about 37× slower in our timing; robots.txt disallows automated finance crawlers | rejected |

This notebook's live mode calls the DOC API (`src/gdelt.py`); the full build, `python -m src.build_dataset`, uses the GKG route (`src/gkg.py`) by default. Both select articles by the same theme codes.

### The decision that matters most: query by theme, not by keyword

A free-text keyword list drifts even over five years. "De-dollarisation" barely exists before 2022; "trade war" spikes in 2018 for reasons of *journalistic vocabulary* as much as of events. Retrieving on such a list silently makes a five-year corpus non-stationary and hands the model a spurious time trend that has nothing to do with geopolitics.

GDELT's theme taxonomy is machine-coded against a fixed codebook, so `ECON_SANCTIONS` means the same thing in 2021 and in 2026. Free-text market terms are still used — but *later*, in the Stage-2 precision gate, where they can be tuned without re-downloading anything.

### Why each theme family gets its own cap

We keep at most 250 articles per family per day. One shared cap would be filled by whichever family dominated the day, so on a heavy conflict-news day our monetary-policy articles would silently vanish. A cap per family makes the corpus composition a **design choice rather than an artefact of the cap**. In the GKG route the 250 are a seeded random draw from evenly spaced time-of-day slices, not the DOC API's "most recent 250", which biased every capped day toward late-evening coverage.
""")

code(r"""
for fam, themes in cfg.dig("theme_families").items():
    print(f"{fam:>17} :  {gdelt.build_theme_query(themes)[:110]}")
""")

code(r"""
if OFFLINE:
    from src.synthetic import synthetic_articles, synthetic_timelines
    articles  = synthetic_articles(cfg, trading_days)
    timelines = synthetic_timelines(cfg, trading_days)
else:
    sess = CachedSession(Path(cfg.path("cache")),
                         sleep=cfg.dig("gdelt", "sleep_seconds", default=1.2))
    articles  = gdelt.fetch_articles(cfg, session=sess, stride_days=STRIDE,
                                     sub_day_chunks=SUB_DAY_CHUNKS)
    timelines = gdelt.timelines_to_wide(gdelt.fetch_timelines(cfg, session=sess))

print(f"\narticles fetched: {len(articles):,}")
articles.head(3)[["title", "domain", "sourcecountry", "theme_family", "seendate_utc"]]
""")

md(r"""
**What we store, and why that is a legal decision as much as a technical one.** GDELT exposes headline, URL and metadata — never body text. We therefore redistribute *only* metadata and links, which keeps us clear of copyright, and we scope the NLP stack to headlines: Task 2 scores them with VADER, Loughran-McDonald and TF-IDF, with no long-document models. This is a constraint, but it is a *declared* constraint rather than a workaround.
""")


# ===========================================================================
md(r"""
---
## 5. Stage 3 — deduplication that keeps what it removes

A single Reuters story about an oil embargo appears in GDELT hundreds of times: mirrored by syndication partners, re-titled by aggregators, re-crawled under a different URL parameter.

Leave them in and three things break at once:

1. **"News volume" measures syndication reach**, not event importance — a quantity with no macroeconomic meaning that nonetheless trends upward over the years and is easily mistaken for signal.
2. **TF-IDF is poisoned** — a copied phrase looks like a strong repeated pattern rather than one observation.
3. **The train/test split leaks** — the same story lands on both sides.

But naive deletion throws away real information. *How widely* a story was copied is a genuine proxy for how important editors judged it to be. So we do not delete and forget: each cluster collapses to its **earliest** member (first-mover timing matters for an event study) and the cluster size survives as **`dup_count`, a feature in its own right**.

### Why SimHash and not sentence embeddings

Near-duplicate detection here is a *lexical* problem — the same sentence with a different outlet tag. SimHash over character 4-grams solves it in $O(n)$ with one 64-bit integer per document, runs on a laptop over a million headlines, and is fully deterministic, so a teammate re-running the pipeline gets byte-identical clusters.

A sentence-transformer would be slower, need a GPU, introduce a model dependency and a random seed — and, decisively, it would **over-merge**: *"Fed raises rates"* and *"Fed cuts rates"* are semantically close and are opposite events. Cheap and deterministic is the right engineering call.

All three levels apply a ±48 h window. Without it, a headline like *"Oil prices rise as tensions mount"* — which recurs verbatim every few months for years — would collapse into a single cluster, deleting a genuine repeated signal.
""")

code(r"""
articles = dedup.deduplicate(articles, cfg)
print(f"\nduplicate rate: {articles['is_duplicate'].mean():.1%}   "
      f"clusters: {articles['cluster_id'].nunique():,}")
display(dedup.dedup_report(articles, top=6)[["title", "dup_count"]])
""")

md(r"""
**A qualitative sanity check that is worth more than it looks.** If the most-syndicated clusters are recognisable major geopolitical events, the retrieval and dedup stack is behaving. If they are horoscopes or football results, it is not — and no summary statistic would have told us.
""")


# ===========================================================================
md(r"""
---
## 6. The filtering funnel

"Geopolitical news that moves the dollar" is neither a keyword nor a single theme. Any **single-stage** filter fails in one of two directions:

- broad (`theme:MILITARY`) floods the corpus with military-parade coverage and video-game reviews;
- narrow (`"dollar" AND "sanctions"`) returns only articles that *already mention the outcome we are trying to predict* — which makes the "prediction" a tautology.

So: four cheap, independently auditable stages. Each writes a boolean column rather than deleting rows, so Task 2 can ablate any stage and measure what it was worth.

### Stage 0 — source gate

GDELT indexes a long tail of content farms. We use **soft** mode: non-listed domains are kept but tagged `tier=3`. Hard deletion would destroy our ability to *test* whether the gate helped, and would bias coverage against non-Western outlets that simply are not on our (inevitably Anglophone) allowlist.

State-affiliated outlets (TASS, RT, Xinhua, Global Times, Press TV) are **kept on purpose** and flagged. They publish propaganda — but propaganda is itself a geopolitical signal, and a framing shift in TASS is information. Excluding them would leave us with only the Western narrative of every event, which is a bias, not a cleaning step.

### Stage 2 — materiality gate

An article passes if **any** of three conditions holds: a G20 / reserve-currency / pivotal actor country; a monetary, trade or security institution in the headline; or an explicit market term.

**Why OR and not AND.** AND would be far more precise and badly wrong. The articles with the most predictive value are often the ones that do *not* yet mention the dollar — by the time a headline reads *"dollar rises on sanctions news"*, the move has already happened and there is nothing left to predict. Requiring a market term would select for **post-hoc explanation articles** and manufacture exactly the reverse-causality problem we have to guard against.
""")

code(r"""
articles = filtering.run_funnel(articles, cfg)
display(filtering.funnel_report(articles))
""")

code(r"""
display(filtering.family_balance(articles))
""")

md(r"""
That second table is the check for a specific silent failure: **a filter that wipes out one theme family has changed the research question without anyone noticing.** If `monetary_policy` retention collapses while `conflict` survives intact, we are no longer testing the hypothesis we wrote down.
""")


# ===========================================================================
md(r"""
---
## 7. Temporal alignment — the section that decides whether any of this is valid

Almost every published failure of "news predicts markets" research is a misalignment bug, not a modelling bug. There are exactly three ways to get it wrong, and **all three make the backtest look better**, which is why they survive code review.

**1 · Same-day pairing.** Join news from day D to the rate fixed on day D. A story published at 20:00 WIB then sits next to that day's JISDOR, which BI had already fixed that afternoon, so the story "predicts" a number that existed before it did. Silent, devastating look-ahead. Taking the date in UTC instead of WIB makes it worse, because it moves late-UTC stories (much US news) onto the wrong day.

**2 · Weekend and holiday leakage.** JISDOR is not fixed on weekends or Indonesian holidays. Naive resampling either drops that news entirely — losing precisely the geopolitical events that governments time for a Friday night — or attaches it to a day with no rate.

**3 · Contemporaneous "prediction".** Using news from day $t$ to explain the return of day $t$ is a *nowcast*, not a forecast, and it cannot distinguish "news moved the dollar" from "the dollar moved, so journalists wrote about it".

### Our convention: a day-level rule in WIB

JISDOR is fixed once per Indonesian business day, but its publication time was not constant over the period. Rather than defend a different intraday cutoff for each regime, we use a rule that never needs to know the fixing time:

1. Convert GDELT's UTC timestamp to WIB (UTC+7, no daylight saving) **first**, then take the calendar date D.
2. Bucket the article on the last trading day on or before D, so Friday + Saturday + Sunday news share one bucket.
3. Shift every bucket forward one trading day (`predict_lag_days: 1`).

Net effect: news from WIB day D is paired with **the first JISDOR fix strictly after D**, and everything that piled up while the market was closed meets the same next fix. The fix a story is paired with was always set after the story existed, whichever fixing-time regime was in force.

We use GDELT's `seendate` (crawl time), not the article's self-declared publication date. `seendate` is weakly *later* than true publication, so any error it introduces is **conservative**: it can delay a story into a later window, never leak it into an earlier one.

Weekend news accumulates into one bucket that meets Monday's fix, and `window_hours` is exposed as a feature. Otherwise the model reads that mechanically larger bucket (about 72 h instead of 24 h) as a genuine news spike, every single week.
""")

code(r"""
articles = align.attach_news_day(articles, trading_days, cfg)
w = articles.groupby("news_day")["window_hours"].first()
print(f"window length: median {w.median():.0f}h   |   >30h (post-weekend): "
      f"{(w > 30).mean():.1%} of sessions")
articles[["title", "seendate_utc", "news_day", "window_hours", "is_post_weekend"]].head(4)
""")

md(r"""
This is asserted, not assumed. `tests/test_pipeline.py` checks the rule directly: 16:59 UTC on Tue 5 Mar 2024 (23:59 WIB) is Tuesday's news, while 18:30 UTC the same day (01:30 WIB Wed) is Wednesday's; Fri + Sat + Sun news shares one bucket and meets Monday's fix; and news during the Idul Fitri 2024 closure (8–12 Apr) meets the 15 Apr fix.
""")

code(r"""
import subprocess
print(subprocess.run([sys.executable, str(ROOT / "tests" / "test_pipeline.py")],
                     capture_output=True, text=True, cwd=str(ROOT)).stdout[-1400:])
""")


# ===========================================================================
md(r"""
---
## 8. Preprocessing — two tracks, because preprocessing is model-dependent

The reflex answer to "preprocess the text" is: lowercase, strip punctuation, remove stopwords, lemmatise. Applied to VADER, which reads capitals, "!" and "not", that reflex throws away exactly what it reads, and for a sentiment task it can be simply **wrong**.

**Failure 1 — negation destruction.**

> `"Fed will not cut rates"` → standard NLTK stopword removal → `"fed cut rates"`

The sentiment has inverted and the economic meaning is now the opposite of the source. Our Track-A configuration therefore carries an explicit **keep-list** of negators and directional words (`not`, `no`, `never`, `against`, `up`, `down`, …). It is the single highest-value line in the config file.

**Failure 2 — casing as information.** "US" (the country) versus "us" (the pronoun); "Fed" versus "fed". Lowercasing merges them.

So: one corpus, two **views** of it.

| | Track A — TF-IDF, Loughran-McDonald | Track B — VADER |
|---|---|---|
| unicode repair | ✔ | ✔ |
| strip outlet suffix | ✔ | ✔ |
| lowercase | ✔ | ✘ casing carries entity information and emphasis |
| remove stopwords | ✔ *with keep-list* | ✘ negators and function words carry meaning |
| lemmatise | ✔ | ✘ words stay as written |
| number masking | `<NUM>` / `<PCT>` / `<MONEY>` | ✘ |

Both tracks share one pre-step: **entity canonicalisation**. "U.S.", "US", "United States" and "Washington" become one symbol — otherwise the country with the most aliases looks like four rare entities instead of one dominant one.

**On number masking.** The exact figure "4.25%" is a hapax that TF-IDF cannot use, but *the presence of a percentage in a headline* is a strong signal that it is a rate or inflation story. Masking keeps the signal, drops the noise, and removes thousands of useless vocabulary types.
""")

code(r"""
display(preprocess.demo_negation_failure())
""")

code(r"""
articles = preprocess.preprocess_frame(articles, cfg)
display(preprocess.preprocessing_examples(articles, n=6))
""")


# ===========================================================================
md(r"""
---
## 9. Daily features — never a raw count

One principle governs every feature:

> **A raw article count is not a feature.**

GDELT's indexed source list expanded materially around 2018, and our own per-family daily cap truncates busy days. Both put non-stationary, non-economic trend into raw volume. A model fed raw counts will learn *"later years have more articles"* and call it signal.

So every volume feature is expressed **relatively**:

- **z-score against a trailing 30/90-day window** — "is today unusual *for this era*?"
- **share of the day's total** — composition, immune to level shifts
- **`articles_per_24h`** — corrects for the 65-hour Monday window
- **`log1p`** — tames the heavy right tail

Beyond volume:

- **`syndication_*`** — recovered from the dedup stage; the newsroom's own importance vote
- **`theme_entropy`, `geo_entropy`, `geo_hhi`** — *concentration*. Low entropy means the day's news is focused on one country or one theme (a single large event); high entropy means diffuse background chatter. One big shock moves markets; a hundred small stories do not.
- **`share_tier1`, `share_state_media`** — who is doing the reporting
- **`gdelt_volume_*`** — daily volume counted **before** the cap, our control for it

All rolling statistics are trailing and are computed *before* the alignment shift, so no future information can enter a feature. That is enforced by assertion, not assumed.
""")

code(r"""
feats = features.build_daily_features(articles, cfg, timeline_wide=timelines)
feats = features.reindex_to_trading_days(feats, trading_days)
print(f"daily features: {feats.shape[0]} days x {feats.shape[1]} columns")
display(features.feature_dictionary(feats).groupby("group")
        .agg(n_columns=("column", "size")).sort_values("n_columns", ascending=False))
""")

md(r"""
**Zero-fill, not forward-fill**, for genuinely newsless days. A day with no matching articles really had none. Forward-filling would invent yesterday's news flow and — worse — smear an event across days, blurring exactly the event-study timing the hypothesis depends on.
""")

code(r"""
fig, ax = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
ax[0].plot(feats.index, feats["articles_per_24h"], lw=0.7)
ax[0].set_title("Article flow per 24h (raw — NOT used directly as a feature)")
ax[1].plot(feats.index, feats["vol_z30"], lw=0.7, color="darkorange")
ax[1].axhline(0, color="k", lw=0.5); ax[1].axhline(2, color="r", lw=0.5, ls="--")
ax[1].set_title("vol_z30 — trailing 30-day z-score (this is what the model sees)")
plt.tight_layout(); plt.show()
""")


# ===========================================================================
md(r"""
---
## 10. Assembly — and the leak we actually found

The features are shifted forward by one session and joined to the targets.

**This step caught a real bug on the first run.** At the time the target was a broad dollar index, and the target frame contained its own return, `ret_BROAD_USD`, which is *numerically identical* to `y_return`. (With the JISDOR target the same column is `ret_USD_IDR`, and the same guard covers it.) Joined naively, it handed the model its own answer under a different name and produced a flawless, worthless $R^2$ — and the leakage assertions flagged it immediately:

```
LEAKAGE CHECK FAILED: features with |corr| > 0.95 vs target:
['ret_BROAD_USD', 'ret_AFE_USD', 'ret_DXY', 'ret_EURUSD', ...]
```

Market history is genuinely useful — momentum and volatility are standard controls, and Task 4 needs them to show that news adds information *beyond price itself*. But only from the past. So every market column is now lagged exactly like the news features and renamed with a `lag1_` prefix that makes its timing **visible in any coefficient table**.

That is the argument for keeping this check in the pipeline rather than in a code review: a reviewer reading a 79-column frame would not have spotted it.
""")

code(r"""
dataset = align.make_prediction_frame(feats, targets, cfg)
dataset = align.chronological_split(dataset, cfg)

print("\nsplit boundaries:")
for s in ("train", "val", "test"):
    ix = dataset.index[dataset["split"] == s]
    if len(ix):
        print(f"  {s:5s} {len(ix):5d} rows   {ix.min().date()} .. {ix.max().date()}")
""")

md(r"""
**Chronological splits, never shuffled.** A random split on autocorrelated data lets the model interpolate between a Monday and a Wednesday it has already seen to "predict" the Tuesday in between. The resulting accuracy is real and completely meaningless.
""")

code(r"""
problems = align.leakage_assertions(dataset, cfg)
print("LEAKAGE ASSERTIONS:", "ALL PASSED" if not problems else "FAILED")
for p in problems:
    print("  -", p)
""")


# ===========================================================================
md(r"""
---
## 11. Quality assurance

Every check below exists because of a specific, named way this dataset can lie to us. A QA suite that just prints `df.describe()` catches none of them.
""")

code(r"""
display(qa.validity_summary(articles, feats, targets, problems))
""")

code(r"""
display(qa.coverage_report(feats, trading_days))
""")

md(r"""
### The cap-saturation check, and why it is the subtlest one here

When a family hits its 250-article cap, we did **not** see that day in full — we kept 250 of its articles.

Saturation is *correlated with news intensity*, which is precisely our predictor. So the measurement error is **not random noise; it is systematic censoring that compresses exactly the big-news days the hypothesis cares about**. If saturation is common, the honest fixes are to raise the cap (`gkg.per_family_daily_cap`) or to lean on the volume counted before the cap. This table tells us which.
""")

code(r"""
sat = qa.cap_saturation_report(articles)
display(sat if not sat.empty else "no per-request accounting available")
""")

code(r"""
display(qa.structural_break_report(feats))
""")

md(r"""
Yearly median volume is the **GDELT crawler-expansion check**. If the medians jump by a factor unrelated to world events, raw counts are contaminated and only relative features may be used downstream. This table is the evidence for that decision — not a hunch about it.
""")

code(r"""
display(qa.fx_sanity_report(targets))
display(qa.corpus_quality_report(articles))
""")


# ===========================================================================
md(r"""
---
## 12. Stage 4 — the human audit

Stages 0–3 are **our opinion** about what a dollar-relevant geopolitical article is, expressed as code. The audit is where that opinion meets evidence.

Two annotators independently label a stratified sample of 300 articles drawn from **both sides** of the filter — auditing only what we kept would measure precision and be blind to recall, and a filter that discards half the real signal is exactly as broken as one that keeps garbage.

```bash
python scripts/make_audit_sample.py build
# ... two team members label reports/audit_annotator_{A,B}.csv independently ...
python scripts/make_audit_sample.py score --a reports/audit_annotator_A.csv \
                                          --b reports/audit_annotator_B.csv
```

Three numbers come out, and **the order in which we read them matters**:

1. **Cohen's $\kappa$ first.** If two humans cannot agree on what "material" means, the label is ill-defined and no precision figure computed from it means anything. Raw agreement will not do: two annotators who both answer "1" 90% of the time agree 82% of the time by luck alone.
2. **Precision** — of what the filter kept, what share is genuinely material?
3. **Recall** — of what it discarded, what share should have been kept?

Acceptance thresholds, fixed in the config *before* seeing any results: $\kappa \ge 0.60$, precision $\ge 0.85$. Stratification is by (year × theme family), because a simple random sample would be dominated by the largest family in the busiest years and would say nothing about whether the filter behaves **consistently across the five years** — which is the failure mode we are actually worried about.

The annotators never see the filter's own verdict. Showing it would anchor them and make the audit worthless.
""")


# ===========================================================================
md(r"""
---
## 13. Handover to Task 2

### Ready

- `data/processed/dataset.parquet` — one row per trading day, lagged news features + labels + chronological split
- `data/interim/articles_flagged.parquet` — article-level text in both preprocessing tracks, ready for TF-IDF and Loughran-McDonald (Track A) or VADER (Track B)
- Every filter decision preserved as a boolean column, so **any stage can be ablated** and its contribution measured
- 16 property tests covering alignment, leakage, negation handling and dedup

### Baselines Task 2 must beat

Any model has to outperform all three, or the result means nothing:

1. **Always predict the majority class** — surprisingly strong with a `FLAT` dead-zone
2. **Lagged market features only** (`lag1_*`) — no news at all; this is the honest test of whether news adds *anything*
3. **News volume z-score only** — one feature, no NLP; if an NLP model cannot beat it, the NLP is decorative

### Open threats, carried forward deliberately

| Threat | Status | Where it must be resolved |
|---|---|---|
| **Reverse causality** — FX moves cause news | mitigated by the OR-gate design, *not eliminated* | Task 4: lead-lag and Granger tests |
| English-source skew | flagged (`state_affiliated`, `sourcecountry`) | Task 3: multilingual ablation |
| 250-article daily cap per family | measured; volume counted before the cap | Task 2: raise the cap if saturation is high |
| Coverage ≠ market impact | unresolved by design | Task 4: interpret with care |
| Daily granularity | structural | acknowledge; no intraday causal claim is available |

### The honest statement of what this dataset can support

This pipeline can establish **daily-horizon predictability** and can test whether news features add information beyond price and generic risk appetite. It **cannot** establish intraday causality, and it cannot rule out that a portion of any measured relationship runs from markets to news rather than the other way round.

Saying so now is not hedging. It is what stops Task 4 from overclaiming — and a Task 4 that overclaims is the most likely way this project goes wrong.
""")

nb["cells"] = C
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT}  ({len(C)} cells)")
