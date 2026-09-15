# Predicting Global Geopolitical Events: Impact on the USD/IDR Exchange Rate
### Data Acquisition & Strategic Preprocessing

**Hypothesis under test.** Global geopolitical news carries significant information about, and is useful for predicting, fluctuations in the US dollar exchange rate — measured here as USD/IDR, Bank Indonesia's JISDOR reference rate.

Task 1 does not test that hypothesis. It builds the dataset that makes testing it *possible*, and makes a false positive *hard*. Every design choice is recorded so it can be defended, ablated, or overturned.

Period: **1 September 2021 – 1 September 2026**, as the assignment requires.

---

## Quick start

```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet omw-1.4    # optional, improves Track A

# 0. One manual input, already in the repo: Bank Indonesia's JISDOR export.
#    https://www.bi.go.id/id/statistik/informasi-kurs/jisdor/default.aspx
#    period 01/09/2021 - 01/09/2026 -> "Unduh" -> data/raw/Informasi Kurs Jisdor.xlsx

# 1. Build the dataset (GDELT bulk files; cached and resumable, ~30 min)
python -m src.build_dataset
```

`python -m src.build_dataset --offline` runs the whole pipeline on synthetic data with no network. It overwrites `data/processed/` and marks the manifest `offline_synthetic: true`, so run step 1 again afterwards.

No API keys are required.

---

## Data sources

| Stream | Source | How it is acquired |
|---|---|---|
| **USD/IDR target** | Bank Indonesia, JISDOR | Official export from bi.go.id, parsed by `src/fx_data.py`. Downloaded in a browser: bi.go.id resets scripted connections and ignores scripted form submissions. |
| Cross-check and controls | Yahoo Finance: `IDR=X`, DXY, VIX, Brent | Public chart endpoint, cached |
| **News** | GDELT Project, GKG 2.1 bulk files | 4 of the 96 daily 15-minute files, filtered by theme on download, `src/gkg.py` |

---

## What gets built

| Output | Grain | Contents | On GitHub |
|---|---|---|---|
| `data/raw/Informasi Kurs Jisdor.xlsx` | trading day | BI's JISDOR export, unmodified | yes |
| `data/raw/fx_panel.parquet` | day | JISDOR, Yahoo IDR=X cross-check, DXY, VIX, Brent | yes |
| `data/raw/articles_raw_sample.csv` | article | 2,000-row random sample of the raw news | yes |
| `data/raw/articles_raw.parquet` | article | every article collected (1,896,380), unfiltered | no (> 100 MB) |
| `data/interim/articles_flagged.parquet` | article | + every filter flag, dedup cluster, both text tracks | no (> 100 MB) |
| `data/processed/articles_flagged_sample.csv` | article | 4,968-row sample of the file above, stratified by year × theme family (70% kept / 30% dropped) | yes |
| `data/processed/daily_features.parquet` | trading day | news features on the JISDOR calendar | yes |
| `data/processed/targets.parquet` | trading day | returns, direction labels, multi-horizon targets | yes |
| **`data/processed/dataset.parquet`** | trading day | **the final aligned dataset**: 1,202 JISDOR days, lagged news features + labels + split | yes |
| `data/processed/dataset_aligned.csv` | trading day | the same table as CSV, for Excel | yes |
| `data/processed/manifest.json` | — | run provenance: period, counts, seed, leakage verdict | yes |
| `reports/*.csv` | — | QA tables (funnel, coverage, validity), written by every build | no (rebuilt) |

The files marked "no" and the `data/cache/` folder are not in the repo: GitHub rejects files over 100 MB. `python -m src.build_dataset` rebuilds them.

**Reading the article flags** (`articles_flagged_sample.csv`): `pass_source` = stage 0 (not a blocklisted aggregator); `theme_family` = stage 1; `has_actor_country` / `has_institution` / `has_market_term` are OR-ed into `pass_materiality` = stage 2; `is_duplicate`, `cluster_id`, `dup_count` = stage 3; `keep` = passed every stage (only these rows feed the features); `news_day` = the JISDOR trading day the article is grouped with; `text_track_a` / `text_track_b` = the two preprocessing tracks.

---

##

```
  BI JISDOR export ─► fx_data (parse, validate) ─► targets (log returns, UP / FLAT / DOWN)
  Yahoo controls  ─┘          │
                              └─► trading calendar = days BI published JISDOR ─┐
                                                                               │
  GDELT GKG files ─► gkg (theme filter, cache) ─► dedup ─► filtering ─► align (WIB day rule)
                                                  Stage 3   Stages 0, 2        │
                                                                               ▼
                                                                preprocess (Track A + B)
                                                                               │
                                                                               ▼
                                                     features ──► dataset.parquet ──► QA
```

A drawn version is in the team's report.

---

## The six decisions that define this dataset

**1. The target is Bank Indonesia's JISDOR rate.**
The assignment names Bank Indonesia as the source, so the target is USD/IDR, not a broad dollar index or DXY. That also fits the question: the rupiah moves on Indonesia-specific news (BI policy, domestic politics, commodity exports) that a broad index would average away. Yahoo's `IDR=X` is kept only as a cross-check. Its levels agree with JISDOR closely, but its daily moves correlate only 0.53 with JISDOR's, because it closes at a different hour from BI's afternoon fix (15:15 or 16:15 WIB). So it cannot stand in for BI's rate.

**2. Retrieval is by GDELT theme code, never by keyword.**
A keyword list drifts over five years as words go in and out of fashion, which would give the model a spurious time trend. GDELT's theme taxonomy is machine-coded against a fixed codebook, so `ECON_SANCTIONS` means the same thing in 2021 and 2026. There are 30 codes in six families: conflict, sanctions & trade, diplomacy, energy, monetary policy, and political risk.

**3. Filtering is a four-stage funnel, and no stage deletes anything.**
Each stage writes a yes/no column, so Task 2 can switch any stage off and measure what it was worth. Every build writes the attrition table to `reports/funnel_attrition.csv`.

| Stage | Purpose | Column |
|---|---|---|
| 0 · source | drop aggregators and content farms; tier the wires | `pass_source`, `source_tier` |
| 1 · topic | recall: applied when the files are read | `theme_family` |
| 2 · relevance | precision: does it touch a USD/IDR channel? | `pass_materiality` |
| 3 · duplicates | collapse syndication, keep breadth as a feature | `is_duplicate`, `dup_count` |

Stage 2 is global plus Indonesia-specific. It checks for key countries (including Indonesia and its ASEAN neighbours), key institutions (including Bank Indonesia, OJK and ASEAN), and market terms (including rupiah, JISDOR, nickel, coal, palm oil and export bans). It uses OR, not AND. Requiring a market term would keep only articles written after the currency had already moved.

**4. Deduplication keeps what it removes.**
One wire story appears in GDELT hundreds of times. Counting every copy would make "news volume" a measure of syndication reach, not of event importance. But how widely a story was copied is newsrooms voting on importance, so each cluster collapses to its earliest member and the cluster size survives as `dup_count`. Matching runs at three levels (URL, then normalised title, then SimHash over character 4-grams) within ±48 hours.

**5. Alignment is a day-level rule in WIB.**
BI fixes JISDOR once per business day, and the publication time has not been constant (moved from 10:00 to 16:15 WIB on 5 April 2021; 15:15 WIB under shortened market hours), so there is no single intraday cutoff to use. Instead:
- GDELT's UTC timestamp is converted to WIB **first**, then the calendar date D is taken.
- News from D is paired with the **first JISDOR fix after D**.
- Weekend and holiday news joins the preceding trading day's group and is paired with the same next fix.

This works whatever the fixing time was and makes lookahead impossible. The trading calendar is JISDOR's own, so Indonesian holidays are handled with no hand-made list. For example, news from Friday 8 March 2024 and that weekend is paired with the fix on Wednesday 13 March, because BI did not publish on 11–12 March (Nyepi). `window_hours` records how many hours of news each group holds, so the model does not read the larger weekend group as a news spike. `src/align.py` has the full argument.

**6. Preprocessing is dual-track, because preprocessing is model-dependent.**
One corpus, two views. Track A (for the Task 2 baselines: TF-IDF, LDA, VADER) lowercases, removes stopwords, lemmatises, and masks numbers as `<NUM>` / `<PCT>`. Track B (for Task 3 transformers such as FinBERT) keeps the headline almost as written, because those models were trained on cased text with function words intact.

Track A's stopword list has an explicit **keep-list** of negators and direction words. Without it:

> `"Fed will not cut rates"` → `"fed cut rates"`, and the meaning has flipped.

---

## Why GDELT's bulk files and not its API

The GDELT DOC 2.0 API allows one request every five seconds per IP. The full period needs thousands of requests, and exceeding the limit got this project's IP blocked for more than a day. GDELT publishes the same index as static 15-minute GKG files and recommends them for heavy use. `src/gkg.py` downloads four files per day, keeps only the articles tagged with the 30 theme codes, and caches those rows, so a rerun downloads nothing new. It takes at most 250 articles per family per day, a random draw with a fixed seed, rather than "the latest 250". The API route is still in `src/gdelt.py` (`--source doc`).

---

## Guards against fooling ourselves

- **Leakage assertions** (`src/align.py`) run on every build. They fail loudly on contemporaneous features, unsorted or duplicated dates, overlapping splits, or any feature correlating above 0.95 with the target. That check caught a real leak on the first run: the raw return of the target series was identical to `y_return`. So market history enters only as `lag1_*` columns.
- **Chronological splits, never shuffled.** Train up to 31 Aug 2024, validation up to 31 Aug 2025, test after. A random split on autocorrelated data lets the model interpolate between days it has already seen.
- **No raw counts as features.** Volumes are z-scored against a trailing window or expressed as shares, and `window_hours` adjusts for bigger weekend groups.
- **Reverse causality is named, not hidden.** FX moves cause news as well as the reverse. Stage 2 admits early, currency-agnostic coverage on purpose, and Task 4 must still run lead-lag and Granger tests.

---

## Known limitations

1. **English-language sources only**, which brings a Western-media framing bias. State-affiliated outlets are included and flagged, not removed.
2. **Headlines only.** GDELT does not provide body text, and republishing it would breach copyright.
3. **Sampled day.** Four of GDELT's 96 files per day are read, not all of them.
4. **GDELT's own outages.** No files exist for all of 11 Nov 2022, parts of 9 and 22–23 Mar 2023, and several days in mid-June 2025.
5. **JISDOR is a browser download**, because BI blocks scripts. The steps above make it repeatable.
6. **"Country" means the most-mentioned location** in the article. GDELT's bulk files do not record the outlet's home country.
7. **Daily granularity.** The data can show next-day predictability, not intraday cause.

---

## Repository layout

```
data/
  raw/                      BI JISDOR export, FX panel, raw news sample
  processed/                final aligned dataset (+ CSV copy), features, targets,
                            filtered-news sample, manifest
config/config.yaml          every tunable, with the rationale in comments
requirements.txt
src/
  config.py                 YAML loader + path management
  http_cache.py             cached, retrying HTTP client + one-run-at-a-time lock
  fx_data.py                TARGET: BI JISDOR loader, Yahoo cross-check + controls
  gkg.py                    NEWS: GDELT GKG bulk files (the route used)
  gdelt.py                  NEWS: GDELT DOC API route (kept for reproducibility)
  filtering.py              Stages 0 and 2 of the funnel
  dedup.py                  Stage 3: SimHash near-duplicate clustering
  align.py                  WIB day-level alignment + leakage assertions
  preprocess.py             dual-track text preprocessing
  features.py               daily panel aggregation
  qa.py                     data-quality checks
  synthetic.py              offline fixtures
  build_dataset.py          pipeline driver
```

---

## Reproducibility

Every random step is seeded from `project.random_seed`. Downloads are cached in `data/cache/`, which freezes the corpus, so a rerun sees exactly the same input. `data/processed/manifest.json` records the period, row counts, seed, news source and leakage verdict of every run.

---

## AI-assistance disclosure

As the project rules allow, AI assistance was used for coding and debugging. The design decisions were made by the team: target and source, the filtering scope, the alignment rule, and the preprocessing tracks. They are justified in the team's report and in the comments of `config/config.yaml`.
