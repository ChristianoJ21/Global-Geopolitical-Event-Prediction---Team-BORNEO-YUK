# Predicting Global Geopolitical Events: Impact on the USD/IDR Exchange Rate
### Task 1: Data Acquisition & Strategic Preprocessing · Task 2: Pipeline Proposal & Baseline Experimentation

**Hypothesis under test.** Global geopolitical news carries significant information about, and is useful for predicting, fluctuations in the US dollar exchange rate, measured here as USD/IDR, Bank Indonesia's JISDOR reference rate.

- **Task 1** builds the dataset that makes testing that hypothesis *possible*, and makes a false positive *hard*.
- **Task 2** turns the headlines into numeric features, builds a baseline that only sees past exchange rates, builds a combined model that also sees the news, and compares them on a strict chronological split.

Every design choice is recorded so it can be defended, ablated, or overturned.

Period: **1 September 2021 to 1 September 2026**, as the assignment requires.

---

## Quick start

```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet omw-1.4 vader_lexicon

# 0. One manual input, already in the repo: Bank Indonesia's JISDOR export.
#    https://www.bi.go.id/id/statistik/informasi-kurs/jisdor/default.aspx
#    period 01/09/2021 - 01/09/2026 -> "Unduh" -> data/raw/Informasi Kurs Jisdor.xlsx

# 1. Task 1: build the dataset (GDELT bulk files; cached and resumable, ~30 min)
python -m src.build_dataset

# 2. Task 2: NLP features -> splits -> baselines -> combined models (~3 min)
python -m src.run_task2

# 3. Check the invariants (alignment, leakage, parsing, Task 2 timing): 35 tests
python -m pytest tests/ -q
```

`python -m src.build_dataset --offline` runs the Task 1 pipeline on synthetic data with no network. It overwrites `data/processed/` and marks the manifest `offline_synthetic: true`, so run step 1 again afterwards.

Step 2 needs `data/interim/articles_flagged.parquet`, which step 1 builds (it is over 100 MB, so it is not on GitHub). Everything step 2 produces is already in the repo, so you can read the results without running it.

No API keys are required.

---

## Data sources

| Stream | Source | How it is acquired |
|---|---|---|
| **USD/IDR target** | Bank Indonesia, JISDOR | Official export from bi.go.id, parsed by `src/fx_data.py`. Downloaded in a browser: bi.go.id resets scripted connections and ignores scripted form submissions. |
| Cross-check and controls | Yahoo Finance: `IDR=X`, DXY, VIX, Brent | Public chart endpoint, cached |
| **News** | GDELT Project, GKG 2.1 bulk files | 4 of the 96 daily 15-minute files, filtered by theme on download, `src/gkg.py` |
| Finance dictionary (Task 2) | Loughran-McDonald 2014 master dictionary | `data/lexicons/loughran_mcdonald_2014.csv`, built by `scripts/make_lm_lexicon.py` |

---

## What gets built

### Task 1

| Output | Grain | Contents | On GitHub |
|---|---|---|---|
| `data/raw/Informasi Kurs Jisdor.xlsx` | trading day | BI's JISDOR export, unmodified | yes |
| `data/raw/fx_panel.parquet` | day | JISDOR, Yahoo IDR=X cross-check, DXY, VIX, Brent | yes |
| `data/raw/articles_raw_sample.csv` | article | 2,000-row random sample of the raw news | yes |
| `data/raw/articles_raw.parquet` | article | every article collected (1,896,380), unfiltered | no (> 100 MB) |
| `data/interim/articles_flagged.parquet` | article | + every filter flag, dedup cluster, both text tracks | no (> 100 MB) |
| `data/samples/articles_flagged_sample.csv` | article | 4,968-row sample of the file above, stratified by year × theme family (70% kept / 30% dropped) | yes |
| `data/processed/daily_features.parquet` | trading day | news features on the JISDOR calendar | yes |
| `data/processed/targets.parquet` | trading day | returns, direction labels, multi-horizon targets | yes |
| **`data/processed/dataset.parquet`** | trading day | **the final aligned dataset**: 1,202 JISDOR days, lagged news features + labels + split | yes |
| `data/samples/dataset_aligned.csv` | trading day | the same table as CSV, for Excel | yes |
| `data/processed/manifest.json` | run | provenance: period, counts, seed, leakage verdict | yes |
| `reports/*.csv` | table | QA tables (funnel, coverage, validity), written by every build | yes |

### Task 2

| Output | Grain | Contents | On GitHub |
|---|---|---|---|
| **`data/splits/train.csv`, `val.csv`, `test.csv`** | trading day | **the modelling table, split by date**: 726 / 235 / 240 days, every feature group + labels | yes |
| `data/splits/feature_groups.json` | column | which columns belong to which group: price 10, market 4, Task 1 news 53, NLP 46 | yes |
| `data/processed/nlp_daily_features.parquet` | trading day | the 46 NLP features, before the one-day shift | yes |
| `data/interim/article_scores.parquet` | article | VADER and Loughran-McDonald score per headline (a cache) | no (rebuilt) |
| `reports/task2/*.csv` | table | results, news gain with intervals, grid search, predictions, confusion matrices, feature importance, lexicon coverage, SVD top terms, class balance | yes |
| `reports/task2/manifest.json` | run | rows per split, feature-group sizes, TF-IDF fit details, ARIMA order, leakage verdict | yes |
| `notebook/task2/task2_eda_and_baselines.ipynb` | notebook | EDA and every result table, executed | yes |
| `docs/task2_pipeline.png` | figure | the Task 2 pipeline diagram (`scripts/make_task2_diagram.py`) | yes |

The files marked "no" and the `data/cache/` folder are not in the repo: GitHub rejects files over 100 MB. `python -m src.build_dataset` and `python -m src.run_task2` rebuild them.

**Reading the article flags** (`articles_flagged_sample.csv`): `pass_source` = stage 0 (not a blocklisted aggregator); `theme_family` = stage 1; `has_actor_country` / `has_institution` / `has_market_term` are OR-ed into `pass_materiality` = stage 2; `is_duplicate`, `cluster_id`, `dup_count` = stage 3; `keep` = passed every stage (only these rows feed the features); `news_day` = the JISDOR trading day the article is grouped with; `text_track_a` / `text_track_b` = the two preprocessing tracks.

---

## Pipeline

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
                                                                               │
  Task 2 ──────────────────────────────────────────────────────────────────────┘
     headlines ─► nlp_features (VADER, Loughran-McDonald, TF-IDF -> SVD) ─► daily, shifted 1 day
     JISDOR    ─► price_features (return lags, 5/20-day mean and volatility)
                          │
                          ▼
     data/splits (train / val / test) ─► models (naive, ARIMA, logistic/ridge, XGBoost)
                                      ─► evaluate (macro-F1, RMSE, block bootstrap) ─► reports/task2
```

A drawn version is in `docs/task2_pipeline.png` and in the team's report.

---

## The six decisions that define this dataset (Task 1)

**1. The target is Bank Indonesia's JISDOR rate.**
The assignment names Bank Indonesia as the source, so the target is USD/IDR, not a broad dollar index or DXY. That also fits the question: the rupiah moves on Indonesia-specific news (BI policy, domestic politics, commodity exports) that a broad index would average away. Yahoo's `IDR=X` is kept only as a cross-check. Its levels agree with JISDOR closely, but its daily moves correlate only 0.53 with JISDOR's, because it closes at a different hour from BI's afternoon fix (15:15 or 16:15 WIB). So it cannot stand in for BI's rate.

**2. Retrieval is by GDELT theme code, never by keyword.**
A keyword list drifts over five years as words go in and out of fashion, which would give the model a spurious time trend. GDELT's theme taxonomy is machine-coded against a fixed codebook, so `ECON_SANCTIONS` means the same thing in 2021 and 2026. There are 30 codes in six families: conflict, sanctions & trade, diplomacy, energy, monetary policy, and political risk.

**3. Filtering is a four-stage funnel, and no stage deletes anything.**
Each stage writes a yes/no column, so any stage can be switched off later to measure what it was worth. Every build writes the attrition table to `reports/funnel_attrition.csv`.

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
One corpus, two views. Track A (used in Task 2 by TF-IDF and Loughran-McDonald) lowercases, removes stopwords, lemmatises, and masks numbers as `<NUM>` / `<PCT>`. Track B (used in Task 2 by VADER) keeps the headline almost as written, because VADER reads capitals, "!" and negations. Track B would also suit transformer models, but the Task 2 brief does not allow them.

Track A's stopword list has an explicit **keep-list** of negators and direction words. Without it:

> `"Fed will not cut rates"` → `"fed cut rates"`, and the meaning has flipped.

---

## Task 2: NLP features and baseline experiments

**What we predict.**
- **Main target: `y_direction`**, the direction of the next JISDOR fix: UP above +0.10%, DOWN below −0.10%, FLAT in between (the dead-zone from Task 1). Scored with **macro-F1**, because the classes are about 40/30/30 and plain accuracy rewards always saying UP.
- **Second target: `y_return`**, the next-day log return in percent. Scored with RMSE and MAE.

**How text becomes features** (`src/nlp_features.py`). No pre-trained embeddings or transformers, as the brief requires:

| Extractor | Text | Features |
|---|---|---|
| VADER | Track B | average sentiment, its spread, share of negative and positive headlines |
| Loughran-McDonald | Track A | rate of negative, positive, uncertainty, litigious and constraining words; net tone |
| TF-IDF → SVD | Track A | 20 topic axes per day, fitted on training-period news only (539,429 headlines, 13,532 terms) |

Headlines are averaged per trading day, weighted by `log1p(dup_count)`, with per-theme scores and 30-day z-scores, then shifted one trading day like the Task 1 features.

**Models** (`src/models.py`). Naive baselines (majority class, persistence, zero return), ARIMA(2,0,2), logistic regression (Ridge for returns) and XGBoost. The learned models are each run on six feature sets: price only (the baseline), price + NLP (the combined model), price + market, price + market + NLP, price + Task 1 news, and all features.

**Protocol.** Train 2 Sep 2021 to 30 Aug 2024 (726 days), validation to 29 Aug 2025 (235), test to 1 Sep 2026 (240). Settings are chosen on validation, then the chosen model is refitted on train + validation and scored on test once. Every with-news vs without-news comparison gets a 95% block-bootstrap interval (`src/evaluate.py`).

**Results (test, macro-F1).** Majority 0.190, persistence 0.370, ARIMA 0.208.

| Model | Price | Price + NLP | Price + market | Price + market + NLP | All |
|---|---|---|---|---|---|
| Logistic regression | 0.321 | 0.342 | **0.458** | 0.388 | 0.340 |
| XGBoost | 0.331 | 0.313 | 0.436 | 0.430 | **0.456** |

- The overnight dollar move (`lag1_ret_DXY`) is the strongest predictor. Its timing was checked: it closes before the JISDOR fix, so there is no leakage.
- The news features did not add reliable directional power yet: every test interval for the news gain includes zero.
- News relates more to how big the next move is than to its direction.

Full tables are in `reports/task2/` and `notebook/task2/`.

---

## Why GDELT's bulk files and not its API

The GDELT DOC 2.0 API allows one request every five seconds per IP. The full period needs thousands of requests, and exceeding the limit got this project's IP blocked for more than a day. GDELT publishes the same index as static 15-minute GKG files and recommends them for heavy use. `src/gkg.py` downloads four files per day, keeps only the articles tagged with the 30 theme codes, and caches those rows, so a rerun downloads nothing new. It takes at most 250 articles per family per day, a random draw with a fixed seed, rather than "the latest 250". The API route is still in `src/gdelt.py` (`--source doc`).

---

## Guards against fooling ourselves

- **Leakage assertions** (`src/align.py`) run on every build, and again on the Task 2 table. They fail loudly on contemporaneous features, unsorted or duplicated dates, overlapping splits, or any feature correlating above 0.95 with the target. That check caught a real leak on the first run: the raw return of the target series was identical to `y_return`. So market history enters only as `lag1_*` columns.
- **Chronological splits, never shuffled.** Train up to 31 Aug 2024, validation up to 31 Aug 2025, test after. A random split on autocorrelated data lets the model interpolate between days it has already seen.
- **Text models see only training news.** The TF-IDF vocabulary, IDF weights and SVD axes are fitted on news before the end of the training period.
- **Test is scored once.** Every setting is chosen on validation.
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
8. **General-purpose dictionaries.** VADER calls 36.5% of headlines exactly neutral, and Loughran-McDonald finds no word at all in 53.9%. Neither was built for geopolitics.

---

## Repository layout

```
data/
  raw/                      BI JISDOR export, FX panel, raw news sample
  processed/                final aligned dataset, features, targets, NLP features, manifest
  samples/                  CSV copies: filtered-news sample, aligned dataset
  lexicons/                 Loughran-McDonald dictionary (Task 2)
  splits/                   Task 2 train / val / test tables + feature groups
config/config.yaml          every tunable, with the rationale in comments (section 12 = Task 2)
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
  build_dataset.py          Task 1 pipeline driver
  price_features.py         Task 2: JISDOR-history features (baseline inputs)
  nlp_features.py           Task 2: VADER, Loughran-McDonald, TF-IDF -> SVD
  models.py                 Task 2: naive, ARIMA, logistic/ridge, XGBoost
  evaluate.py               Task 2: macro-F1, hit rate, RMSE, block bootstrap
  run_task2.py              Task 2 pipeline driver
notebook/
  task1/                    Task 1 notebook: data acquisition and preprocessing
  task2/                    Task 2 notebook: EDA and results
reports/                    Task 1 QA tables; reports/task2/ = Task 2 results
docs/                       Task 1 report, Task 2 pipeline diagram
scripts/                    notebook, diagram, sample and lexicon builders
tests/                      35 tests: alignment, leakage, JISDOR/GKG parsing, Task 2 timing
```

---

## Reproducibility

Every random step is seeded from `project.random_seed`. Downloads are cached in `data/cache/`, which freezes the corpus, so a rerun sees exactly the same input. `data/processed/manifest.json` records the period, row counts, seed, news source and leakage verdict of every Task 1 run, and `reports/task2/manifest.json` does the same for Task 2.

---

## AI-assistance disclosure

As the project rules allow, AI assistance was used for coding and debugging. The design decisions were made by the team: target and source, the filtering scope, the alignment rule, the preprocessing tracks, and for Task 2 the target formulation, the feature extractors and the model set. They are justified in the team's reports and in the comments of `config/config.yaml`.
