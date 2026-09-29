// Generates docs/TASK1_REPORT.docx
const fs = require("fs");
const path = require("path");
const d = require("docx");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  PageBreak, Header, Footer, PageNumber, TableOfContents, LevelFormat,
  convertInchesToTwip,
} = d;

const OUT = path.join(__dirname, "..", "docs", "TASK1_REPORT.docx");
const ACCENT = "1F3864";
const GREY = "595959";
const LIGHT = "F2F5F9";

// ---------------------------------------------------------------- helpers
const P = (text, o = {}) =>
  new Paragraph({
    alignment: o.align,
    spacing: { after: o.after ?? 120, line: o.line ?? 276 },
    indent: o.indent,
    border: o.border,
    children: [new TextRun({ text, bold: o.bold, italics: o.italics,
      size: o.size ?? 21, color: o.color, font: o.font })],
  });

// Rich paragraph: array of [text, {bold, italics, code}]
const RP = (parts, o = {}) =>
  new Paragraph({
    alignment: o.align,
    spacing: { after: o.after ?? 120, line: 276 },
    indent: o.indent,
    children: parts.map(([t, s = {}]) =>
      new TextRun({
        text: t, bold: s.b, italics: s.i,
        size: s.code ? 19 : 21,
        font: s.code ? "Consolas" : undefined,
        color: s.code ? "8B2500" : s.color,
      })),
  });

const H1 = (t) => new Paragraph({
  heading: HeadingLevel.HEADING_1, spacing: { before: 320, after: 160 },
  children: [new TextRun({ text: t, bold: true, size: 30, color: ACCENT })],
});
const H2 = (t) => new Paragraph({
  heading: HeadingLevel.HEADING_2, spacing: { before: 260, after: 120 },
  children: [new TextRun({ text: t, bold: true, size: 24, color: ACCENT })],
});
const H3 = (t) => new Paragraph({
  heading: HeadingLevel.HEADING_3, spacing: { before: 200, after: 100 },
  children: [new TextRun({ text: t, bold: true, size: 22, color: "2E4A7D" })],
});

const BULLET = (t, level = 0) => new Paragraph({
  numbering: { reference: "bullets", level },
  spacing: { after: 70, line: 276 },
  children: [new TextRun({ text: t, size: 21 })],
});
const RBULLET = (parts, level = 0) => new Paragraph({
  numbering: { reference: "bullets", level },
  spacing: { after: 70, line: 276 },
  children: parts.map(([t, s = {}]) => new TextRun({
    text: t, bold: s.b, italics: s.i, size: s.code ? 19 : 21,
    font: s.code ? "Consolas" : undefined, color: s.code ? "8B2500" : undefined,
  })),
});
const NUM = (t) => new Paragraph({
  numbering: { reference: "numbers", level: 0 },
  spacing: { after: 70, line: 276 },
  children: [new TextRun({ text: t, size: 21 })],
});

const CODE = (lines) => lines.map((l, i) => new Paragraph({
  spacing: { after: i === lines.length - 1 ? 140 : 0, line: 240 },
  indent: { left: convertInchesToTwip(0.25) },
  shading: { type: ShadingType.CLEAR, fill: "F4F4F4" },
  children: [new TextRun({ text: l || " ", font: "Consolas", size: 18 })],
}));

const CALLOUT = (label, text) => new Paragraph({
  spacing: { before: 140, after: 160, line: 276 },
  indent: { left: convertInchesToTwip(0.12) },
  shading: { type: ShadingType.CLEAR, fill: LIGHT },
  border: { left: { style: BorderStyle.SINGLE, size: 18, color: ACCENT, space: 8 } },
  children: [
    new TextRun({ text: label + "  ", bold: true, size: 21, color: ACCENT }),
    new TextRun({ text, size: 21 }),
  ],
});

// Table. widths in DXA, must sum to total.
const TBL = (headers, rows, widths) => {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (txt, { head = false, w, bold = false, alignRight = false } = {}) =>
    new TableCell({
      width: { size: w, type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, fill: head ? ACCENT : "FFFFFF" },
      margins: { top: 60, bottom: 60, left: 90, right: 90 },
      children: [new Paragraph({
        alignment: alignRight ? AlignmentType.RIGHT : AlignmentType.LEFT,
        spacing: { after: 0, line: 240 },
        children: [new TextRun({
          text: String(txt), bold: head || bold, size: 19,
          color: head ? "FFFFFF" : undefined,
        })],
      })],
    });
  return new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      new TableRow({
        tableHeader: true,
        children: headers.map((h, i) => cell(h, { head: true, w: widths[i] })),
      }),
      ...rows.map((r, ri) => new TableRow({
        children: r.map((c, i) => new TableCell({
          width: { size: widths[i], type: WidthType.DXA },
          shading: { type: ShadingType.CLEAR, fill: ri % 2 ? "F7F9FC" : "FFFFFF" },
          margins: { top: 60, bottom: 60, left: 90, right: 90 },
          children: [new Paragraph({
            spacing: { after: 0, line: 240 },
            alignment: (i > 0 && /^[-\d.,%<>= ]+$/.test(String(c))) ? AlignmentType.RIGHT : AlignmentType.LEFT,
            children: [new TextRun({ text: String(c), size: 19 })],
          })],
        })),
      })),
    ],
  });
};

const CAPTION = (t) => new Paragraph({
  spacing: { before: 60, after: 200 },
  children: [new TextRun({ text: t, size: 18, italics: true, color: GREY })],
});

const FULL = 9360; // 6.5" in DXA

// ================================================================== content
const body = [];

// ---- title page
body.push(
  new Paragraph({ spacing: { before: 1800, after: 0 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "NATURAL LANGUAGE PROCESSING — GROUP PROJECT",
      size: 20, color: GREY, bold: true })] }),
  new Paragraph({ spacing: { before: 280, after: 80 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Predicting Global Geopolitical Events:",
      size: 40, bold: true, color: ACCENT })] }),
  new Paragraph({ spacing: { after: 320 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Analysing the Impact on the US Dollar Exchange Rate",
      size: 32, bold: true, color: ACCENT })] }),
  new Paragraph({ spacing: { after: 600 }, alignment: AlignmentType.CENTER,
    border: { top: { style: BorderStyle.SINGLE, size: 8, color: ACCENT, space: 10 } },
    children: [new TextRun({ text: "TASK 1 — Strategic Data Acquisition & Preprocessing",
      size: 26, bold: true })] }),
  P("Hypothesis under test", { align: AlignmentType.CENTER, bold: true, color: GREY, size: 20 }),
  P("“Global geopolitical news carries significant information about, and is useful for predicting, fluctuations in the US dollar exchange rate.”",
    { align: AlignmentType.CENTER, italics: true, size: 22, after: 700 }),
);

body.push(TBL(
  ["Item", "Detail"],
  [
    ["Deliverable", "Task 1 of 4 — data acquisition and strategic preprocessing"],
    ["Study period", "2017-01-01 to 2026-08-31 (2,430 trading sessions)"],
    ["Primary target", "Fed nominal broad trade-weighted USD index (DTWEXBGS), daily log return"],
    ["News corpus", "GDELT 2.0 DOC API + Timeline API, 30 theme codes in 6 families"],
    ["Deliverables", "Runnable pipeline (12 modules), notebook, 15 property tests, this report"],
    ["Repository", "geo-usd/ — see README.md"],
    ["AI assistance", "Used for code scaffolding and debugging. All architectural decisions are the team's own and are justified in §3–§7 below."],
  ],
  [1900, 7460],
));

body.push(new Paragraph({ children: [new PageBreak()] }));

// ---- TOC
body.push(H1("Contents"));
body.push(P("If the list below appears blank, click it and press F9 (Word) or choose Tools → Update → Update All Fields (LibreOffice) to populate it.",
  { italics: true, color: GREY, size: 18, after: 200 }));
body.push(new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-3" }));
body.push(new Paragraph({ children: [new PageBreak()] }));

// ================================================================== §1
body.push(H1("1.  Executive Summary"));

body.push(P("This report documents the design and construction of the dataset that the remaining three tasks will use to test our hypothesis. It is a design document, not a results document: Task 1 produces no finding about the dollar."));

body.push(P("The pipeline has two goals, and the second one shaped almost every decision recorded here."));

body.push(RP([["First, assemble a defensible corpus: ", { b: true }], ["roughly ten years of geopolitically-themed news headlines, aligned to the correct trading session, joined to a well-specified measure of the dollar."]]));
body.push(RP([["Second — and this is the harder goal — ", { b: true }], ["make a false positive difficult to reach.", { b: true }], [" It is easy to build a pipeline of this kind that reports impressive accuracy. Bin news by calendar date; shuffle the train/test split; feed raw article counts. Each of those is a bug. Each one "], ["improves", { i: true }], [" the reported metric. And none of them is visible in the final number. A dataset that cannot tell a real relationship from a leaked one is worse than no dataset, because it produces confident conclusions."]]));

body.push(H2("Key decisions"));
body.push(TBL(
  ["Decision", "Choice", "Reason in one line"],
  [
    ["Which dollar?", "Fed broad trade-weighted index", "DXY is 57.6% euro — a DXY model is a EUR/USD model in disguise"],
    ["Retrieval key", "GDELT theme codes", "Keyword lists drift over a decade and make the corpus non-stationary"],
    ["Filtering", "4-stage funnel, non-destructive", "Every stage is a flag, not a deletion, so any stage can be ablated"],
    ["Materiality rule", "OR, not AND", "Requiring a market term selects for post-hoc explanation articles"],
    ["Deduplication", "SimHash; keep dup_count", "Syndication breadth is the newsroom's own importance vote"],
    ["Alignment", "17:00 ET boundary, 1-session lag", "Calendar-day binning is silent look-ahead"],
    ["Preprocessing", "Two tracks, one corpus", "Aggressive cleaning helps TF-IDF and harms transformers"],
    ["Validation", "Two-annotator human audit", "Stages 0–3 are our opinion in code; the audit is evidence"],
  ],
  [1500, 2450, 5410],
));
body.push(CAPTION("Table 1. The eight decisions that define this dataset. Each is justified in §3–§7."));

body.push(H2("Status"));
body.push(P("The pipeline runs end-to-end. Fifteen property tests pass, covering temporal alignment to the minute, leakage detection, negation preservation and deduplication behaviour. The leakage assertions caught a genuine defect on the first run — documented in §5.4 — which is the clearest evidence we can offer that the guards are doing real work rather than decorating the report."));

// ================================================================== §2
body.push(H1("2.  Scope, and What This Dataset Cannot Do"));

body.push(P("We state the limits first, because they constrain what Task 4 is permitted to conclude, and a limitation discovered in week twelve is worth far less than one declared in week two."));

body.push(TBL(
  ["Limitation", "Consequence", "Mitigation"],
  [
    ["English-language sources only", "Western media framing bias; coverage of non-Anglophone events is systematically thinner", "State-affiliated outlets deliberately retained and flagged; multilingual ablation deferred to Task 3"],
    ["Headlines only, never body text", "Short-text NLP throughout; no long-document methods available", "Scoped deliberately: 128-token models, headline-tuned sentiment. Also keeps us clear of copyright"],
    ["Daily granularity", "Cannot resolve intraday causality", "No intraday causal claim will be made"],
    ["Coverage ≠ market impact", "A heavily covered event may be fully priced in already", "Syndication breadth and concentration features added as partial proxies"],
    ["250 records per API call", "Busy days are censored — and saturation correlates with news intensity, so the error is systematic, not random", "Per-family quotas; sub-day chunking; cross-check against uncapped timeline volume"],
    ["Reverse causality", "FX moves cause news as well as the reverse", "Mitigated by the OR-gate design (§4.3); must be tested by lead-lag and Granger analysis in Task 4"],
  ],
  [2100, 3630, 3630],
));
body.push(CAPTION("Table 2. Declared limitations. The last row is the central threat to the hypothesis."));

body.push(CALLOUT("What this dataset can support.",
  "Daily-horizon predictability, and a test of whether news features add information beyond price history and generic risk appetite. It cannot establish intraday causality, and it cannot by itself rule out that part of any measured relationship runs from markets to news. Saying so now is what stops Task 4 from overclaiming — which is the most likely way this project goes wrong."));

// ================================================================== §3
body.push(H1("3.  Target and Source Selection"));

body.push(H2("3.1  Which dollar?"));
body.push(RP([["“The USD exchange rate” is under-specified, and the specification matters more than it appears to."]]));
body.push(RP([["The DXY trap. ", { b: true }], ["The ICE Dollar Index is what traders quote, and it is the obvious default. It is also "], ["57.6% euro", { b: true }], [". A model trained on DXY is largely a EUR/USD model: it will attribute eurozone-specific news to “the dollar”, and a headline about ECB policy will register as evidence for our hypothesis when it is really evidence about the euro."]]));
body.push(RP([["Our choice. ", { b: true }], ["The Federal Reserve's nominal broad trade-weighted index ("], ["DTWEXBGS", { code: true }], [") weights 26 partners by actual trade volume. That is the closest available measure of “the dollar against the world”, which is what the hypothesis is about. DXY and six bilateral pairs are retained for Task 4's error analysis, because geopolitical shocks are frequently bilateral: a Taiwan Strait headline should move USDCNY more than it moves USDIDR, and a model that cannot show this is not capturing a mechanism."]]));

body.push(H3("Controls are acquired here, not bolted on later"));
body.push(RP([["VIX and the 10-year Treasury yield are downloaded in the same step, because the honest version of our hypothesis is that geopolitical news adds information "], ["beyond", { i: true }], [" generic risk appetite and rate differentials. Without those columns, Task 4 cannot make that claim — and a result that merely rediscovers “risk-off days move the dollar” would be worthless."]]));

body.push(H2("3.2  Source evaluation"));
body.push(TBL(
  ["Source", "History", "Cost", "Verdict"],
  [
    ["GDELT 2.0 DOC API", "2017–present", "Free, no key", "PRIMARY — article headlines"],
    ["GDELT Timeline API", "2017–present", "Free, no key", "PRIMARY — uncapped daily volume/tone"],
    ["FRED (fredgraph.csv)", "2006–present", "Free, no key", "PRIMARY — broad USD index, VIX, UST 10y"],
    ["Stooq CSV", "long", "Free, no key", "PRIMARY — DXY, bilateral pairs, oil, gold"],
    ["NewsAPI.org", "1 month (free tier)", "Freemium", "Rejected — cannot span macro regimes"],
    ["Kaggle news dumps", "Frozen snapshot", "Free", "Rejected — no recency, no reproducible refresh"],
    ["Wire RSS feeds", "~30 days", "Free", "Secondary — spot validation only"],
    ["ECB / Frankfurter", "1999–present", "Free, no key", "Fallback only — fixes at 16:00 CET, a different day convention"],
  ],
  [2100, 1700, 1500, 4060],
));
body.push(CAPTION("Table 3. Source evaluation. Every primary source is keyless, so any team member can reproduce the fetch with no shared secret and no free tier that expires mid-semester."));

body.push(H3("Why GDELT, in one paragraph"));
body.push(RP([["Three properties decided it. "], ["History: ", { b: true }], ["the DOC 2.0 index starts 2017-01-01, which is long enough to span five distinct macro regimes, so a model cannot succeed by memorising one. "], ["Machine-coded structure: ", { b: true }], ["a fixed theme taxonomy and a per-article tone score. "], ["Cost and reproducibility: ", { b: true }], ["no key, no quota, no per-seat licence."]]));

body.push(CALLOUT("Legal note.",
  "GDELT exposes headline, URL and metadata — never body text. We therefore redistribute only metadata and links, which keeps the project clear of copyright, and we scope the entire NLP stack to short text. This is a constraint, but a declared one rather than a workaround discovered later."));

body.push(H2("3.3  Temporal scope"));
body.push(RP([["2017-01-01 to 2026-08-31 — the maximum span that is "], ["methodologically homogeneous", { b: true }], [" (one crawler generation, one theme taxonomy). The window deliberately covers regimes that behave differently, so that a model cannot succeed by learning a single macro environment:"]]));
body.push(BULLET("2017–18 — US–China trade war; tariff escalation"));
body.push(BULLET("2020 — COVID dollar funding squeeze"));
body.push(BULLET("2021–22 — Fed hiking cycle; Russia–Ukraine; the SWIFT sanctions regime"));
body.push(BULLET("2023–24 — Middle East escalation; global disinflation"));
body.push(BULLET("2025–26 — renewed tariff regime; de-dollarisation debate"));

body.push(new Paragraph({ children: [new PageBreak()] }));

// ================================================================== §4
body.push(H1("4.  The Filtering Strategy"));

body.push(RP([["“Geopolitical news that moves the dollar” is neither a keyword nor a single theme. Any "], ["single-stage", { b: true }], [" filter fails in one of two directions:"]]));
body.push(RBULLET([["Too broad", { b: true }], [" — "], ["theme:MILITARY", { code: true }], [" floods the corpus with military-parade coverage and video-game reviews."]]));
body.push(RBULLET([["Too narrow", { b: true }], [" — "], ["\"dollar\" AND \"sanctions\"", { code: true }], [" returns only articles that "], ["already mention the outcome we are trying to predict", { i: true }], [", which makes the “prediction” a tautology."]]));

body.push(RP([["We therefore use a funnel of four cheap, independently auditable stages. Each stage writes a "], ["boolean column rather than deleting rows", { b: true }], [". Nothing is discarded irreversibly, so Task 2 can ablate any stage and measure what it was worth — which is exactly the evidence we want when asked to justify the design."]]));

body.push(TBL(
  ["Stage", "Purpose", "Orientation", "Column written"],
  [
    ["0 · Source gate", "Remove content farms, tier the wires", "Precision", "pass_source, source_tier"],
    ["1 · Thematic gate", "Retrieve by GDELT theme code", "Recall", "theme_family (at query time)"],
    ["2 · Materiality gate", "Does it touch a USD channel?", "Precision", "pass_materiality"],
    ["3 · Deduplication", "Collapse syndication", "Noise", "is_duplicate, dup_count"],
    ["4 · Human audit", "Validate 0–3 against two annotators", "Validation", "audit_results.csv"],
  ],
  [1750, 2900, 1400, 3310],
));
body.push(CAPTION("Table 4. The four-stage funnel."));

body.push(H2("4.1  Stage 0 — source gate"));
body.push(P("GDELT indexes a long tail of content farms that republish a single wire story hundreds of times. If we count raw articles, our “news volume” feature is really “how many SEO sites copied Reuters today” — a quantity with no macroeconomic meaning that nonetheless trends upward across the decade and would be happily mistaken for signal."));
body.push(RP([["We use ", {}], ["soft", { code: true }], [" mode: non-listed domains are kept but tagged "], ["tier=3", { code: true }], [". Hard deletion would destroy our ability to "], ["test", { i: true }], [" whether the gate helped, and would bias coverage against non-Western outlets that simply are not on our — inevitably Anglophone — allowlist."]]));

body.push(CALLOUT("State media are kept on purpose.",
  "TASS, RT, Xinhua, Global Times and Press TV are retained and flagged with state_affiliated. They publish propaganda, but propaganda is itself a geopolitical signal: a framing shift in TASS is information about Russian intent. Excluding them would leave us with only the Western narrative of every event, which is a bias, not a cleaning step."));

body.push(H2("4.2  Stage 1 — thematic gate, and why not keywords"));
body.push(RP([["This is the decision we would defend hardest. A free-text keyword list "], ["drifts over a decade", { b: true }], [". “De-dollarisation” barely exists before 2022. “Trade war” spikes in 2018 partly for reasons of journalistic vocabulary rather than of events. Retrieving on such a list silently makes a ten-year corpus non-stationary and hands the model a spurious time trend that has nothing to do with geopolitics — and, critically, one that would be invisible in any accuracy metric."]]));
body.push(RP([["GDELT's theme taxonomy is machine-coded against a fixed codebook, so "], ["ECON_SANCTIONS", { code: true }], [" means the same thing in 2017 and in 2026. Free-text market terms are still used — but "], ["later", { i: true }], [", in the Stage-2 precision gate, where they can be tuned without re-downloading anything."]]));

body.push(TBL(
  ["Family", "Representative theme codes"],
  [
    ["conflict", "ARMEDCONFLICT, MILITARY, WB_2457_TERRORISM, CRISISLEX_CRISISLEXREC, SIEGE"],
    ["sanctions_trade", "ECON_SANCTIONS, ECON_TRADE_DISPUTE, ECON_TARIFF, ECON_FREETRADE, WTO"],
    ["diplomacy", "DIPLOMATIC_COOPERATION, NEGOTIATIONS, TREATY, ALLIANCE, SUMMIT"],
    ["energy", "ENV_OIL, ENV_GAS, ECON_OILPRICE, ENV_NUCLEARPOWER"],
    ["monetary_policy", "ECON_INTEREST_RATES, ECON_CENTRALBANK, ECON_INFLATION, ECON_CURRENCY_EXCHANGE_RATE, ECON_CURRENCY_RESERVES"],
    ["political_risk", "ECON_UNCERTAINTY, PROTEST, ELECTION, GOVERNMENT_INSTABILITY, CORRUPTION"],
  ],
  [2000, 7360],
));
body.push(CAPTION("Table 5. Six theme families, 30 codes. Each family becomes its own feature column downstream, so composition is measurable rather than assumed."));

body.push(H3("Why one request per family, not one union query"));
body.push(RP([["The API caps every call at 250 records. A union query hits that ceiling and returns whatever family dominated the day — so on a heavy conflict-news day, our monetary-policy articles silently vanish. Per-family iteration gives each family its own quota, producing a corpus whose composition is a "], ["design choice rather than an artefact of the cap", { b: true }], [". The cost is roughly six times the requests; the on-disk cache makes that a one-time expense."]]));

body.push(H2("4.3  Stage 2 — materiality gate"));
body.push(P("An article survives if ANY of three independent conditions holds:"));
body.push(NUM("actor_country — it originates from or concerns a G20 / reserve-currency / geopolitically pivotal state (25 FIPS codes);"));
body.push(NUM("institution — its headline names a monetary, trade or security institution (Fed, ECB, BoJ, PBoC, IMF, OPEC, NATO, WTO, BRICS, G7, G20, UNSC, US Treasury);"));
body.push(NUM("market_term — its headline contains an explicitly market-relevant term (tariff, sanction, embargo, rate hike, oil price, reserves, SWIFT, devaluation, …)."));

body.push(CALLOUT("Why OR and not AND.",
  "AND would be far more precise and badly wrong. The articles with the most predictive value are frequently the ones that do not yet mention the dollar: by the time a headline reads “dollar rises on sanctions news”, the move has already happened and there is nothing left to predict. Requiring a market term would select for post-hoc explanation articles and manufacture exactly the reverse-causality problem we have to guard against. Condition (a) deliberately admits early, currency-agnostic coverage."));

body.push(RP([["The country list is not a value judgement. It is a statement about the transmission mechanism we hypothesised: a cabinet crisis in a state with 0.05% of world trade generates headlines but no dollar flow. We record the decision as a flag so the assumption is "], ["testable", { i: true }], [", not baked in."]]));

body.push(H2("4.4  Stage 3 — deduplication that keeps what it removes"));
body.push(P("A single Reuters story about an oil embargo appears in GDELT hundreds of times: mirrored by syndication partners, re-titled by aggregators, re-crawled under a different URL parameter. Leaving them in breaks three things at once."));
body.push(NUM("“News volume” measures syndication reach rather than event importance — a quantity with no macroeconomic meaning that trends upward across the decade."));
body.push(NUM("TF-IDF is poisoned: a copied phrase looks like a strong repeated pattern rather than one observation."));
body.push(NUM("The train/test split leaks, because the same story lands on both sides."));

body.push(RP([["But naive deletion throws away real information. "], ["How widely", { i: true }], [" a story was copied is a genuine proxy for how important editors judged it to be. So we do not delete and forget: each cluster collapses to its "], ["earliest", { b: true }], [" member — first-mover timing matters for an event study — and the cluster size survives as "], ["dup_count", { code: true }], [", a feature in its own right."]]));

body.push(TBL(
  ["Level", "Method", "Catches"],
  [
    ["1", "Exact canonical URL (tracking params stripped)", "Re-crawls, mirrors, UTM variants"],
    ["2", "Exact normalised title, within ±48 h", "Verbatim syndication under a new URL"],
    ["3", "SimHash over character 4-grams, Hamming ≤ 6, within ±48 h", "Re-titled copies, localised spelling, minor edits"],
  ],
  [900, 4560, 3900],
));
body.push(CAPTION("Table 6. Three matching levels. All three apply the ±48-hour window."));

body.push(H3("Why SimHash and not sentence embeddings"));
body.push(RP([["Near-duplicate detection here is a "], ["lexical", { i: true }], [" problem — the same sentence with a different outlet tag. SimHash over character 4-grams solves it in O(n) with one 64-bit integer per document, runs on a laptop over a million headlines, and is fully deterministic, so a teammate re-running the pipeline gets byte-identical clusters."]]));
body.push(RP([["A sentence-transformer would be slower, need a GPU, introduce a model dependency and a random seed — and, decisively, it would "], ["over-merge", { b: true }], [": “Fed raises rates” and “Fed cuts rates” are semantically close and are opposite events. Cheap and deterministic is the correct engineering call."]]));

body.push(CALLOUT("The ±48-hour window is not an optimisation.",
  "Without it, a headline such as “Oil prices rise as Middle East tensions mount” — which recurs verbatim every few months for a decade — would collapse into a single cluster, deleting a genuine repeated signal. Two articles are the same story only if they are also close in time. This was a real bug in our first implementation, caught by tests/test_pipeline.py."));

body.push(H2("4.5  Stage 4 — the human audit"));
body.push(RP([["Stages 0–3 are ", {}], ["our opinion", { b: true }], [" about what a dollar-relevant geopolitical article is, expressed as code. The audit is where that opinion meets evidence."]]));
body.push(RP([["Two annotators independently label a stratified sample of 300 articles drawn from "], ["both sides", { b: true }], [" of the filter (70% kept / 30% discarded). Auditing only what we kept would measure precision and be blind to recall — and a filter that discards half the real signal is exactly as broken as one that keeps garbage."]]));

body.push(TBL(
  ["Metric", "Question", "Threshold"],
  [
    ["Cohen's κ (read FIRST)", "Do two humans agree on what “material” means?", "≥ 0.60"],
    ["Precision", "Of what the filter kept, what share is genuinely material?", "≥ 0.85"],
    ["Recall (estimated)", "Of what it discarded, what share should have been kept?", "reported, not gated"],
  ],
  [2200, 5260, 1900],
));
body.push(CAPTION("Table 7. Acceptance criteria, fixed in config/config.yaml before any labelling took place."));

body.push(RP([["The order matters. If two humans cannot agree on the label, the label is ill-defined and "], ["no precision figure computed from it means anything", { b: true }], [" — so κ is read first, and a failing κ sends us back to the guidelines rather than to the filter. Raw agreement will not substitute: two annotators who both answer “1” 90% of the time agree 82% of the time by luck alone."]]));
body.push(RP([["Stratification is by (year × theme family). A simple random sample would be dominated by the largest family in the busiest years and would say nothing about whether the filter behaves "], ["consistently across the decade", { i: true }], [" — which is the failure mode we are actually worried about. Annotators never see the filter's own verdict; showing it would anchor them and make the audit worthless."]]));

body.push(new Paragraph({ children: [new PageBreak()] }));

// ================================================================== §5
body.push(H1("5.  Temporal Alignment"));

body.push(CALLOUT("This is the most important section of the report.",
  "Almost every published failure of “news predicts markets” research is a misalignment bug, not a modelling bug. There are exactly three ways to get it wrong, and all three make the backtest look BETTER — which is precisely why they survive code review."));

body.push(H2("5.1  The three failure modes"));
body.push(RP([["1 · Calendar-day binning. ", { b: true }], ["Assign news to its UTC calendar date and join on the FX date. A story published at 21:00 UTC on Monday — which is 17:00 ET, "], ["after", { i: true }], [" the close — then sits in Monday's bucket and “predicts” a return that had already printed. Silent, devastating look-ahead."]]));
body.push(RP([["2 · Weekend leakage. ", { b: true }], ["FX closes Friday 17:00 ET and reopens Sunday 17:00 ET. Naive resampling either drops weekend news entirely — losing precisely the geopolitical events that governments time for a Friday night — or attaches it to a Saturday row that does not exist."]]));
body.push(RP([["3 · Contemporaneous “prediction”. ", { b: true }], ["Using news from day t to explain the return of day t is a "], ["nowcast", { i: true }], [", not a forecast, and it cannot distinguish “news moved the dollar” from “the dollar moved, so journalists wrote about it”."]]));

body.push(H2("5.2  Our convention"));
body.push(...CODE([
  "news_day(t) = ( close(t-1) , close(t) ]      close = 17:00 America/New_York",
  "",
  "features from news_day(t)  ->  predict return of session t+1",
]));
body.push(RP([["Every feature is therefore "], ["strictly in the past", { b: true }], [" relative to its label, so the setup is tradeable in principle: a desk could have acted on it at the close. Implementation is a single "], ["searchsorted", { code: true }], [" over the vector of session close instants, which encodes the half-open rule, weekends and holidays at once — the Friday-to-Monday gap simply makes Monday's window 65 hours long instead of 24."]]));

body.push(TBL(
  ["Choice", "Alternative rejected", "Why"],
  [
    ["GDELT seendate (crawl time)", "Article's self-declared publication date", "seendate is weakly LATER than true publication, so its error is conservative: it can delay a story into a later window, never leak it into an earlier one"],
    ["Calendar derived from the target series", "Hard-coded holiday list", "US, UK and Japanese holidays handled for free; the calendar cannot drift out of sync with the price series"],
    ["window_hours exposed as a feature", "Silently accept unequal windows", "Otherwise the model reads Monday's mechanically larger bucket as a genuine news spike, every single week"],
    ["Chronological split", "Random split", "A random split lets the model interpolate between a Monday and a Wednesday it has seen to “predict” the Tuesday in between"],
  ],
  [2300, 2400, 4660],
));
body.push(CAPTION("Table 8. Alignment decisions and the alternatives rejected."));

body.push(H2("5.3  Asserted, not assumed"));
body.push(RP([["The boundary is verified to the minute by the test suite. An article at 21:59 UTC on 2024-03-05 belongs to that session; one at 22:01 UTC belongs to the next; one exactly at 22:00 belongs to the earlier session, because the window is half-open. Weekend news is asserted to land on Monday. These tests exist because every one of these statements is the kind of thing that is obviously true right up until it is quietly wrong."]]));

body.push(H2("5.4  A leak we actually found"));
body.push(RP([["The leakage assertions run on the joined frame and fail loudly on contemporaneous features, unsorted or duplicated dates, split overlap, or any feature correlating above 0.95 with the target. "], ["On the very first run, they fired:", { b: true }]]));
body.push(...CODE([
  "LEAKAGE CHECK FAILED: features with |corr| > 0.95 vs target (likely leak):",
  "  ['ret_BROAD_USD', 'ret_AFE_USD', 'ret_EME_USD', 'ret_DXY',",
  "   'ret_EURUSD', 'ret_USDJPY', 'ret_GBPUSD', 'ret_USDCNY']",
]));
body.push(RP([["The target frame contained "], ["ret_BROAD_USD", { code: true }], [", which is numerically "], ["identical", { i: true }], [" to "], ["y_return", { code: true }], [". Joined naively, it handed the model its own answer under a different name and would have produced a flawless, worthless R²."]]));
body.push(RP([["Market history is genuinely useful — momentum and volatility are standard controls, and Task 4 needs them to show that news adds information "], ["beyond price itself", { i: true }], [". But only from the past. Every market column is now lagged exactly like the news features and renamed with a "], ["lag1_", { code: true }], [" prefix that makes its timing visible in any coefficient table."]]));
body.push(CALLOUT("Why this matters beyond the fix.",
  "A reviewer reading a 79-column frame would not have spotted it. That is the argument for keeping automated leakage checks inside the pipeline rather than relying on code review — and it is the single most useful thing Task 1 produced."));

body.push(new Paragraph({ children: [new PageBreak()] }));

// ================================================================== §6
body.push(H1("6.  Text Preprocessing"));

body.push(H2("6.1  The argument: preprocessing is model-dependent"));
body.push(P("The reflex answer to “preprocess the text” is: lowercase, strip punctuation, remove stopwords, lemmatise. Applied to a transformer that reflex is actively harmful, and applied to a sentiment task it is sometimes simply wrong. So we maintain one corpus and two views of it."));

body.push(TBL(
  ["Step", "Track A — TF-IDF, LDA, VADER", "Track B — FinBERT, DeBERTa"],
  [
    ["Unicode repair (ftfy, NFKC)", "Yes", "Yes"],
    ["Strip outlet suffix", "Yes", "Yes"],
    ["Entity canonicalisation", "Yes", "Optional (off by default)"],
    ["Lowercase", "Yes", "NO — casing carries entity information"],
    ["Remove stopwords", "Yes, with keep-list", "NO — attention needs function words"],
    ["Lemmatise", "Yes", "NO — subword tokenisers handle morphology"],
    ["Number masking", "<NUM> / <PCT> / <MONEY>", "No"],
    ["Length limit", "min 3 tokens", "max 128 tokens"],
  ],
  [2600, 3380, 3380],
));
body.push(CAPTION("Table 9. The two tracks. Track B does almost nothing, and that is the point."));

body.push(RP([["Track B is minimal because a subword transformer was pre-trained on ordinary cased prose with function words intact. Feeding it a stemmed bag of content words moves the input off-distribution and reliably "], ["costs", { i: true }], [" several points of F1 — the opposite of what “cleaning” is supposed to achieve."]]));

body.push(H2("6.2  The negation keep-list"));
body.push(P("This is the single highest-value line in the configuration file."));
body.push(TBL(
  ["Original headline", "Naive stopword removal", "Ours (with keep-list)"],
  [
    ["Fed will not cut rates in December", "fed cut rates december", "fed will not cut rates december"],
    ["Sanctions are not expected to be lifted", "sanctions expected lifted", "sanctions are not expected lifted"],
    ["China says it will never devalue the yuan", "china says devalue yuan", "china says will never devalue yuan"],
    ["Oil prices down as OPEC talks collapse", "oil prices opec talks collapse", "oil prices down opec talks collapse"],
  ],
  [3100, 3130, 3130],
));
body.push(CAPTION("Table 10. Standard stopword removal inverts the meaning of the first three headlines and destroys the direction of the fourth."));

body.push(RP([["The keep-list preserves negators and directional words: "], ["not, no, nor, never, none, cannot, without, against, down, up, over, under, above, below, off, out, more, less", { code: true }], [". A test in the suite asserts that “not” survives Track A, so this cannot silently regress."]]));

body.push(H2("6.3  Number masking and entity canonicalisation"));
body.push(RP([["The exact figure “4.25%” is a hapax that TF-IDF cannot use, but "], ["the presence of a percentage in a headline", { i: true }], [" is a strong signal that it is a rate or inflation story. Masking to "], ["<PCT>", { code: true }], [" keeps the signal, drops the noise, and removes thousands of useless vocabulary types."]]));
body.push(RP([["Entity canonicalisation runs before everything else on both tracks: “U.S.”, “US”, “United States” and “Washington” all become "], ["UNITED_STATES", { code: true }], [". Without it, the country with the most aliases looks like four rare entities instead of one dominant one. Aliases are matched longest-first, or “united states” would be partially consumed by the shorter alias “us”."]]));

body.push(CALLOUT("Graceful degradation.",
  "NLTK is optional. If its corpora are unavailable — a common situation on a locked-down lab machine — the module falls back to a bundled stopword list and a suffix-rule lemmatiser, and the pipeline still runs end-to-end. A preprocessing step that only works on one team member's laptop is a reproducibility bug."));

// ================================================================== §7
body.push(H1("7.  Schema and Features"));

body.push(H2("7.1  Three tables"));
body.push(TBL(
  ["Table", "Grain", "Purpose"],
  [
    ["articles_raw", "article", "Everything GDELT returned, before any judgement"],
    ["articles_flagged", "article", "Every filter flag, dedup cluster, and both text tracks — the audit trail"],
    ["dataset", "trading day", "THE modelling table: lagged features + labels + chronological split"],
  ],
  [2100, 1500, 5760],
));
body.push(CAPTION("Table 11. Three tables, one per stage of judgement. The middle one is what makes every decision reversible."));

body.push(H2("7.2  The feature principle: never a raw count"));
body.push(RP([["GDELT's indexed source list expanded materially around 2018, and our own 250-records-per-call ceiling truncates busy days. Both put non-stationary, non-economic trend into raw volume. A model fed raw counts will learn "], ["“later years have more articles”", { i: true }], [" and call it signal."]]));
body.push(P("So every volume feature is expressed relatively:"));
body.push(RBULLET([["z-score against a trailing 30/90-day window", { b: true }], [" — “is today unusual "], ["for this era", { i: true }], ["?”"]]));
body.push(RBULLET([["share of the day's total", { b: true }], [" — composition, immune to level shifts"]]));
body.push(RBULLET([["articles_per_24h", { code: true }], [" — corrects for the 65-hour Monday window"]]));
body.push(RBULLET([["log1p", { code: true }], [" — tames the heavy right tail"]]));

body.push(TBL(
  ["Group", "Examples", "What it captures"],
  [
    ["Volume (relative)", "vol_z30, vol_z90, articles_per_24h", "Is the news flow unusual for this period?"],
    ["Syndication", "syndication_mean / _max / _p90", "The newsroom's own importance vote, recovered from dedup"],
    ["Theme composition", "share_conflict … share_monetary_policy, z_*", "What KIND of news, not just how much"],
    ["Concentration", "theme_entropy, geo_entropy, geo_hhi", "One big event vs diffuse chatter"],
    ["Source composition", "share_tier1, share_state_media", "Who is doing the reporting"],
    ["Calendar", "window_hours, is_post_weekend", "Prevents the Monday artefact from reading as a spike"],
    ["GDELT uncapped", "gdelt_volume_*, gdelt_tone_*", "Control for the 250-record ceiling"],
    ["Lagged market", "lag1_ret_*, lag1_lvl_VIX, lag1_chg_DGS10", "Momentum and risk-appetite controls, strictly past"],
  ],
  [2000, 3680, 3680],
));
body.push(CAPTION("Table 12. Feature groups (53 columns in the reference run). All rolling statistics are trailing and computed before the alignment shift."));

body.push(CALLOUT("Zero-fill, not forward-fill.",
  "A trading day with no matching articles genuinely had none. Forward-filling would invent yesterday's news flow and — worse — smear an event across days, blurring exactly the event-study timing the hypothesis depends on."));

body.push(H2("7.3  Target construction"));
body.push(RP([["Log returns, not levels. ", { b: true }], ["An FX index level is a near unit-root process. A model fed levels scores a spectacular R² by predicting “tomorrow ≈ today” while learning nothing about geopolitics. Log returns are also additive across time and scale-free, so DXY (~100) and USDIDR (~16,000) become directly comparable."]]));
body.push(RP([["A ternary label with a dead-zone. ", { b: true }], ["Moves smaller than 10 bp are labelled FLAT. Without the dead-zone, the label on a quiet day is sign(noise), and the model burns its capacity on microstructure it cannot possibly predict from news."]]));
body.push(RP([["A stale-quote guard. ", { b: true }], ["An "], ["exactly", { i: true }], [" zero log return on an index of 26 currencies is a repeated print carried over a holiday, not a genuine no-change day. Left in, these fake days inflate the FLAT class and flatter any classifier."]]));

body.push(new Paragraph({ children: [new PageBreak()] }));

// ================================================================== §8
body.push(H1("8.  Quality Assurance"));

body.push(P("Every check exists because of a specific, named way this dataset can lie to us. A QA suite that prints df.describe() catches none of them."));

body.push(TBL(
  ["Check", "The failure it is looking for"],
  [
    ["coverage_report", "A silent outage — a changed parameter or an overnight rate limit — whose only symptom is a quietly thinner year"],
    ["cap_saturation_report", "Systematic censoring of busy days (see 8.1)"],
    ["structural_break_report", "GDELT's crawler expansion contaminating raw counts"],
    ["fx_sanity_report", "Stale prints, impossible returns, and whether the return series behaves as FX theory predicts"],
    ["corpus_quality_report", "Duplicates, empty text, language leakage"],
    ["leakage_assertions", "Look-ahead, split overlap, target-in-features"],
    ["validity_summary", "One pass/fail table over all of the above"],
  ],
  [2500, 6860],
));
body.push(CAPTION("Table 13. The QA suite."));

body.push(H2("8.1  Cap saturation — the subtlest check here"));
body.push(RP([["When a request returns the full 250 records, we did not "], ["sample", { i: true }], [" that day — we took the first 250 the API chose to return."]]));
body.push(RP([["Saturation is "], ["correlated with news intensity", { b: true }], [", which is precisely our predictor. So the measurement error is not random noise; it is "], ["systematic censoring that compresses exactly the big-news days the hypothesis cares about", { b: true }], [". If saturation proves common, the honest fixes are to raise "], ["sub_day_chunks", { code: true }], [" or to lean on the uncapped timeline volume. The report tells us which."]]));

body.push(H2("8.2  Structural break"));
body.push(P("Yearly median volume is the crawler-expansion check. If the medians jump by a factor unrelated to world events, raw counts are contaminated and only relative features may be used downstream. The table is the evidence for that decision rather than a hunch about it."));

body.push(H2("8.3  Reference run"));
body.push(RP([["The figures below come from a synthetic-fixture run (386 sessions) used to verify the pipeline end-to-end without network access. "], ["They validate the code path, not the hypothesis", { b: true }], [" — "], ["manifest.json", { code: true }], [" records "], ["offline_synthetic: true", { code: true }], [" so a fixture run can never be mistaken for results. The live census reproduces the same tables."]]));

body.push(TBL(
  ["Funnel stage", "Removed", "Surviving", "% of raw"],
  [
    ["0. Retrieved (post-thematic query)", "—", "19,190", "100.0"],
    ["0a. minus blocklisted domains", "3,423", "15,767", "82.2"],
    ["2. minus non-material", "0", "15,767", "82.2"],
    ["3. minus duplicates", "3,030", "12,737", "66.4"],
  ],
  [4060, 1700, 1900, 1700],
));
body.push(CAPTION("Table 14. Funnel attrition. Roughly one third of retrieved articles is removed, the majority by deduplication."));

body.push(TBL(
  ["Validity check", "Status", "Detail"],
  [
    ["Corpus non-empty after filtering", "PASS", "12,737 articles kept"],
    ["News coverage of trading days ≥ 90%", "PASS", "100.0% of sessions"],
    ["No degenerate class (< 5%)", "PASS", "DOWN 40%, UP 38%, FLAT 22%"],
    ["Duplicate rate under control (< 60%)", "PASS", "19.2% flagged duplicate"],
    ["No leakage assertion failures", "PASS", "all checks passed"],
    ["Target series usable (> 1,000 obs)", "FAIL", "381 returns — correct: the fixture spans 18 months"],
  ],
  [3760, 1300, 4300],
));
body.push(CAPTION("Table 15. Validity summary. The single FAIL is the expected and correct response to a deliberately short fixture window; it passes on the full period."));

body.push(RP([["Retention is even across five of the six theme families (66–71%) and lower for "], ["monetary_policy", { code: true }], [" (54%), which is the expected consequence of that family producing more near-duplicate wire copy. We flag it here because "], ["a filter that quietly wipes out one family has changed the research question without anyone noticing", { b: true }], [" — and that check is worth more than the number itself."]]));

// ================================================================== §9
body.push(H1("9.  Ethics, Bias and Reproducibility"));

body.push(H2("9.1  Coverage bias"));
body.push(RP([["Restricting to English sources produces a Western-media framing bias, and GDELT's own source list is denser in some regions than others. "], ["Coverage is not importance", { b: true }], [": GDELT counts articles, not market impact. A heavily covered event may be fully priced in; a lightly covered one may not be. Our partial mitigations — retaining and flagging state-affiliated outlets, recording "], ["sourcecountry", { code: true }], [", and using concentration rather than raw volume — reduce but do not remove the problem."]]));

body.push(H2("9.2  Legal and ethical use"));
body.push(BULLET("We store and redistribute headlines, URLs and metadata only. No article body text is republished."));
body.push(BULLET("GDELT is a free public research service. The client identifies itself honestly in its User-Agent, rate-limits to roughly one request per 1.2 s, and caches aggressively so that a rerun costs the service nothing."));
body.push(BULLET("No personal data is collected. Named individuals appear only incidentally, in public political reporting."));
body.push(BULLET("State-affiliated media are included as evidence of framing, not as endorsed fact, and are flagged so downstream analysis can treat them accordingly."));

body.push(H2("9.3  Reverse causality — the central threat"));
body.push(RP([["FX moves cause news as well as the reverse. A large dollar move generates explanatory coverage within hours, so a naive corpus will contain articles that "], ["describe", { i: true }], [" the move we are trying to predict."]]));
body.push(P("Three defences are built into the dataset, and one is deferred:"));
body.push(NUM("The Stage-2 gate deliberately admits early, currency-agnostic coverage rather than requiring a market term, so the corpus is not selected for post-hoc explanation."));
body.push(NUM("The one-session lag means every feature precedes its label."));
body.push(NUM("Lagged market controls let Task 4 ask whether news adds information beyond price itself."));
body.push(RP([["4. Deferred: ", { b: true }], ["lead-lag and Granger-causality tests in Task 4. Until those are run, "], ["the hypothesis is not established, only testable", { b: true }], ["."]]));

body.push(H2("9.4  Reproducibility"));
body.push(BULLET("Every stochastic step is seeded from project.random_seed."));
body.push(BULLET("HTTP responses are cached to disk. This freezes the corpus — GDELT back-fills its index, so the same query issued a week apart returns slightly different results. The cache directory is therefore the scientific artefact and should be shipped alongside the code, or at least a checksum manifest of it."));
body.push(BULLET("manifest.json records period, row counts, seed, parameters and leakage verdict for every run."));
body.push(BULLET("No API keys are required by any source, so reproduction needs no shared secret."));
body.push(BULLET("Fifteen property tests run in seconds and cover the invariants that matter."));

// ================================================================== §10
body.push(H1("10.  Handover to Task 2"));

body.push(H2("10.1  Ready"));
body.push(BULLET("dataset.parquet — one row per trading day: lagged news features, labels at three horizons, chronological split column."));
body.push(BULLET("articles_flagged.parquet — article-level text in both preprocessing tracks, ready for TF-IDF (Track A) or a transformer (Track B)."));
body.push(BULLET("Every filter decision preserved as a boolean column, so any stage can be ablated and its contribution measured."));
body.push(BULLET("Fifteen property tests covering alignment, leakage, negation handling and deduplication."));

body.push(H2("10.2  Baselines Task 2 must beat"));
body.push(P("Any model must outperform all three, or the result means nothing:"));
body.push(NUM("Majority-class prediction — surprisingly strong given the FLAT dead-zone."));
body.push(NUM("Lagged market features only (lag1_*), with no news at all. This is the honest test of whether news adds anything."));
body.push(NUM("News-volume z-score alone — one feature, no NLP. If a transformer cannot beat it, the NLP is decorative."));

body.push(H2("10.3  Open threats carried forward"));
body.push(TBL(
  ["Threat", "Status after Task 1", "Where it must be resolved"],
  [
    ["Reverse causality", "Mitigated by design, not eliminated", "Task 4 — lead-lag and Granger tests"],
    ["English-source skew", "Flagged and measurable", "Task 3 — multilingual ablation"],
    ["250-record ceiling", "Measured; mitigated via timelines", "Task 2 — raise sub_day_chunks if saturation is high"],
    ["Coverage ≠ market impact", "Unresolved by design", "Task 4 — interpret with care"],
    ["Daily granularity", "Structural", "Acknowledge; no intraday causal claim is available"],
  ],
  [2300, 3400, 3660],
));
body.push(CAPTION("Table 16. Threats deliberately carried forward, with an owner."));

body.push(H2("10.4  Closing statement"));
body.push(RP([["Task 1 has produced a dataset on which the hypothesis "], ["can", { i: true }], [" be tested, and — equally deliberately — one on which it is difficult to be tested "], ["wrongly", { i: true }], [". The leakage assertions have already caught a defect that would have produced a spurious near-perfect result, and the deduplication window bug found by the test suite would have silently thinned the corpus of genuine repeated events."]]));
body.push(P("Neither of those would have been visible in a final accuracy number. That, more than any table in this report, is the case for having spent Task 1 on guards rather than on volume."));

// ================================================================== appendix
body.push(new Paragraph({ children: [new PageBreak()] }));
body.push(H1("Appendix A.  How to Reproduce"));
body.push(...CODE([
  "pip install -r requirements.txt",
  "python -m nltk.downloader stopwords wordnet omw-1.4    # optional",
  "",
  "# 1. verify the pipeline with no network access (~60 s)",
  "python -m src.build_dataset --offline",
  "",
  "# 2. verify the invariants",
  "python -m pytest tests/ -v",
  "",
  "# 3. fast live run (every 7th day, ~8 min)",
  "python -m src.build_dataset --stride 7 --families conflict,sanctions_trade",
  "",
  "# 4. full census (~60-90 min first time; cached thereafter)",
  "python -m src.build_dataset",
  "",
  "# 5. Stage-4 human audit",
  "python scripts/make_audit_sample.py build",
  "#    ... two members label reports/audit_annotator_{A,B}.csv independently ...",
  "python scripts/make_audit_sample.py score \\",
  "    --a reports/audit_annotator_A.csv --b reports/audit_annotator_B.csv",
]));

body.push(H1("Appendix B.  Module Map"));
body.push(TBL(
  ["Module", "Responsibility"],
  [
    ["config/config.yaml", "Every tunable, with its rationale as a comment. No magic numbers in code."],
    ["src/config.py", "YAML loader and path management"],
    ["src/http_cache.py", "Cached, retrying, self-identifying HTTP client"],
    ["src/fx_data.py", "Target: FX series, controls, return and label construction"],
    ["src/gdelt.py", "News: article headlines (DOC API) and theme timelines"],
    ["src/filtering.py", "Funnel stages 0 and 2, attrition accounting"],
    ["src/dedup.py", "Stage 3 — SimHash near-duplicate clustering"],
    ["src/align.py", "News-day windowing, prediction frame, leakage assertions"],
    ["src/preprocess.py", "Dual-track text preprocessing"],
    ["src/features.py", "Daily panel aggregation"],
    ["src/qa.py", "Data-quality checks"],
    ["src/synthetic.py", "Offline fixtures for testing"],
    ["src/build_dataset.py", "Pipeline driver"],
    ["scripts/make_audit_sample.py", "Stage 4 human audit — build and score"],
    ["tests/test_pipeline.py", "15 property tests"],
    ["notebooks/task1_data_acquisition.ipynb", "Narrated end-to-end walkthrough"],
  ],
  [3000, 6360],
));

// ================================================================== document
const doc = new Document({
  creator: "NLP Project Team",
  title: "Task 1 — Strategic Data Acquisition & Preprocessing",
  description: "Predicting Global Geopolitical Events: Impact on the USD Exchange Rate",
  styles: {
    default: {
      document: { run: { font: "Calibri", size: 21 }, paragraph: { spacing: { line: 276 } } },
    },
  },
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [
          { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 400, hanging: 220 } } } },
          { level: 1, format: LevelFormat.BULLET, text: "◦", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 760, hanging: 220 } } } },
        ],
      },
      {
        reference: "numbers",
        levels: [
          { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 400, hanging: 240 } } } },
        ],
      },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1200, right: 1440, bottom: 1200, left: 1440 },
      },
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "C9D3E3", space: 4 } },
          children: [new TextRun({
            text: "Task 1 — Strategic Data Acquisition & Preprocessing",
            size: 16, color: GREY })],
        })],
      }),
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ children: ["Page ", PageNumber.CURRENT, " of ", PageNumber.TOTAL_PAGES],
            size: 16, color: GREY })],
        })],
      }),
    },
    children: body,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  fs.writeFileSync(OUT, buf);
  console.log("wrote", OUT, (buf.length / 1024).toFixed(0) + " KB");
});
