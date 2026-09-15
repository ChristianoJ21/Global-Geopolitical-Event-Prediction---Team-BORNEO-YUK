"""Task 1 — Strategic Data Acquisition & Preprocessing.

Package layout
--------------
    config       YAML-backed configuration (no magic numbers in code)
    http_cache   polite, cached, retrying HTTP client
    fx_data      TARGET: USD exchange-rate series + macro controls
    gdelt        NEWS: article headlines + daily theme timelines
    filtering    the four-stage filtering funnel
    dedup        three-level deduplication with SimHash
    align        news-to-trading-day alignment and leakage assertions
    preprocess   dual-track text preprocessing (sparse vs transformer)
    features     daily panel aggregation
    qa           data-quality checks
    synthetic    offline fixtures for testing
    build_dataset  pipeline driver (python -m src.build_dataset)
"""
__version__ = "1.0.0"
