"""
Tests that don't require an ANTHROPIC_API_KEY -- data prep, retrieval,
baselines, and metrics computation. Classification/drafting/judging
(anything that calls the LLM) is covered by eval/harness.py + manual runs,
not here, since mocking the API well enough to be meaningful is basically
duplicating the mock dry-run already shown in the README.

Run with: python -m pytest tests/ -v
"""
import sys
sys.path.insert(0, ".")

import pandas as pd

from src.data_prep import load_raw, build_brand_pairs, clean_text
from src.retrieval import ResolutionIndex, HistoricalCase
from src.baselines import trivial_predict, simple_predict
from src.intents import INTENT_LIST, BASE_RISK


def test_clean_text_strips_urls_and_whitespace():
    assert clean_text("hi   there https://x.co/abc   now") == "hi there now"


def test_build_brand_pairs_matches_replies():
    raw = load_raw("data/raw/sample_twcs.csv")
    pairs = build_brand_pairs(raw, "AmazonHelp")
    assert len(pairs) > 0
    assert pairs["brand_reply_text_clean"].notna().sum() > 0
    # every customer row should actually mention the brand
    assert pairs["text"].str.contains("AmazonHelp", case=False).all()


def test_resolution_index_retrieves_relevant_case():
    cases = [
        HistoricalCase("my order is late", "sorry, DM your order number"),
        HistoricalCase("I want a refund for my broken item", "DM us for a refund"),
        HistoricalCase("how do I cancel my prime trial", "cancel from account settings"),
    ]
    index = ResolutionIndex(cases)
    results = index.search("where is my package, it's late", k=1)
    assert len(results) == 1
    assert "late" in results[0].customer_text


def test_trivial_baseline_always_escalates():
    pred = trivial_predict("anything at all")
    assert pred["decision"] == "escalate_to_human"


def test_simple_baseline_classifies_account_access():
    pred = simple_predict("I can't log in, locked out of my account")
    assert pred["intent"] == "account_access_issue"
    assert pred["decision"] == "escalate_to_human"


def test_simple_baseline_classifies_routine_order_status_as_low_risk():
    pred = simple_predict("where is my order, tracking hasn't updated")
    assert pred["intent"] == "order_status_delay"
    assert pred["decision"] == "auto_handle"


def test_intent_list_matches_base_risk_keys():
    assert set(INTENT_LIST) == set(BASE_RISK.keys())


def test_golden_set_is_in_expected_size_range():
    golden = pd.read_csv("data/golden_eval_set.csv")
    assert 150 <= len(golden) <= 250, f"golden set has {len(golden)} rows, expected 150-250"
    assert set(golden["gold_intent"].unique()).issubset(set(INTENT_LIST))
