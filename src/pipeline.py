"""
End-to-end agent: classify -> retrieve+draft -> escalation decision.
This is the single entry point used by both eval/harness.py and any
interactive use.
"""
from __future__ import annotations

import argparse
import json
import re

import pandas as pd

from .data_prep import load_raw, build_brand_pairs
from .classify import classify
from .draft_reply import draft_reply
from .escalation import decide_escalation
from .retrieval import ResolutionIndex


def build_index_from_history(history_df: pd.DataFrame) -> ResolutionIndex:
    """history_df must already have an 'intent' column (from classify or
    golden labels) to enable intent-filtered retrieval."""
    return ResolutionIndex.from_dataframe(history_df, intent_col="intent")


def extract_handle(text: str) -> str | None:
    m = re.search(r"@(\w+)", text)
    # crude: skip the brand's own handle if it's first
    for handle in re.findall(r"@(\w+)", text):
        if handle.lower() != "amazonhelp":
            return handle
    return None


def run_one(customer_text: str, index: ResolutionIndex) -> dict:
    cls = classify(customer_text)
    draft = draft_reply(
        customer_text, index,
        intent=cls["intent"],
        customer_handle=extract_handle(customer_text),
    )
    esc = decide_escalation(customer_text, cls["intent"], cls["confidence"], draft)
    return {
        "customer_text": customer_text,
        "intent": cls["intent"],
        "intent_confidence": cls["confidence"],
        "intent_rationale": cls.get("rationale"),
        "reply": draft["reply"],
        "grounded_on_examples": draft.get("grounded_on_examples"),
        "n_examples_retrieved": draft.get("n_examples_retrieved"),
        "escalate": esc["escalate"],
        "decision": esc["decision"],
        "escalation_reasons": esc["reasons"],
        "anger_score": esc["anger_score"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/sample_twcs.csv")
    ap.add_argument("--brand", default="AmazonHelp")
    ap.add_argument("--message", required=True, help="A single customer message to run through the agent")
    args = ap.parse_args()

    raw = load_raw(args.data)
    pairs = build_brand_pairs(raw, args.brand)
    pairs = pairs[pairs["brand_reply_text_clean"].notna()].copy()

    # Bootstrap intents on the history using the same classifier, so
    # retrieval can be intent-filtered. In a real deployment you'd cache
    # this (see README "productionizing").
    print("Classifying historical cases for retrieval index (one-time)...")
    pairs["intent"] = pairs["customer_text_clean"].map(lambda t: classify(t)["intent"])
    index = build_index_from_history(pairs)

    result = run_one(args.message, index)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
