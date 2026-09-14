"""
LLM-as-judge rubric for reply quality, plus the human-agreement check the
assignment requires.

Rubric (each 1-5, judge sees customer message + candidate reply + the real
historical brand reply as a *reference*, not a required match):
  - helpfulness: does the reply move the customer toward resolution
  - tone_fit: does it match a professional, empathetic brand-support voice
  - faithfulness: does it avoid inventing facts (order numbers, amounts,
    policies) not present in the customer message or the reference
  - actionability: is there a concrete next step (not just an apology)

We keep the rubric to 4 axes on purpose -- more axes without more human
labels just adds judge noise we can't validate. See DECISION_LOG.md.
"""
from __future__ import annotations

import json
import sys

import pandas as pd

sys.path.insert(0, ".")
from src.llm_client import call_llm_json  # noqa: E402

JUDGE_SYSTEM = """You are grading a customer-support reply for the brand \
@AmazonHelp. You'll see the customer's message, a reference reply (how the \
brand actually resolved a similar case historically -- NOT a required exact \
match, just context), and the candidate reply to grade.

Score the candidate reply on these 4 axes, 1 (poor) to 5 (excellent):
- helpfulness: does it move the customer toward resolution
- tone_fit: professional, empathetic brand-support voice
- faithfulness: does NOT invent order numbers, refund amounts, dates, or \
policy claims not supported by the customer message or reference
- actionability: gives a concrete next step, not just an apology

Return ONLY JSON:
{"helpfulness": <1-5>, "tone_fit": <1-5>, "faithfulness": <1-5>, \
"actionability": <1-5>, "overall": <1-5>, "justification": "<one sentence>"}
"""


def judge_reply(customer_text: str, candidate_reply: str, reference_reply: str = "") -> dict:
    user = f"""Customer message: {customer_text}

Reference reply (historical, for context only): {reference_reply or '(none available)'}

Candidate reply to grade: {candidate_reply}
"""
    return call_llm_json(JUDGE_SYSTEM, user, max_tokens=500)


def judge_results_file(results_csv: str, out_csv: str):
    df = pd.read_csv(results_csv)
    scores = []
    for _, r in df.iterrows():
        try:
            s = judge_reply(r["customer_text"], r["pred_reply"], r.get("reference_reply", ""))
        except Exception as e:  # noqa: BLE001
            s = {"helpfulness": None, "tone_fit": None, "faithfulness": None,
                 "actionability": None, "overall": None, "justification": f"judge error: {e}"}
        scores.append(s)
    scored = pd.concat([df.reset_index(drop=True), pd.DataFrame(scores)], axis=1)
    scored.to_csv(out_csv, index=False)
    means = scored[["helpfulness", "tone_fit", "faithfulness", "actionability", "overall"]].mean(numeric_only=True)
    print("Mean judge scores:")
    print(means.round(2).to_string())
    return scored


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="eval/results/full_results.csv")
    ap.add_argument("--out", default="eval/results/full_results_judged.csv")
    args = ap.parse_args()
    judge_results_file(args.results, args.out)
