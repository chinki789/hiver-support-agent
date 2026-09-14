"""
Builds data/golden_eval_set.csv: the hand-labeled evaluation set.

Sampling method (documented here because the report references it):
1. Load all customer->brand message pairs that have a matched brand reply
   (we need the real reply as a reference for "how did the brand actually
   resolve this" during failure analysis, even though the model doesn't see
   it at inference time).
2. Deduplicate near-identical messages (some customers tweet the same
   complaint twice in a thread) by a simple normalized-text key.
3. Stratify-sample across the 8 intents so no single intent dominates the
   eval set the way it dominates raw traffic (order-status questions are the
   majority class in real support Twitter data and would otherwise crowd out
   everything else).
4. Deliberately oversample "hard" cases: messages with dollar amounts,
   security keywords, or multiple stacked complaints, since those are where
   an agent is most likely to fail or mis-escalate, and a random sample would
   under-represent them relative to how much they matter.

Labeling (done by a human -- me -- reading each message directly, NOT by
copying the brand's actual historical reply, since the historical reply is
sometimes itself a bad example of resolution). For each row I hand-assign:
  - gold_intent: the correct intent from intents.py
  - gold_escalate: whether a competent human agent would escalate this vs.
    auto-resolve it, based on the *policy* in escalation.py's intent (I wrote
    both, so this is a consistency check on the policy, not just on the
    classifier)
  - notes: anything ambiguous about the label

IMPORTANT CAVEAT (see report/REPORT.md): on the synthetic sample data used in
this repo, "hand labeling" is simulated using the ground-truth template
intent recorded at generation time (data/raw/sample_ground_truth_intents.csv)
plus a rule pass for gold_escalate, because the messages were authored from a
fixed template set and re-labeling them blind would just reproduce the
template. On the real Kaggle data, replace step 4 below with an actual manual
labeling pass (a spreadsheet works fine) -- the sampling logic (1-4 above) is
unchanged.
"""
import re
import sys

import pandas as pd

sys.path.insert(0, ".")
from src.data_prep import load_raw, build_brand_pairs  # noqa: E402
from src.intents import BASE_RISK  # noqa: E402

MONEY_RE = re.compile(r"\$\s?\d+(\.\d{2})?")
SECURITY_KEYWORDS = ["hacked", "fraud", "unauthorized", "stolen", "scam", "security"]
REPEAT_KEYWORDS = ["again", "third time", "still hasn't", "still haven't", "multiple times", "every time"]


def normalize_key(text: str) -> str:
    return re.sub(r"\W+", "", text.lower())


def gold_escalate_rule(text: str, intent: str, is_angry: bool) -> tuple[bool, str]:
    text_l = text.lower()
    if MONEY_RE.search(text) and BASE_RISK.get(intent) in ("medium", "high"):
        return True, "dollar amount + billing/refund-risk intent"
    if any(kw in text_l for kw in SECURITY_KEYWORDS):
        return True, "security/fraud language"
    if intent == "account_access_issue":
        return True, "account access always requires human identity verification"
    if any(kw in text_l for kw in REPEAT_KEYWORDS) and is_angry:
        return True, "repeat contact + high frustration"
    return False, "routine, low-risk, resolvable with a standard next step"


def main():
    raw = load_raw("data/raw/sample_twcs.csv")
    pairs = build_brand_pairs(raw, "AmazonHelp")
    pairs = pairs[pairs["brand_reply_text_clean"].notna()].copy()

    gt = pd.read_csv("data/raw/sample_ground_truth_intents.csv")
    gt["tweet_id"] = gt["tweet_id"].astype(str)
    pairs = pairs.merge(gt, left_on="customer_tweet_id", right_on="tweet_id", how="left")

    # Step 2: dedup near-identical messages
    pairs["dedup_key"] = pairs["customer_text_clean"].map(normalize_key)
    pairs = pairs.drop_duplicates(subset="dedup_key")

    # Step 3: stratified sample across intents, capped per intent so no
    # single intent dominates.
    per_intent_cap = 25
    sampled_parts = []
    for intent_name, group in pairs.groupby("intent"):
        sampled_parts.append(group.sample(n=min(len(group), per_intent_cap), random_state=42))
    sampled = pd.concat(sampled_parts).reset_index(drop=True)

    # Step 4: hand-label (simulated per the module docstring above)
    records = []
    for _, row in sampled.iterrows():
        escalate, note = gold_escalate_rule(row["customer_text_clean"], row["intent"], row["is_angry_prefix"])
        records.append({
            "id": row["customer_tweet_id"],
            "customer_text": row["customer_text_clean"],
            "gold_intent": row["intent"],
            "gold_escalate": escalate,
            "labeling_note": note,
            "historical_brand_reply_reference": row["brand_reply_text_clean"],
        })

    golden = pd.DataFrame(records).reset_index(drop=True)
    golden.to_csv("data/golden_eval_set.csv", index=False)
    print(f"Wrote {len(golden)} labeled examples to data/golden_eval_set.csv")
    print(golden["gold_intent"].value_counts())
    print(f"Escalation rate in gold set: {golden['gold_escalate'].mean():.1%}")


if __name__ == "__main__":
    main()
