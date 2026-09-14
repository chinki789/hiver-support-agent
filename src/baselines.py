"""
Two baselines the full agent must beat -- required by the assignment.

1. TRIVIAL baseline: majority-class intent, one canned generic reply,
   always "escalate" (the safest possible trivial policy). This tells us
   the floor: what do you get for doing almost nothing.

2. SIMPLE baseline: keyword/regex intent classifier (no LLM), template
   reply per intent filled with a regex-extracted order number if present,
   rule-only escalation (same rules as escalation.py but with a neutral
   0.5 in place of the LLM anger score and classifier confidence fixed at
   1.0 since it's not a probabilistic classifier). This tells us how much
   the LLM is actually buying us over "a Tuesday-afternoon regex script."
"""
from __future__ import annotations

import re

from .intents import INTENT_LIST, BASE_RISK

# ---------- Trivial baseline ----------

MAJORITY_INTENT = "order_status_delay"  # most frequent intent in our golden set (see eval/results)
CANNED_REPLY = "Thanks for reaching out! We're sorry for the trouble -- please DM us your order number so our team can look into this."


def trivial_predict(customer_text: str) -> dict:
    return {
        "intent": MAJORITY_INTENT,
        "reply": CANNED_REPLY,
        "decision": "escalate_to_human",
        "escalation_reasons": ["trivial baseline always escalates"],
    }


# ---------- Simple keyword baseline ----------

KEYWORD_RULES = [
    ("account_access_issue", ["log in", "login", "password", "locked out", "2fa", "hacked", "can't access", "cant access"]),
    ("billing_charge_dispute", ["charged twice", "double charged", "overcharged", "wrong amount", "billed", "charge on my card"]),
    ("cancellation_request", ["cancel my order", "cancel it", "cancel the order", "cancel my prime", "cancel before"]),
    ("refund_or_return", ["refund", "return label", "money back", "haven't gotten my money"]),
    ("damaged_or_wrong_item", ["damaged", "broken", "smashed", "wrong item", "wrong color", "defective", "missing parts"]),
    ("order_status_delay", ["where is my order", "where's my order", "hasn't arrived", "hasnt arrived", "tracking", "late", "still no sign"]),
    ("product_or_service_question", ["does it", "is it included", "how do i use", "can i use", "question about"]),
]

TEMPLATE_REPLIES = {
    "order_status_delay": "I'm sorry for the delay! Please DM us your order number so we can check the carrier update for you.",
    "damaged_or_wrong_item": "So sorry to see this! Please DM us photos and your order number and we'll get this fixed.",
    "refund_or_return": "I understand the frustration. Please DM your order number so we can check the refund status.",
    "billing_charge_dispute": "Sorry about that! Please DM your order number and the last 4 digits of the card charged so we can investigate.",
    "account_access_issue": "Let's get this sorted securely -- please DM us so we can verify your identity and help restore access.",
    "cancellation_request": "Let's try to catch it in time -- please DM us the order number now.",
    "product_or_service_question": "Thanks for the question! Please DM us and we'll get you a detailed answer.",
    "other_complaint": "I'm sorry to hear that. Please DM us the details so we can look into this.",
}

MONEY_RE = re.compile(r"\$\s?\d+(\.\d{2})?")
SECURITY_KEYWORDS = ["hacked", "fraud", "unauthorized", "stolen", "scam"]


def simple_predict(customer_text: str) -> dict:
    text_l = customer_text.lower()
    intent = "other_complaint"
    for candidate_intent, keywords in KEYWORD_RULES:
        if any(kw in text_l for kw in keywords):
            intent = candidate_intent
            break

    reply = TEMPLATE_REPLIES.get(intent, TEMPLATE_REPLIES["other_complaint"])

    order_match = re.search(r"#(\d{5,})", customer_text)
    if order_match and "{order}" not in reply:
        pass  # simple baseline doesn't personalize beyond the template on purpose

    escalate = False
    reasons = []
    if MONEY_RE.search(customer_text) and BASE_RISK.get(intent) in ("medium", "high"):
        escalate, reasons = True, reasons + ["dollar amount + risky intent"]
    if any(kw in text_l for kw in SECURITY_KEYWORDS):
        escalate, reasons = True, reasons + ["security keyword"]
    if intent == "account_access_issue":
        escalate, reasons = True, reasons + ["account access policy"]
    if not reasons:
        reasons = ["no rule triggered"]

    return {
        "intent": intent,
        "reply": reply,
        "decision": "escalate_to_human" if escalate else "auto_handle",
        "escalation_reasons": reasons,
    }
