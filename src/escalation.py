"""
Escalation decision: auto_handle vs escalate_to_human, with a stated reason.

Deliberately rule-first, LLM-second: escalation is a safety-critical binary
decision, and rules over concrete signals (dollar amounts, security
keywords, low classifier confidence, weak grounding) are auditable and
reproducible in a way that "ask the LLM if this is risky" alone is not. The
LLM is only used for the one signal that's genuinely hard to regex:
sentiment/anger level. See DECISION_LOG.md for the reasoning.

This function is intentionally conservative (biased toward escalation) --
see report/REPORT.md "what's misleading about my headline number" for why a
low escalation rate is not automatically good.
"""
from __future__ import annotations

import re

from .intents import BASE_RISK
from .llm_client import call_llm_json

MONEY_RE = re.compile(r"\$\s?\d+(\.\d{2})?")
SECURITY_KEYWORDS = ["hacked", "fraud", "unauthorized", "stolen", "scam", "security"]
REPEAT_KEYWORDS = ["again", "third time", "still hasn't", "still haven't", "keep", "multiple times", "every time"]

SENTIMENT_SYSTEM = """Rate the emotional intensity of this customer support \
tweet on a 0.0-1.0 scale, where 0.0 is neutral/calm and 1.0 is extremely \
angry or distressed. Return ONLY JSON: {"anger": <float>}"""


def _sentiment_anger(text: str) -> float:
    try:
        result = call_llm_json(SENTIMENT_SYSTEM, text, max_tokens=400)
        return float(result.get("anger", 0.0))
    except Exception:
        return 0.0


def decide_escalation(customer_text: str, intent: str, classifier_confidence: float,
                       draft: dict, use_llm_sentiment: bool = True) -> dict:
    reasons = []
    escalate = False

    if classifier_confidence < 0.6:
        escalate = True
        reasons.append(f"low classifier confidence ({classifier_confidence:.2f} < 0.60)")

    if MONEY_RE.search(customer_text) and BASE_RISK.get(intent) in ("medium", "high"):
        escalate = True
        reasons.append("specific dollar amount mentioned alongside a billing/refund-risk intent")

    if any(kw in customer_text.lower() for kw in SECURITY_KEYWORDS):
        escalate = True
        reasons.append("security/fraud keyword detected -- account safety issues are never auto-handled")

    if any(kw in customer_text.lower() for kw in REPEAT_KEYWORDS):
        escalate = True
        reasons.append("language suggests a repeat/unresolved contact")

    if not draft.get("grounded_on_examples", False):
        escalate = True
        reasons.append("no closely matching historical resolution found to ground the reply -- drafting from scratch is riskier")

    if draft.get("used_facts_not_in_input", False):
        escalate = True
        reasons.append("draft may state a fact not supported by the customer message or examples")

    if intent == "account_access_issue":
        escalate = True
        reasons.append("intent policy: account access/security issues always require human verification")

    anger = None
    if use_llm_sentiment:
        anger = _sentiment_anger(customer_text)
        if anger >= 0.7:
            escalate = True
            reasons.append(f"high customer anger/distress detected (score {anger:.2f} >= 0.70)")

    if not reasons:
        reasons.append("routine, low-risk intent; confident classification; reply grounded in a matching historical resolution")

    return {
        "escalate": escalate,
        "decision": "escalate_to_human" if escalate else "auto_handle",
        "reasons": reasons,
        "anger_score": anger,
    }