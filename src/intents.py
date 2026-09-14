"""
Intent taxonomy for the chosen brand: AmazonHelp.

How this taxonomy was built (see report/REPORT.md "Problem framing" for the
full rationale): I read ~250 raw customer-initiated tweets directed at
@AmazonHelp, open-coded them, merged near-duplicate labels, and collapsed
anything with < 4% frequency into "other". The result is 8 intents that
cover the observed traffic and map to genuinely different reply strategies
and different escalation risk profiles -- that second property is the actual
design constraint, not just "looks like a clean taxonomy."
"""

INTENTS = {
    "order_status_delay": (
        "Customer is asking where an order/package is, or reporting it is "
        "late / hasn't arrived by the promised date."
    ),
    "damaged_or_wrong_item": (
        "Customer received an item that is broken, defective, missing parts, "
        "or is not what they ordered."
    ),
    "refund_or_return": (
        "Customer is asking for a refund, a return label, or the status of a "
        "refund/return already in progress."
    ),
    "billing_charge_dispute": (
        "Customer is disputing a charge, was billed incorrectly, double "
        "charged, or charged after cancelling."
    ),
    "account_access_issue": (
        "Customer cannot log in, is locked out, suspects fraud/unauthorized "
        "access, or has a password/2FA problem."
    ),
    "cancellation_request": (
        "Customer wants to cancel an order, subscription (e.g. Prime), or a "
        "pending action before it ships/renews."
    ),
    "product_or_service_question": (
        "Pre-purchase or how-to question about a product, Prime, or a "
        "service feature. No error or loss has occurred."
    ),
    "other_complaint": (
        "General complaint, venting, or anything that doesn't cleanly fit "
        "the categories above (e.g. app bug reports, delivery driver "
        "behavior, packaging complaints)."
    ),
}

INTENT_LIST = list(INTENTS.keys())

# Escalation risk tier per intent -- used as one input signal to escalation.py.
# This is a prior, not the final decision: escalation.py also looks at
# sentiment, monetary amounts mentioned, repeat-contact signals, and
# classifier confidence.
BASE_RISK = {
    "order_status_delay": "low",
    "damaged_or_wrong_item": "medium",
    "refund_or_return": "medium",
    "billing_charge_dispute": "high",
    "account_access_issue": "high",
    "cancellation_request": "medium",
    "product_or_service_question": "low",
    "other_complaint": "medium",
}


def intent_prompt_block() -> str:
    lines = [f"- {name}: {desc}" for name, desc in INTENTS.items()]
    return "\n".join(lines)
