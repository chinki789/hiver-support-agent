"""
Drafts a reply grounded in retrieved historical resolutions for similar
issues (see retrieval.py). The prompt explicitly instructs the model to
match the brand's tone and typical resolution pattern from the examples,
but NOT to invent order numbers, refund amounts, or policy claims that
aren't supported by the examples or the customer's own message.
"""
from __future__ import annotations

from .llm_client import call_llm_json
from .retrieval import ResolutionIndex

SYSTEM = """You are drafting a reply as the @AmazonHelp Twitter support \
account. You will be given the customer's message and 1-2 examples of how \
this brand has actually resolved similar issues in the past. Write a reply \
that:
- Matches the brand's real tone and structure from the examples (empathy \
line, then a concrete next step, often asking the customer to DM specific \
info).
- Does NOT invent facts: no specific order numbers, refund amounts, dates, \
or policy claims that aren't in the customer's message or the examples.
- Is a single tweet-length reply (under 280 characters), starting with \
@{{customer_handle}} if a handle is given.
- If the examples don't clearly cover this situation, fall back to a safe, \
generic acknowledgment + request for details via DM rather than guessing.

Return ONLY a JSON object:
{"reply": "<the drafted reply text>",
 "grounded_on_examples": <true/false -- true only if you actually used the \
resolution pattern from the examples, not just brand voice>,
 "used_facts_not_in_input": <true/false -- true if you are aware you had to \
state anything not directly supported by the customer message or examples>}
"""


def draft_reply(customer_text: str, index: ResolutionIndex, intent: str | None = None,
                 customer_handle: str | None = None, k: int = 2) -> dict:
    examples = index.search(customer_text, k=k, intent_filter=intent)
    example_block = "\n\n".join(
        f"Example {i+1}:\nCustomer: {c.customer_text}\nBrand reply: {c.brand_reply}"
        for i, c in enumerate(examples)
    ) or "(no closely matching historical example found)"

    user = f"""Customer message: {customer_text}
Customer handle: {customer_handle or '(unknown)'}

Historical examples of how this brand resolved similar issues:
{example_block}
"""
    try:
        result = call_llm_json(SYSTEM, user, max_tokens=700)
    except (ValueError, KeyError) as e:
        # Fail safe rather than crash the whole eval run: a generic,
        # non-committal reply is always safer to send than nothing, and
        # escalation.py will (correctly) flag this as not grounded, which
        # pushes it toward escalation anyway.
        result = {
            "reply": "Thanks for reaching out! Please DM us your order number "
                      "and we'll look into this right away.",
            "grounded_on_examples": False,
            "used_facts_not_in_input": False,
            "draft_error": str(e),
        }
    result["n_examples_retrieved"] = len(examples)
    result["example_texts"] = [c.brand_reply for c in examples]
    return result