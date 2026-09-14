"""
Intent classification via LLM with a fixed label set (see intents.py).
We ask for a confidence score too, because escalation.py needs it -- a
low-confidence classification is itself an escalation signal, independent of
which label won.
"""
from __future__ import annotations

from .intents import INTENT_LIST, intent_prompt_block
from .llm_client import call_llm_json

SYSTEM = """You are an intent classifier for customer support tweets sent to \
the Amazon support Twitter account (@AmazonHelp). Classify the customer's \
message into exactly one of the following intents:

{intents}

Return ONLY a JSON object, no other text:
{{"intent": "<one of the intent keys above, exactly as written>",
  "confidence": <float 0.0-1.0>,
  "rationale": "<one short sentence>"}}
""".format(intents=intent_prompt_block())


def classify(customer_text: str) -> dict:
    # max_tokens is generous (not just enough for the JSON itself) because
    # some providers/models spend a variable, sometimes large chunk of the
    # token budget on internal reasoning before the visible answer, and a
    # tight limit truncates the JSON before it's complete. Kept moderate
    # (not huge) since total tokens also count against free-tier daily quotas.
    try:
        result = call_llm_json(SYSTEM, customer_text, max_tokens=700)
    except (ValueError, KeyError) as e:
        # The model returned unparseable/truncated JSON. Rather than crash
        # the entire evaluation run over one bad response, fail safe into a
        # low-confidence "other_complaint" -- low confidence is itself an
        # escalation trigger in escalation.py, which is the right behavior
        # here: we genuinely don't know what this message is about.
        return {
            "intent": "other_complaint",
            "confidence": 0.0,
            "rationale": f"classifier call failed, defaulting safely: {e}",
        }

    intent = result.get("intent", "").strip()
    if intent not in INTENT_LIST:
        # Model returned something off-list -- fail safe into "other_complaint"
        # rather than crash the pipeline, but keep the raw value for debugging.
        result["raw_intent"] = intent
        result["intent"] = "other_complaint"
        result["confidence"] = min(float(result.get("confidence", 0.5)), 0.4)
    return result