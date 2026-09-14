# Report: AI Support Agent for @AmazonHelp

*(README.md has the reproduction instructions and an important caveat about
this repo being built without internet access — read that first if you
haven't. This report assumes you have.)*

## 1. Problem framing

**Brand chosen**: AmazonHelp. It's one of the highest-volume brands in the
dataset, its traffic spans genuinely different failure modes (logistics,
billing, account security, product questions) rather than one narrow issue,
and Amazon's actual support tone (brief, empathetic, "DM us and we'll fix
it") is distinctive enough to be worth imitating rather than generating
generic corporate-speak.

**What "good" means here**: I'm optimizing for a specific, narrow claim —
*"for the subset of messages this agent is confident are routine and
low-risk, it should draft a reply as good as a human agent's first response,
and it should almost never mis-classify a risky message as routine."* That
second half is the actual product requirement. A support automation system
that's 90% accurate on replies but occasionally auto-closes a fraud report
or a billing dispute is not a good system — it's a liability with a good demo.
So the three things I actually optimized for, in order:

1. **Escalation recall on messages that should escalate** (don't miss the
   dangerous ones). This is the metric I'd block a launch on.
2. **Reply faithfulness** (don't invent order numbers, refund amounts, or
   policies). A wrong intent label is annoying; a fabricated promise to a
   customer is a trust and possibly a legal problem.
3. **Coverage** (how much of the traffic can actually be auto-handled). This
   is the business-value metric, and it's explicitly last — I'd rather ship
   a system that auto-handles 30% of traffic safely than one that
   auto-handles 70% and gets the risky 5% of that wrong.

**What I chose not to build**:
- **Multi-turn conversation state.** Each decision is made from a single
  customer message, not a reconstructed thread. Real support conversations
  are multi-turn (customer replies again after DMing details), and a
  production system needs thread state. I scoped this out because thread
  reconstruction on the raw Twitter data has its own hard data-quality
  problems (broken reply chains, customers tweeting the same complaint fresh
  instead of replying) that deserved dedicated time I didn't have this week
  — better to be honest about a real single-turn system than a broken
  fake multi-turn one.
- **Actually sending replies or taking actions** (issuing real refunds,
  cancelling real orders). The agent drafts and decides; it never executes.
  Any real deployment needs a human-in-the-loop send step regardless of how
  good the model is, so building auto-send wasn't a good use of a week.
- **Fine-tuning anything.** With ~150-250 labeled examples, fine-tuning a
  classifier would overfit before it beat a well-prompted LLM with the same
  data used as retrieval examples instead. Banking77 (77 labeled intents,
  13k examples) was available as an optional secondary dataset but I didn't
  use it — its intent taxonomy is for retail banking, not e-commerce
  logistics/billing, and mapping it over would have cost more time than it
  saved versus just hand-defining 8 intents from the actual data.
- **Non-English support.** The dataset and this agent are English-only.

## 2. Results vs. baselines

| System | Intent Acc | Intent Macro-F1 | Escalation Acc | Escalation Recall (should-escalate) | Over-escalation rate | Lexical grounding |
|---|---|---|---|---|---|---|
| **Trivial** (majority intent, canned reply, always escalate) | 0.152 | 0.033 | 0.378 | 1.000 | 1.000 | 0.170 |
| **Simple** (keyword rules + templates, rule-based escalation) | 0.445 | 0.442 | 0.768 | 0.387 | 0.000 | 0.198 |
| **Full agent (LLM)** | *fill in after running `eval.harness --system full` with a real API key — not run in the sandbox this repo was built in, see README* | | | | | |

(Numbers above for trivial/simple were actually run against the golden set
in this repo; see `eval/results/trivial_results_metrics.json` and
`eval/results/simple_results_metrics.json`.)

**Reading the baseline numbers correctly, before you look at the full
system**: the simple baseline's `over_escalation_rate = 0.000` looks great
in isolation and is not great in context. It means the keyword rules
*never* escalate anything that isn't already caught by an obvious
keyword (security words, dollar signs, "account access"). Its escalation
*recall* on things that truly should escalate is only 0.387 — it misses
**6 out of 10** cases that a human labeler flagged as needing a human. A
system that only escalates on hard keyword triggers will always look
"conservative" on the over-escalation axis while quietly failing at the one
job escalation exists to do. This is exactly the kind of headline-metric
trap Section 4 is about.

The full agent should be evaluated primarily on whether it beats the simple
baseline's escalation recall (0.387) by a wide margin while keeping
over-escalation reasonable (not 0.000 — some over-escalation is the correct
price of catching more true positives) and beats both baselines' reply
quality on the LLM-judge rubric, not just on intent accuracy alone.

## 3. Failure analysis: top 5 failure modes (with hypotheses)

*(These are the failure modes I identified while designing the eval harness
and rules, cross-checked against the golden set. Once you run the full
system on real data, replace/augment this list with the actual failures you
observe — I expect the general categories to hold but the specific examples
and rates will differ on messier real tweets.)*

1. **Sarcasm / rhetorical questions read as literal requests.**
   e.g. *"oh great, ANOTHER broken item, love this brand"* — the surface
   keywords ("broken item") point to `damaged_or_wrong_item`, which is
   correct, but a naive sentiment pass can under- or over-read the anger
   signal depending on how literally it parses the sarcasm. Hypothesis: an
   LLM classifier handles this far better than the keyword baseline (which
   has zero sarcasm handling), but will still occasionally under-escalate
   dry, deadpan-angry messages that don't contain "obvious" anger words.

2. **Stacked complaints in one tweet** (e.g. billing dispute *and* a rude
   delivery driver in the same message). The taxonomy forces a single
   intent label, so whichever issue is mentioned first or most saliently
   wins, and the other issue's context is dropped from the reply. Hypothesis:
   this under-serves the secondary issue and is a real limit of a
   single-label taxonomy — a production system probably needs multi-label
   intent, which I scoped out (see Section 1).

3. **Vague messages with no order number / no specifics**
   (e.g. *"still nothing, so done with this company"* with no context).
   The retrieval step has almost nothing to match against, so
   `grounded_on_examples` correctly comes back low-confidence and the
   escalation rule correctly kicks in — but the *reply itself* becomes a
   generic "please DM more details," which is safe but not very useful.
   Hypothesis: this is a dataset-quality issue as much as a model one — a
   real production system would have order history to join against and
   wouldn't need the customer to self-report an order number at all.

4. **Overconfident wrong intent on borderline cases** — e.g.
   *"asked for a refund three weeks ago, still shows processing"* is
   arguably both `refund_or_return` and `order_status_delay`-adjacent
   ("processing" is a status). Hypothesis: the classifier will pick one
   confidently (likely correctly, since `refund` is the more specific
   term), but confidence calibration on genuinely ambiguous cases is
   something I haven't validated — the confidence score is a stated
   probability from the model, not a calibrated one, and the escalation
   rule's `< 0.6` threshold is a guess, not a tuned value (see
   "what's misleading," item 3 below).

5. **Grounding retrieval finding a superficially similar but
   substantively wrong historical case.** BM25 matches on lexical overlap,
   not resolution logic — a message about a *late* order and a message
   about a *cancelled* order can share enough vocabulary ("package,"
   "never arrived") that the wrong historical resolution gets retrieved and
   the drafted reply borrows the wrong next step. Hypothesis: this is worse
   on the synthetic sample data (limited template vocabulary makes lexical
   overlap especially misleading) and better, but not solved, on real data
   with more lexical variety. A semantic (embedding) retriever would
   probably help here at the cost of the simplicity/debuggability tradeoff
   documented in `DECISION_LOG.md`.

## 4. What is misleading about my headline number

*(Mandatory section — the assignment is explicit that this matters more
than the number itself.)*

If I report a single "intent accuracy: X%" or "escalation accuracy: Y%" as
*the* headline, here's what it hides:

1. **The golden set is 164 examples on synthetic, template-generated data
   in this build.** Even once re-run on real Kaggle data, 164 examples
   split across 8 intents is ~20 examples per intent — enough to catch
   gross failures, not enough to trust a difference of a few percentage
   points between two systems as statistically meaningful. A macro-F1 of
   0.85 vs 0.80 on this sample size could easily flip with a different
   random seed.

2. **Escalation *accuracy* is the wrong single number to headline at all** —
   see Section 2. A system that never escalates anything risky can still
   post a high accuracy number if most of the traffic genuinely is routine
   (which it is — most support traffic is boring). The real headline number
   should always be reported as a pair: escalation recall on the
   should-escalate subset, and over-escalation rate on the shouldn't
   subset, never accuracy alone. I've tried to enforce this by having
   `eval/harness.py` compute both and print both, but it would be very easy
   for someone (including future-me) to quote just the top-line accuracy in
   a slide deck and mislead a stakeholder.

3. **The escalation rule thresholds are hand-picked, not tuned.** The
   `confidence < 0.6` cutoff, the anger score `>= 0.7` cutoff, and the
   "always escalate account_access_issue" policy are defensible defaults,
   not values I validated against a held-out set by sweeping thresholds and
   picking an ROC-optimal point. Any headline number is conditional on
   these arbitrary-ish thresholds; a different (also defensible) threshold
   choice would shift escalation recall and over-escalation rate in
   opposite directions, and I haven't shown the tradeoff curve.

4. **The LLM-as-judge score is only as good as its agreement with a human
   rater, and that agreement check needs to actually be run** (see
   `eval/human_agreement.py` and the caveat in README — this requires a
   real API key and a real human sitting down to score 30 examples, which
   wasn't possible in the sandbox this repo was built in). Until that
   agreement number exists and is reasonable (I'd want to see Spearman
   correlation ≥ 0.6 and within-1-point agreement ≥ 80% before trusting the
   judge at all), any judge-based "reply quality" score should be treated
   as unvalidated.

5. **Grounding ("`grounded_on_examples: true`") is self-reported by the
   drafting model**, not independently verified. The model is asked to
   report whether it actually used the retrieved examples' pattern, but
   nothing stops it from saying "true" while actually freelancing. A more
   rigorous version would independently check textual/structural similarity
   between the draft and the retrieved examples rather than trusting the
   model's own claim.

6. **Real-world traffic distribution isn't the golden set's distribution.**
   The golden set is deliberately stratified to have roughly equal
   representation per intent (25 cap each) and to oversample "hard" cases.
   Real @AmazonHelp traffic is not evenly distributed across intents (order
   status questions dominate), so a deployment-weighted accuracy number
   would look different — probably better on intent accuracy (order status
   is the easiest intent) and worse on average escalation recall (the hard,
   rare intents that get proportionally more escalation-critical decisions
   are underrepresented in raw volume terms).

## 5. What I'd do next with one more week

1. **Run everything on real data** and rebuild the golden set with actual
   independent human labeling (not simulated from generation templates) —
   this is the single highest-priority item; everything else is secondary
   until this is done.
2. **Threshold tuning with a tradeoff curve.** Sweep the confidence and
   anger-score escalation thresholds, plot escalation recall vs.
   over-escalation rate (a precision-recall-style curve), and pick a point
   with a stated business justification instead of a guess.
3. **Independent grounding verification** instead of trusting the model's
   self-report (Section 4, item 5) — e.g. an embedding-similarity check
   between the draft and the retrieved example(s).
4. **Multi-label intent** for stacked-complaint messages (Section 3, item 2).
5. **A second, independently-built judge rubric or a second judge model**,
   to check whether judge-vs-judge agreement is higher than judge-vs-human
   (which would suggest the judge and I share a blind spot, not that the
   judge is reliable).
6. **Calibration analysis on the classifier's confidence score** — is 0.9
   actually right ~90% of the time? Right now it's used as a raw threshold
   with no calibration check.
7. **Embedding-based retrieval as an A/B against BM25**, now that there'd be
   enough real historical data to make the comparison meaningful.
