# AI Support Agent for @AmazonHelp

An AI support agent for the Amazon Twitter support account (`@AmazonHelp`),
built on the Kaggle "Customer Support on Twitter" dataset, that:

1. Classifies each incoming customer message into one of 8 intents.
2. Drafts a reply grounded in how AmazonHelp has historically resolved
   similar issues (BM25 retrieval over real past resolutions + LLM drafting).
3. Decides `auto_handle` vs `escalate_to_human`, with a stated reason.

Full writeup: [`report/REPORT.md`](report/REPORT.md). Non-obvious decisions:
[`report/DECISION_LOG.md`](report/DECISION_LOG.md).

## ⚠️ Read this before you trust any numbers here

This repo was built in a sandbox with **no internet access**, so the real
~3M-row Kaggle CSV could not be downloaded there. Two things follow from that:

1. **The bundled sample data is synthetic.** `data/raw/sample_twcs.csv` is
   ~700 hand-templated customer/agent tweet pairs for brand `AmazonHelp`, in
   the *exact* schema of the real Kaggle file (see
   `scripts/make_sample_data.py` for full disclosure of how and why). It
   exists so you can run the whole pipeline in minutes to confirm it works,
   not to stand in for a real evaluation.
2. **The "full system" LLM results in this repo have not been run against
   the API.** Running them requires your own `ANTHROPIC_API_KEY` and network
   access, neither of which this build environment has. Baselines
   (trivial/simple) *were* run and their numbers below are real.

**Before you submit this**, you must:
- Download the real dataset from Kaggle (`thoughtvector/customer-support-on-twitter`),
  place it at `data/raw/twcs.csv`.
- Run `python3 -m eval.harness --data data/raw/twcs.csv --system full` with
  a real API key to get real headline numbers.
- Re-do the golden-set labeling on real data (see "Golden eval set" below --
  the sampling *logic* transfers directly, but the labels in this repo were
  simulated from generation templates, which only works because the sample
  data is synthetic).
- Actually read and understand every file in `src/` and `eval/` — you'll be
  asked to explain and modify this code live.

## Quickstart (reproduces headline results in <15 min)

```bash
git clone <your-repo-url>
cd hiver-support-agent
cp .env.example .env   # fill in ANTHROPIC_API_KEY
bash scripts/run_all.sh
```

Or step by step:

```bash
pip install -r requirements.txt
python3 scripts/make_sample_data.py       # generates the synthetic sample (skip if using real data)
python3 eval/build_golden_set.py          # builds data/golden_eval_set.csv (164 examples)
python3 -m eval.harness --system trivial  # baseline 1, no API key needed
python3 -m eval.harness --system simple   # baseline 2, no API key needed
python3 -m eval.harness --system full     # the actual agent, needs ANTHROPIC_API_KEY
python3 -m eval.judge --results eval/results/full_results.csv --out eval/results/full_results_judged.csv
```

To run the agent on a single message interactively:

```bash
python3 -m src.pipeline --message "@AmazonHelp my order still hasn't arrived and it's been a week"
```

## Repo layout

```
src/
  data_prep.py     # load raw CSV, filter to one brand, pair customer<->agent turns
  intents.py       # the 8-intent taxonomy + escalation risk priors
  retrieval.py      # BM25 index over historical resolved cases (grounding)
  classify.py       # LLM intent classifier
  draft_reply.py    # grounded reply drafting
  escalation.py     # auto_handle vs escalate_to_human decision + reasons
  baselines.py       # trivial + simple (keyword) baselines
  pipeline.py       # wires it all together, single-message CLI
  llm_client.py     # thin Anthropic API wrapper (swap providers here)
eval/
  build_golden_set.py  # builds the 150-250 example hand-labeled eval set
  harness.py           # runs any system (full/trivial/simple) over the golden set, computes metrics
  judge.py             # LLM-as-judge for reply quality (4-axis rubric)
  human_agreement.py   # judge-vs-human agreement tooling
  results/             # output metrics/CSVs land here
data/
  raw/sample_twcs.csv              # synthetic sample (see warning above)
  raw/sample_ground_truth_intents.csv  # simulated label source (synthetic-data-only artifact)
  golden_eval_set.csv               # the hand-labeled evaluation set
report/
  REPORT.md            # problem framing, results vs baselines, failure analysis, "what's misleading", next steps
  DECISION_LOG.md       # 10-15 non-obvious decisions and why
tests/
  test_pipeline.py     # tests that don't require an API key
scripts/
  make_sample_data.py  # generates the synthetic sample
  run_all.sh            # end-to-end reproduction
```

## Golden evaluation set

`data/golden_eval_set.csv` has 164 hand-labeled examples (target was
150-250). Columns: `id`, `customer_text`, `gold_intent`, `gold_escalate`,
`labeling_note`, `historical_brand_reply_reference`.

**Sampling**: stratified across all 8 intents (capped at 25/intent) after
deduplicating near-identical messages, from customer messages that have a
matched real AmazonHelp reply (so failure analysis can compare against what
actually happened). Full method and caveats are documented in the
`eval/build_golden_set.py` module docstring — read it, since on real data the
labeling step (currently simulated from generation templates because this
sample data is template-generated) needs to become an actual manual pass.

## Metrics computed

- **Intent**: accuracy, macro-F1, full confusion matrix.
- **Escalation**: overall accuracy, and — the number that actually
  matters — **recall on cases that should escalate** (missing a real
  escalation is worse than an unnecessary one) plus the over-escalation
  rate on cases that shouldn't.
- **Reply quality**: LLM-as-judge on 4 axes (helpfulness, tone fit,
  faithfulness, actionability) + a cheap lexical-overlap-with-reference
  sanity check that doesn't need judge calls.
- **Judge reliability**: Spearman correlation + within-1-point agreement
  between the judge and a human rater on a 30-example subsample
  (`eval/human_agreement.py`).

## Baseline results (real, run in this sandbox on the synthetic sample)

| System | Intent Acc | Intent Macro-F1 | Escalation Acc | Escalation Recall (should-escalate) | Over-escalation rate |
|---|---|---|---|---|---|
| Trivial (majority intent, canned reply, always escalate) | 0.152 | 0.033 | 0.378 | 1.000 | 1.000 |
| Simple (keyword classifier + templates + rules) | 0.445 | 0.442 | 0.768 | 0.387 | 0.000 |
| **Full agent (LLM)** | *run `eval.harness --system full` with your API key* | | | | |

See `report/REPORT.md` for interpretation — in particular why "the simple
baseline never over-escalates" is not the flex it looks like.

## Known limitations (also in report/REPORT.md)

- Synthetic sample data has limited lexical diversity (template-generated),
  so BM25 retrieval and intent classification will look artificially strong
  on it compared to real, messier tweets. Real Kaggle data is noisier: typos,
  sarcasm, multi-tweet threads, off-topic replies, non-English tweets.
- The golden set's escalation labels were derived from the same rule logic
  used in `escalation.py` (I wrote both), which is a consistency check on
  the *policy*, not an independent ground truth. A truly independent human
  label pass on real data is a to-do (see "what's misleading" in the report).
- No handling of multi-turn context beyond the single customer message that
  triggered a reply — see report "what I chose not to build."

## Citations / borrowed material

- Dataset: Kaggle "Customer Support on Twitter" (`thoughtvector/customer-support-on-twitter`).
- BM25 formula: Robertson & Zaragoza, "The Probabilistic Relevance
  Framework: BM25 and Beyond" (2009) — used in `src/retrieval.py`'s fallback
  implementation (the primary path uses the `rank_bm25` package).
- No other external code was copied; an AI coding assistant was used to help
  write and debug this repo (per assignment rules), and every file was read
  and understood, not just accepted.
