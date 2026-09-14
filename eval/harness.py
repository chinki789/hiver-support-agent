"""
Runs the full agent (and both baselines) over data/golden_eval_set.csv and
computes:
  - Intent classification: accuracy, macro-F1, per-intent confusion counts
  - Escalation decision: accuracy vs gold_escalate, and separately recall on
    cases that SHOULD escalate (this is the number that actually matters --
    see report/REPORT.md; missing a real escalation is much worse than an
    unnecessary one).
  - Reply quality: delegated to judge.py (LLM-as-judge), reported alongside
    a lexical grounding metric (token overlap with the retrieved historical
    reply) as a cheap sanity check that doesn't depend on judge calls.

Usage:
    python -m eval.harness --system full
    python -m eval.harness --system full --limit 20
    python -m eval.harness --system full --offset 80 --limit 84
    python -m eval.harness --system trivial
    python -m eval.harness --system simple
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

sys.path.insert(0, ".")
from src.data_prep import load_raw, build_brand_pairs  # noqa: E402
from src.classify import classify  # noqa: E402
from src.draft_reply import draft_reply  # noqa: E402
from src.escalation import decide_escalation  # noqa: E402
from src.retrieval import ResolutionIndex  # noqa: E402
from src.baselines import trivial_predict, simple_predict  # noqa: E402
from src.intents import INTENT_LIST  # noqa: E402


def token_overlap(a: str, b: str) -> float:
    ta, tb = set(a.lower().split()), set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def run_full_system(golden: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    print(f"Classifying {len(history_df)} historical cases to build retrieval index "
          f"(progress prints below so it doesn't look stuck)...")
    history_df = history_df.copy()
    intents = []
    for i, t in enumerate(history_df["customer_text_clean"], start=1):
        intents.append(classify(t)["intent"])
        print(f"  history classified: {i}/{len(history_df)}", flush=True)
    history_df["intent"] = intents
    index = ResolutionIndex.from_dataframe(history_df, intent_col="intent")
    print("Retrieval index built. Now processing golden examples...")

    rows = []
    for _, r in golden.iterrows():
        cls = classify(r["customer_text"])
        draft = draft_reply(r["customer_text"], index, intent=cls["intent"])
        esc = decide_escalation(r["customer_text"], cls["intent"], cls["confidence"], draft)
        rows.append({
            "id": r["id"],
            "customer_text": r["customer_text"],
            "gold_intent": r["gold_intent"],
            "gold_escalate": r["gold_escalate"],
            "pred_intent": cls["intent"],
            "pred_confidence": cls["confidence"],
            "pred_reply": draft["reply"],
            "grounded_on_examples": draft.get("grounded_on_examples"),
            "pred_escalate": esc["escalate"],
            "escalation_reasons": "; ".join(esc["reasons"]),
            "reference_reply": r.get("historical_brand_reply_reference", ""),
        })
        print(f"  processed {len(rows)}/{len(golden)}", flush=True)
    return pd.DataFrame(rows)


def run_baseline(golden: pd.DataFrame, kind: str) -> pd.DataFrame:
    predict_fn = trivial_predict if kind == "trivial" else simple_predict
    rows = []
    for _, r in golden.iterrows():
        pred = predict_fn(r["customer_text"])
        rows.append({
            "id": r["id"],
            "customer_text": r["customer_text"],
            "gold_intent": r["gold_intent"],
            "gold_escalate": r["gold_escalate"],
            "pred_intent": pred["intent"],
            "pred_confidence": 1.0,
            "pred_reply": pred["reply"],
            "grounded_on_examples": False,
            "pred_escalate": pred["decision"] == "escalate_to_human",
            "escalation_reasons": "; ".join(pred["escalation_reasons"]),
            "reference_reply": r.get("historical_brand_reply_reference", ""),
        })
    return pd.DataFrame(rows)


def compute_metrics(results: pd.DataFrame) -> dict:
    intent_acc = accuracy_score(results["gold_intent"], results["pred_intent"])
    intent_f1 = f1_score(results["gold_intent"], results["pred_intent"],
                          labels=INTENT_LIST, average="macro", zero_division=0)

    esc_acc = accuracy_score(results["gold_escalate"], results["pred_escalate"])

    should_escalate = results[results["gold_escalate"]]
    escalation_recall = (
        should_escalate["pred_escalate"].mean() if len(should_escalate) else float("nan")
    )
    should_not = results[~results["gold_escalate"]]
    over_escalation_rate = (
        should_not["pred_escalate"].mean() if len(should_not) else float("nan")
    )

    results["lexical_grounding"] = results.apply(
        lambda r: token_overlap(str(r["pred_reply"]), str(r["reference_reply"])), axis=1
    )

    return {
        "n_examples": len(results),
        "intent_accuracy": round(intent_acc, 4),
        "intent_macro_f1": round(intent_f1, 4),
        "escalation_accuracy": round(esc_acc, 4),
        "escalation_recall_on_should_escalate": round(float(escalation_recall), 4),
        "over_escalation_rate_on_should_auto_handle": round(float(over_escalation_rate), 4),
        "mean_lexical_grounding_to_reference_reply": round(results["lexical_grounding"].mean(), 4),
        "confusion_matrix_labels": INTENT_LIST,
        "confusion_matrix": confusion_matrix(
            results["gold_intent"], results["pred_intent"], labels=INTENT_LIST
        ).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/sample_twcs.csv")
    ap.add_argument("--brand", default="AmazonHelp")
    ap.add_argument("--golden", default="data/golden_eval_set.csv")
    ap.add_argument("--system", choices=["full", "trivial", "simple"], default="full")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None,
                     help="Only process the first N golden examples (and cap "
                          "history size similarly for --system full). Useful "
                          "for a quick smoke test, or for splitting a run "
                          "across multiple days on a free-tier daily token "
                          "quota (combine with --offset).")
    ap.add_argument("--offset", type=int, default=0,
                     help="Skip the first N golden examples before applying "
                          "--limit. Lets you process the golden set in "
                          "chunks across sessions/days if your provider has "
                          "a daily quota too small for all 164 examples in "
                          "one run, e.g.: --offset 0 --limit 80 today, "
                          "--offset 80 --limit 84 tomorrow. Use --out to give "
                          "each chunk a distinct filename, then concatenate "
                          "the resulting CSVs before computing final metrics.")
    ap.add_argument("--history-size", type=int, default=50,
                     help="Max number of historical (customer,reply) pairs to "
                          "classify when building the retrieval index for "
                          "--system full. Classifying ALL history uses one "
                          "LLM call per historical row before you even reach "
                          "the golden examples, which can exhaust a free-tier "
                          "daily token quota on its own. 50 is enough for "
                          "reasonable BM25 retrieval quality on this dataset "
                          "size; raise it only if you have quota to spare.")
    args = ap.parse_args()

    golden = pd.read_csv(args.golden)
    golden["gold_escalate"] = golden["gold_escalate"].astype(bool)
    if args.offset:
        golden = golden.iloc[args.offset:].copy()
        print(f"--offset set: skipping the first {args.offset} golden examples.")
    if args.limit:
        golden = golden.head(args.limit).copy()
        print(f"--limit set: only processing {len(golden)} golden examples "
              f"(offset {args.offset}).")

    if args.system == "full":
        raw = load_raw(args.data)
        pairs = build_brand_pairs(raw, args.brand)
        pairs = pairs[pairs["brand_reply_text_clean"].notna()].copy()
        history_cap = args.history_size
        if args.limit:
            history_cap = min(history_cap, max(args.limit * 2, 10))
        if len(pairs) > history_cap:
            print(f"Capping retrieval-index history to {history_cap} of {len(pairs)} "
                  f"available historical pairs (see --history-size) to avoid "
                  f"burning through a free-tier daily token quota just on setup.")
            pairs = pairs.head(history_cap).copy()
        results = run_full_system(golden, pairs)
    else:
        results = run_baseline(golden, args.system)

    metrics = compute_metrics(results)
    out_path = args.out or f"eval/results/{args.system}_results.csv"
    metrics_path = out_path.replace(".csv", "_metrics.json")
    results.to_csv(out_path, index=False)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n=== {args.system.upper()} SYSTEM METRICS ===")
    for k, v in metrics.items():
        if "confusion" not in k:
            print(f"{k}: {v}")
    print(f"\nDetailed results: {out_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()