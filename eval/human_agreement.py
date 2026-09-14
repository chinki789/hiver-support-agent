"""
Computes agreement between the LLM judge (eval/judge.py) and a human rater
on the same set of (customer_text, candidate_reply) pairs, using the
"overall" 1-5 score.

Workflow:
1. `python -m eval.human_agreement sample --judged eval/results/full_results_judged.csv --n 30`
   -> writes eval/results/human_labeling_sheet.csv with a blank
      `human_overall` column for a human to fill in (open it in
      Excel/Sheets/a text editor and score each row 1-5 yourself, blind to
      the judge's score -- the judge columns are intentionally NOT included
      in this sheet to avoid anchoring).
2. Fill in human_overall by hand.
3. `python -m eval.human_agreement score --sheet eval/results/human_labeling_sheet.csv --judged eval/results/full_results_judged.csv`
   -> joins back to the judge scores by id and reports Spearman correlation
      + exact-match rate + within-1-point agreement rate.

We report within-1-point agreement alongside exact correlation because for a
1-5 rubric, "exact match" is a harsh bar that even two human raters often
don't clear -- within-1 is the more standard way to report this humanely
while still being falsifiable (a judge that's often off by 2+ points is
disqualifying).
"""
from __future__ import annotations

import argparse

import pandas as pd
from scipy.stats import spearmanr


def make_sample(judged_csv: str, n: int, out_csv: str, seed: int = 13):
    df = pd.read_csv(judged_csv)
    sample = df.sample(n=min(n, len(df)), random_state=seed)
    sheet = sample[["id", "customer_text", "pred_reply"]].copy()
    sheet["human_overall"] = ""  # fill in 1-5 by hand
    sheet.to_csv(out_csv, index=False)
    print(f"Wrote {len(sheet)} rows to {out_csv}. Fill in human_overall (1-5), "
          f"then run the 'score' command.")


def score(sheet_csv: str, judged_csv: str):
    sheet = pd.read_csv(sheet_csv)
    judged = pd.read_csv(judged_csv)
    sheet = sheet[sheet["human_overall"].notna() & (sheet["human_overall"] != "")]
    if len(sheet) < 10:
        print(f"WARNING: only {len(sheet)} human-labeled rows found. This is too "
              f"few for a stable agreement estimate -- aim for at least 25-30.")
    merged = sheet.merge(judged[["id", "overall"]], on="id", how="left", suffixes=("", "_judge"))
    merged["human_overall"] = merged["human_overall"].astype(float)

    rho, pval = spearmanr(merged["human_overall"], merged["overall"])
    exact = (merged["human_overall"] == merged["overall"]).mean()
    within_1 = (merged["human_overall"] - merged["overall"]).abs().le(1).mean()

    print(f"n = {len(merged)}")
    print(f"Spearman correlation (judge vs human, 'overall' score): {rho:.3f} (p={pval:.3g})")
    print(f"Exact-match agreement rate: {exact:.1%}")
    print(f"Within-1-point agreement rate: {within_1:.1%}")

    out = "eval/results/judge_human_agreement.csv"
    merged.to_csv(out, index=False)
    print(f"Row-level comparison saved to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_sample = sub.add_parser("sample")
    p_sample.add_argument("--judged", default="eval/results/full_results_judged.csv")
    p_sample.add_argument("--n", type=int, default=30)
    p_sample.add_argument("--out", default="eval/results/human_labeling_sheet.csv")

    p_score = sub.add_parser("score")
    p_score.add_argument("--sheet", default="eval/results/human_labeling_sheet.csv")
    p_score.add_argument("--judged", default="eval/results/full_results_judged.csv")

    args = ap.parse_args()
    if args.cmd == "sample":
        make_sample(args.judged, args.n, args.out)
    else:
        score(args.sheet, args.judged)
