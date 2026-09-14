#!/usr/bin/env bash
# Reproduces the headline results end to end. Should finish in well under
# 15 minutes on the sample data (the bulk of the time is LLM API latency for
# ~164 golden examples x a few calls each).
set -e

echo "== 1/6: installing dependencies =="
pip install -r requirements.txt --break-system-packages -q

echo "== 2/6: generating sample data (synthetic stand-in for the real Kaggle CSV -- see data/raw/sample_twcs.csv docstring in scripts/make_sample_data.py) =="
python3 scripts/make_sample_data.py

echo "== 3/6: building golden evaluation set =="
python3 eval/build_golden_set.py

echo "== 4/6: running baselines (no API calls, no network needed) =="
python3 -m eval.harness --system trivial
python3 -m eval.harness --system simple

echo "== 5/6: running the full agent (requires ANTHROPIC_API_KEY in .env) =="
python3 -m eval.harness --system full

echo "== 6/6: running LLM-as-judge over the full system's replies =="
python3 -m eval.judge --results eval/results/full_results.csv --out eval/results/full_results_judged.csv

echo "Done. See eval/results/*_metrics.json for headline numbers."
echo "For judge-vs-human agreement: python -m eval.human_agreement sample   (then fill in scores, then 'score')"
