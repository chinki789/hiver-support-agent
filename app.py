"""
Local web dashboard (Streamlit) for the AmazonHelp support agent project.

This is NOT part of the Hiver assignment deliverables -- the assignment
explicitly wants a CLI-reproducible pipeline (see README.md / scripts/run_all.sh
for that). This file is a convenience/demo layer on top of the same
functions and result files the CLI pipeline already uses.

Run with:
    streamlit run app.py
"""
import json
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, ".")

from src.data_prep import load_raw, build_brand_pairs
from src.classify import classify
from src.draft_reply import draft_reply
from src.escalation import decide_escalation
from src.retrieval import ResolutionIndex
from src import llm_client

RESULTS_DIR = "eval/results"
GOLDEN_PATH = "data/golden_eval_set.csv"
DATA_PATH = "data/raw/sample_twcs.csv"
BRAND = "AmazonHelp"

st.set_page_config(
    page_title="AmazonHelp AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero {
    background: linear-gradient(120deg, #232f3e 0%, #37475a 60%, #ff9900 160%);
    padding: 2.2rem 2rem;
    border-radius: 18px;
    color: white;
    margin-bottom: 1.6rem;
    box-shadow: 0 8px 24px rgba(0,0,0,0.18);
}
.hero h1 { margin: 0; font-size: 2.1rem; font-weight: 800; }
.hero p { margin: 0.4rem 0 0 0; opacity: 0.9; font-size: 1.02rem; }

.pill {
    display: inline-block;
    padding: 0.15rem 0.7rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 0.4rem;
}
.pill-green { background: #d7f5df; color: #146c2e; }
.pill-red   { background: #fbdada; color: #9c1c1c; }
.pill-blue  { background: #dbe9ff; color: #16408a; }
.pill-gray  { background: #eaeaea; color: #444; }

div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #eee;
    border-radius: 14px;
    padding: 0.8rem 1rem 0.4rem 1rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}

section[data-testid="stSidebar"] {
    background: #16202b;
}
section[data-testid="stSidebar"] * { color: #e8edf2 !important; }
</style>
""", unsafe_allow_html=True)


def load_csv_if_exists(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def load_json_if_exists(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def pill(text, kind="gray"):
    return f'<span class="pill pill-{kind}">{text}</span>'


@st.cache_resource(show_spinner="Building retrieval index from historical resolutions...")
def get_retrieval_index():
    if not os.path.exists(DATA_PATH):
        return None
    raw = load_raw(DATA_PATH)
    pairs = build_brand_pairs(raw, BRAND)
    pairs = pairs[pairs["brand_reply_text_clean"].notna()].copy()
    return ResolutionIndex.from_dataframe(pairs)


with st.sidebar:
    st.markdown("### 🤖 AmazonHelp Agent")
    st.caption(f"Provider: **{llm_client.PROVIDER}** · Model: `{llm_client.MODEL}`")
    page = st.radio(
        "Navigate",
        ["📊 Overview", "💬 Chat with Agent", "🔍 Browse Examples",
         "⚖️ Baseline Comparison", "🧑‍⚖️ Judge & Human Agreement"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(
        "This dashboard is a convenience layer over the CLI pipeline used "
        "for the actual Hiver take-home deliverables. See README.md for "
        "the graded reproduction steps."
    )

golden = load_csv_if_exists(GOLDEN_PATH)

st.markdown("""
<div class="hero">
  <h1>AI Support Agent — AmazonHelp</h1>
  <p>Classifies customer tweets, drafts grounded replies from historical resolutions,
  and decides auto-handle vs. escalate — with a stated reason for every call.</p>
</div>
""", unsafe_allow_html=True)

if golden is None:
    st.error(f"Couldn't find {GOLDEN_PATH}. Run `python eval/build_golden_set.py` first, then reload this page.")
    st.stop()

if page == "📊 Overview":
    st.subheader("Golden evaluation set")
    c1, c2, c3 = st.columns(3)
    c1.metric("Labeled examples", len(golden))
    c2.metric("Distinct intents", golden["gold_intent"].nunique())
    c3.metric("Escalation rate (gold)", f"{golden['gold_escalate'].mean():.1%}")

    st.markdown("#### Intent distribution")
    st.bar_chart(golden["gold_intent"].value_counts())

    full_metrics = load_json_if_exists(f"{RESULTS_DIR}/full_results_metrics.json")
    if full_metrics:
        st.markdown("#### Full LLM agent — latest run")
        m = full_metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Intent accuracy", f"{m['intent_accuracy']:.1%}")
        c2.metric("Intent macro-F1", f"{m['intent_macro_f1']:.2f}")
        rec = m['escalation_recall_on_should_escalate']
        c3.metric("Escalation recall", f"{rec:.1%}" if rec == rec else "n/a")
        over = m['over_escalation_rate_on_should_auto_handle']
        c4.metric("Over-escalation rate", f"{over:.1%}" if over == over else "n/a")
        st.caption(f"Based on {m['n_examples']} of 164 golden examples.")
    else:
        st.info("No full-system results yet. Run `python -m eval.harness --system full` in your terminal, then reload.")

elif page == "💬 Chat with Agent":
    st.subheader("Talk to the agent like a customer would")
    st.caption(
        "Type a message as if you were tweeting @AmazonHelp. The agent will "
        "classify it, draft a grounded reply, and decide whether it would "
        "auto-handle or escalate it -- live, using your configured LLM provider."
    )

    index = get_retrieval_index()
    if index is None:
        st.error(f"Couldn't find {DATA_PATH} to build the retrieval index.")
        st.stop()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
            st.write(msg["content"])
            if msg["role"] == "assistant" and "meta" in msg:
                meta = msg["meta"]
                decision_kind = "red" if meta["escalate"] else "green"
                decision_label = "🚨 ESCALATE TO HUMAN" if meta["escalate"] else "✅ AUTO-HANDLE"
                confidence_text = f"confidence {meta['confidence']:.2f}"
                badges = (
                    pill(meta["intent"], "blue")
                    + pill(confidence_text, "gray")
                    + pill(decision_label, decision_kind)
                )
                st.markdown(badges, unsafe_allow_html=True)
                with st.expander("Why this decision?"):
                    for r in meta["reasons"]:
                        st.write(f"- {r}")

    user_input = st.chat_input("e.g. @AmazonHelp my order still hasn't arrived, it's been a week")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑"):
            st.write(user_input)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Classifying, retrieving similar past cases, and drafting a reply..."):
                cls = classify(user_input)
                draft = draft_reply(user_input, index, intent=cls["intent"])
                esc = decide_escalation(user_input, cls["intent"], cls["confidence"], draft)
            st.write(draft["reply"])
            decision_kind = "red" if esc["escalate"] else "green"
            decision_label = "🚨 ESCALATE TO HUMAN" if esc["escalate"] else "✅ AUTO-HANDLE"
            confidence_text = f"confidence {cls['confidence']:.2f}"
            badges = (
                pill(cls["intent"], "blue")
                + pill(confidence_text, "gray")
                + pill(decision_label, decision_kind)
            )
            st.markdown(badges, unsafe_allow_html=True)
            with st.expander("Why this decision?"):
                for r in esc["reasons"]:
                    st.write(f"- {r}")

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": draft["reply"],
            "meta": {
                "intent": cls["intent"],
                "confidence": cls["confidence"],
                "escalate": esc["escalate"],
                "reasons": esc["reasons"],
            },
        })

    if st.session_state.chat_history:
        if st.button("🗑️ Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()

elif page == "🔍 Browse Examples":
    full_results = load_csv_if_exists(f"{RESULTS_DIR}/full_results.csv")
    if full_results is None:
        st.info("No results to browse yet. Run `python -m eval.harness --system full` in your terminal, then reload.")
    else:
        c1, c2 = st.columns([2, 1])
        with c1:
            intents = ["All"] + sorted(full_results["gold_intent"].dropna().unique().tolist())
            selected_intent = st.selectbox("Filter by gold intent", intents)
        with c2:
            show_escalated_only = st.checkbox("Escalated only", value=False)

        view = full_results.copy()
        if selected_intent != "All":
            view = view[view["gold_intent"] == selected_intent]
        if show_escalated_only:
            view = view[view["pred_escalate"] == True]  # noqa: E712

        st.caption(f"Showing {len(view)} of {len(full_results)} examples")

        for _, row in view.iterrows():
            correct = row["gold_intent"] == row["pred_intent"]
            icon = "✅" if correct else "❌"
            esc = row["pred_escalate"]
            with st.expander(f"{icon} [{row['gold_intent']}] {str(row['customer_text'])[:80]}..."):
                st.write(f"**Customer message:** {row['customer_text']}")
                gold_text = f"gold: {row['gold_intent']}"
                pred_text = f"predicted: {row['pred_intent']}"
                esc_text = "🚨 escalate" if esc else "✅ auto-handle"
                badges = (
                    pill(gold_text, "gray")
                    + pill(pred_text, "blue" if correct else "red")
                    + pill(esc_text, "red" if esc else "green")
                )
                st.markdown(badges, unsafe_allow_html=True)
                st.write(f"**Escalation reasons:** {row['escalation_reasons']}")
                st.write(f"**Drafted reply:** {row['pred_reply']}")
                if isinstance(row.get("reference_reply"), str) and row["reference_reply"]:
                    st.write(f"**Historical reference reply:** {row['reference_reply']}")

elif page == "⚖️ Baseline Comparison":
    st.subheader("System comparison")
    rows = []
    for system in ["trivial", "simple", "full"]:
        m = load_json_if_exists(f"{RESULTS_DIR}/{system}_results_metrics.json")
        if m:
            rows.append({
                "System": system,
                "N": m["n_examples"],
                "Intent Acc": m["intent_accuracy"],
                "Intent Macro-F1": m["intent_macro_f1"],
                "Escalation Acc": m["escalation_accuracy"],
                "Escalation Recall": m["escalation_recall_on_should_escalate"],
                "Over-escalation": m["over_escalation_rate_on_should_auto_handle"],
                "Lexical grounding": m["mean_lexical_grounding_to_reference_reply"],
            })
    if rows:
        df = pd.DataFrame(rows).set_index("System")
        st.dataframe(df.style.format("{:.3f}", subset=df.columns[1:]), use_container_width=True)
        st.markdown("#### Escalation recall vs. over-escalation")
        st.bar_chart(df[["Escalation Recall", "Over-escalation"]])
        st.caption(
            "Escalation recall on should-escalate cases is the metric that matters most -- "
            "see report/REPORT.md Section 2 for why over-escalation rate alone is misleading."
        )
    else:
        st.info(
            "No metrics yet. In your terminal, run each system: "
            "`eval.harness --system trivial`, `--system simple`, `--system full`, then reload."
        )

elif page == "🧑‍⚖️ Judge & Human Agreement":
    full_results_path = f"{RESULTS_DIR}/full_results.csv"
    judged_path = f"{RESULTS_DIR}/full_results_judged.csv"

    full_results = load_csv_if_exists(full_results_path)
    judged = load_csv_if_exists(judged_path)

    if full_results is None:
        st.info("Run the full agent first (Overview tab will tell you when results exist).")
        st.stop()

    st.subheader("LLM-as-judge reply quality")
    if judged is None:
        st.write("No judge scores yet for the current results.")
        if st.button("▶️ Run judge scoring now", type="primary"):
            from eval.judge import judge_results_file
            with st.spinner("Scoring each reply on helpfulness, tone, faithfulness, actionability..."):
                judged = judge_results_file(full_results_path, judged_path)
            st.success("Done!")
            st.rerun()
    else:
        score_cols = [c for c in ["helpfulness", "tone_fit", "faithfulness", "actionability", "overall"]
                      if c in judged.columns]
        st.bar_chart(judged[score_cols].mean())
        with st.expander("Per-example judge scores"):
            cols = ["customer_text", "pred_reply"] + score_cols
            if "justification" in judged.columns:
                cols.append("justification")
            st.dataframe(judged[cols], use_container_width=True)

        st.divider()
        st.subheader("Judge-vs-human agreement")
        st.caption(
            "Score a sample of replies yourself, blind to the judge's scores, "
            "then compare. This is a required check, not optional polish -- "
            "an LLM judge is only trustworthy if it roughly agrees with a person."
        )

        sample_path = f"{RESULTS_DIR}/human_labeling_sheet.csv"
        agreement_path = f"{RESULTS_DIR}/judge_human_agreement.csv"

        if "human_sample" not in st.session_state:
            existing = load_csv_if_exists(sample_path)
            st.session_state.human_sample = existing

        if st.session_state.get("human_sample") is None:
            n = st.slider("How many examples to sample?", 10, min(50, len(judged)), min(30, len(judged)))
            if st.button("🎲 Sample examples for review", type="primary"):
                sample = judged.sample(n=n, random_state=13)[["id", "customer_text", "pred_reply"]].copy()
                sample["human_overall"] = None
                st.session_state.human_sample = sample
                sample.to_csv(sample_path, index=False)
                st.rerun()
        else:
            st.write("Score each reply 1 (poor) to 5 (excellent) in the **Your score** column below, "
                     "without peeking at the judge's own scores.")
            edited = st.data_editor(
                st.session_state.human_sample,
                column_config={
                    "human_overall": st.column_config.NumberColumn(
                        "Your score", min_value=1, max_value=5, step=1
                    ),
                    "customer_text": st.column_config.TextColumn("Customer message", width="large"),
                    "pred_reply": st.column_config.TextColumn("Agent's reply", width="large"),
                },
                disabled=["id", "customer_text", "pred_reply"],
                use_container_width=True,
                num_rows="fixed",
                key="human_score_editor",
            )
            st.session_state.human_sample = edited

            scored = edited[edited["human_overall"].notna()]
            st.caption(f"{len(scored)} of {len(edited)} scored")

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("💾 Save my scores"):
                    edited.to_csv(sample_path, index=False)
                    st.success("Saved.")
            with col_b:
                if st.button("📊 Compute agreement", type="primary", disabled=len(scored) < 10):
                    from scipy.stats import spearmanr
                    merged = scored.merge(judged[["id", "overall"]], on="id", how="left")
                    rho, pval = spearmanr(merged["human_overall"], merged["overall"])
                    within_1 = (merged["human_overall"] - merged["overall"]).abs().le(1).mean()
                    exact = (merged["human_overall"] == merged["overall"]).mean()
                    merged.to_csv(agreement_path, index=False)
                    st.session_state.agreement_result = {
                        "rho": rho, "pval": pval, "within_1": within_1, "exact": exact, "n": len(merged)
                    }

            if len(scored) < 10:
                st.caption("Score at least 10 examples to compute agreement.")

            if "agreement_result" in st.session_state:
                r = st.session_state.agreement_result
                st.markdown("#### Result")
                c1, c2, c3 = st.columns(3)
                c1.metric("Spearman correlation", f"{r['rho']:.2f}")
                c2.metric("Within-1-point agreement", f"{r['within_1']:.1%}")
                c3.metric("Exact-match agreement", f"{r['exact']:.1%}")
                st.caption(f"Based on {r['n']} human-scored examples.")

            if st.button("🔄 Start a new sample"):
                st.session_state.human_sample = None
                if "agreement_result" in st.session_state:
                    del st.session_state["agreement_result"]
                st.rerun()