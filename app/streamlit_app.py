"""Live demo: score one transaction, show the block/approve decision and why (SRS FR-13)."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

MODELS = ROOT / "models"
FIGS = ROOT / "reports" / "figures"

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")


@st.cache_resource
def load_artifacts():
    pipe = joblib.load(MODELS / "best_model.joblib")
    meta = json.loads((MODELS / "threshold.json").read_text())
    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    samples = pd.read_csv(Path(__file__).parent / "demo_samples.csv", index_col="row_id")
    return pipe, meta, explainer, samples


pipe, meta, explainer, samples = load_artifacts()
FEATURES = meta["feature_order"]
THRESHOLD = meta["threshold"]


def parse_values(text):
    """Turn pasted text into a 1-row DataFrame, or raise ValueError with a clear message."""
    parts = [p.strip() for p in text.replace("\n", ",").split(",") if p.strip()]
    if not parts:
        raise ValueError("Input is empty. Paste 30 numbers: Time, V1 … V28, Amount.")
    if len(parts) != len(FEATURES):
        raise ValueError(f"Expected {len(FEATURES)} numbers (Time, V1 … V28, Amount) but got {len(parts)}.")
    try:
        values = [float(p) for p in parts]
    except ValueError:
        bad = next(p for p in parts if not _is_number(p))
        raise ValueError(f"'{bad}' is not a number. Use only digits, '.', and '-'.")
    if values[-1] < 0:
        raise ValueError("Amount cannot be negative.")
    return pd.DataFrame([values], columns=FEATURES)


def _is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def explain_chart(row):
    pre = pipe.named_steps["scale"]
    names = list(pre.get_feature_names_out())
    xt = pd.DataFrame(pre.transform(row), columns=names)
    sv = explainer(xt)[0]
    contrib = pd.Series(sv.values, index=names)
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index[:10])[::-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#c44e52" if v > 0 else "#4c72b0" for v in top.values]
    ax.barh(top.index, top.values, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("← pushes towards LEGIT      pushes towards FRAUD →")
    ax.set_title("Top 10 reasons (SHAP)")
    fig.tight_layout()
    return fig, contrib


st.title("💳 Credit Card Fraud Detection")
st.caption(f"Model: {meta['model']} · decision threshold t = {THRESHOLD:.2f} · "
           "block if fraud probability ≥ t, otherwise approve")

tab_check, tab_results = st.tabs(["🔍 Check a transaction", "📊 Model results"])

with tab_check:
    mode = st.radio("How do you want to give a transaction?",
                    ["Pick a sample from the test set", "Paste values"], horizontal=True)

    row, actual = None, None
    if mode == "Pick a sample from the test set":
        c1, c2 = st.columns([1, 2])
        kind = c1.selectbox("Show", ["All samples", "Fraud samples", "Legit samples"])
        pool = samples if kind == "All samples" else samples[samples.Class == (1 if kind == "Fraud samples" else 0)]
        if "pick" not in st.session_state or st.session_state.pick not in pool.index:
            st.session_state.pick = int(pool.index[0])
        if c1.button("🎲 Random sample"):
            st.session_state.pick = int(pool.sample(1).index[0])
        options = list(pool.index)
        choice = c2.selectbox(
            "Transaction", options, index=options.index(st.session_state.pick),
            format_func=lambda i: f"#{i} · Amount {pool.loc[i, 'Amount']:.2f}")
        st.session_state.pick = int(choice)
        row = samples.loc[[choice], FEATURES]
        actual = int(samples.loc[choice, "Class"])
        st.caption("The model sees only these 30 values. The true label stays hidden "
                   "and is shown only after the prediction, to check it.")
        with st.expander("See the 30 input values"):
            st.dataframe(row, hide_index=True)
    else:
        example = ", ".join(f"{v:.4f}" for v in samples.iloc[0][FEATURES])
        text = st.text_area("Paste 30 comma-separated numbers: Time, V1 … V28, Amount", value=example, height=110)
        try:
            row = parse_values(text)
        except ValueError as e:
            st.error(f"⚠️ {e}")

    if row is not None and st.button("Check transaction", type="primary"):
        t0 = time.perf_counter()
        proba = float(pipe.predict_proba(row)[0, 1])
        ms = (time.perf_counter() - t0) * 1000

        left, right = st.columns([1, 1.3])
        with left:
            if proba >= THRESHOLD:
                st.error("## 🚫 BLOCK\nLikely fraud")
            else:
                st.success("## ✅ APPROVE\nLooks legit")
            m1, m2 = st.columns(2)
            m1.metric("Fraud probability", f"{proba:.1%}")
            m2.metric("Threshold", f"{THRESHOLD:.0%}")
            m3, m4 = st.columns(2)
            m3.metric("Time taken", f"{ms:.0f} ms")
            if actual is not None:
                correct = (proba >= THRESHOLD) == (actual == 1)
                m4.metric("Actual label", "Fraud" if actual == 1 else "Legit",
                          "model correct" if correct else "model wrong",
                          delta_color="normal" if correct else "inverse")
        with right:
            fig, contrib = explain_chart(row)
            st.pyplot(fig)
            up = contrib[contrib > 0].sort_values(ascending=False).index[:3]
            if proba >= THRESHOLD and len(up):
                st.write("**Main reasons for blocking:** " + ", ".join(f"{f} pushed risk up" for f in up) + ".")
            st.caption("V1–V28 are anonymised PCA components from the bank, so reasons name components, not business terms.")

with tab_results:
    t = meta["test"]
    st.subheader("Final results on the untouched test set")
    c = st.columns(5)
    c[0].metric("Recall (fraud caught)", f"{t['recall']:.1%}", "goal ≥ 80%")
    c[1].metric("Precision (alerts that are fraud)", f"{t['precision']:.1%}", "goal ≥ 70%")
    c[2].metric("PR-AUC", f"{t['pr_auc']:.3f}")
    c[3].metric("Frauds caught", f"{t['tp']} / {t['tp'] + t['fn']}")
    saved = meta["test_no_model_cost"] - meta["test_cost"]
    c[4].metric("Money saved vs no model", f"{saved:,.0f}", f"{saved / meta['test_no_model_cost']:.0%} of fraud losses")

    table = ROOT / "reports" / "comparison_table.csv"
    if table.exists():
        st.subheader("Model comparison (5-fold CV on training data)")
        st.dataframe(pd.read_csv(table), hide_index=True, use_container_width=True)

    figures = [
        ("01_accuracy_trap.png", "The accuracy trap"),
        ("04_model_comparison.png", "Model comparison"),
        ("05_cost_vs_threshold.png", "Business cost vs threshold"),
        ("05_confusion_matrix_test.png", "Test confusion matrix"),
        ("05_shap_summary.png", "Global explanation (SHAP)"),
        ("04_pr_curves_all.png", "Precision–recall curves"),
    ]
    cols = st.columns(2)
    for i, (f, cap) in enumerate(x for x in figures if (FIGS / x[0]).exists()):
        cols[i % 2].image(str(FIGS / f), caption=cap, use_container_width=True)
