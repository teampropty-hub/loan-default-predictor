"""Interactive Canadian credit-risk demo.

    streamlit run app.py

Loads the calibrated model + metadata produced by `train.py` (run that first, after
`python data/canadianize.py`). Scores an applicant, explains the score with SHAP
(adverse-action reasons), lets you move the cost assumptions, and shows model performance,
calibration and a Canadian Human Rights Act fairness read-out.
"""

import json
import os

import joblib
import pandas as pd
import streamlit as st

import loan_default as L
import threshold as TH
import fairness as F
import explain as E

HERE = os.path.dirname(os.path.abspath(__file__))
PROVINCES = ["ON", "QC", "BC", "AB", "MB", "SK", "NS", "NB", "NL", "PE"]

st.set_page_config(page_title="Canadian Credit Risk", page_icon="🍁", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(os.path.join(HERE, "model.joblib"))


@st.cache_resource
def load_rf_base():
    return joblib.load(os.path.join(HERE, "rf_base.joblib"))


@st.cache_data
def load_meta():
    with open(os.path.join(HERE, "model_meta.json"), encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_scored():
    return F.load_scored()


model, rf_base, meta, scored = load_model(), load_rf_base(), load_meta(), load_scored()
medians = meta["medians"]

st.title("🍁 Canadian Credit Risk Model")
st.caption(f"Isotonic-calibrated Random Forest · ROC AUC {meta['metrics']['roc_auc']} · "
           f"~24k loans (CAD). Aligned to OSFI Guideline E-23 (model risk, explainability, "
           f"fairness). Educational demo on adapted data — not a production credit model.")

# ---- sidebar ----------------------------------------------------------------------
st.sidebar.header("Applicant")
age = st.sidebar.slider("Age", 18, 80, 30)
province = st.sidebar.selectbox("Province", PROVINCES)
credit_score = st.sidebar.slider("Credit score (Equifax/TransUnion, 300–900)", 300, 900, 700)
income = st.sidebar.number_input("Annual income (CAD $)", 4000, 500000, 60000, step=1000)
home = st.sidebar.selectbox("Home ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
emp_len = st.sidebar.slider("Employment length (years)", 0, 40, 5)
intent = st.sidebar.selectbox("Loan intent",
    ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
grade = st.sidebar.selectbox("Loan grade", list("ABCDEFG"))
amount = st.sidebar.number_input("Loan amount (CAD $)", 500, 40000, 10000, step=500)
rate = st.sidebar.slider("Interest rate (%)", 5.0, 24.0, 11.0, step=0.1)
prior_default = st.sidebar.selectbox("Prior default on file", ["N", "Y"])
hist_len = st.sidebar.slider("Credit history length (years)", 2, 30, 5)

st.sidebar.header("Cost assumptions")
lgd = st.sidebar.slider("Loss given default (LGD)", 0.1, 1.0, TH.DEFAULT_LGD, step=0.05)
term = st.sidebar.slider("Loan term (years)", 1.0, 7.0, TH.DEFAULT_TERM, step=0.5)

best = TH.optimal_threshold(scored["actual"], scored["proba"],
                            scored["loan_amnt"], scored["loan_int_rate"], lgd, term)
t = best["threshold"]
review_floor = max(0.0, t - 0.15)

applicant = pd.DataFrame([{
    "person_age": age, "person_income": income, "person_home_ownership": home,
    "person_emp_length": emp_len, "loan_intent": intent, "loan_grade": grade,
    "loan_amnt": amount, "loan_int_rate": rate,
    "loan_percent_income": round(amount / max(income, 1), 3),
    "cb_person_default_on_file": prior_default, "cb_person_cred_hist_length": hist_len,
    "province": province, "credit_score": credit_score,
}])
feats = L.prepare_features(applicant, medians)
risk = float(model.predict_proba(feats)[:, 1][0])

if risk >= t:
    decision, color = "DECLINE", "#dc2626"
elif risk >= review_floor:
    decision, color = "REVIEW", "#d97706"
else:
    decision, color = "APPROVE", "#16a34a"

tab_score, tab_why, tab_perf, tab_cost, tab_fair = st.tabs(
    ["Score applicant", "Why this score?", "Model performance", "Cost & threshold", "Fairness"])

with tab_score:
    c1, c2 = st.columns(2)
    c1.metric("Default risk (calibrated)", f"{risk:.1%}")
    c1.progress(min(risk, 1.0))
    c2.markdown(
        f"<div style='padding:1rem;border-radius:.5rem;background:{color};color:white;"
        f"text-align:center;font-size:1.6rem;font-weight:700'>{decision}</div>",
        unsafe_allow_html=True)
    c2.caption(f"Decline at or above {t:.0%} · review from {review_floor:.0%} · approve below")
    st.info(f"Threshold **{t:.2f}** is the cost-minimising cutoff for the current LGD/term "
            f"assumptions — move the sliders to see it shift.")

with tab_why:
    st.subheader("Why this score? (SHAP)")
    st.caption("The features that pushed this applicant's risk up or down — the basis for an "
               "adverse-action notice.")
    reasons = E.top_reasons(rf_base, feats, k=5)
    for feat, val in reasons:
        bump = "raises risk ↑" if val > 0 else "lowers risk ↓"
        st.write(f"- **{feat}** — {bump} (`{val:+.3f}`)")
    p = os.path.join(L.REPORTS, "shap_summary.png")
    if os.path.exists(p):
        st.image(p, caption="Global feature impact (SHAP)", width=620)

with tab_perf:
    st.subheader("Held-out metrics")
    st.table(pd.Series(meta["metrics"]).to_frame("value"))
    st.caption("Brier/ECE measure probability calibration (lower = better).")
    cols = st.columns(3)
    for col, img in zip(cols, ["roc_curves.png", "confusion_matrix.png", "calibration_curve.png"]):
        p = os.path.join(L.REPORTS, img)
        if os.path.exists(p):
            col.image(p, use_container_width=True)

with tab_cost:
    st.subheader("Cost-based threshold (CAD)")
    a, b, c = st.columns(3)
    a.metric("Optimal threshold", f"{t:.2f}")
    b.metric("Expected cost", f"${best['cost']:,.0f}")
    c.metric("Saved vs 0.50", f"${best['saved']:,.0f}")
    p = os.path.join(L.REPORTS, "cost_threshold.png")
    if os.path.exists(p):
        st.image(p, width=560)

with tab_fair:
    st.subheader(f"Fairness at threshold {t:.2f} — Canadian Human Rights Act grounds")
    st.caption("Age is a protected ground; province is regional access. No sex/gender field "
               "in the data, so that ground is not audited.")
    for by in ["age_band", "province"]:
        st.markdown(f"**By {by}** · disparities: `{F.disparities(scored, by, t)}`")
        st.dataframe(F.group_metrics(scored, by, t).round(3),
                     use_container_width=True, hide_index=True)
