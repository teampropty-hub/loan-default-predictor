"""Interactive loan-risk demo.

    streamlit run app.py

Loads the model + metadata produced by `train.py` (run that first), scores an applicant
you describe in the sidebar, and lets you move the cost assumptions to see the
money-optimal decision threshold shift. Extra tabs show model performance, the cost
trade-off and a fairness read-out.
"""

import json
import os

import joblib
import pandas as pd
import streamlit as st

import loan_default as L
import threshold as TH
import fairness as F

HERE = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Loan Default Risk", page_icon="💳", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(os.path.join(HERE, "model.joblib"))


@st.cache_data
def load_meta():
    with open(os.path.join(HERE, "model_meta.json"), encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_scored():
    return F.load_scored()


model, meta, scored = load_model(), load_meta(), load_scored()
medians = meta["medians"]

st.title("💳 Loan Default Risk")
st.caption("Random Forest · ROC AUC {auc} · trained on ~24k loans. "
           "Educational demo, not a production credit model.".format(auc=meta["metrics"]["roc_auc"]))

# ---- sidebar: applicant + cost assumptions ----------------------------------------
st.sidebar.header("Applicant")
age = st.sidebar.slider("Age", 18, 80, 30)
income = st.sidebar.number_input("Annual income ($)", 4000, 500000, 60000, step=1000)
home = st.sidebar.selectbox("Home ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
emp_len = st.sidebar.slider("Employment length (years)", 0, 40, 5)
intent = st.sidebar.selectbox("Loan intent",
    ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
grade = st.sidebar.selectbox("Loan grade", list("ABCDEFG"))
amount = st.sidebar.number_input("Loan amount ($)", 500, 40000, 10000, step=500)
rate = st.sidebar.slider("Interest rate (%)", 5.0, 24.0, 11.0, step=0.1)
prior_default = st.sidebar.selectbox("Prior default on file", ["N", "Y"])
hist_len = st.sidebar.slider("Credit history length (years)", 2, 30, 5)

st.sidebar.header("Cost assumptions")
lgd = st.sidebar.slider("Loss given default (LGD)", 0.1, 1.0, TH.DEFAULT_LGD, step=0.05,
                        help="Fraction of principal lost when a loan defaults.")
term = st.sidebar.slider("Loan term (years)", 1.0, 7.0, TH.DEFAULT_TERM, step=0.5,
                         help="Years of interest foregone when a good loan is declined.")

# Recompute the money-optimal threshold live from the held-out scored set.
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
}])
risk = float(model.predict_proba(L.prepare_features(applicant, medians))[:, 1][0])

if risk >= t:
    decision, color = "DECLINE", "#dc2626"
elif risk >= review_floor:
    decision, color = "REVIEW", "#d97706"
else:
    decision, color = "APPROVE", "#16a34a"

tab_score, tab_perf, tab_cost, tab_fair = st.tabs(
    ["Score applicant", "Model performance", "Cost & threshold", "Fairness"])

with tab_score:
    c1, c2 = st.columns([1, 1])
    c1.metric("Default risk", f"{risk:.1%}")
    c1.progress(min(risk, 1.0))
    c2.markdown(
        f"<div style='padding:1rem;border-radius:.5rem;background:{color};color:white;"
        f"text-align:center;font-size:1.6rem;font-weight:700'>{decision}</div>",
        unsafe_allow_html=True)
    c2.caption(f"Decline at or above {t:.0%} · review from {review_floor:.0%} · "
               f"approve below")
    st.info(f"Threshold **{t:.2f}** is the cost-minimising cutoff for the current "
            f"LGD/term assumptions — move the sliders to see it shift.")

with tab_perf:
    st.subheader("Held-out metrics")
    st.table(pd.Series(meta["metrics"]).to_frame("value"))
    cols = st.columns(3)
    for col, img in zip(cols, ["roc_curves.png", "confusion_matrix.png", "feature_importance.png"]):
        p = os.path.join(L.REPORTS, img)
        if os.path.exists(p):
            col.image(p, use_container_width=True)

with tab_cost:
    st.subheader("Cost-based threshold")
    a, b, c = st.columns(3)
    a.metric("Optimal threshold", f"{t:.2f}")
    b.metric("Expected cost", f"${best['cost']:,.0f}")
    c.metric("Saved vs 0.50", f"${best['saved']:,.0f}")
    p = os.path.join(L.REPORTS, "cost_threshold.png")
    if os.path.exists(p):
        st.image(p, width=560)
    st.caption("Approving a defaulter costs principal × LGD; declining a good borrower "
               "costs the foregone interest. The optimal cutoff minimises the total.")

with tab_fair:
    st.subheader(f"Group fairness at threshold {t:.2f}")
    st.caption("Age is protected under ECOA; home ownership is a wealth proxy. The "
               "dataset has no race or gender, so those cannot be audited.")
    for by in ["age_band", "person_home_ownership"]:
        table = F.group_metrics(scored, by, t)
        st.markdown(f"**By {by}**")
        st.dataframe(table.round(3), use_container_width=True, hide_index=True)
