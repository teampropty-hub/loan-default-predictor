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
RED, AMBER, GREEN = "#D52B1E", "#D97706", "#15803D"

st.set_page_config(page_title="Canadian Credit Risk", page_icon="🍁", layout="wide")

st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; max-width: 1150px;}
  .hero {display:flex; align-items:center; gap:.6rem;}
  .hero h1 {font-size:2rem; margin:0; font-weight:800; letter-spacing:-.5px;}
  .pill {display:inline-block; padding:.2rem .6rem; border-radius:999px;
         background:#F3F5F7; color:#475569; font-size:.74rem; font-weight:600; margin-right:.3rem;}
  .card {background:#FFFFFF; border:1px solid #E6EAEE; border-radius:14px;
         padding:1.1rem 1.3rem; box-shadow:0 1px 2px rgba(16,24,40,.04);}
  .badge {padding:.55rem 0; border-radius:10px; color:#fff; text-align:center;
          font-size:1.5rem; font-weight:800; letter-spacing:.5px;}
  .muted {color:#64748B; font-size:.85rem;}
  .reason {display:flex; justify-content:space-between; padding:.45rem .7rem; margin:.3rem 0;
           border-radius:8px; background:#F8FAFC; border:1px solid #EEF2F6;}
  div[data-testid="stMetricValue"] {font-size:1.6rem;}
  .stTabs [data-baseweb="tab"] {font-weight:600;}
</style>
""", unsafe_allow_html=True)


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


def gauge(pct, color):
    """A compact SVG donut showing the default-risk percentage."""
    r, c = 54, 2 * 3.14159 * 54
    dash = pct * c
    return f"""
    <svg viewBox="0 0 140 140" width="150" height="150">
      <circle cx="70" cy="70" r="{r}" fill="none" stroke="#EEF2F6" stroke-width="14"/>
      <circle cx="70" cy="70" r="{r}" fill="none" stroke="{color}" stroke-width="14"
        stroke-linecap="round" stroke-dasharray="{dash} {c-dash}"
        transform="rotate(-90 70 70)"/>
      <text x="70" y="66" text-anchor="middle" font-size="30" font-weight="800"
        fill="#141A1F">{pct*100:.0f}%</text>
      <text x="70" y="88" text-anchor="middle" font-size="12" fill="#64748B">default risk</text>
    </svg>"""


model, rf_base, meta, scored = load_model(), load_rf_base(), load_meta(), load_scored()
medians = meta["medians"]
m = meta["metrics"]

st.markdown(
    f"<div class='hero'><span style='font-size:2rem'>🍁</span>"
    f"<h1>Canadian Credit Risk Model</h1></div>"
    f"<div style='margin:.5rem 0 .2rem'>"
    f"<span class='pill'>ROC AUC {m['roc_auc']}</span>"
    f"<span class='pill'>Calibrated · ECE {m['ece']}</span>"
    f"<span class='pill'>SHAP explainable</span>"
    f"<span class='pill'>Fairlearn audited</span>"
    f"<span class='pill'>OSFI E-23 aligned</span></div>",
    unsafe_allow_html=True)

with st.expander("About this model — data, scope and sources"):
    st.markdown(
        "Predicts loan-default probability for the **Canadian** context (CAD, province, "
        "Equifax/TransUnion **300–900** score). Built to the expectations of **OSFI "
        "Guideline E-23** (model risk, calibration, explainability, fairness) and audited "
        "for bias under the **Canadian Human Rights Act**.\n\n"
        "**Data:** a public credit-risk dataset *adapted* to a Canadian framing "
        "(`data/canadianize.py`) — a transparent demonstration proxy, **not real Canadian "
        "bank data** (none is open under PIPEDA). Real Canadian references worth grounding "
        "a production model on: [CMHC mortgage-delinquency tables]"
        "(https://www.cmhc-schl.gc.ca/en/professionals/housing-markets-data-and-research/"
        "housing-data/data-tables/mortgage-and-debt), Statistics Canada debt-service-ratio "
        "data, and the Bank of Canada's loan-level mortgage dataset.")

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
grade = st.sidebar.selectbox("Internal loan grade", list("ABCDEFG"))
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
decision, color = (("DECLINE", RED) if risk >= t
                   else ("REVIEW", AMBER) if risk >= review_floor
                   else ("APPROVE", GREEN))

tab_score, tab_why, tab_perf, tab_cost, tab_fair = st.tabs(
    ["🧮 Score", "🔎 Why this score?", "📊 Performance", "💰 Cost & threshold", "⚖️ Fairness"])

with tab_score:
    left, right = st.columns([1, 1.15])
    with left:
        st.markdown(f"<div class='card' style='text-align:center'>{gauge(risk, color)}"
                    f"<div class='badge' style='background:{color};margin-top:.6rem'>{decision}</div>"
                    f"</div>", unsafe_allow_html=True)
    with right:
        st.markdown(
            f"<div class='card'>"
            f"<b>{province}</b> · age {age} · credit score <b>{credit_score}</b><br>"
            f"<span class='muted'>CAD ${amount:,} loan · {rate:.1f}% · {intent.title()} · "
            f"income ${income:,}</span><hr style='border:none;border-top:1px solid #EEF2F6;margin:.7rem 0'>"
            f"Decision cutoff <b>{t:.0%}</b> &nbsp;·&nbsp; review {review_floor:.0%}–{t:.0%} "
            f"&nbsp;·&nbsp; approve below<br>"
            f"<span class='muted'>Threshold is the cost-minimising cutoff for the current "
            f"LGD/term — change them in the sidebar.</span></div>",
            unsafe_allow_html=True)

with tab_why:
    st.subheader("Why this score? (SHAP)")
    st.caption("The features that pushed this applicant's risk up or down — the basis for an "
               "adverse-action notice required of Canadian lenders.")
    for feat, val in E.top_reasons(rf_base, feats, k=5):
        arrow, col = ("▲ raises risk", RED) if val > 0 else ("▼ lowers risk", GREEN)
        st.markdown(f"<div class='reason'><span><b>{feat}</b></span>"
                    f"<span style='color:{col};font-weight:600'>{arrow} ({val:+.3f})</span></div>",
                    unsafe_allow_html=True)
    p = os.path.join(L.REPORTS, "shap_summary.png")
    if os.path.exists(p):
        st.image(p, caption="Global feature impact (SHAP)", width=640)

with tab_perf:
    c = st.columns(len(m))
    labels = {"roc_auc": "ROC AUC", "accuracy": "Accuracy", "precision": "Precision",
              "recall": "Recall", "f1": "F1", "brier": "Brier", "ece": "ECE"}
    for col, (k, v) in zip(c, m.items()):
        col.metric(labels.get(k, k), f"{v:.3f}")
    st.caption("Brier/ECE measure probability calibration (lower = better).")
    cols = st.columns(3)
    for col, img in zip(cols, ["roc_curves.png", "confusion_matrix.png", "calibration_curve.png"]):
        p = os.path.join(L.REPORTS, img)
        if os.path.exists(p):
            col.image(p, use_container_width=True)

with tab_cost:
    st.subheader("Cost-based threshold (CAD)")
    a, b, cc = st.columns(3)
    a.metric("Optimal threshold", f"{t:.2f}")
    b.metric("Expected cost", f"${best['cost']:,.0f}")
    cc.metric("Saved vs 0.50", f"${best['saved']:,.0f}")
    p = os.path.join(L.REPORTS, "cost_threshold.png")
    if os.path.exists(p):
        st.image(p, width=560)
    st.caption("Approving a defaulter costs principal × LGD; declining a good borrower costs "
               "foregone interest. The optimal cutoff minimises the total.")

with tab_fair:
    st.subheader(f"Fairness at threshold {t:.2f}")
    st.caption("Canadian Human Rights Act grounds. Age is a protected ground; province is "
               "regional access. No sex/gender field in the data, so that ground is not audited.")
    for by in ["age_band", "province"]:
        d = F.disparities(scored, by, t)
        st.markdown(f"**By {by}** · demographic-parity diff `{d['demographic_parity_difference']}` "
                    f"· equalized-odds diff `{d['equalized_odds_difference']}`")
        st.dataframe(F.group_metrics(scored, by, t).round(3),
                     use_container_width=True, hide_index=True)
