"""Interactive Canadian credit-risk demo.

    streamlit run app.py

Loads the calibrated model + metadata produced by `train.py` (run `python data/canadianize.py`
and `python train.py` first). Scores an applicant, explains it with SHAP, lets you move the
cost assumptions, and shows performance, calibration, a fairness audit and a portfolio view.
"""

import json
import os

import joblib
import pandas as pd
import streamlit as st
from streamlit_extras.metric_cards import style_metric_cards

import loan_default as L
import threshold as TH
import fairness as F
import explain as E

HERE = os.path.dirname(os.path.abspath(__file__))
PROVINCES = ["ON", "QC", "BC", "AB", "MB", "SK", "NS", "NB", "NL", "PE"]
RED, AMBER, GREEN, INK, MUTED = "#D52B1E", "#B45309", "#15803D", "#0F172A", "#64748B"

st.set_page_config(page_title="Canadian Credit Risk", page_icon="🍁", layout="wide")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
  html, body, [class*="css"], .stMarkdown, button, input, select { font-family:'Inter',-apple-system,sans-serif; }
  .block-container { padding-top:2rem; padding-bottom:3rem; max-width:1180px; }
  #MainMenu, footer, header [data-testid="stToolbar"] { visibility:hidden; }

  .brand { display:flex; align-items:center; gap:.7rem; }
  .brand .leaf { font-size:1.9rem; }
  .brand h1 { font-size:1.9rem; font-weight:800; letter-spacing:-1px; margin:0; color:var(--ink,#0F172A); }
  .sub { color:#64748B; font-size:.92rem; margin:.35rem 0 .9rem; max-width:720px; }
  .tags { display:flex; flex-wrap:wrap; gap:.4rem; margin-bottom:.4rem; }
  .tag { font-size:.72rem; font-weight:600; color:#334155; background:#F1F5F9;
         border:1px solid #E2E8F0; border-radius:999px; padding:.22rem .66rem; }
  .tag.red { color:#B91C1C; background:#FEF2F2; border-color:#FECACA; }

  .card { background:#FFFFFF; border:1px solid #E2E8F0; border-radius:16px; padding:1.3rem 1.4rem;
          box-shadow:0 1px 3px rgba(15,23,42,.05), 0 1px 2px rgba(15,23,42,.03); }
  .badge { padding:.6rem 0; border-radius:12px; color:#fff; text-align:center;
           font-size:1.35rem; font-weight:800; letter-spacing:1px; }
  .kv { color:#0F172A; font-size:.95rem; line-height:1.9; }
  .muted { color:#64748B; font-size:.85rem; }
  .reason { display:flex; justify-content:space-between; align-items:center; padding:.6rem .9rem;
            margin:.4rem 0; border-radius:10px; background:#F8FAFC; border:1px solid #EEF2F6; font-size:.92rem; }

  .stTabs [data-baseweb="tab-list"] { gap:.4rem; border-bottom:1px solid #E2E8F0; }
  .stTabs [data-baseweb="tab"] { font-weight:600; color:#64748B; padding:.4rem .2rem; }
  .stTabs [aria-selected="true"] { color:#0F172A; }
  section[data-testid="stSidebar"] { background:#FBFCFD; border-right:1px solid #E2E8F0; }
  section[data-testid="stSidebar"] h2 { font-size:.78rem; text-transform:uppercase; letter-spacing:.08em;
            color:#94A3B8; font-weight:700; }
  div[data-testid="stMetric"] { background:#FFF; border:1px solid #E2E8F0; border-radius:14px; padding:.9rem 1rem; }
  div[data-testid="stMetricValue"] { font-size:1.4rem; font-weight:700; }
  div[data-testid="stMetricLabel"] p { font-size:.8rem; color:#64748B; font-weight:600; }
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


@st.cache_data
def load_macro():
    return pd.read_csv(os.path.join(HERE, "data", "boc_macro.csv"))


def gauge(pct, color):
    r, c = 52, 2 * 3.14159 * 52
    dash = pct * c
    return f"""
    <svg viewBox="0 0 150 150" width="168" height="168" style="display:block;margin:0 auto">
      <circle cx="75" cy="75" r="{r}" fill="none" stroke="#EEF2F6" stroke-width="11"/>
      <circle cx="75" cy="75" r="{r}" fill="none" stroke="{color}" stroke-width="11"
        stroke-linecap="round" stroke-dasharray="{dash} {c-dash}" transform="rotate(-90 75 75)"/>
      <text x="75" y="72" text-anchor="middle" font-size="34" font-weight="800"
        font-family="Inter" fill="#0F172A">{pct*100:.0f}%</text>
      <text x="75" y="94" text-anchor="middle" font-size="12" font-family="Inter"
        fill="#64748B">default risk</text>
    </svg>"""


model, rf_base, meta, scored = load_model(), load_rf_base(), load_meta(), load_scored()
medians = meta["medians"]
mt = meta["metrics"]

st.markdown(
    f"<div class='brand'><span class='leaf'>🍁</span><h1>Canadian Credit Risk Model</h1></div>"
    f"<div class='sub'>Supervised ML for Canadian lending — calibrated default probability, "
    f"SHAP explanations and a fairness audit, aligned to OSFI Guideline E-23.</div>"
    f"<div class='tags'>"
    f"<span class='tag'>ROC AUC {mt['roc_auc']}</span>"
    f"<span class='tag'>Calibrated · ECE {mt['ece']}</span>"
    f"<span class='tag'>SHAP</span><span class='tag'>Fairlearn</span>"
    f"<span class='tag'>Bank of Canada data</span>"
    f"<span class='tag red'>OSFI E-23</span></div>",
    unsafe_allow_html=True)

with st.expander("About this model — data, scope and sources"):
    st.markdown(
        "Predicts loan-default probability for the **Canadian** context (CAD, province, "
        "Equifax/TransUnion **300–900** score) with real **Bank of Canada** policy-rate data. "
        "Built to **OSFI Guideline E-23** and audited for bias under the **Canadian Human "
        "Rights Act**.\n\n**Data:** a public credit-risk dataset *adapted* to a Canadian "
        "framing (`data/canadianize.py`) — a transparent demonstration proxy, **not real "
        "Canadian bank data** (none is open under PIPEDA). Real references: "
        "[CMHC delinquency tables](https://www.cmhc-schl.gc.ca/en/professionals/housing-markets-data-and-research/housing-data/data-tables/mortgage-and-debt), "
        "[OSFI B-20](https://www.osfi-bsif.gc.ca/en/guidance/guidance-library/residential-mortgage-underwriting-practices-procedures-guideline-b-20), "
        "[Bank of Canada](https://www.bankofcanada.ca/valet/).")

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
gds = float(feats["gds_ratio"].iloc[0])
stress_gds = float(feats["b20_stress_gds"].iloc[0])
stress_pass = bool(feats["b20_stress_pass"].iloc[0])
decision, color = (("DECLINE", RED) if risk >= t
                   else ("REVIEW", AMBER) if risk >= review_floor
                   else ("APPROVE", GREEN))

tab_score, tab_why, tab_perf, tab_cost, tab_fair, tab_port = st.tabs(
    ["Score", "Why this score?", "Performance", "Cost & threshold", "Fairness", "Portfolio"])

with tab_score:
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.markdown(f"<div class='card' style='text-align:center'>{gauge(risk, color)}"
                    f"<div class='badge' style='background:{color};margin-top:.9rem'>{decision}</div>"
                    f"</div>", unsafe_allow_html=True)
    with right:
        st.markdown(
            f"<div class='card'><div class='kv'>"
            f"<b style='font-size:1.05rem'>{province}</b> &nbsp;·&nbsp; age {age} &nbsp;·&nbsp; "
            f"credit score <b>{credit_score}</b><br>"
            f"<span class='muted'>CAD ${amount:,} · {rate:.1f}% · {intent.title()} · income ${income:,}</span>"
            f"<hr style='border:none;border-top:1px solid #EEF2F6;margin:.8rem 0'>"
            f"Affordability (GDS) <b>{gds:.0%}</b> &nbsp;·&nbsp; "
            f"OSFI B-20 stress: <b style='color:{GREEN if stress_pass else RED}'>"
            f"{'PASS' if stress_pass else 'FAIL'}</b> <span class='muted'>(stressed {stress_gds:.0%})</span><br>"
            f"Decision cutoff <b>{t:.0%}</b> &nbsp;·&nbsp; review {review_floor:.0%}–{t:.0%} "
            f"&nbsp;·&nbsp; approve below"
            f"</div></div>", unsafe_allow_html=True)
        st.caption("The threshold is the cost-minimising cutoff for the current LGD/term — "
                   "adjust them in the sidebar.")

with tab_why:
    st.markdown("##### Why this score? — SHAP")
    st.caption("Features that pushed this applicant's risk up or down — the basis for an "
               "adverse-action notice required of Canadian lenders.")
    for feat, val in E.top_reasons(rf_base, feats, k=5):
        arrow, col = ("▲ raises risk", RED) if val > 0 else ("▼ lowers risk", GREEN)
        st.markdown(f"<div class='reason'><span><b>{feat}</b></span>"
                    f"<span style='color:{col};font-weight:700'>{arrow} &nbsp;{val:+.3f}</span></div>",
                    unsafe_allow_html=True)
    p = os.path.join(L.REPORTS, "shap_summary.png")
    if os.path.exists(p):
        st.image(p, caption="Global feature impact (SHAP)", width=660)

with tab_perf:
    labels = {"roc_auc": "ROC AUC", "accuracy": "Accuracy", "precision": "Precision",
              "recall": "Recall", "f1": "F1", "brier": "Brier", "ece": "ECE"}
    items = [(labels.get(k, k), v) for k, v in mt.items()]
    for row in (items[:4], items[4:]):
        cols = st.columns(4)
        for col, (name, v) in zip(cols, row):
            col.metric(name, f"{v:.3f}")
    style_metric_cards(background_color="#FFFFFF", border_left_color=RED,
                       border_color="#E2E8F0", box_shadow=False)
    st.caption("Brier/ECE measure probability calibration (lower = better).")
    cols = st.columns(3)
    for col, img in zip(cols, ["roc_curves.png", "confusion_matrix.png", "calibration_curve.png"]):
        p = os.path.join(L.REPORTS, img)
        if os.path.exists(p):
            col.image(p, use_container_width=True)

with tab_cost:
    st.markdown("##### Cost-based threshold (CAD)")
    a, b, cc = st.columns(3)
    a.metric("Optimal threshold", f"{t:.2f}")
    b.metric("Expected cost", f"${best['cost']:,.0f}")
    cc.metric("Saved vs 0.50", f"${best['saved']:,.0f}")
    style_metric_cards(background_color="#FFFFFF", border_left_color=RED,
                       border_color="#E2E8F0", box_shadow=False)
    p = os.path.join(L.REPORTS, "cost_threshold.png")
    if os.path.exists(p):
        st.image(p, width=560)
    st.caption("Approving a defaulter costs principal × LGD; declining a good borrower costs "
               "foregone interest. The optimal cutoff minimises the total.")

with tab_fair:
    st.markdown(f"##### Fairness at threshold {t:.2f} — Canadian Human Rights Act")
    st.caption("Age is a protected ground; province is regional access. No sex/gender field "
               "in the data, so that ground is not audited.")
    for by in ["age_band", "province"]:
        d = F.disparities(scored, by, t)
        st.markdown(f"**By {by}** · demographic-parity diff `{d['demographic_parity_difference']}` "
                    f"· equalized-odds diff `{d['equalized_odds_difference']}`")
        st.dataframe(F.group_metrics(scored, by, t).round(3),
                     use_container_width=True, hide_index=True)

with tab_port:
    st.markdown("##### Portfolio overview")
    st.caption("The held-out loan book scored by the model — the same views exported for "
               "Power BI via `python export.py`.")
    port = scored.copy()
    port["decision"] = pd.cut(port["proba"], [-0.01, review_floor, t, 1.01],
                              labels=["approve", "review", "decline"])
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Loans", f"{len(port):,}")
    k2.metric("Approved", f"{(port['decision'] == 'approve').mean():.0%}")
    k3.metric("Default rate", f"{port['actual'].mean():.0%}")
    k4.metric("Avg risk", f"{port['proba'].mean():.0%}")
    style_metric_cards(background_color="#FFFFFF", border_left_color=RED,
                       border_color="#E2E8F0", box_shadow=False)
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Decision mix")
        st.bar_chart(port["decision"].value_counts(), color=GREEN)
    with c2:
        st.caption("Default rate by province")
        st.bar_chart(port.groupby("province", observed=True)["actual"].mean()
                     .sort_values(ascending=False), color=RED)
    c3, c4 = st.columns(2)
    with c3:
        st.caption("Risk distribution")
        bands = pd.cut(port["proba"], [0, .1, .25, .5, 1.01],
                       labels=["<10%", "10–25%", "25–50%", ">50%"]).value_counts().sort_index()
        st.bar_chart(bands, color=INK)
    with c4:
        st.caption("Bank of Canada policy rate — real history")
        st.line_chart(load_macro().set_index("month")["boc_policy_rate"], color=RED)
