# 🍁 Canadian Credit Risk Model

> A supervised **machine-learning** credit-risk model built for the **Canadian banking**
> context: it predicts loan-default probability in **CAD**, uses an **Equifax/TransUnion
> 300–900 credit score**, and is aligned to **OSFI Guideline E-23** (model risk,
> explainability, fairness) and the **Canadian Human Rights Act**.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-calibrated%20RF-F7931E?logo=scikitlearn&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-explainability-6f42c1)
![Fairlearn](https://img.shields.io/badge/Fairlearn-fairness-0A7BBB)
![Streamlit](https://img.shields.io/badge/Streamlit-demo-FF4B4B?logo=streamlit&logoColor=white)

**ROC AUC 0.93 · calibrated (ECE 0.006) · explained with SHAP · fairness-audited with Fairlearn.**

**→ [`analysis.ipynb`](./analysis.ipynb) for the walkthrough · `streamlit run app.py` for the demo.**

![Demo](reports/demo.png)

---

## What it does

Given a Canadian loan application (age, **province**, **credit score 300–900**, income in
CAD, loan details…), the model outputs a **calibrated default probability** and a decision
— **approve / review / decline** — plus the **reasons** behind the score (for an
adverse-action notice). It's trained on ~24,000 loans where the outcome is known.

This is **binary classification**: the model learns the patterns that separate good loans
from bad ones, instead of following hand-written rules.

## Aligned to OSFI Guideline E-23

Canadian federally regulated banks must manage model risk under **OSFI Guideline E-23**
(effective May 2027), which explicitly covers credit-scoring models. Each part of this repo
maps to one of its expectations:

| E-23 expectation | In this project |
|---|---|
| Sound methodology & validation | Cross-validation + held-out test, `train.py` |
| Calibrated, reliable outputs | Isotonic calibration, Brier/ECE — `calibration.py` |
| Explainability | SHAP global + per-applicant reasons — `explain.py` |
| Bias & fairness controls | Fairlearn audit under the Canadian Human Rights Act — `fairness.py` |
| Data governance & privacy | Documented provenance; no real personal data (PIPEDA) |

## Quickstart

```bash
pip install -r requirements.txt
python data/canadianize.py   # build the Canadian-framed dataset
python train.py              # train + calibrate, pick threshold, write model + charts
streamlit run app.py         # interactive demo
```

## Results

Held-out test set (20% never seen in training), calibrated Random Forest:

| metric | value |
|---|---|
| ROC AUC | **0.929** |
| accuracy | 93.0% |
| precision | 95.7% |
| recall | 71.1% |
| F1 | 81.6% |
| Brier score | 0.059 |
| **ECE (calibration error)** | **0.006** |

| ROC curves | Calibration (reliability) | SHAP — feature impact |
|---|---|---|
| ![ROC](reports/roc_curves.png) | ![Calibration](reports/calibration_curve.png) | ![SHAP](reports/shap_summary.png) |

The near-zero ECE means a predicted "20% risk" really does default ~20% of the time — the
probabilities can feed IFRS 9 expected-credit-loss provisioning and risk-based pricing. The
**credit score** is the strongest driver, as a lender would expect.

## Cost-based threshold

The two mistakes cost different amounts: approving a loan that defaults loses
`principal × LGD`; declining a good borrower loses the foregone interest. Because the model
is **calibrated**, the money-optimal cutoff already sits near 0.49 — the cost analysis
fine-tunes it (and lets you re-derive it live in the demo when LGD/term change).

![Cost vs threshold](reports/cost_threshold.png)

## Fairness (Canadian Human Rights Act)

Audited with **Fairlearn** (`MetricFrame`, demographic-parity and equalized-odds
differences) across **age** (a protected ground) and **province** (regional access). Both
pass the four-fifths rule — the largest demographic-parity gap is 0.06 by age and 0.13 by
province. The dataset has **no sex/gender** field, so that protected ground is not audited —
stated plainly rather than faked.

![Fairness by province](reports/fairness_province.png)

## Repo map

| File | Purpose |
|---|---|
| `loan_default.py` | Data loading, feature engineering, models, scoring (shared core) |
| `data/canadianize.py` | Adds CAD framing, province and a 300–900 credit score |
| `train.py` | Trains + calibrates, picks the threshold, writes `model.joblib` + charts |
| `calibration.py` | Probability calibration (Brier / ECE / reliability curve) |
| `explain.py` | SHAP explainability + per-applicant adverse-action reasons |
| `threshold.py` | Cost-minimising decision threshold |
| `fairness.py` | Fairlearn fairness audit (age + province) |
| `app.py` | Streamlit demo |
| `analysis.ipynb` | Narrative walkthrough with charts |

## Data & disclaimer

No open dataset of **individual Canadian loan defaults** exists (privacy / PIPEDA), so this
uses a public US-style credit-risk dataset **adapted to a Canadian framing** — CAD,
provinces sampled from StatsCan population shares, and a derived Equifax/TransUnion-style
300–900 score. It is a **transparent demonstration proxy, not real Canadian bank data**, and
this is a learning project, not a production credit model.

## License

[MIT](./LICENSE)
