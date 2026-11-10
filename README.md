# Loan Default Predictor

Estimate the probability that a loan will default, from borrower and loan features
(age, income, home ownership, loan intent, grade, amount, interest rate, credit
history…). Trains on ~24k historical loans, then scores a file of pending requests
and ranks applicants by risk.

## Run

```bash
pip install -r requirements.txt
python loan_default.py
```

## Pipeline

1. **Load** `data/credit_risk_train.csv` (24,435 past loans; the label is
   `loan_status`, 1 = defaulted).
2. **Preprocess** inside a scikit-learn `Pipeline` — one-hot encode the categorical
   columns, fill missing numerics with the median.
3. **Train** a balanced `RandomForestClassifier`.
4. **Evaluate** on a 20% held-out slice: accuracy, ROC AUC, recall and false alarms.
5. **Score** `data/loan_requests.csv` and rank applicants, flagging each
   approve / review / decline.

## Example result

```
Held-out performance:
  accuracy: 92.8%    ROC AUC: 0.932
  caught 794 of 1059 real defaults (75% recall), 88 false alarms

Pending loan requests, riskiest first:
  Elena Petrova     99.5%  -> DECLINE
  ...
  Johan Svensson     2.0%  -> approve
```

## What it demonstrates

- A leakage-safe `Pipeline` where encoding/imputation are fit on training data only,
  so the exact same transforms apply to new applicants.
- Judging an imbalanced classifier by **ROC AUC and recall**, not just accuracy —
  with ~22% defaults, catching real defaults is what matters.
- Turning model probabilities into an actionable decision (approve / review / decline).

## Data

The `data/` folder holds an anonymized public-style credit-risk dataset used purely
for demonstration. This is a learning project, not a production credit model.
