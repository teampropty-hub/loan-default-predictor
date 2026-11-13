"""Pick the decision threshold that minimises expected money lost.

A default probability of 0.5 is an arbitrary cutoff. In lending the two mistakes cost
very different amounts:

  * approving a loan that defaults   -> lose ``loan_amnt * LGD``   (LGD = loss given default)
  * declining a loan that would repay -> lose the interest we would have earned,
                                         ``loan_amnt * rate * term``

So the right threshold is the one that minimises total expected cost on held-out data,
not the one that maximises accuracy. This module finds it and plots the trade-off.

Decision rule: decline when ``risk >= t``, otherwise approve.
"""

import os

import matplotlib
matplotlib.use("Agg")  # headless chart generation
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(HERE, "reports")
DEFAULT_LGD = 0.60    # fraction of principal lost when a loan defaults
DEFAULT_TERM = 3.0    # years of interest foregone when a good loan is declined


def expected_cost(y_true, proba, amounts, rates, t, lgd=DEFAULT_LGD, term=DEFAULT_TERM):
    """Total cost (in loaned-currency units) of using threshold ``t``."""
    y_true = np.asarray(y_true); proba = np.asarray(proba)
    amounts = np.asarray(amounts, dtype=float); rates = np.asarray(rates, dtype=float)
    declined = proba >= t
    approved = ~declined
    # Approved loans that actually default -> principal lost.
    fn_cost = (amounts * lgd)[approved & (y_true == 1)].sum()
    # Declined loans that would have repaid -> interest foregone.
    fp_cost = (amounts * (rates / 100.0) * term)[declined & (y_true == 0)].sum()
    return fn_cost + fp_cost


def sweep(y_true, proba, amounts, rates, lgd=DEFAULT_LGD, term=DEFAULT_TERM, n=97):
    """Return (thresholds, costs) across a grid of thresholds in [0.02, 0.98]."""
    ts = np.linspace(0.02, 0.98, n)
    costs = np.array([expected_cost(y_true, proba, amounts, rates, t, lgd, term) for t in ts])
    return ts, costs


def optimal_threshold(y_true, proba, amounts, rates, lgd=DEFAULT_LGD, term=DEFAULT_TERM):
    """The threshold minimising expected cost, plus that cost and the cost at 0.5."""
    ts, costs = sweep(y_true, proba, amounts, rates, lgd, term)
    i = int(np.argmin(costs))
    naive = expected_cost(y_true, proba, amounts, rates, 0.5, lgd, term)
    return {"threshold": float(ts[i]), "cost": float(costs[i]),
            "cost_at_0.5": float(naive), "saved": float(naive - costs[i])}


def plot_cost_curve(y_true, proba, amounts, rates, lgd=DEFAULT_LGD, term=DEFAULT_TERM,
                    path=None):
    import matplotlib.pyplot as plt
    ts, costs = sweep(y_true, proba, amounts, rates, lgd, term)
    best = optimal_threshold(y_true, proba, amounts, rates, lgd, term)
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(ts, costs / 1e6, color="#2563eb", label="expected cost")
    plt.axvline(best["threshold"], color="#16a34a", linestyle="--",
                label=f"optimal t = {best['threshold']:.2f}")
    plt.axvline(0.5, color="#dc2626", linestyle=":", label="naive t = 0.50")
    plt.xlabel("decision threshold"); plt.ylabel("expected cost (millions)")
    plt.title("Cost vs threshold"); plt.legend()
    plt.tight_layout()
    path = path or os.path.join(REPORTS, "cost_threshold.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=120); plt.close()
    return best


def main():
    """Standalone run: train quickly, find the threshold, draw the curve."""
    from sklearn.model_selection import train_test_split
    import loan_default as L

    df = L.engineer_features(L.load_data())
    X, y = df[L.RAW_CATEGORICAL + L.NUMERIC], df[L.TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42,
                                              stratify=y)
    model = L.build_models()["Random Forest"].fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    best = plot_cost_curve(y_te, proba, X_te["loan_amnt"], X_te["loan_int_rate"])
    print(f"Optimal threshold: {best['threshold']:.2f}")
    print(f"Expected cost at optimal: ${best['cost']:,.0f}")
    print(f"Expected cost at 0.50   : ${best['cost_at_0.5']:,.0f}")
    print(f"Saved vs naive 0.50     : ${best['saved']:,.0f}")
    print(f"Chart -> {os.path.relpath(os.path.join(REPORTS, 'cost_threshold.png'), HERE)}")


if __name__ == "__main__":
    main()
