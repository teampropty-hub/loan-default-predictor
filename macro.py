"""Real Bank of Canada macro data (policy rate) and derived risk features.

Credit risk is strongly cyclical: defaults rise after monetary tightening. This module
pulls the **real** Bank of Canada overnight policy-rate history from the public Valet API
(https://www.bankofcanada.ca/valet/) and turns it into monthly macro features:

  * boc_policy_rate     — the policy rate that month
  * boc_rate_change_12m — 12-month change (positive = tightening)
  * boc_rate_vol        — 12-month rolling volatility of the rate

These are joined to each loan by a synthetic origination month (see data/canadianize.py),
and a brand-new applicant is assessed under the *current* environment (CURRENT_MACRO).

Run with:  python macro.py   (refreshes data/boc_macro.csv)
"""

import json
import os
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MACRO_CSV = os.path.join(HERE, "data", "boc_macro.csv")
SERIES = "V39079"  # Bank of Canada target for the overnight rate
VALET = f"https://www.bankofcanada.ca/valet/observations/{SERIES}/json?start_date=2010-01-01"


def fetch_boc():
    """Download the real policy-rate history and build monthly macro features."""
    with urllib.request.urlopen(VALET, timeout=30) as r:
        raw = json.load(r)
    obs = [(o["d"], float(o[SERIES]["v"])) for o in raw["observations"] if o[SERIES]["v"]]
    s = pd.DataFrame(obs, columns=["date", "policy_rate"])
    s["date"] = pd.to_datetime(s["date"])
    monthly = (s.set_index("date").resample("MS")["policy_rate"].last().ffill().to_frame())
    monthly["boc_policy_rate"] = monthly["policy_rate"]
    monthly["boc_rate_change_12m"] = monthly["policy_rate"].diff(12)
    monthly["boc_rate_vol"] = monthly["policy_rate"].rolling(12, min_periods=3).std()
    monthly = monthly.dropna().reset_index()
    monthly["month"] = monthly["date"].dt.strftime("%Y-%m")
    return monthly[["month", "boc_policy_rate", "boc_rate_change_12m", "boc_rate_vol"]]


def load_macro():
    """Load the saved macro table (download + save it if missing)."""
    if not os.path.exists(MACRO_CSV):
        save()
    return pd.read_csv(MACRO_CSV)


def current_macro():
    """The most recent month's macro values — the environment a new applicant faces."""
    m = load_macro().iloc[-1]
    return {"boc_policy_rate": float(m["boc_policy_rate"]),
            "boc_rate_change_12m": float(m["boc_rate_change_12m"]),
            "boc_rate_vol": float(m["boc_rate_vol"])}


def save():
    df = fetch_boc()
    os.makedirs(os.path.dirname(MACRO_CSV), exist_ok=True)
    df.to_csv(MACRO_CSV, index=False)
    print(f"saved {len(df)} months -> {os.path.relpath(MACRO_CSV, HERE)}")
    print(f"current: {current_macro()}")
    return df


if __name__ == "__main__":
    save()
