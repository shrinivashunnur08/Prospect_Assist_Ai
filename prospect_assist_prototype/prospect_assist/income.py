"""
DATA PROCESSING — income assessment + existing-obligation detection.

Income philosophy: VERIFIED beats DECLARED beats TOTALLED.
- Salaried: latest-3-months recurring salary mean (current income, survives
  mid-history increments) with a tight confidence test.
- Variable income: median-month aggregation with self-transfers and interest
  excluded — the exclusion IS the income-padding defense.
- We also compute the NAIVE total-credits figure a lazy statement read would
  produce; the verified-vs-naive gap is surfaced as a padding tell.
"""
import re

import numpy as np

from .categorizer import NON_INCOME_CREDITS


def estimate_income(df) -> dict:
    """Recurring-credit detection -> verified monthly income + confidence band."""
    credits = df[df["type"] == "CREDIT"]
    naive_monthly = credits.groupby("month")["amount"].sum().mean() if len(credits) else 0.0
    verified = credits[~credits["category"].isin(NON_INCOME_CREDITS)]
    monthly = verified.groupby("month")["amount"].sum()
    sal = verified[verified["category"] == "SALARY"].groupby("month")["amount"].sum()

    if len(sal) >= 3:  # salaried: tight recurring pattern; latest 3 months = current income
        recent = sal.iloc[-3:]
        est, spread = recent.mean(), recent.std(ddof=0)
        source = "Verified recurring salary credit (latest 3 months)"
        confidence = "HIGH" if (spread / max(est, 1)) < 0.05 else "MEDIUM"
    else:              # variable income: trimmed monthly aggregation
        vals = monthly.values
        est = np.median(vals) if len(vals) else 0.0
        spread = np.std(vals) if len(vals) else 0.0
        cv = spread / max(est, 1)
        confidence = "MEDIUM" if cv < 0.35 else "LOW"
        source = "Aggregated business/gig inflows (median-month basis, self-transfers excluded)"
    return dict(monthly_income=float(est), band=float(1.96 * spread / max(np.sqrt(max(len(monthly), 1)), 1)),
                confidence=confidence, source=source, naive_monthly=float(naive_monthly))


def detect_obligations(df) -> dict:
    """Recurring EMI debits -> lender names, monthly EMI outgo, rent outflow."""
    emis = df[df["category"] == "EMI"]
    monthly_emi = emis.groupby("month")["amount"].sum().mean() if len(emis) else 0.0
    lenders = sorted(set(re.sub(r"ACH-EMI-([A-Z]+)-.*", r"\1", n) for n in emis["narration"]))
    rent = df[df["category"] == "RENT"].groupby("month")["amount"].sum().mean() if (df["category"] == "RENT").any() else 0.0
    return dict(monthly_emi=float(monthly_emi or 0), lenders=lenders, monthly_rent=float(rent or 0))
