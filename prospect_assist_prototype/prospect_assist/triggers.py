"""
RIGHT-TIME TRIGGERS — WHEN to call, not just WHO.

Transparent heuristics over the transaction history (salary hike, rent rise,
idle-balance build-up, windfall credit). In production these become
event-driven: a new salary credit lands -> the customer is re-scored the same
day (EventBridge/queue worker), so outreach happens in the week intent peaks.
"""
import pandas as pd

from .categorizer import NON_INCOME_CREDITS


def detect_triggers(df, inc, obl) -> list:
    trig = []
    sal = df[(df["type"] == "CREDIT") & (df["category"] == "SALARY")].groupby("month")["amount"].sum()
    if len(sal) >= 5:
        early, late = sal.iloc[:3].mean(), sal.iloc[-2:].mean()
        if late > early * 1.06:
            trig.append(f"Salary hike detected: ₹{early:,.0f} → ₹{late:,.0f}/month (+{(late/early-1):.0%}) — capacity just improved")
    rent = df[df["category"] == "RENT"].groupby("month")["amount"].sum()
    if len(rent) >= 5:
        early, late = rent.iloc[:3].mean(), rent.iloc[-2:].mean()
        if late > early * 1.06:
            trig.append(f"Rent increased to ₹{late:,.0f}/month — home-loan conversation is timely")
    bal = df.groupby("month")["currentBalance"].mean()
    if len(bal) >= 6 and bal.iloc[-2:].mean() > max(bal.iloc[:3].mean() * 1.4, 2 * inc["monthly_income"]):
        trig.append("Idle balance building up — accumulating for a down payment; approach with an offer now")
    recent = df[(df["type"] == "CREDIT") & (df["ts"] >= df["ts"].max() - pd.Timedelta(days=60))]
    windfall = recent[(recent["amount"] >= 3 * max(inc["monthly_income"], 1)) &
                      (~recent["category"].isin(NON_INCOME_CREDITS))]
    if len(windfall):
        trig.append(f"Windfall credit of ₹{windfall['amount'].max():,.0f} in the last 60 days — down-payment ready")
    return trig
