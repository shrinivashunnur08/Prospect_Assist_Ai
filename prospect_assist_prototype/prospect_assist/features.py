"""
DATA PROCESSING — behavioral feature vector.

One customer envelope -> 13 standardized features. This is the contract
between the processing layer and the model layer: the feature list is
versioned with the trained weights in the model registry, so a scorer can
never silently receive features it wasn't trained on.
"""
from .categorizer import to_frame
from .income import estimate_income, detect_obligations

FEATURES = ["income", "income_stability", "foir", "rent_ratio", "savings_rate",
            "balance_trend", "n_emis", "upi_intensity", "invests",
            "commute_ratio", "fuel_ratio", "property_owner", "cash_crunch"]


def extract_features(customer, cfg=None):
    df = to_frame(customer, cfg)
    inc = estimate_income(df)
    obl = detect_obligations(df)
    income = max(inc["monthly_income"], 1.0)
    monthly_out = df[df["type"] == "DEBIT"].groupby("month")["amount"].sum().mean() or 0
    bal = df.sort_values("ts")["currentBalance"]
    trend = (bal.iloc[-1] - bal.iloc[0]) / max(bal.iloc[0], 1) if len(bal) > 1 else 0
    n_months = max(df["month"].nunique(), 1)
    commute = df[df["category"] == "COMMUTE"]["amount"].sum() / n_months
    fuel = df[df["category"] == "FUEL"]["amount"].sum() / n_months
    month_min_bal = df.groupby("month")["currentBalance"].min()
    feats = dict(
        income=income,
        income_stability={"HIGH": 1.0, "MEDIUM": 0.5, "LOW": 0.15}[inc["confidence"]],
        foir=min(obl["monthly_emi"] / income, 1.5),
        rent_ratio=min(obl["monthly_rent"] / income, 1.0),
        savings_rate=max(min(1 - monthly_out / income, 1), -1),
        balance_trend=max(min(trend, 3), -1),
        n_emis=len(obl["lenders"]),
        upi_intensity=float((df["mode"] == "UPI").mean()),
        invests=float((df["category"] == "SIP_INVEST").any()),
        commute_ratio=min(commute / income, 0.3),
        fuel_ratio=min(fuel / income, 0.3),
        property_owner=float((df["category"] == "PROPERTY_TAX").any()),
        cash_crunch=float((month_min_bal < 0.3 * income).mean()),
    )
    return feats, inc, obl, df
