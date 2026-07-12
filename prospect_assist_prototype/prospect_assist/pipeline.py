"""
SCORING PIPELINE — the composition root.

One customer envelope in -> one fully-explained decision out:
  features -> intent probability -> reason codes -> trust screen ->
  capacity gate -> pre-approved offer -> right-time triggers.

Every downstream surface (Streamlit workboard, REST API, batch job, prospect
report) calls score_customer(); there is exactly one scoring path to audit.
"""
import numpy as np

from .config import load_config
from .features import FEATURES, extract_features
from .modeling import reason_contributions
from .policy import capacity_gate, preapproved_offer
from .triggers import detect_triggers
from .trust import trust_screen

REASON_TEXT = {
    "income": ("Strong verified income of ₹{income:,.0f}/month", "Low assessed income limits eligibility"),
    "income_stability": ("Highly regular income pattern (salary-grade stability)", "Irregular income — verify before underwriting"),
    "foir": ("Existing EMIs already consume {foir:.0%} of income — limited headroom", "Low existing obligations ({foir:.0%} FOIR) — room for a new EMI"),
    "rent_ratio": ("Pays rent of {rent_ratio:.0%} of income — an EMI could replace rent (home-loan trigger)", ""),
    "savings_rate": ("Saves {savings_rate:.0%} of income monthly — strong repayment capacity", "Spends nearly all inflows — thin repayment buffer"),
    "balance_trend": ("Balances rising steadily — accumulating for a big purchase", "Declining balances — financial stress signal"),
    "n_emis": ("Servicing {n_emis} active loans concurrently", "No existing loans — fresh credit capacity"),
    "invests": ("Active SIP investor — financially engaged profile", ""),
    "commute_ratio": ("Spends {commute_ratio:.1%} of income on cabs/commute — own-vehicle economics favour an auto loan", ""),
    "fuel_ratio": ("Regular fuel spends — already runs a vehicle", "No own-vehicle running costs — first-vehicle candidate"),
    "property_owner": ("Pays property tax — owns collateralizable property (LAP-ready)", ""),
    "cash_crunch": ("Recurring month-end cash crunches — working-capital need a LAP can solve", "Comfortable liquidity through the month"),
}


def score_customer(customer, bundle, product="home", cfg=None) -> dict:
    cfg = cfg or load_config()
    f, inc, obl, df = extract_features(customer, cfg)
    x = np.array([[f[k] for k in FEATURES]])
    xn = (x - bundle["mu"]) / bundle["sd"]
    prob = float(bundle["models"][product].predict_proba(xn)[0, 1])
    contrib = reason_contributions(bundle, product, xn)
    order = np.argsort(-np.abs(contrib))
    reasons = []
    for idx in order[:5]:
        feat = FEATURES[idx]
        pos, neg = REASON_TEXT.get(feat, ("", ""))
        text = pos if contrib[idx] > 0 else neg
        if text:
            reasons.append(("+" if contrib[idx] > 0 else "−", text.format(**f)))
    trust = trust_screen(df, cfg)
    offer = preapproved_offer(f, product, cfg)
    triggers = detect_triggers(df, inc, obl)
    # "genuinely interested" (PS language): expressed-interest event = hottest trigger
    engagement = customer.get("engagement")
    if engagement:
        triggers.insert(0, f"🔥 Expressed interest: {engagement['event']} "
                           f"{engagement['days_ago']} day(s) ago — call while it's hot")
    # declared-vs-assessed: "accurate assessment of borrowers' ACTUAL income" (PS outcome 2)
    declared = customer.get("declaredIncomeMonthly")
    income_check = None
    if declared:
        gap = (declared - inc["monthly_income"]) / max(inc["monthly_income"], 1)
        income_check = dict(declared=float(declared), gap=float(gap),
                            verdict=("OVERSTATED" if gap > 0.25 else
                                     "UNDERSTATED" if gap < -0.25 else "CONSISTENT"))
    # two gates, both prudent-underwriting: capacity (FOIR/income) and trust (anomaly screen)
    capacity_ok = capacity_gate(f, inc, cfg)
    eligible = capacity_ok and trust["clean"]
    holder = (customer.get("account", {}).get("profile", {})
              .get("holders", {}).get("holder") or [{}])[0]
    contact = dict(mobile=holder.get("mobile", "—"), email=holder.get("email", "—"))
    return dict(name=customer["customerName"], persona=customer.get("persona", "unknown"), prob=prob,
                contact=contact,
                income=inc, obligations=obl, features=f, reasons=reasons[:3],
                eligible=eligible, capacity_ok=capacity_ok, trust=trust,
                offer=offer, triggers=triggers, engagement=engagement,
                income_check=income_check, df=df)
