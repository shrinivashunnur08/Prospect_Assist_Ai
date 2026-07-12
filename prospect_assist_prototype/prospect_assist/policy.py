"""
POLICY & DECISION ENGINE — gates and offer sizing.

Everything here reads from config.yaml (policy.*): the FOIR ceiling, income
floor, and per-product rate/tenor/cap are risk-team-owned parameters, not
code. Deliberately NOT machine learning: an offer amount the bank must stand
behind contractually comes from deterministic, reviewable arithmetic.
"""
from .config import load_config


def capacity_gate(feats, inc, cfg=None) -> bool:
    """Prudent-underwriting gate: FOIR headroom + income floor."""
    cfg = cfg or load_config()
    pol = cfg["policy"]
    return feats["foir"] < pol["foir_ceiling"] and inc["monthly_income"] > pol["min_income"]


def preapproved_offer(feats, product, cfg=None) -> dict:
    """Max EMI headroom under the FOIR ceiling -> loan amount via annuity formula."""
    cfg = cfg or load_config()
    pol = cfg["policy"]
    terms = pol["products"][product]
    income = feats["income"]
    current_emi = feats["foir"] * income
    max_emi = max(pol["foir_ceiling"] * income - current_emi, 0.0)
    r = terms["rate"] / 12
    n = terms["tenor_months"]
    amount = max_emi * (1 - (1 + r) ** -n) / r
    amount = min(amount, terms["cap"])
    amount = int(amount // 10000 * 10000)  # round down to nearest ₹10k
    return dict(max_emi=round(max_emi), amount=amount,
                rate=terms["rate"], tenor_months=n, label=terms["label"])
