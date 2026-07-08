"""
TRUST MODULE — Hunter-style anomaly screen at PROSPECT stage.

Deliberately rule-based, not ML: a fraud gate must be auditable line-by-line.
These are first-line versions of what Perfios FSA alerts and Experian Hunter
rules catch at APPLICATION stage — run here so a padded statement never
becomes a "high-quality lead". Rules can be toggled and penalties tuned in
config.yaml (trust.*) by the bank's fraud-risk team.

Production enrichment path (Phase 2): Hunter closed-user-group feed and an
anomaly-detection model as additional voters; this rules layer stays as the
explainable floor.
"""
import numpy as np
import pandas as pd

from .config import load_config


def trust_screen(df, cfg=None) -> dict:
    cfg = cfg or load_config()
    tc = cfg["trust"]
    if not tc.get("enabled", True):
        return dict(flags=[], trust_score=100, clean=True)
    rules = tc["rules"]
    flags = []
    credits = df[df["type"] == "CREDIT"]

    # 1. circular transfers: round-trip credits that exit within days (income padding)
    if rules.get("circular_transfers", True):
        tin = df[(df["category"] == "SELF_TRANSFER") & (df["type"] == "CREDIT")]
        tout = df[(df["category"] == "SELF_TRANSFER") & (df["type"] == "DEBIT")]
        circular = 0
        for _, r in tin.iterrows():
            window = tout[(tout["ts"] >= r["ts"]) & (tout["ts"] <= r["ts"] + pd.Timedelta(days=4))]
            if len(window) and (abs(window["amount"] - r["amount"]) / max(r["amount"], 1) < 0.06).any():
                circular += 1
        if circular >= 2:
            flags.append(("HIGH", f"{circular} round-trip transfers: credits exit to the same counterparty within days — classic income padding"))

    # 2. repeated large round-figure non-salary credits
    if rules.get("round_figure_credits", True):
        rf = credits[(credits["amount"] >= 20000) & (credits["amount"] % 5000 == 0) & (credits["category"] != "SALARY")]
        if len(rf) >= 3:
            flags.append(("MEDIUM", f"{len(rf)} large round-figure credits (₹25k/₹50k-type) outside salary — verify source"))

    # 3. salary credited on Sundays (employers don't pay on bank holidays)
    if rules.get("holiday_salary", True):
        sun = df[(df["category"] == "SALARY") & (df["ts"].dt.weekday == 6)]
        if len(sun) >= 2:
            flags.append(("HIGH", f"{len(sun)} salary credits dated on Sundays — payroll doesn't run on bank holidays; likely fabricated entries"))

    # 4. running-balance tally (tampered statement detection)
    if rules.get("balance_tally", True):
        # stable sort: statements list same-timestamp txns in ledger order - keep it
        s = df.sort_values("ts", kind="stable")
        signed = np.where(s["type"] == "CREDIT", s["amount"], -s["amount"])
        expected = s["currentBalance"].shift(1) + signed
        mism = int(((expected >= 0) & ((expected - s["currentBalance"]).abs() > 1000)).sum())
        if mism >= 2:
            flags.append(("HIGH", f"Running balance fails to tally on {mism} records — statement-tampering signature"))

    # 5. cash-deposit-heavy inflows
    if rules.get("cash_heavy", True):
        cash_cr = credits[credits["category"] == "CASH"]["amount"].sum()
        if len(credits) and credits["amount"].sum() > 0 and cash_cr / credits["amount"].sum() > 0.30:
            flags.append(("MEDIUM", "Over 30% of inflows are cash deposits — income source unverifiable from statement alone"))

    hi, med = tc.get("high_penalty", 40), tc.get("medium_penalty", 15)
    trust = max(0, 100 - sum(hi if sev == "HIGH" else med for sev, _ in flags))
    return dict(flags=flags, trust_score=trust,
                clean=not any(sev == "HIGH" for sev, _ in flags))
