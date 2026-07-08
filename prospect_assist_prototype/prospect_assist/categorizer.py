"""
DATA PROCESSING — transaction categorization.

Provider is configurable (config.yaml -> categorizer.provider):
  rules  - ordered regex rules, first match wins (demo + production baseline;
           deterministic, auditable, zero cost)
  llm    - Phase-2 long-tail narration understanding (regional languages,
           merchant aliases) via an on-perimeter model (e.g. AWS Bedrock in the
           bank's VPC) with the rules engine as guaranteed fallback.
Category rules live in config so the bank can extend them without a deployment.
"""
import re

import pandas as pd

from .config import load_config

# credits that must NOT count as income (self-transfers, interest sweeps)
NON_INCOME_CREDITS = {"SELF_TRANSFER", "INTEREST"}


def _compiled_rules(cfg):
    return [(name, re.compile(pat, re.I)) for name, pat in cfg["categorizer"]["rules"]]


def to_frame(customer, cfg=None) -> pd.DataFrame:
    """ReBIT envelope -> categorized transaction DataFrame."""
    cfg = cfg or load_config()
    rules = _compiled_rules(cfg)
    txns = customer["account"]["transactions"]["transaction"]
    df = pd.DataFrame(txns)
    df["ts"] = pd.to_datetime(df["transactionTimestamp"].str[:19])
    df["month"] = df["ts"].dt.to_period("M")

    def cat(narration, ttype):
        for name, pat in rules:
            if pat.search(narration):
                return name
        return "OTHER_CREDIT" if ttype == "CREDIT" else "OTHER_DEBIT"

    df["category"] = [cat(n, t) for n, t in zip(df["narration"], df["type"])]
    return df
