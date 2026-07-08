"""
Prospect Assist AI — modular scoring engine.

Layer map (one module per architectural responsibility):
    config.py       config.yaml loader — risk policy as configuration
    ingestion.py    INPUT: pluggable data sources (synthetic | DWH batch | AA FIU)
    categorizer.py  PROCESSING: narration -> category (rules | llm provider)
    income.py       PROCESSING: verified income + obligations/FOIR
    features.py     PROCESSING: 13-feature behavioral vector (model contract)
    trust.py        TRUST: Hunter-style prospect-stage anomaly screen
    modeling.py     MODELS: training pipeline, weight registry, scorers (logistic | gbm)
    policy.py       DECISION: capacity gate + offer sizing (risk-team-owned params)
    triggers.py     DECISION: right-time outreach triggers
    pipeline.py     composition root — the single scoring path
    output.py       OUTPUT: lead feed (json | csv | crm) + Prospect Report
"""
from .config import load_config, products, product_terms
from .ingestion import (DataSource, SyntheticAAFeed, CSVBatchSource, AAFIUClient,
                        get_source, generate_customer, generate_cohort, PERSONAS)
from .categorizer import to_frame, NON_INCOME_CREDITS
from .income import estimate_income, detect_obligations
from .features import FEATURES, extract_features
from .trust import trust_screen
from .modeling import train_models, train_or_load, save_bundle, load_bundle
from .policy import capacity_gate, preapproved_offer
from .triggers import detect_triggers
from .pipeline import score_customer
from .output import lead_feed, build_report
