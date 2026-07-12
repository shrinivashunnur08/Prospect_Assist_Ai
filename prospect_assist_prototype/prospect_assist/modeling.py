"""
MODEL LAYER — training pipeline, model registry, scorers.

Configurable model family (config.yaml -> model.type):
  logistic - champion. Fully interpretable: reason codes are EXACT per-feature
             coefficient contributions, which is what a bank's model-risk
             committee (RBI model-governance expectations) can sign off.
  gbm      - challenger (sklearn GradientBoostingClassifier). Same features,
             same registry; promoted only if it beats the champion on live
             conversions (champion/challenger governance).

Registry: trained weights + metadata (model type, feature list, AUCs,
timestamp) are persisted under model.registry_dir via joblib. Serving loads
from the registry; if weights are missing or the config no longer matches the
metadata, the pipeline retrains and re-registers — so a scorer can never run
on stale or mismatched weights.
"""
import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .config import load_config, products
from .features import FEATURES, extract_features

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _make_model(cfg):
    if cfg["model"]["type"] == "gbm":
        return GradientBoostingClassifier(random_state=cfg["model"]["split_seed"])
    return LogisticRegression(max_iter=1000)


def train_models(cohort, cfg=None) -> dict:
    """Training pipeline: features -> standardize -> per-product fit -> held-out AUC."""
    cfg = cfg or load_config()
    prods = products(cfg)
    rows, labels = [], {p: [] for p in prods}
    for c in cohort:
        f, *_ = extract_features(c, cfg)
        rows.append([f[k] for k in FEATURES])
        for p in prods:
            labels[p].append(c["label_" + p])
    X = np.array(rows)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xn = (X - mu) / sd
    models, aucs = {}, {}
    for name in prods:
        y = np.array(labels[name])
        Xtr, Xte, ytr, yte = train_test_split(
            Xn, y, test_size=cfg["model"]["test_size"],
            random_state=cfg["model"]["split_seed"], stratify=y)
        m = _make_model(cfg).fit(Xtr, ytr)
        aucs[name] = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
        models[name] = m
    return dict(models=models, mu=mu, sd=sd, aucs=aucs,
                model_type=cfg["model"]["type"], features=list(FEATURES),
                trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))


# ---------------------------- model registry ------------------------------
def _registry_paths(cfg):
    d = os.path.join(_ROOT, cfg["model"]["registry_dir"])
    return d, os.path.join(d, "bundle.joblib"), os.path.join(d, "metadata.json")


def save_bundle(bundle, cfg=None):
    cfg = cfg or load_config()
    d, weights_path, meta_path = _registry_paths(cfg)
    os.makedirs(d, exist_ok=True)
    joblib.dump(bundle, weights_path)
    meta = {k: bundle[k] for k in ("model_type", "features", "trained_at")}
    meta["aucs"] = {k: round(v, 4) for k, v in bundle["aucs"].items()}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return weights_path


def load_bundle(cfg=None):
    """Load registered weights if they exist AND match the active config."""
    cfg = cfg or load_config()
    _, weights_path, meta_path = _registry_paths(cfg)
    if not (os.path.exists(weights_path) and os.path.exists(meta_path)):
        return None
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("model_type") != cfg["model"]["type"] or meta.get("features") != list(FEATURES):
        return None  # config drifted from registered weights -> force retrain
    return joblib.load(weights_path)


def train_or_load(cohort, cfg=None, force_retrain=False):
    """Serving entry point: registry hit -> load; miss/mismatch -> train + register."""
    cfg = cfg or load_config()
    if not force_retrain:
        bundle = load_bundle(cfg)
        if bundle is not None:
            bundle["from_registry"] = True
            return bundle
    bundle = train_models(cohort, cfg)
    save_bundle(bundle, cfg)
    bundle["from_registry"] = False
    return bundle


def reason_contributions(bundle, product, xn):
    """Per-feature contribution vector used for plain-English reason codes.
    Logistic: exact coefficient contributions. GBM: global importances signed
    by the standardized feature deviation (documented approximation)."""
    m = bundle["models"][product]
    if hasattr(m, "coef_"):
        return xn[0] * m.coef_[0]
    return xn[0] * m.feature_importances_
