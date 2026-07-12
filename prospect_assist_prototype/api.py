"""
Prospect Assist AI — REST scoring service (FastAPI).

The integration surface for the bank's systems: the SAME pipeline the
dashboard uses, exposed as versioned endpoints. Interactive OpenAPI docs are
auto-generated at /docs — that page IS the integration contract.

Run:  uvicorn api:app --port 8000

Production notes: this container deploys inside the bank perimeter (on-prem
VPC or the bank's AWS account — ECS/Fargate); TLS terminates at the bank's
gateway; authn/authz via the bank's IAM (OAuth2 client-credentials between
systems); every request is audit-logged with a correlation ID.
"""
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from prospect_assist import (load_config, products, get_source, train_or_load,
                             score_customer, lead_feed, build_report)

cfg = load_config()
app = FastAPI(
    title="Prospect Assist AI — Scoring API",
    version="1.0.0",
    description="Prospect-stage intent + capacity + trust scoring for retail lending. "
                "Consumes ReBIT FI-Deposit JSON (Account Aggregator canonical schema).",
)

# serving bootstrap: registry hit -> load weights; miss -> train once + register
_cohort = get_source(cfg).fetch_cohort()
_bundle = train_or_load(_cohort, cfg)


class CustomerEnvelope(BaseModel):
    """ReBIT FI-Deposit-shaped customer record (see /schema for a sample)."""
    customerName: str
    account: dict
    persona: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok", "model_type": _bundle["model_type"],
            "weights_from_registry": _bundle.get("from_registry", False),
            "trained_at": _bundle["trained_at"]}


@app.get("/config")
def active_policy():
    """The risk policy currently in force (config.yaml, sans internals)."""
    return {"policy": cfg["policy"], "trust": cfg["trust"], "model": cfg["model"]["type"],
            "output_format": cfg["output"]["lead_feed_format"]}


@app.get("/models")
def model_card():
    """Model registry metadata — what's serving, how good, trained when."""
    return {"model_type": _bundle["model_type"], "features": _bundle["features"],
            "aucs": {k: round(v, 3) for k, v in _bundle["aucs"].items()},
            "trained_at": _bundle["trained_at"], "products": products(cfg)}


@app.post("/score")
def score(customer: CustomerEnvelope, product: str = Query("home", enum=products(cfg))):
    """Score ONE customer envelope for one product. This is the endpoint a CRM
    or LOS calls in real time (e.g. when a customer uses the EMI calculator)."""
    s = score_customer(customer.model_dump(), _bundle, product=product, cfg=cfg)
    s.pop("df")  # internal frame, not part of the contract
    return s


@app.get("/leads")
def leads(product: str = Query("home", enum=products(cfg)),
          fmt: Optional[str] = Query(None, enum=["json", "csv", "crm"])):
    """The aggregated ranked lead feed for a product, shaped per the configured
    output format (or ?fmt= override). This is the nightly-batch integration."""
    scores = [score_customer(c, _bundle, product=product, cfg=cfg) for c in _cohort[:80]]
    body = lead_feed(scores, product, cfg, fmt=fmt)
    fmt = fmt or cfg["output"]["lead_feed_format"]
    if fmt == "csv":
        return PlainTextResponse(body, media_type="text/csv")
    return PlainTextResponse(body, media_type="application/json")


@app.get("/report/{customer_name}")
def report(customer_name: str, product: str = Query("home", enum=products(cfg))):
    """Prospect Report (markdown) for one lead — the LOS/underwriting handoff."""
    match = next((c for c in _cohort if c["customerName"] == customer_name), None)
    if match is None:
        raise HTTPException(404, f"customer '{customer_name}' not in cohort")
    s = score_customer(match, _bundle, product=product, cfg=cfg)
    return PlainTextResponse(build_report(s, product, cfg), media_type="text/markdown")
