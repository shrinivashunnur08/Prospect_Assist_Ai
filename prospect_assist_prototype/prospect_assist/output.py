"""
OUTPUT & AGGREGATION LAYER — decisions become bank-consumable artifacts.

Two artifact families, both format-configurable:
  lead_feed()    - the aggregated ranked call list, shaped per
                   config.yaml -> output.lead_feed_format:
                     json - REST/queue consumers
                     csv  - Phase-0 pilot (RMs work from a sheet on day one)
                     crm  - CRM-ingestible records (lead ID, owner queue,
                            disposition fields pre-added)
  build_report() - the per-lead Prospect Report: a one-page underwriting
                   handoff (income basis, obligations, trust flags, offer)
                   that pre-fills the application-stage stack (LOS/Perfios).
"""
import csv
import io
import json as _json

from .config import load_config


def _lead_row(s, product, cfg):
    return {
        "customer": s["name"],
        "mobile": s.get("contact", {}).get("mobile", "—"),
        "email": s.get("contact", {}).get("email", "—"),
        "product": cfg["policy"]["products"][product]["label"],
        "intent_score": round(s["prob"], 3),
        "assessed_income": round(s["income"]["monthly_income"]),
        "income_confidence": s["income"]["confidence"],
        "foir": round(s["features"]["foir"], 2),
        "trust_score": s["trust"]["trust_score"],
        "eligible": s["eligible"],
        "preapproved_amount": s["offer"]["amount"] if s["eligible"] else 0,
        "top_reason": s["reasons"][0][1] if s["reasons"] else "",
        "triggers": s["triggers"],
    }


def lead_feed(scores, product, cfg=None, fmt=None):
    """Aggregate ranked scores into the configured output format."""
    cfg = cfg or load_config()
    fmt = fmt or cfg["output"]["lead_feed_format"]
    ranked = sorted(scores, key=lambda s: (-s["eligible"], -s["prob"]))
    rows = [_lead_row(s, product, cfg) for s in ranked]
    for rank, r in enumerate(rows, 1):
        r["rank"] = rank
    if fmt == "json":
        return _json.dumps({"product": product, "leads": rows}, indent=2, ensure_ascii=False)
    if fmt == "csv":
        buf = io.StringIO()
        flat = [{**r, "triggers": " | ".join(r["triggers"])} for r in rows]
        w = csv.DictWriter(buf, fieldnames=list(flat[0].keys()))
        w.writeheader()
        w.writerows(flat)
        return buf.getvalue()
    if fmt == "crm":
        return _json.dumps({"source": "prospect_assist", "campaign": f"{product}_cross_sell",
                            "records": [{"lead_id": f"PA-{product.upper()}-{r['rank']:04d}",
                                         "status": "NEW", "owner_queue": "RM_DEFAULT",
                                         "disposition": None, **r} for r in rows]},
                           indent=2, ensure_ascii=False)
    raise ValueError(f"unknown lead feed format: {fmt}")


def build_report(s, product, cfg=None) -> str:
    """Perfios-style underwriting handoff pack — one lead, one page, LOS-ready."""
    cfg = cfg or load_config()
    label = cfg["policy"]["products"][product]["label"]
    foir_ceiling = cfg["policy"]["foir_ceiling"]
    inc, offer, trust = s["income"], s["offer"], s["trust"]
    contact = s.get("contact", {})
    lines = [
        f"# Prospect Report — {s['name']}",
        f"**Contact:** 📱 {contact.get('mobile', '—')} · ✉️ {contact.get('email', '—')}",
        f"**Product:** {label} · **Intent score:** {s['prob']:.2f} · "
        f"**Status:** {'ELIGIBLE' if s['eligible'] else 'DEPRIORITIZED'}",
        "",
        "## Income assessment (from AA cash-flows)",
        f"- Verified monthly income: ₹{inc['monthly_income']:,.0f} ± ₹{inc['band']:,.0f} ({inc['confidence']} confidence)",
        f"- Basis: {inc['source']}",
        f"- Naive total-credits figure: ₹{inc['naive_monthly']:,.0f}"
        + (" ⚠️ materially above verified income — padding suspected" if inc["naive_monthly"] > 1.35 * max(inc["monthly_income"], 1) else ""),
        (f"- Declared vs assessed: customer declares ₹{s['income_check']['declared']:,.0f}/month → "
         f"**{s['income_check']['verdict']}** by {abs(s['income_check']['gap']):.0%} — underwrite on the assessed figure"
         if s.get("income_check") else "- No declared-income figure on record (net-new prospect)"),
        "",
        "## Obligations & capacity",
        f"- Existing EMIs: ₹{s['obligations']['monthly_emi']:,.0f}/month"
        + (f" (lenders: {', '.join(s['obligations']['lenders'])})" if s["obligations"]["lenders"] else " (none detected)"),
        f"- FOIR: {s['features']['foir']:.0%} (ceiling {foir_ceiling:.0%})",
        f"- Rent outflow: ₹{s['obligations']['monthly_rent']:,.0f}/month",
        "",
        "## Pre-approved offer (FOIR-headroom annuity sizing)",
        f"- Up to **₹{offer['amount']:,.0f}** at {offer['rate']:.1%} for {offer['tenor_months']//12} years "
        f"(max EMI headroom ₹{offer['max_emi']:,.0f}/month)" if s["eligible"] else "- Not extended (failed capacity/trust gate)",
        "",
        "## Trust screen (Hunter-style, prospect-stage)",
        f"- Trust score: {trust['trust_score']}/100",
    ]
    lines += [f"- [{sev}] {msg}" for sev, msg in trust["flags"]]
    if not trust["flags"]:
        lines.append("- No anomalies detected")
    lines += ["", "## Right-time triggers"]
    lines += [f"- {t}" for t in s["triggers"]] if s["triggers"] else ["- None — standard outreach cadence"]
    lines += ["", "## Why this lead (reason codes)"]
    lines += [f"- ({sign}) {r}" for sign, r in s["reasons"]]
    lines += ["", "---",
              "_Generated by Prospect Assist AI from consented AA transaction data. "
              "Complements (not replaces) the application-stage Perfios FSA report, bureau pull "
              "and Hunter fraud check during formal underwriting._"]
    return "\n".join(lines)
