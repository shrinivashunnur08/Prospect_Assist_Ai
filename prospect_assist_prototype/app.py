"""Prospect Assist AI — RM-facing demo. Team Fintech Frontiers · IDBI Innovate PS2.

Thin presentation layer: every score on screen comes from the same
prospect_assist.pipeline the REST API serves — one scoring path to audit.
"""
import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from prospect_assist import (load_config, products, get_source, train_or_load,
                             score_customer, generate_customer, lead_feed, build_report,
                             PERSONAS)

st.set_page_config(page_title="Prospect Assist AI", page_icon="🎯", layout="wide")

cfg = load_config()
PRODUCTS = products(cfg)
PRODUCT_LABEL = {k: v["label"] for k, v in cfg["policy"]["products"].items()}
FOIR_CEILING = cfg["policy"]["foir_ceiling"]
MIN_INCOME = cfg["policy"]["min_income"]

GLOSSARY = {
    "Intent score": "Model probability (0–100%) that this customer takes THIS loan in the next ~6 months, "
                    "read from behavior: rent, savings build-up, spend patterns. Higher = more likely to say yes.",
    "Assessed income": "₹/month VERIFIED from actual bank credits (recurring salary / business inflows) — "
                       "not what the customer declares on a form. Self-transfers are excluded.",
    "Confidence": "How reliable the income figure is. HIGH = tight recurring salary. "
                  "LOW = irregular inflows — verify before lending.",
    "FOIR": "Fixed Obligation to Income Ratio — the share of monthly income ALREADY going out as loan EMIs. "
            f"Bank policy here: must stay under {FOIR_CEILING:.0%}. Above that, a new loan is imprudent "
            "no matter how interested the customer is.",
    "Trust score": "0–100 statement-authenticity score from the fraud screen (circular transfers, tampered "
                   "balances, salary dated on bank holidays). Anything flagged HIGH fails the gate.",
    "Pre-approved": f"The largest loan the customer can service while keeping FOIR under {FOIR_CEILING:.0%}, "
                    "at the product's rate and tenor. Computed, not promised.",
    "Decision": "✅ CALL — eligible & ranked · 🚫 SKIP — fails capacity policy · 🛡️ REVIEW — fraud anomalies.",
}


@st.cache_resource(show_spinner="Loading model registry (training on first boot)…")
def load_models():
    cohort = get_source(cfg).fetch_cohort()
    bundle = train_or_load(cohort, cfg)
    return cohort, bundle


cohort, bundle = load_models()
st.session_state.setdefault("contact_log", [])

st.title("🎯 Prospect Assist AI")
st.caption(
    "Cross-sell intelligence for the bank's existing customer base (Tier 1 — no new consent needed). "
    "Data is synthetic but AA-schema-faithful (ReBIT FI-Deposit): swap in a licensed AA pipe and this runs unchanged."
)

# ---------------- sidebar: product + live config surface ----------------
product = st.sidebar.radio("Loan product", PRODUCTS, format_func=lambda k: PRODUCT_LABEL[k])

st.sidebar.markdown("---")
st.sidebar.markdown("**⚙️ Active policy — `config.yaml`**")
st.sidebar.caption(
    f"Risk team owns this file; no deployment needed.\n\n"
    f"- FOIR ceiling: **{FOIR_CEILING:.0%}** · income floor: **₹{MIN_INCOME:,}**\n"
    f"- {PRODUCT_LABEL[product]}: **{cfg['policy']['products'][product]['rate']:.1%}** · "
    f"{cfg['policy']['products'][product]['tenor_months']//12} yrs · "
    f"cap ₹{cfg['policy']['products'][product]['cap']:,}\n"
    f"- Trust screen: **{'ON' if cfg['trust']['enabled'] else 'OFF'}** "
    f"({sum(cfg['trust']['rules'].values())}/5 rules active)\n"
    f"- Model: **{bundle['model_type']}** "
    f"({'loaded from registry' if bundle.get('from_registry') else 'trained + registered this boot'})\n"
    f"- Lead feed format: **{cfg['output']['lead_feed_format'].upper()}**"
)
st.sidebar.markdown("---")
st.sidebar.caption("🔌 **API-ready:** the same pipeline serves REST — "
                   "`uvicorn api:app` → interactive contract at `/docs`.")


# ---------------- score everyone ----------------
@st.cache_data(show_spinner="Scoring customer base…")
def score_all(prod):
    rows = []
    for i, c in enumerate(cohort[:80]):  # demo subset
        s = score_customer(c, bundle, product=prod, cfg=cfg)
        rows.append(dict(idx=i, Name=s["name"], Segment=s["persona"].replace("_", " ").title(),
                         IntentScore=round(s["prob"], 3),
                         AssessedIncome=round(s["income"]["monthly_income"]),
                         Confidence=s["income"]["confidence"],
                         FOIR=round(s["features"]["foir"], 2),
                         Trust=s["trust"]["trust_score"],
                         PreApproved=s["offer"]["amount"] if s["eligible"] else 0,
                         Status="✅" if s["eligible"] else ("🛡️" if not s["trust"]["clean"] else "🚫"),
                         TopReason=s["reasons"][0][1] if s["reasons"] else ""))
    df = pd.DataFrame(rows)
    df["Rank"] = (df["Status"].eq("✅") * df["IntentScore"]).rank(ascending=False).astype(int)
    return df.sort_values("Rank")


leads = score_all(product)
contacted_names = {e["name"] for e in st.session_state.contact_log if e["product"] == product}
base_rate = float(np.mean([cohort[int(i)]["label_" + product] for i in leads["idx"]]))
elig_ranked = leads[leads.Status == "✅"].sort_values("IntentScore", ascending=False)
top_n = max(len(elig_ranked) // 5, 1)
top_rate = float(np.mean([cohort[int(i)]["label_" + product] for i in elig_ranked.head(top_n)["idx"]])) if len(elig_ranked) else 0.0


def decision_banner(s, rank=None, total=None):
    """The one thing an RM needs: call, skip, or escalate — and why, in plain words."""
    pct_better_than = float((leads["IntentScore"] < s["prob"]).mean())
    if s["eligible"]:
        rank_txt = f"**Rank #{rank} of {total}** · " if rank else ""
        st.success(
            f"### 📞 CALL THIS CUSTOMER\n"
            f"{rank_txt}Intent **{s['prob']:.0%}** — more likely to convert than **{pct_better_than:.0%}** of the base · "
            f"FOIR **{s['features']['foir']:.0%}** (policy limit {FOIR_CEILING:.0%} — safe headroom) · "
            f"Trust **{s['trust']['trust_score']}/100** (statement clean)\n\n"
            f"**Open with:** pre-approved {PRODUCT_LABEL[product]} up to **₹{s['offer']['amount']:,.0f}** "
            f"at {s['offer']['rate']:.1%} for {s['offer']['tenor_months']//12} years"
            + (f" · **This week because:** {s['triggers'][0]}" if s["triggers"] else "")
        )
    elif not s["trust"]["clean"]:
        st.error(
            f"### 🛡️ DO NOT CALL — ESCALATE TO FRAUD REVIEW\n"
            f"Trust score **{s['trust']['trust_score']}/100**: the statement shows "
            f"{len(s['trust']['flags'])} anomaly signature(s) — details below. Intent may look high "
            f"(**{s['prob']:.0%}**) but the income behind it is not real. Route to verification, not sales."
        )
    else:
        why = (f"FOIR **{s['features']['foir']:.0%}** exceeds the {FOIR_CEILING:.0%} policy limit"
               if s["features"]["foir"] >= FOIR_CEILING
               else f"verified income **₹{s['income']['monthly_income']:,.0f}/month** is below the ₹{MIN_INCOME:,} floor")
        st.error(
            f"### 🚫 DO NOT CALL — FAILS LENDING POLICY\n"
            f"{why}. The customer may want the loan (intent {s['prob']:.0%}) but cannot safely service it — "
            f"calling wastes RM time and risks a bad loan. System deprioritized automatically."
        )


# Navigation: a keyed radio (NOT st.tabs) so the active page survives the rerun
# every button/radio interaction triggers — essential for the worklist flow.
PAGES = ["🎬 Start Here", "📋 Ranked Leads", "🔍 Customer Deep-Dive", "🧪 Try Your Own Data",
         "💰 Business Impact", "📈 Model Validation", "🧾 AA Schema & API"]
page = st.radio("nav", PAGES, horizontal=True, label_visibility="collapsed", key="nav")
st.markdown("---")

# ================= STORY =================
if page == "🎬 Start Here":
    left, right = st.columns([3, 2])
    with left:
        st.subheader("The problem, in one sentence")
        st.markdown(
            "Retail lending outreach is **spray-and-pray**: campaigns built on age/salary bands "
            "convert in single digits because they measure neither **intent** nor **true capacity** "
            "— and every declined call costs RM time and customer goodwill."
        )
        st.subheader("What this engine does differently")
        st.markdown(
            f"""
1. **Reads behavior, not demographics.** Every transaction is categorized; income is
   **verified from cash-flows** (not declared), obligations become a live **FOIR**.
2. **Scores intent per product** — {', '.join(PRODUCT_LABEL.values())} — with
   plain-English reason codes an RM can actually dial with.
3. **Gates before it recommends.** A capacity gate (FOIR < {FOIR_CEILING:.0%}) and a
   **Hunter-style trust screen** (income padding, tampered statements) run at *prospect*
   stage — high intent alone is never a lead.
4. **Sizes the offer and times the call** — FOIR-headroom pre-approval + triggers like
   a salary hike or rent increase say *call this week, not this quarter*.
"""
        )
    with right:
        st.subheader("This cohort, this product — live")
        m1, m2 = st.columns(2)
        m1.metric("Spray-and-pray conversion", f"{base_rate:.0%}", help="If RMs called everyone in the base")
        m2.metric(f"Our top-{top_n} call list", f"{top_rate:.0%}",
                  f"{(top_rate / max(base_rate, 1e-9)):.1f}× lift")
        m3, m4 = st.columns(2)
        m3.metric("Fraud caught by trust screen", f"{int((leads.Status == '🛡️').sum())} customers")
        m4.metric("Over-leveraged auto-rejected", f"{int((leads.Status == '🚫').sum())} customers")
        st.info(
            "**90-second tour** → ① *Ranked Leads*: who to call, in order. "
            "② *Deep-Dive*: use the **worklist** — call verdicts, mark contacted, next lead. "
            "Try **“a fraud catch”**: ₹1L/month of fake income collapses to ₹35k verified. "
            "③ *Try Your Own Data*: generate a sample file, upload it, watch it get scored. "
            "④ *Business Impact*: the >30% conversion math."
        )

# ================= RANKED LEADS =================
if page == "📋 Ranked Leads":
    st.subheader(f"Top prospects — {PRODUCT_LABEL[product]}")
    with st.expander("📖 New here? The 30-second read on every column"):
        for term, desc in GLOSSARY.items():
            st.markdown(f"**{term}** — {desc}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers scored", len(leads))
    c2.metric("Call-ready (✅)", int((leads.Status == "✅").sum()),
              help="Pass BOTH gates: capacity (FOIR/income) and trust (fraud screen)")
    c3.metric("Capacity-gated 🚫", int((leads.Status == "🚫").sum()),
              help=f"FOIR ≥ {FOIR_CEILING:.0%} or income < ₹{MIN_INCOME:,} — cannot safely service a new EMI")
    c4.metric("Trust-flagged 🛡️", int((leads.Status == "🛡️").sum()),
              help="Statement anomalies (padding/tampering) — routed to fraud review, not sales")
    show = leads.copy()
    show["Contacted"] = show["Name"].map(lambda n: "☎️" if n in contacted_names else "")
    show["PreApproved"] = show["PreApproved"].map(lambda v: f"₹{v:,.0f}" if v else "—")
    show["IntentScore"] = (show["IntentScore"] * 100).round(0).astype(int)
    show["FOIR"] = (show["FOIR"] * 100).round(0).astype(int)
    st.dataframe(
        show[["Rank", "Name", "Segment", "IntentScore", "AssessedIncome", "Confidence",
              "FOIR", "Trust", "PreApproved", "Status", "Contacted", "TopReason"]],
        use_container_width=True, hide_index=True, height=430,
        column_config={
            "IntentScore": st.column_config.ProgressColumn("Intent %", min_value=0, max_value=100,
                                                           format="%d%%", help=GLOSSARY["Intent score"]),
            "AssessedIncome": st.column_config.NumberColumn("Income ₹/mo", format="₹%d",
                                                            help=GLOSSARY["Assessed income"]),
            "Confidence": st.column_config.TextColumn("Income conf.", help=GLOSSARY["Confidence"]),
            "FOIR": st.column_config.NumberColumn("FOIR %", format="%d%%", help=GLOSSARY["FOIR"]),
            "Trust": st.column_config.ProgressColumn("Trust", min_value=0, max_value=100, format="%d",
                                                     help=GLOSSARY["Trust score"]),
            "PreApproved": st.column_config.TextColumn("Pre-approved", help=GLOSSARY["Pre-approved"]),
            "Status": st.column_config.TextColumn("Decision", help=GLOSSARY["Decision"]),
            "Contacted": st.column_config.TextColumn("☎️", help="Marked contacted in this session (see Deep-Dive worklist)"),
            "TopReason": st.column_config.TextColumn("Why (top reason)"),
        },
    )
    st.info(
        "🚫 = **capacity gate** (over-leveraged / low income) · 🛡️ = **trust screen** "
        "(income-padding or tampering — the checks Perfios/Hunter run at application stage, "
        "applied here at prospect stage). Both auto-deprioritized: rejecting high-intent but "
        "unsafe customers is the point."
    )
    scored_all = [score_customer(cohort[int(i)], bundle, product=product, cfg=cfg) for i in leads["idx"]]
    fmt = cfg["output"]["lead_feed_format"]
    st.download_button(
        f"⬇️ Export ranked lead feed ({fmt.upper()} — configured in config.yaml; CRM/batch-ready)",
        lead_feed(scored_all, product, cfg),
        file_name=f"lead_feed_{product}.{'csv' if fmt == 'csv' else 'json'}",
        mime="text/csv" if fmt == "csv" else "application/json",
    )


def render_customer(s, rank=None, total=None, key_prefix="dd"):
    """Full decision view for one scored customer — identical fields to the Ranked Leads table."""
    decision_banner(s, rank, total)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Intent score", f"{s['prob']:.0%}", help=GLOSSARY["Intent score"])
    m2.metric("Assessed income", f"₹{s['income']['monthly_income']:,.0f}",
              s["income"]["confidence"] + " confidence", help=GLOSSARY["Assessed income"])
    m3.metric("FOIR", f"{s['features']['foir']:.0%}",
              f"{(s['features']['foir'] - FOIR_CEILING):+.0%} vs {FOIR_CEILING:.0%} limit",
              delta_color="inverse", help=GLOSSARY["FOIR"])
    m4.metric("Trust score", f"{s['trust']['trust_score']}/100", help=GLOSSARY["Trust score"])
    m5.metric("Pre-approved", f"₹{s['offer']['amount']:,.0f}" if s["eligible"] else "—",
              help=GLOSSARY["Pre-approved"])

    a, b = st.columns([1, 1])
    with a:
        naive = s["income"]["naive_monthly"]
        if naive > 1.35 * max(s["income"]["monthly_income"], 1):
            st.warning(f"⚠️ Naive statement total shows ₹{naive:,.0f}/month in credits — "
                       f"₹{naive - s['income']['monthly_income']:,.0f} is NOT verifiable income "
                       "(self-transfers / padding excluded by the engine).")
        if s["obligations"]["lenders"]:
            st.write(f"**Existing EMIs:** ₹{s['obligations']['monthly_emi']:,.0f}/month to "
                     + ", ".join(s["obligations"]["lenders"]))
        st.markdown("**Why this lead (reason codes):**")
        for sign, r in s["reasons"]:
            st.markdown(f"- ({sign}) {r}")
        st.caption(f"Income basis: {s['income']['source']}")
        t = s["trust"]
        if t["flags"]:
            st.error(f"🛡️ Trust screen — {len(t['flags'])} anomaly(ies):")
            for sev, msg in t["flags"]:
                st.markdown(f"- **{sev}** — {msg}")
        else:
            st.success("🛡️ Trust screen — statement internally consistent, no anomalies.")
        if s["triggers"]:
            st.markdown("**⏰ Right-time triggers — why call this week:**")
            for trg in s["triggers"]:
                st.info(trg)
        st.download_button(
            "⬇️ Download Prospect Report (underwriting handoff)",
            build_report(s, product, cfg),
            file_name=f"prospect_report_{s['name'].replace(' ', '_')}.md",
            mime="text/markdown", key=f"{key_prefix}_report_{s['name']}",
        )
    with b:
        df = s["df"]
        cat = df[df.type == "DEBIT"].groupby("category")["amount"].sum().reset_index()
        st.plotly_chart(px.pie(cat, names="category", values="amount", title="Spend mix (from AA transactions)"),
                        use_container_width=True, key=f"{key_prefix}_pie_{s['name']}")
        monthly = df[df.type == "CREDIT"].groupby(df["month"].astype(str))["amount"].sum().reset_index()
        st.plotly_chart(px.bar(monthly, x="month", y="amount", title="Monthly inflows — income pattern"),
                        use_container_width=True, key=f"{key_prefix}_bar_{s['name']}")


# ================= DEEP-DIVE =================
if page == "🔍 Customer Deep-Dive":
    st.markdown("**Show me:**")
    pick_mode = st.radio(
        "quick pick",
        ["📇 Let me choose", "☎️ Worklist — next lead to call", "🕵️ A fraud catch", "⛔ An over-leveraged reject"],
        index=0, horizontal=True, label_visibility="collapsed",
    )
    names = leads["Name"].tolist()
    worklist = elig_ranked[~elig_ranked["Name"].isin(contacted_names)]
    if pick_mode == "📇 Let me choose":
        pick = st.selectbox("Choose a customer", names)
    elif pick_mode == "☎️ Worklist — next lead to call":
        done = len(elig_ranked) - len(worklist)
        st.progress(done / max(len(elig_ranked), 1),
                    text=f"Worklist: {done} of {len(elig_ranked)} call-ready leads contacted this session")
        if len(worklist):
            pick = worklist.iloc[0]["Name"]
            st.caption(f"Next up: **{pick}**")
        else:
            st.success("🎉 Worklist complete — every call-ready lead has been contacted this session.")
            pick = elig_ranked.iloc[0]["Name"] if len(elig_ranked) else names[0]
    elif pick_mode == "🕵️ A fraud catch" and (leads.Status == "🛡️").any():
        pick = leads[leads.Status == "🛡️"].iloc[0]["Name"]
        st.caption(f"Showing **{pick}**")
    elif pick_mode == "⛔ An over-leveraged reject" and (leads.Status == "🚫").any():
        pick = leads[leads.Status == "🚫"].iloc[0]["Name"]
        st.caption(f"Showing **{pick}**")
    else:
        pick = names[0]

    row = leads[leads.Name == pick].iloc[0]
    cust = cohort[int(row["idx"])]
    s = score_customer(cust, bundle, product=product, cfg=cfg)
    render_customer(s, rank=int(row["Rank"]), total=len(leads))

    # ---- call workflow: disposition buttons + session log ----
    if s["eligible"]:
        if pick in contacted_names:
            st.caption("☎️ Already marked contacted this session.")
        else:
            st.markdown("**Log the call outcome (moves worklist to the next lead):**")
            b1, b2, b3, _ = st.columns([1, 1, 1, 2])
            for col, label, disp in ((b1, "✅ Interested", "INTERESTED"),
                                     (b2, "❌ Not interested", "NOT_INTERESTED"),
                                     (b3, "📵 No answer", "NO_ANSWER")):
                if col.button(label, key=f"disp_{disp}_{pick}"):
                    st.session_state.contact_log.append(dict(
                        name=pick, product=product, disposition=disp,
                        time=datetime.now().strftime("%H:%M:%S"),
                        intent=f"{s['prob']:.0%}", offer=s["offer"]["amount"]))
                    st.rerun()
    with st.expander(f"📒 Contact log — {len(st.session_state.contact_log)} call(s) this session "
                     "(feeds the retraining loop in production)"):
        if st.session_state.contact_log:
            st.dataframe(pd.DataFrame(st.session_state.contact_log), hide_index=True, use_container_width=True)
            if st.button("🗑️ Clear log"):
                st.session_state.contact_log = []
                st.rerun()
        else:
            st.caption("No calls logged yet. Use the worklist above — each disposition you log is exactly "
                       "the conversion feedback that retrains the model monthly in production.")

# ================= TRY YOUR OWN DATA =================
if page == "🧪 Try Your Own Data":
    st.subheader("Not canned data — score any customer file in the AA (ReBIT) format")
    gen_col, up_col = st.columns([1, 1])
    with gen_col:
        st.markdown("**① Generate a sample customer file**")
        st.caption("Pick a profile, download the JSON — this is exactly the shape a licensed AA pipe delivers.")
        persona = st.selectbox("Profile", list(PERSONAS.keys()),
                               format_func=lambda k: k.replace("_", " ").title())
        seed = st.number_input("Random seed (any number → a different customer)", 1, 99999, 101)
        sample = generate_customer(persona, months=9, seed=int(seed))
        st.download_button("⬇️ Download sample customer JSON",
                           json.dumps(sample, indent=2),
                           file_name=f"sample_{persona}_{seed}.json", mime="application/json")
        st.caption(f"{len(sample['account']['transactions']['transaction'])} transactions over 9 months.")
    with up_col:
        st.markdown("**② Upload customer file(s) → instant decision**")
        uploads = st.file_uploader("ReBIT FI-Deposit JSON (the file from step ① works)",
                                   type="json", accept_multiple_files=True)
        if uploads:
            st.caption(f"Scoring {len(uploads)} file(s) for **{PRODUCT_LABEL[product]}** with the registered model…")
    if uploads:
        st.markdown("---")
        for up in uploads:
            try:
                cust_up = json.loads(up.getvalue().decode("utf-8"))
                s_up = score_customer(cust_up, bundle, product=product, cfg=cfg)
                st.markdown(f"#### 📄 {up.name} → {s_up['name']}")
                render_customer(s_up, key_prefix=f"up_{up.name}")
            except Exception as e:
                st.error(f"**{up.name}** could not be scored: {e}. "
                         "Expected a ReBIT FI-Deposit envelope — generate one on the left to see the shape.")
    else:
        st.info("This tab is the 'it's not hard-coded' proof: generate a file with any seed, upload it back "
                "(or hand-edit amounts first), and the same pipeline that serves the API scores it live.")

# ================= BUSINESS IMPACT =================
if page == "💰 Business Impact":
    st.subheader("What this does to the funnel — same RM effort, better targets")
    c1, c2, c3 = st.columns(3)
    c1.metric("Spray-and-pray conversion (call everyone)", f"{base_rate:.0%}")
    c2.metric(f"Prospect Assist top-{top_n} call list", f"{top_rate:.0%}",
              f"{(top_rate / max(base_rate, 1e-9)):.1f}× lift")
    c3.metric("Calls saved", f"{(1 - top_n / len(leads)):.0%}",
              "of the base never needs a cold call")
    st.markdown(
        f"""
**The >30% conversion path, honestly framed.** On this cohort, calling everyone converts at
**{base_rate:.0%}**; calling only the top-ranked eligible list converts at **{top_rate:.0%}**.
The mechanism — rank by intent, gate by capacity and trust, call the top — is what carries to
production. We propose a **branch-level A/B pilot** (RM team A gets our call list, team B gets
the usual campaign list) as the honest production test of the 30% target.

**Where the money is:** an RM makes ~25 calls/day. Redirecting one month of one RM's calls from
random outreach to this list at the lift above means the same salary cost produces
**{(top_rate / max(base_rate, 1e-9)):.1f}× the sanctioned loans** — before counting the
underwriting time saved by pre-assessed income and the losses avoided by the trust screen.
"""
    )

# ================= VALIDATION =================
if page == "📈 Model Validation":
    st.subheader("Honest validation on held-out synthetic cohort")
    cols = st.columns(len(PRODUCTS))
    for col, p in zip(cols, PRODUCTS):
        col.metric(f"AUC — {PRODUCT_LABEL[p]}", f"{bundle['aucs'][p]:.2f}",
                   help="Area Under the ROC Curve on customers the model never saw in training. "
                        "1.0 = perfect ranking, 0.5 = coin toss. 0.7+ is deployable for lead ranking.")
    probs = leads.sort_values("IntentScore", ascending=False)
    labels = np.array([cohort[int(i)]["label_" + product] for i in probs["idx"]])
    deciles = np.array_split(labels, 10)
    base = labels.mean()
    lift = [d.mean() / max(base, 1e-9) for d in deciles]
    fig = go.Figure(go.Bar(x=[f"D{i+1}" for i in range(10)], y=lift))
    fig.add_hline(y=1.0, line_dash="dash", annotation_text="baseline (random outreach)")
    fig.update_layout(title=f"Lift by score decile — {PRODUCT_LABEL[product]} (D1 = highest scores)",
                      yaxis_title="conversion lift ×", xaxis_title="score decile")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "We do not claim '30% conversion' from a lab. We show top-decile lift over baseline and propose a "
        "branch-level A/B pilot as the production test. Methodology over magic numbers. "
        f"Model card: {bundle['model_type']} · trained {bundle['trained_at']} · weights in model registry."
    )

# ================= SCHEMA & API =================
if page == "🧾 AA Schema & API":
    left, right = st.columns([1, 1])
    with left:
        st.subheader("The exact shape a licensed AA pipe delivers (ReBIT FI-Deposit)")
        sample = generate_customer("salaried_regular", months=2, seed=7)
        sample["account"]["transactions"]["transaction"] = sample["account"]["transactions"]["transaction"][:5]
        st.json(sample, expanded=False)
        st.caption("Production swap: replace the synthetic feed with the bank's FIU AA client "
                   "(`ingestion.AAFIUClient`) — zero engine changes.")
    with right:
        st.subheader("Integration surface — same pipeline over REST")
        st.code(
            "uvicorn api:app --port 8000   # interactive contract at /docs\n\n"
            "GET  /health                → serving status + registry info\n"
            "GET  /config                → risk policy in force (config.yaml)\n"
            "GET  /models                → model card: type, features, AUCs\n"
            "POST /score?product=home    → score ONE ReBIT envelope (CRM real-time)\n"
            "GET  /leads?product=home    → ranked feed (json | csv | crm)\n"
            "GET  /report/{customer}     → Prospect Report (LOS handoff)",
            language="text",
        )
        st.caption(
            "Deploys as one container inside the bank perimeter (on-prem VPC or bank AWS "
            "account). TLS at the gateway, OAuth2 service auth, every request audit-logged. "
            "Bank systems integrate against /docs — no shared database, no core-banking writes."
        )
