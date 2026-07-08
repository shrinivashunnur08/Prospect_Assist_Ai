
# Prospect Assist AI — Team Fintech Frontiers (IDBI Innovate · PS2)

Prospect-stage lead intelligence for retail lending, covering **all four PS loan
products** — Personal, Home, Auto, Mortgage/LAP. Ranks the bank's existing customers
by **behavioral intent**, assesses **actual income + FOIR** from AA-schema transaction
data, screens every statement through a **Hunter-style trust check**, sizes a
**pre-approved offer**, and hands each lead to the RM/LOS with plain-English reasons.

Data is synthetic but **ReBIT FI-Deposit schema-faithful**: swap the feed for a
licensed AA (FIU) pipe and the engine runs unchanged.

## Architecture = code layout (one module per responsibility)

```
config.yaml                 ← risk policy AS CONFIG: gates, product terms, trust rules,
                              model choice, data source, output format (risk-team owned)
prospect_assist/
├── config.py               config loader
├── ingestion.py            INPUT — pluggable sources: SyntheticAAFeed (demo) ·
│                           CSVBatchSource (Phase-0 DWH drop) · AAFIUClient (Phase-2)
├── categorizer.py          PROCESSING — narration → category (rules | llm provider)
├── income.py               PROCESSING — verified income ± confidence · obligations/FOIR
├── features.py             PROCESSING — 13-feature vector (the model contract)
├── trust.py                TRUST — 5 Hunter-style prospect-stage anomaly rules
├── modeling.py             MODELS — training pipeline · registry (weights+metadata) ·
│                           logistic champion | GBM challenger via config
├── policy.py               DECISION — capacity gate + FOIR-headroom offer sizing
├── triggers.py             DECISION — right-time outreach triggers
├── pipeline.py             composition root — the single scoring path
└── output.py               OUTPUT — lead feed (json|csv|crm per config) · Prospect Report
app.py                      RM dashboard (Streamlit) — storytelling demo over the pipeline
api.py                      REST scoring service (FastAPI) — the integration surface
verify_metrics.py           reproduces every deck/README number in one command
```

## Run

```bash
pip install -r requirements.txt
streamlit run app.py                 # RM dashboard (demo link)
uvicorn api:app --port 8000          # REST API → interactive contract at /docs
python verify_metrics.py             # reproduce all benchmark numbers
```

First boot trains the four models (~seconds) and **registers weights + metadata**
under `models/`; subsequent boots (and the API) load from the registry.

## Integration surface (how a bank consumes this)

| Endpoint | Consumer | Use |
|---|---|---|
| `POST /score?product=` | CRM / channels | real-time score of one ReBIT envelope |
| `GET /leads?product=&fmt=json\|csv\|crm` | campaign/CRM batch | ranked lead feed, format per config |
| `GET /report/{customer}` | LOS | Prospect Report — underwriting handoff |
| `GET /models` · `/config` · `/health` | ops / model risk | model card, active policy, serving status |

Read-only and upstream: no core-banking writes, ever. Deploys as one container inside
the bank perimeter (on-prem VPC or the bank's AWS account: S3/Glue → SageMaker →
ECS/Fargate → EventBridge for trigger-driven re-scoring). TLS at gateway, OAuth2
service auth, every score audit-loggable; no customer data to external LLMs.

## Deploy the demo link (free, ~5 minutes)
1. Push this folder to a **public GitHub repo** (also your GitHub submission link).
2. share.streamlit.io → "New app" → pick the repo → main file: `app.py` → Deploy.
3. Copy the live URL into the deck, slide 13.

## Honest benchmark (reproduce: `python verify_metrics.py`)
- AUC — Home **0.96** · Auto **0.77** · Mortgage-LAP **0.75** · Personal **0.71**
  (home intent is the most behaviorally legible; PL intent is diffuse — consistent with industry experience)
- Top-ranked call list converts at **31–46%** across products vs **8–30%** spray-and-pray
  baseline — every product clears the PS's 30% bar on this cohort; a branch-level A/B
  pilot is the honest production test
- Trust screen: **22/22** income-padders caught, **0/218** honest customers false-flagged
- Capacity gate: **36/36** over-leveraged customers auto-rejected
