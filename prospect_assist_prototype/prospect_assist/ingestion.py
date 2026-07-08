"""
INPUT LAYER — pluggable data sources behind one interface.

Every source emits the same canonical shape: ReBIT FI-Deposit JSON
(Account -> Transactions -> Transaction records). The engine downstream is
therefore source-agnostic: the synthetic feed used for the demo, a batch drop
from the bank's DWH, or a licensed AA (FIU) pipe are interchangeable.

Sources:
  SyntheticAAFeed  - demo generator (default). Six personas covering all four
                     PS loan products + one fraud persona for the trust screen.
  CSVBatchSource   - Phase-0 production path: normalized JSON drops extracted
                     read-only from the bank's DWH/CBS.
  AAFIUClient      - Phase-2 production path: consented cross-bank pulls via a
                     licensed Account Aggregator (Setu / Finvu / OneMoney).
"""
import glob
import json
import os
import random
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

EMPLOYERS = ["INFOSYS LTD", "TCS LTD", "WIPRO LTD", "HDFC LIFE", "ACCENTURE", "RELIANCE RETAIL"]
LENDERS = ["HDFCBANK", "BAJAJFIN", "ICICIBANK", "AXISBANK", "TATACAP"]
UPI_MERCHANTS = ["SWIGGY", "ZOMATO", "AMAZON", "FLIPKART", "BIGBASKET", "JIOMART", "MYNTRA"]
GIG_PLATFORMS = ["UBER INDIA", "ZOMATO PARTNER", "URBANCLAP", "SWIGGY DELIVERY"]
CAB_NARRATIONS = ["OLACABS", "UBER TRIP", "RAPIDO RIDE"]
FUEL_STATIONS = ["HPCL PETROL", "INDIANOIL FUEL", "SHELL PETROLPUMP", "IOCL FUELSTN"]
PROPERTY_TAX = ["BBMP-PROPERTYTAX", "MCGM-PROPERTY TAX", "GHMC-PROPERTYTAX"]

PERSONAS = {
    "salaried_regular": dict(
        salary=(70000, 110000), salary_jitter=0.02, rent=(22000, 35000),
        emis=1, upi_freq=(18, 30), savings_bias=0.25, sip=True,
        cab_rides=(8, 16),                       # commutes by cab -> auto-loan signal
    ),
    "self_employed": dict(
        salary=None, biz_credits=(8, 18), biz_amount=(15000, 90000), rent=None,
        emis=1, upi_freq=(12, 22), savings_bias=0.15, sip=False,
        property_tax=True, biz_volatile=True,    # owns property + lumpy cash -> LAP signal
        fuel_visits=(2, 5),
    ),
    "gig_worker": dict(
        salary=None, gig_credits=(14, 26), gig_amount=(800, 4500), rent=(8000, 15000),
        emis=0, upi_freq=(20, 40), savings_bias=0.05, sip=False,
        fuel_visits=(6, 12),                     # runs own two-wheeler
    ),
    "over_leveraged": dict(
        salary=(55000, 85000), salary_jitter=0.02, rent=(18000, 26000),
        emis=4, upi_freq=(15, 25), savings_bias=-0.05, sip=False,
        cab_rides=(2, 6),                        # wants money, can't afford it
    ),
    "dormant": dict(
        salary=(40000, 60000), salary_jitter=0.03, rent=None,
        emis=0, upi_freq=(2, 6), savings_bias=0.4, sip=False,
    ),
    "income_padder": dict(
        salary=(28000, 40000), salary_jitter=0.02, rent=(10000, 16000),
        emis=1, upi_freq=(10, 20), savings_bias=0.0, sip=False,
        pad_credits=True, sunday_salary=True, tamper_balance=True,
    ),
}

FIRST = ["Aarav", "Vivaan", "Diya", "Ananya", "Kabir", "Ishaan", "Meera", "Rohan", "Sneha", "Arjun",
         "Priya", "Karan", "Nisha", "Aditya", "Pooja", "Rahul", "Tanvi", "Dev", "Riya", "Manish"]
LAST = ["Sharma", "Verma", "Iyer", "Reddy", "Khan", "Patel", "Nair", "Gupta", "Das", "Singh"]


def _txn(ts, amount, ttype, narration, balance, mode="OTHERS"):
    return {
        "txnId": uuid.uuid4().hex[:12].upper(),
        "type": ttype,                      # CREDIT / DEBIT
        "mode": mode,                       # UPI / ATM / OTHERS / FT
        "amount": round(amount, 2),
        "currentBalance": round(max(balance, 0), 2),
        "transactionTimestamp": ts.strftime("%Y-%m-%dT%H:%M:%S+05:30"),
        "valueDate": ts.strftime("%Y-%m-%d"),
        "narration": narration,
        "reference": uuid.uuid4().hex[:10].upper(),
    }


def generate_customer(persona_key, months=9, seed=None, name=None):
    rng = random.Random(seed)
    p = PERSONAS[persona_key]
    name = name or f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    end = datetime(2026, 6, 30)
    start = end - timedelta(days=30 * months)
    balance = rng.uniform(30000, 150000)
    txns = []

    salary_amt = rng.uniform(*p["salary"]) if p.get("salary") else None
    rent_amt = rng.uniform(*p["rent"]) if p.get("rent") else None
    # life events that make "right-time triggers" real: increments, rent hikes, windfalls
    long_enough = months >= 5
    hike_month = rng.randint(months // 2 + 1, months - 1) if (salary_amt and long_enough and rng.random() < 0.30) else None
    hike_factor = rng.uniform(1.08, 1.18)
    rent_hike_month = rng.randint(months // 2 + 1, months - 1) if (rent_amt and long_enough and rng.random() < 0.25) else None
    rent_hike_factor = rng.uniform(1.08, 1.15)
    windfall_month = months - rng.randint(1, 2) if (salary_amt and long_enough and rng.random() < 0.15) else None
    employer = rng.choice(EMPLOYERS)
    my_lenders = rng.sample(LENDERS, k=p.get("emis", 0)) if p.get("emis") else []
    emi_amts = {l: rng.uniform(6000, 18000) for l in my_lenders}
    savings_growth = p["savings_bias"]
    pad_partner = f"{rng.choice(FIRST).upper()} {rng.choice(LAST).upper()}"

    month_cursor = start.replace(day=1)
    month_no = 0
    while month_cursor < end:
        mdays = 28
        month_no += 1
        # salary credit (1st +/- 1 day; fraud persona lands some on Sundays)
        if salary_amt:
            if hike_month and month_no == hike_month:
                salary_amt *= hike_factor          # annual increment lands
            amt = salary_amt * rng.uniform(1 - p["salary_jitter"], 1 + p["salary_jitter"])
            ts = month_cursor.replace(day=1) + timedelta(days=rng.randint(0, 1), hours=10)
            if p.get("sunday_salary") and rng.random() < 0.5:
                while ts.weekday() != 6:   # fabricated entry: payroll "runs" on a Sunday
                    ts += timedelta(days=1)
            else:
                while ts.weekday() == 6:   # honest payroll shifts off bank holidays
                    ts += timedelta(days=1)
            balance += amt
            txns.append(_txn(ts, amt, "CREDIT", f"NEFT-SAL-{employer}-{ts.strftime('%b%y').upper()}", balance, "FT"))
        # business credits (volatile personas get lean months -> cash-crunch/LAP signal)
        if p.get("biz_credits"):
            lean = p.get("biz_volatile") and (month_no % 3 == 0)
            lo, hi = p["biz_credits"]
            n_credits = rng.randint(max(2, int(lo * 0.35)), max(3, int(hi * 0.4))) if lean else rng.randint(lo, hi)
            for _ in range(n_credits):
                amt = rng.uniform(*p["biz_amount"])
                ts = month_cursor + timedelta(days=rng.randint(0, mdays), hours=rng.randint(9, 20))
                balance += amt
                txns.append(_txn(ts, amt, "CREDIT", f"UPI-{rng.choice(['PAYMENT FROM CLIENT','INV-'+str(rng.randint(1000,9999)),'GPAY BUSINESS'])}", balance, "UPI"))
        # gig credits
        if p.get("gig_credits"):
            for _ in range(rng.randint(*p["gig_credits"])):
                amt = rng.uniform(*p["gig_amount"])
                ts = month_cursor + timedelta(days=rng.randint(0, mdays), hours=rng.randint(8, 23))
                balance += amt
                txns.append(_txn(ts, amt, "CREDIT", f"IMPS-{rng.choice(GIG_PLATFORMS)}-PAYOUT", balance))
        # circular round-figure credits that exit within days (income padding / fraud)
        if p.get("pad_credits"):
            for _ in range(rng.randint(2, 3)):
                amt = rng.choice([25000.0, 30000.0, 50000.0])
                day = rng.randint(3, 20)
                ts_in = month_cursor + timedelta(days=day, hours=rng.randint(9, 18))
                balance += amt
                txns.append(_txn(ts_in, amt, "CREDIT", f"IMPS-TRANSFER FROM {pad_partner}", balance))
                ts_out = ts_in + timedelta(days=rng.randint(1, 2), hours=3)
                out_amt = amt * rng.uniform(0.96, 1.0)
                balance -= out_amt
                txns.append(_txn(ts_out, out_amt, "DEBIT", f"IMPS-TRANSFER TO {pad_partner}", balance))
        # windfall credit (FD maturity / bonus payout) in a recent month
        if windfall_month and month_no == windfall_month:
            amt = salary_amt * rng.uniform(3.5, 6.0)
            ts = month_cursor + timedelta(days=rng.randint(5, 20), hours=11)
            balance += amt
            txns.append(_txn(ts, amt, "CREDIT", "NEFT-FD MATURITY PROCEEDS", balance, "FT"))
        # rent
        if rent_amt:
            if rent_hike_month and month_no == rent_hike_month:
                rent_amt *= rent_hike_factor       # landlord raises rent
            ts = month_cursor.replace(day=min(3 + rng.randint(0, 2), 28), hour=11)
            balance -= rent_amt
            txns.append(_txn(ts, rent_amt, "DEBIT", f"UPI-RENT-{rng.choice(['HOUSEOWNER','LANDLORD','PG RENT'])}-{ts.strftime('%b').upper()}", balance, "UPI"))
        # EMIs
        for lender, amt in emi_amts.items():
            ts = month_cursor.replace(day=min(5 + rng.randint(0, 2), 28), hour=6)
            balance -= amt
            txns.append(_txn(ts, amt, "DEBIT", f"ACH-EMI-{lender}-{rng.randint(10000,99999)}", balance))
        # SIP
        if p.get("sip"):
            ts = month_cursor.replace(day=min(7, 28), hour=8)
            amt = rng.uniform(5000, 15000)
            balance -= amt
            txns.append(_txn(ts, amt, "DEBIT", "ACH-SIP-GROWWMF-MONTHLY", balance))
        # cab / commute spends (auto-loan intent signal)
        if p.get("cab_rides"):
            for _ in range(rng.randint(*p["cab_rides"])):
                amt = rng.uniform(200, 800)
                ts = month_cursor + timedelta(days=rng.randint(0, mdays), hours=rng.randint(7, 22))
                balance -= amt
                txns.append(_txn(ts, amt, "DEBIT", f"UPI-{rng.choice(CAB_NARRATIONS)}-{uuid.uuid4().hex[:6].upper()}", balance, "UPI"))
        # fuel spends (already owns a vehicle)
        if p.get("fuel_visits"):
            for _ in range(rng.randint(*p["fuel_visits"])):
                amt = rng.uniform(300, 2200)
                ts = month_cursor + timedelta(days=rng.randint(0, mdays), hours=rng.randint(7, 21))
                balance -= amt
                txns.append(_txn(ts, amt, "DEBIT", f"UPI-{rng.choice(FUEL_STATIONS)}", balance, "UPI"))
        # property tax (ownership signal -> LAP eligibility)
        if p.get("property_tax") and month_no in (2, 8):
            amt = rng.uniform(4000, 15000)
            ts = month_cursor + timedelta(days=rng.randint(5, 20), hours=13)
            balance -= amt
            txns.append(_txn(ts, amt, "DEBIT", f"BBPS-{rng.choice(PROPERTY_TAX)}", balance, "UPI"))
        # UPI spends
        for _ in range(rng.randint(*p["upi_freq"])):
            amt = rng.uniform(80, 3500)
            ts = month_cursor + timedelta(days=rng.randint(0, mdays), hours=rng.randint(8, 23))
            balance -= amt
            txns.append(_txn(ts, amt, "DEBIT", f"UPI-{rng.choice(UPI_MERCHANTS)}-{uuid.uuid4().hex[:6].upper()}", balance, "UPI"))
        # utilities
        for util in ["ELECTRICITYBILL", "AIRTEL POSTPAID", "JIO RECHARGE"]:
            if rng.random() < 0.8:
                amt = rng.uniform(300, 2500)
                ts = month_cursor + timedelta(days=rng.randint(5, 20), hours=12)
                balance -= amt
                txns.append(_txn(ts, amt, "DEBIT", f"BBPS-{util}", balance, "UPI"))
        # monthly savings interest keeps balance trend consistent with persona
        if savings_growth > 0:
            amt = max(balance, 10000) * savings_growth / 12 * rng.uniform(0.6, 1.4)
            ts = month_cursor.replace(day=28, hour=23)
            balance += amt
            txns.append(_txn(ts, amt, "CREDIT", "INT.PD-SAVINGS-QTRLY", balance))
        month_cursor = (month_cursor + timedelta(days=32)).replace(day=1)

    txns.sort(key=lambda t: t["transactionTimestamp"])
    # recompute running balances chronologically so every honest statement tallies
    # exactly (credit adds, debit subtracts) - the trust screen depends on this
    bal = rng.uniform(30000, 150000)
    for t in txns:
        bal = bal + t["amount"] if t["type"] == "CREDIT" else max(bal - t["amount"], 0.0)
        t["currentBalance"] = round(bal, 2)
    # tampered statement: running balance doesn't tally on a few records (fraud persona)
    if p.get("tamper_balance"):
        for idx in rng.sample(range(len(txns)), k=min(4, len(txns))):
            txns[idx]["currentBalance"] = round(txns[idx]["currentBalance"] + rng.uniform(8000, 30000), 2)
    # ReBIT FI-Deposit-shaped envelope
    return {
        "customerName": name,
        "persona": persona_key,
        "account": {
            "linkedAccRef": uuid.uuid4().hex[:16].upper(),
            "maskedAccNumber": "XXXXXXXX" + str(random.randint(1000, 9999)),
            "type": "deposit",
            "fiType": "DEPOSIT",
            "profile": {"holders": {"holder": [{"name": name, "ckycCompliance": True}]}},
            "summary": {"currentBalance": txns[-1]["currentBalance"] if txns else 0,
                        "currency": "INR", "type": "SAVINGS"},
            "transactions": {"startDate": start.strftime("%Y-%m-%d"),
                             "endDate": end.strftime("%Y-%m-%d"),
                             "transaction": txns},
        },
    }


def _sigmoid(z):
    import math
    return 1.0 / (1.0 + math.exp(-z))


def generate_cohort(n=200, seed=42):
    """Cohort of existing bank customers (Tier 1) with intent labels for model training.
    Labels simulate observed outcomes ("took this loan within 6 months") and are driven by
    the customer's actual behavioral signals + noise - as real-world intent is."""
    from .features import extract_features  # local import: features doesn't import ingestion
    rng = random.Random(seed)
    keys = list(PERSONAS.keys())
    weights = [0.30, 0.18, 0.14, 0.14, 0.14, 0.10]
    cohort = []
    for i in range(n):
        k = rng.choices(keys, weights)[0]
        uname = f"{rng.choice(FIRST)} {chr(65 + i % 26)}. {rng.choice(LAST)}"
        cust = generate_customer(k, months=9, seed=seed + i, name=uname)
        f, *_ = extract_features(cust)
        z_home = (-5.0 + 4.0 * f["rent_ratio"] + 1.6 * f["savings_rate"]
                  + 1.2 * f["income_stability"] + 0.9 * min(f["income"] / 100000, 1.5)
                  - 2.5 * f["foir"] + 0.8 * f["balance_trend"] + rng.gauss(0, 0.7))
        z_pl = (-5.1 + 2.5 * f["upi_intensity"] + 2.4 * max(-f["savings_rate"], 0)
                + 1.4 * (f["n_emis"] > 0) - 1.5 * f["income_stability"]
                + 1.0 * max(-f["balance_trend"], 0)
                + 0.8 * min(f["income"] / 80000, 1.0) + rng.gauss(0, 0.55))
        z_auto = (-4.1 + 22.0 * f["commute_ratio"] - 10.0 * f["fuel_ratio"]
                  + 1.1 * f["income_stability"] + 0.9 * min(f["income"] / 80000, 1.3)
                  - 2.2 * f["foir"] + 0.6 * f["savings_rate"] + rng.gauss(0, 0.55))
        z_lap = (-5.2 + 2.6 * f["property_owner"] + 1.6 * f["cash_crunch"]
                 + 1.0 * (1 - f["income_stability"]) + 1.1 * min(f["income"] / 100000, 1.5)
                 - 1.8 * f["foir"] + rng.gauss(0, 0.7))
        cust["label_home"] = int(rng.random() < _sigmoid(z_home))
        cust["label_pl"] = int(rng.random() < _sigmoid(z_pl))
        cust["label_auto"] = int(rng.random() < _sigmoid(z_auto))
        cust["label_lap"] = int(rng.random() < _sigmoid(z_lap))
        cohort.append(cust)
    return cohort


# ---------------------------------------------------------------------------
# Data-source interface: demo feed, DWH batch drop, and AA client are
# interchangeable because all emit the same canonical ReBIT shape.
# ---------------------------------------------------------------------------
class DataSource(ABC):
    """One customer record = one ReBIT FI-Deposit JSON envelope."""

    @abstractmethod
    def fetch_cohort(self) -> list:
        ...


class SyntheticAAFeed(DataSource):
    """Demo source. Schema-faithful synthetic cohort with training labels."""

    def __init__(self, cfg):
        ds = cfg["data_source"]
        self.n, self.seed = ds.get("cohort_size", 240), ds.get("seed", 42)

    def fetch_cohort(self):
        return generate_cohort(n=self.n, seed=self.seed)


class CSVBatchSource(DataSource):
    """Phase-0 production source: normalized JSON drops from the bank DWH.
    Read-only extract, no core-banking writes - the lowest-friction pilot path."""

    def __init__(self, cfg):
        self.path = cfg["data_source"].get("path", "./batch_drop")

    def fetch_cohort(self):
        files = sorted(glob.glob(os.path.join(self.path, "*.json")))
        if not files:
            raise FileNotFoundError(
                f"No normalized customer JSON found in {self.path}. "
                "Expected one ReBIT FI-Deposit envelope per file (see AA Schema tab).")
        return [json.load(open(f, encoding="utf-8")) for f in files]


class AAFIUClient(DataSource):
    """Phase-2 production source: licensed Account Aggregator pull (consented,
    cross-bank). Wire the bank's FIU credentials + TSP SDK (Setu/Finvu/OneMoney)
    here; the response is already in the canonical shape this engine consumes."""

    def __init__(self, cfg):
        self.cfg = cfg

    def fetch_cohort(self):
        raise NotImplementedError(
            "Requires the bank's FIU license + TSP credentials. "
            "The engine consumes the AA response unchanged - this class is the only code that changes.")


_PROVIDERS = {"synthetic": SyntheticAAFeed, "csv_batch": CSVBatchSource, "aa_fiu": AAFIUClient}


def get_source(cfg) -> DataSource:
    provider = cfg["data_source"]["provider"]
    return _PROVIDERS[provider](cfg)
