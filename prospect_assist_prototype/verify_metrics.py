"""
Reproduces every number printed on deck slide 11, the README benchmark section,
and the tracker — from scratch. Run whenever engine/data_gen change, and keep
deck + README + tracker in sync with what this prints.

    python verify_metrics.py            # metrics only
    python verify_metrics.py --chart    # also regenerate the slide-11 lift chart PNG

Provenance of the slide-11 chart: train cohort n=240 seed=42, held-out validation
cohort n=600 seed=777, ranked in operational workboard order (eligible-first,
then intent score).
"""
import sys
import numpy as np
from collections import Counter

from prospect_assist import (generate_cohort, train_models, score_customer,
                             trust_screen, to_frame, load_config, products)

PRODUCTS = products(load_config())

TRAIN_N, TRAIN_SEED = 240, 42
VAL_N, VAL_SEED = 600, 777


def main():
    cohort = generate_cohort(n=TRAIN_N, seed=TRAIN_SEED)
    bundle = train_models(cohort)

    print("=== AUCs (held-out 30% split of train cohort) — deck slide 11, README ===")
    print({k: round(v, 2) for k, v in bundle["aucs"].items()})

    print("\n=== Persona mix / label prevalence ===")
    print(Counter(c["persona"] for c in cohort))
    for p in PRODUCTS:
        ys = [c["label_" + p] for c in cohort]
        print(f"  {p}: positives {sum(ys)}/{len(ys)}")

    print("\n=== Trust screen — deck slide 11 ('22/22 caught, 0/218 false-flagged') ===")
    padders = padders_caught = honest = honest_flagged = 0
    for c in cohort:
        t = trust_screen(to_frame(c))
        if c["persona"] == "income_padder":
            padders += 1
            padders_caught += 0 if t["clean"] else 1
        else:
            honest += 1
            honest_flagged += 0 if t["clean"] else 1
    print(f"  padders caught {padders_caught}/{padders} · honest false-flagged {honest_flagged}/{honest}")

    print("\n=== Capacity gate — deck slide 11 ('36/36 over-leveraged rejected') ===")
    ol = [c for c in cohort if c["persona"] == "over_leveraged"]
    rej = sum(1 for c in ol if not score_customer(c, bundle, product="pl")["capacity_ok"])
    print(f"  over-leveraged capacity-rejected: {rej}/{len(ol)}")

    print("\n=== Per-product funnel (demo subset of 80) — README, Business Impact tab ===")
    for p in PRODUCTS:
        scores = [score_customer(c, bundle, product=p) for c in cohort[:80]]
        name_to_label = {c["customerName"]: c["label_" + p] for c in cohort[:80]}
        ranked = sorted(scores, key=lambda s: -s["prob"])
        labels = np.array([name_to_label[s["name"]] for s in ranked])
        base = labels.mean()
        elig = [s for s in ranked if s["eligible"]]
        top_n = max(len(elig) // 5, 1)
        top_rate = np.mean([name_to_label[s["name"]] for s in elig[:top_n]]) if elig else 0.0
        print(f"  {p}: base {base:.0%} → top-{top_n} call list {top_rate:.0%} "
              f"({top_rate / max(base, 1e-9):.1f}x) · eligible {sum(s['eligible'] for s in scores)}/80 "
              f"· trust-flagged {sum(not s['trust']['clean'] for s in scores)}")

    print("\n=== Slide-11 lift chart data (held-out n=600 seed=777, workboard ranking) ===")
    val = generate_cohort(n=VAL_N, seed=VAL_SEED)
    rows = []
    for c in val:
        s = score_customer(c, bundle, product="home")
        rows.append((s["eligible"], s["prob"], c["label_home"]))
    rows.sort(key=lambda t: (-t[0], -t[1]))
    labels = np.array([l for _, _, l in rows])
    deciles = np.array_split(labels, 10)
    base = labels.mean()
    lift = [d.mean() / base for d in deciles]
    print("  home lift D1..D10:", [round(x, 1) for x in lift])

    if "--chart" in sys.argv:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
        colors = ["#8B1A1A"] + ["#C97B7B"] * 9
        bars = ax.bar([f"D{i+1}" for i in range(10)], lift, color=colors)
        for b, v in zip(bars, lift):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.1f}x", ha="center", fontsize=9)
        ax.axhline(1.0, ls="--", c="gray", lw=1)
        ax.text(9.4, 1.06, "random outreach", ha="right", fontsize=9, color="gray")
        ax.set_ylabel("conversion lift ×")
        ax.set_title(f"Home-loan conversion lift by workboard rank decile (held-out cohort, n={VAL_N})",
                     fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig("lift_home.png", facecolor="white")
        print("  chart written to lift_home.png (swap into deck slide 11 if numbers changed)")


if __name__ == "__main__":
    main()
