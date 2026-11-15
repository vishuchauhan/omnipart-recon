"""Resolution Board cards, department routing, audience-specific message drafts, client one-pager.
Language rules: 'proposed'/'verified'/'unresolved'; 'spend to investigate'; findings are 'investigation leads'."""
from __future__ import annotations
import numpy as np


def money(r):
    f = lambda v: f"${v/1000:,.1f}K" if abs(v) >= 1000 else f"${v:,.0f}"
    return f"{f(r['p10'])}-{f(r['p90'])}"


ROUTING = {
    "VERIFY_LINK": ["Engineering", "Procurement"], "CONSUMPTION_SHIFT": ["Engineering"],
    "PERSISTENT_EXCESS": ["Plant / Quality"], "PRICE_GAP": ["Procurement"],
    "NO_ACTIVE_DEMAND": ["Procurement", "Service"], "UNRESOLVED_SPEND": ["Procurement"],
    "STANDARDIZATION": ["Engineering", "Procurement"],
}


def build_cards(links, findings, price_gaps, std, part_desc, spend12_rng):
    cards, n = [], 0
    rng_seed = np.random.default_rng(5)

    def add(**k):
        nonlocal n
        n += 1
        conf = k.pop("confidence")
        k.update(id=f"F-{n:03d}", owners=ROUTING[k["type"]], state="Proposed", confidence=conf,
                 priority=k["rng"]["p50"] * (1 - conf), age_days=int(rng_seed.integers(1, 12)))
        cards.append(k)

    for l in links:
        if l["evidence_class"] in ("Hidden link", "Look-alike (twin)") or l["warnings"]:
            why = {"Hidden link": "Name evidence is weak; purchase pattern fits this part's expected demand",
                   "Look-alike (twin)": "Demand pattern is shared with another part; name/spec broke the tie"}.get(
                l["evidence_class"], "; ".join(l["warnings"]))
            add(type="VERIFY_LINK", title=f"Verify {l['sku']} -> {l['part']}",
                subtitle=f"\"{l['sku_description']}\" -> {part_desc.get(l['part'], '')}", sku=l["sku"], part=l["part"],
                rng=spend12_rng[l["sku"]], confidence=l["stability"] if l["stability"] is not None else 0.5,
                next_action="Engineering confirms spec/drawing; Procurement updates cross-reference",
                evidence=[why, f"Name {l['name']:.2f} | behavior {l['behavior'] if l['behavior'] is not None else 'n/a'}",
                          (f"Held in {l['stability_pct']:.0%} of bootstrap resamples" if l["stability_pct"] is not None
                           else "Stability not assessed (no purchase history)")])
    for f in findings:
        conf = 0.9 if f["type"] in ("CONSUMPTION_SHIFT", "PERSISTENT_EXCESS") else 0.7
        title = {"CONSUMPTION_SHIFT": f"Usage shift on {f['part']}", "PERSISTENT_EXCESS": f"Excess usage on {f['part']}",
                 "NO_ACTIVE_DEMAND": f"Purchasing {f['skus'][0]} with no BOM demand",
                 "UNRESOLVED_SPEND": f"Unresolved spend: {f['skus'][0]}"}[f["type"]]
        ev = [f["metric"]]
        for im in f.get("implied") or []:
            ev.append(f"{im['product']}: implied {im['implied']:.2f}/unit (90% CI {im['lo']:.2f}-{im['hi']:.2f}) vs BOM {im['bom']:.0f}")
        if f.get("book_minus_shelf"):
            ev.append(f"ERP book stock exceeds shelf count by {f['book_minus_shelf']:,.0f} units")
        add(type=f["type"], title=title, subtitle=part_desc.get(f["part"], "") if f["part"] else "", sku=", ".join(f["skus"]),
            part=f["part"], rng=f["rng"], confidence=conf, next_action=f["action"], evidence=ev)
    for g in price_gaps:
        add(type="PRICE_GAP", title=f"Price gap on {g['part']}", subtitle=part_desc.get(g["part"], ""), sku=", ".join(g["skus"]),
            part=g["part"], rng=g["rng"], confidence=0.8,
            next_action="Benchmark against lowest price; renegotiate or rebalance volume (benchmark, not guaranteed)",
            evidence=[f"{s}: ${p:.3f} x {q:,} units" for s, p, q in zip(g["suppliers"], g["prices"], g["qty_12m"])])
    for c in std["candidates"]:
        ns = c["net_saving"]
        if ns is None:
            continue
        add(type="STANDARDIZATION", title=f"Standardize {c['merge_from']} into {c['merge_into']}",
            subtitle=f"{c['from_desc']} -> {c['into_desc']}", sku="", part=c["merge_into"],
            rng=dict(p10=ns["low"], p50=ns["base"], p90=ns["high"]), confidence=0.5,
            next_action=("Engineering confirms compatibility for core platform" if ns["base"] > 0 else
                         "Net saving is negative at base assumptions - do not standardize unless other benefits apply"),
            evidence=[f"Spec similarity {c['similarity']:.2f}", f"Used in {', '.join(c['products_from'])} / {', '.join(c['products_into'])}"])
    for c in cards:
        c["messages"] = messages(c)
    return sorted(cards, key=lambda c: -c["priority"])


def messages(c):
    r, ev = money(c["rng"]), "\n".join(f"- {e}" for e in c["evidence"])
    t = c["title"]
    return {
        "Engineer": f"Subject: {t} - engineering check needed\n\nHi,\n\nOmniPart flagged an investigation lead that needs an engineering view:\n{c['subtitle']}\n\nEvidence:\n{ev}\n\nRequested action: {c['next_action']}\nCould you confirm or reject in the Resolution Board by Friday? A rejection with a reason is just as useful - it teaches the model.\n\nThanks",
        "Buyer": f"Subject: {t} - procurement follow-up\n\nHi,\n\nSpend involved (12-month range): {r}.\n{c['subtitle']}\n\nEvidence:\n{ev}\n\nNext step: {c['next_action']}\nNote: links are proposed until engineering verifies them; please do not change supplier records before then.",
        "Plant manager": f"Subject: {t}\n\nHi,\n\nPhysical counts point to an investigation lead on your lines (not a confirmed root cause):\n{ev}\n\nCould your team check scrap/rework reporting for this part? Spend involved: {r}.",
        "Executive": f"{t}: {r} of spend to investigate (12-month range). Confidence {c['confidence']:.0%}. Owner: {', '.join(c['owners'])}. Next step: {c['next_action']}.",
    }


def one_pager(company, kpis, cards):
    top = cards[:5]
    lines = [f"OMNIPART RECON - WEEKLY STEERING UPDATE - {company} (synthetic demo data)", "",
             "HEADLINES",
             f"- Proposed linkage coverage: {kpis['coverage']:.1%} of 12-month spend ({kpis['n_verified']} links verified so far)",
             f"- Spend newly attributed to BOMs: {money(kpis['attributed'])} (now visible in product cost; not a saving)",
             f"- Spend Integrity Gap: {money(kpis['gap'])} to investigate ({kpis['gap_pct']['p10']:.1%}-{kpis['gap_pct']['p90']:.1%} of spend)",
             f"- Links: {kpis['n_verified']} verified, {kpis['n_proposed']} proposed, {kpis['n_unresolved']} unresolved",
             "", "TOP 5 DECISIONS NEEDED THIS WEEK"]
    for i, c in enumerate(top, 1):
        lines.append(f"{i}. {c['title']} - {money(c['rng'])} - owner: {', '.join(c['owners'])} - {c['next_action']}")
    lines += ["", "MONEY LADDER: every dollar above is INDICATIVE. It becomes qualified when an owner reviews it, approved when "
              "signed off, and realized only after the action is executed and measured.",
              "All figures are ranges (P10-P90). Findings are investigation leads, not confirmed causes. "
              "Price benchmarks use the lowest observed price and are not guaranteed."]
    return "\n".join(lines)
