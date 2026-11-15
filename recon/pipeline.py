"""Run the whole engine on company A (+ blind test on B) and export bundle.json for the web app."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .config import CONFIG
from .core import load, name_scores, behavior_scores, cascade, decide, annotate, backflush_check, signal_self_check
from .analysis import (calibration, band_of, stability, findings, inventory_builds, integrity, product_costs,
                       price_gaps, standardization, boot_range, reconcile_part)
from .benchmark import benchmark, stress
from .workflow import build_cards, one_pager

ROOT = Path(__file__).resolve().parent.parent


def run(company_dir=ROOT / "data/A", blind_dir=ROOT / "data/B", out=ROOT / "app/bundle.json", reference=None,
        unfamiliar_dir=ROOT / "data/C"):
    """reference: a demo bundle (dict). Used for calibration and Proof when the data has no answer key
    (i.e. a real company's upload)."""
    ds = load(company_dir)
    N = name_scores(ds)
    BEH_ind, SH, SR, RS = behavior_scores(ds)
    dec, BEH = cascade(ds, N, SH, SR, RS)
    dec = annotate(ds, dec, N, BEH_ind, SR)   # twins: can RAW demand patterns tell candidates apart?
    caps = ds.readiness["capabilities"]
    ds.readiness["behavior_signal"] = signal_self_check(ds, N, SH)
    if caps["behavior_matching"]:
        held, B = stability(ds, N, dec)
    else:
        held, B = {s: None for s in ds.skus}, 0
    has_key = ds.truth is not None
    if reference is None and not has_key:
        raise ValueError("Data without an answer key needs a reference (demo) bundle for calibration")
    cal = calibration(dec, ds.truth) if has_key else pd.DataFrame(reference["calibration"]["A"])
    name_only = decide(ds, N, BEH_ind, method="name")
    links_after = dict(zip(dec.sku, dec.part))
    links_before = dict(zip(name_only.sku, name_only.part))
    last12 = ds.months[-12:]
    spend12 = (ds.R[last12].sum(1) * ds.price)
    spend12_rng = {s: boot_range((ds.R.loc[s, last12] * ds.price[s]).values, 12) for s in ds.skus}
    part_desc = ds.parts.set_index("eng_part_id").description.to_dict()
    bf = backflush_check(ds, dec.set_index("sku").part, SR)

    calmap = {r.band: r for r in cal.itertuples()}
    # Signals disagree: clear name match but purchases do not follow the part's demand. Only meaningful
    # when the dataset's behavior signal is strong or moderate (self-check).
    builds = inventory_builds(ds, dec)
    build_skus = {b["sku"]: b for b in builds if b["explained"] >= CONFIG["build_explained"]}
    if ds.readiness["behavior_signal"]["verdict"] in ("strong", "moderate"):
        for i, r in dec.iterrows():
            if r.route == "name-led" and isinstance(r.part, str) and ds.E.loc[r.part].sum() > 0 \
                    and pd.notna(SH.loc[r.sku, r.part]) and SH.loc[r.sku, r.part] < 0.3:
                why = (f" - explained by the inventory build in {build_skus[r.sku]['month']}" if r.sku in build_skus
                       else " - check for substitutions, wrong unit of measure, or a mismatched part")
                dec.at[i, "warnings"] = r.warnings + [f"Signals disagree: strong name match, purchases do not follow demand{why}"]
                dec.at[i, "evidence_class"] = "Signals disagree"
    links = []
    for r in dec.itertuples():
        p = r.part if isinstance(r.part, str) else None
        band = band_of(r.score) if p else None
        c = calmap.get(band)
        links.append(dict(
            sku=r.sku, sku_description=ds.spend.set_index("sku").sku_description[r.sku],
            supplier=ds.spend.set_index("sku").supplier[r.sku], price=float(ds.price[r.sku]),
            part=p, part_description=part_desc.get(p, ""), state=r.state, route=r.route,
            evidence_class=r.evidence_class, name=round(r.name, 3),
            behavior=None if r.behavior is None or pd.isna(r.behavior) else round(r.behavior, 3),
            score=round(r.score, 3), band=band,
            band_precision=None if c is None or c.precision is None else [c.precision, c.lo, c.hi, c.n],
            stability=(held[r.sku] / B) if B else None, stability_pct=(held[r.sku] / B) if B else None,
            stability_txt=f"{held[r.sku]}/{B}" if B else "n/a (no purchase history)",
            warnings=r.warnings, twin_group=r.twin_group, share=r.share,
            best_name_part=r.best_name_part, best_name_score=round(float(N.loc[r.sku].max()), 3),
            spend12=float(spend12[r.sku]), spend12_rng=spend12_rng[r.sku],
            name_only_part=links_before.get(r.sku) if isinstance(links_before.get(r.sku), str) else None,
            backflush=bf.set_index("sku").label.get(r.sku)))

    F, recon = findings(ds, dec)
    integ = integrity(ds, F, builds)
    costs = product_costs(ds, links_before, links_after)
    gaps = price_gaps(ds, links_after)
    std = standardization(ds, links_after)
    cards = build_cards(links, F, gaps, std, part_desc, spend12_rng)

    attributed = {k: sum(c["attributed"][k] for c in costs) for k in ("p10", "p50", "p90")}
    linked12 = float(spend12[[s for s, p in links_after.items() if isinstance(p, str)]].sum())
    kpis = dict(total_spend_12m=integ["total_spend_12m"], coverage=(linked12 / integ["total_spend_12m"]) if integ["total_spend_12m"] else 0.0,
                gap=integ["gap"], gap_pct=integ["gap_pct"], attributed=attributed,
                n_proposed=int((dec.state == "Proposed").sum()), n_verified=int((dec.state == "Verified").sum()),
                n_unresolved=int((dec.state == "Unresolved").sum()),
                n_hidden=int((dec.evidence_class == "Hidden link").sum()),
                n_twins=int((dec.evidence_class == "Look-alike (twin)").sum()),
                explained_by_inventory=integ["explained_by_inventory"],
                backflushed=int(bf.circular.sum()), assessed=int(bf["corr"].notna().sum()))

    # reconciliation tables for all linked parts with demand (UI shows flagged first)
    rec_tables = {}
    for part, skus in (dec[dec.part.notna()].groupby("part").sku.apply(list).items() if caps["reconciliation"] else []):
        if ds.E.loc[part].sum() > 0:
            rec_tables[part] = reconcile_part(ds, part, skus).to_dict("records")

    if reference is None:   # demo build: development set A, blind B, unfamiliar C, stress grid
        sets = {"A": (ds, dec)}
        for key, d in (("B", blind_dir), ("C", unfamiliar_dir)):
            if d is not None and Path(d).exists():
                dsX = load(d)
                NX = name_scores(dsX); _, SHX, SRX, RSX = behavior_scores(dsX)
                decX, _ = cascade(dsX, NX, SHX, SRX, RSX)
                sets[key] = (dsX, decX)
        bench, cals, names, signals = {}, {}, {}, {}
        for key, (dsX, decX) in sets.items():
            NX = name_scores(dsX); _, SHX, _, _ = behavior_scores(dsX)
            signals[key] = signal_self_check(dsX, NX, SHX)
            rows, seg, hard, errs = benchmark(dsX)
            bench[key] = rows; bench["segments_" + key] = seg; bench["hard_" + key] = hard; bench["errors_" + key] = errs
            cals[key] = calibration(decX, dsX.truth).to_dict("records"); names[key] = dsX.name
        proof = dict(benchmark=bench, calibration=cals, stress=stress(ds).to_dict("records"), proof_sets=names, proof_signals=signals,
                     blind_company=names.get("B"), proof_company=ds.name, proof_source="demo datasets")
    else:
        proof = dict(benchmark=dict(reference["benchmark"]), calibration=dict(reference["calibration"]),
                     stress=reference["stress"], proof_sets=dict(reference.get("proof_sets", {"A": reference["company"]})),
                     proof_signals=dict(reference.get("proof_signals", {})),
                     blind_company=reference["blind_company"], proof_company=reference["company"],
                     proof_source="demo datasets (your upload has no answer key)")
        if has_key:   # the upload came with an answer key: score it too
            rows, seg, hard, errs = benchmark(ds)
            proof["benchmark"].update({"U": rows, "segments_U": seg, "hard_U": hard, "errors_U": errs})
            proof["calibration"]["U"] = cal.to_dict("records"); proof["proof_sets"]["U"] = ds.name + " (your upload)"
            proof["proof_signals"]["U"] = ds.readiness["behavior_signal"]
            proof["proof_source"] = "demo datasets + your upload's answer key"

    bundle = dict(
        company=ds.name, months=ds.months, config=CONFIG, **proof,
        kpis=kpis, links=links,
        products=ds.products.to_dict("records"), bom=ds.bom[["product_id", "eng_part_id", "qty_per_unit"]].to_dict("records"),
        parts=ds.parts.to_dict("records"),
        units=ds.Q.to_dict("index"), units12={p: float(ds.Q.loc[p, last12].sum()) for p in ds.Q.index},
        receipts={s: ds.R.loc[s].tolist() for s in ds.skus}, posted={s: ds.POST.loc[s].tolist() for s in ds.skus},
        qty12={s: float(ds.R.loc[s, last12].sum()) for s in ds.skus},
        expected={j: ds.E.loc[j].tolist() for j in ds.E.index},
        counts={s: ds.counts.loc[s].to_dict() for s in ds.skus},
        findings=F, reconciliation=rec_tables, inventory_builds=builds, integrity=integ,
        product_costs=costs, price_gaps=gaps, standardization=std, cards=cards,
        onepager=one_pager(ds.name, kpis, cards),
        backflush=bf.to_dict("records"),
        stability_B=B,
        planted=ds.planted.to_dict("records") if ds.planted is not None else [],
        readiness=ds.readiness,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_clean(bundle), default=_js, allow_nan=False))
    return bundle


def caps_full(ds):
    return ds.readiness["capabilities"]["behavior_matching"]


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    return o


def _js(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if o is None or (isinstance(o, float) and np.isnan(o)):
        return None
    return str(o)


if __name__ == "__main__":
    import time
    t = time.time()
    b = run()
    print(f"bundle.json written in {time.time()-t:.1f}s")
    k = b["kpis"]
    print({x: k[x] for x in ("coverage", "n_proposed", "n_unresolved", "n_hidden", "n_twins", "backflushed", "assessed")})
    print("gap", {x: round(v) for x, v in k["gap"].items()}, "attributed", {x: round(v) for x, v in k["attributed"].items()})
    print("cards", len(b["cards"]), [c["title"] for c in b["cards"][:8]])
