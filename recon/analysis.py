"""
Confidence (Wilson calibration, block-bootstrap stability, implied usage CIs, money ranges),
count-to-count reconciliation, findings, Spend Integrity (mutually exclusive buckets),
true product cost, price harmonization, Track 1 standardization.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .config import CONFIG
from .core import Dataset, behavior_scores, cascade, windows

RNG = lambda: np.random.default_rng(CONFIG["seed"])


# ---------------------------------------------------------------- confidence
def wilson(k: int, n: int, z: float = 1.96):
    """Wilson 95% interval for a proportion k/n -> (p, lo, hi)."""
    if n == 0:
        return (None, None, None)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (p, max(0.0, c - h), min(1.0, c + h))


BANDS = [(0.0, 0.6, "< 0.60"), (0.6, 0.8, "0.60-0.80"), (0.8, 0.9, "0.80-0.90"), (0.9, 1.01, ">= 0.90")]


def calibration(dec: pd.DataFrame, truth: pd.Series) -> pd.DataFrame:
    """Precision of proposed links per score band, with Wilson intervals (needs a labeled set)."""
    d = dec[dec.part.notna()].copy()
    d["ok"] = [truth[s] == p for s, p in zip(d.sku, d.part)]
    rows = []
    for lo, hi, lab in BANDS:
        g = d[(d.score >= lo) & (d.score < hi)]
        p, a, b = wilson(int(g.ok.sum()), len(g))
        rows.append(dict(band=lab, n=len(g), correct=int(g.ok.sum()), precision=p, lo=a, hi=b))
    return pd.DataFrame(rows)


def band_of(score: float):
    for lo, hi, lab in BANDS:
        if lo <= score < hi:
            return lab
    return BANDS[-1][2]


def stability(ds: Dataset, NAME, dec: pd.DataFrame, cfg=CONFIG, B=None):
    """Block bootstrap over months: resample 3-month blocks of the time axis, recompute signals and the
    full cascade; count how often each SKU keeps the same decision."""
    B = B or cfg["bootstrap_B"]
    rng = RNG(); T = len(ds.months); blk = cfg["block"]
    base = dict(zip(dec.sku, dec.part))
    held = {s: 0 for s in ds.skus}
    starts = np.arange(0, T - blk + 1)
    for _ in range(B):
        idx = np.concatenate([np.arange(st, st + blk) for st in rng.choice(starts, T // blk)])
        sub = Dataset(ds.name, [ds.months[i] for i in idx], ds.products, ds.parts, ds.bom, ds.spend,
                      ds.Q.iloc[:, idx], ds.B, ds.E.iloc[:, idx], ds.R.iloc[:, idx], ds.POST.iloc[:, idx],
                      ds.counts, ds.lead)
        _, SH, SR, RS = behavior_scores(sub, cfg=cfg)
        d2, _ = cascade(sub, NAME, SH, SR, RS, cfg)
        for s, p in zip(d2.sku, d2.part):
            same = (p == base[s]) or (not isinstance(p, str) and not isinstance(base[s], str))
            held[s] += int(same)
    return held, B


def boot_range(values, horizon: int, B: int = 2000, scale: float = 1.0):
    """P10/P50/P90 of the total over `horizon` periods, resampling observed periods with replacement."""
    v = np.asarray(values, float)
    if len(v) == 0 or np.all(v == 0):
        return dict(p10=0.0, p50=0.0, p90=0.0)
    rng = RNG()
    sims = rng.choice(v, (B, horizon)).sum(1) * scale
    return dict(p10=float(np.percentile(sims, 10)), p50=float(np.percentile(sims, 50)), p90=float(np.percentile(sims, 90)))


# ---------------------------------------------------------------- reconciliation (count-to-count)
def intervals(ds: Dataset):
    """Count intervals: (label, month indices). First count column is the opening count."""
    cols = list(ds.counts.columns)
    ivs = []
    for a, b in zip(cols[:-1], cols[1:]):
        idx = [i for i, m in enumerate(ds.months) if a < m <= b]
        ivs.append((f"{a} to {b}", a, b, idx))
    return ivs


def reconcile_part(ds: Dataset, part: str, skus: list):
    """Observed_use = Counted0 + Purchased - Counted1 (no returns/transfers in this dataset);
    ratio = Observed_use / Expected per interval. Also posted ratio and book-vs-shelf drift."""
    rows, drift = [], 0.0
    for lab, a, b, idx in intervals(ds):
        c0 = ds.counts.loc[skus, a].sum(); c1 = ds.counts.loc[skus, b].sum()
        pur = ds.R.loc[skus].iloc[:, idx].values.sum()
        post = ds.POST.loc[skus].iloc[:, idx].values.sum()
        exp = ds.E.loc[part].iloc[idx].sum()
        obs = c0 + pur - c1
        drift += (c0 + pur - post) - c1
        rows.append(dict(interval=lab, end=b, purchased=float(pur), expected=float(exp), observed_use=float(obs),
                         posted=float(post), stock_change=float(c1 - c0),
                         ratio_observed=float(obs / exp) if exp else None,
                         ratio_posted=float(post / exp) if exp else None, book_minus_shelf=float(drift)))
    return pd.DataFrame(rows)


def change_point(r: np.ndarray):
    """Best single mean-shift split (min SSE), k in [2, n-2]. Returns (k, before, after)."""
    best = None
    for k in range(2, len(r) - 1):
        a, b = r[:k], r[k:]
        sse = ((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum()
        if best is None or sse < best[0]:
            best = (sse, k, a.mean(), b.mean())
    return best[1:] if best else (None, None, None)


def implied_usage(ds: Dataset, part: str, skus: list, ivs_idx: list):
    """NNLS of observed use on units built of the products using `part`, over chosen intervals,
    bounded 0 <= b <= 3*BOM; bootstrap over intervals for a 90% interval."""
    prods = [p for p in ds.B.columns if ds.B.loc[part, p] > 0]
    ivs = intervals(ds)
    X, y = [], []
    for i in ivs_idx:
        lab, a, b, idx = ivs[i]
        X.append([ds.Q.loc[p].iloc[idx].sum() for p in prods])
        c0 = ds.counts.loc[skus, a].sum(); c1 = ds.counts.loc[skus, b].sum()
        y.append(c0 + ds.R.loc[skus].iloc[:, idx].values.sum() - c1)
    X, y = np.array(X, float), np.array(y, float)
    fit = lambda X_, y_: np.minimum(nnls(X_, y_)[0], 3 * ds.B.loc[part, prods].values)
    est = fit(X, y)
    rng = RNG(); sims = []
    for _ in range(500):
        k = rng.integers(0, len(y), len(y))
        sims.append(fit(X[k], y[k]))
    sims = np.array(sims)
    return [dict(product=p, bom=float(ds.B.loc[part, p]), implied=float(est[i]),
                 lo=float(np.percentile(sims[:, i], 5)), hi=float(np.percentile(sims[:, i], 95)))
            for i, p in enumerate(prods)]


def findings(ds: Dataset, dec: pd.DataFrame, cfg=CONFIG):
    price = ds.price
    links = dec[dec.part.notna()].groupby("part").sku.apply(list)
    out, recon = [], {}
    last12 = ds.months[-12:]
    caps = ds.readiness.get("capabilities", {})
    for part, skus in links.items():
        in_no_bom = ds.B.loc[part].sum() == 0 if part in ds.B.index else True
        if in_no_bom or (caps.get("standardization_savings") and ds.E.loc[part].sum() == 0):
            spend = ds.R.loc[skus, last12].mul(price.loc[skus], axis=0).sum(0).values
            out.append(dict(type="NO_ACTIVE_DEMAND", part=part, skus=skus, rng=boot_range(spend, 12),
                            metric="Purchased, but in no active BOM", owner=["Procurement", "Service"],
                            action="Check spares/service obligations, then reduce or stop purchasing"))
            continue
        if not caps.get("reconciliation", True):
            continue
        rc = reconcile_part(ds, part, skus); recon[part] = rc
        r = rc.ratio_observed.values.astype(float)
        p_avg = float((price.loc[skus] * ds.R.loc[skus, last12].sum(1)).sum() / max(ds.R.loc[skus, last12].values.sum(), 1))
        k, before, after = change_point(r)
        if k is not None and abs(after - before) >= cfg["shift_min"]:
            post_ivs = list(range(k + 1, len(r))) if k + 1 < len(r) else [k]
            imp = implied_usage(ds, part, skus, post_ivs)
            excess_units = (rc.observed_use - rc.expected * before).clip(lower=0).values[-4:]
            out.append(dict(type="CONSUMPTION_SHIFT", part=part, skus=skus, start=rc.end.iloc[k - 1],
                            metric=f"Observed/expected {before:.2f} -> {after:.2f} after count {rc.end.iloc[k-1]}",
                            implied=imp, rng=boot_range(excess_units * p_avg, 4),
                            posted_ratio=float(np.nanmean(rc.ratio_posted.values)),
                            owner=["Engineering"], action="Check engineering change records; update BOM or investigate usage"))
            continue
        recent = r[-4:]
        if np.median(recent) >= cfg["excess_ratio"] and np.mean(recent > 1.03) >= cfg["excess_share"]:
            imp = implied_usage(ds, part, skus, list(range(len(r))))
            excess_units = (rc.observed_use - rc.expected * 1.005).clip(lower=0).values[-4:]
            out.append(dict(type="PERSISTENT_EXCESS", part=part, skus=skus,
                            metric=f"Observed use {np.median(recent):.3f}x expected in recent count intervals; ERP postings show {np.nanmean(rc.ratio_posted.values):.3f}x",
                            implied=imp, rng=boot_range(excess_units * p_avg, 4),
                            book_minus_shelf=float(rc.book_minus_shelf.iloc[-1]),
                            owner=["Plant / Quality"], action="Yield / scrap investigation on consuming lines"))
    for _, r in dec[dec.part.isna()].iterrows():
        spend = (ds.R.loc[r.sku, last12] * price[r.sku]).values
        out.append(dict(type="UNRESOLVED_SPEND", part=None, skus=[r.sku], rng=boot_range(spend, 12),
                        metric="No proposed engineering part", owner=["Procurement"],
                        action="Classify: indirect spend, or map manually"))
    return out, recon


def inventory_builds(ds: Dataset, dec: pd.DataFrame, cfg=CONFIG):
    """Receipt spike > 3x median where the rise in COUNTED stock explains >= 90% of (purchased - expected)."""
    rows = []
    if not ds.readiness.get("capabilities", {}).get("reconciliation", True):
        return rows
    linked = dict(zip(dec.sku, dec.part))
    ivs = intervals(ds)
    for s in ds.skus:
        r = ds.R.loc[s].values
        med = np.median(r[r > 0]) if (r > 0).any() else 0
        for i in np.where(r > cfg["spike_multiple"] * med)[0] if med else []:
            j = linked.get(s)
            iv = next(v for v in ivs if i in v[3])
            pur = r[iv[3]].sum()
            share = (ds.R.loc[s].sum() / ds.E.loc[j].sum()) if isinstance(j, str) and ds.E.loc[j].sum() else 1
            exp = ds.E.loc[j].iloc[iv[3]].sum() * share if isinstance(j, str) else 0
            dstock = ds.counts.loc[s, iv[2]] - ds.counts.loc[s, iv[1]]
            expl = dstock / (pur - exp) if pur - exp > 0 else 0
            rows.append(dict(sku=s, month=ds.months[i], received=int(r[i]), interval=iv[0],
                             stock_rise=float(dstock), explained=float(expl),
                             value=float(dstock * ds.price[s]),
                             verdict="Inventory build - explained, not a variance" if expl >= cfg["build_explained"] else "Unexplained receipt spike"))
    return rows


def integrity(ds: Dataset, f: list, builds: list):
    """Mutually exclusive buckets by precedence: Unresolved > No active demand > Possible BOM discrepancy
    > Unexplained consumption. Each SKU/part's dollars land in exactly one bucket."""
    order = [("UNRESOLVED_SPEND", "Unresolved spend"), ("NO_ACTIVE_DEMAND", "Inactive-part purchasing"),
             ("CONSUMPTION_SHIFT", "Possible BOM discrepancy"), ("PERSISTENT_EXCESS", "Unexplained consumption")]
    used, buckets = set(), []
    for t, lab in order:
        tot = dict(p10=0.0, p50=0.0, p90=0.0); n = 0
        for x in f:
            key = tuple(x["skus"])
            if x["type"] == t and not (set(x["skus"]) & used):
                for k in tot:
                    tot[k] += x["rng"][k]
                used |= set(x["skus"]); n += 1
        buckets.append(dict(bucket=lab, n=n, **tot))
    last12 = ds.months[-12:]
    total = float((ds.R[last12].sum(1) * ds.price).sum())
    gap = {k: sum(b[k] for b in buckets) for k in ("p10", "p50", "p90")}
    explained = sum(b["value"] for b in builds if b["explained"] >= CONFIG["build_explained"])
    return dict(total_spend_12m=total, buckets=buckets, gap=gap,
                gap_pct={k: (v / total if total else 0.0) for k, v in gap.items()}, explained_by_inventory=explained)


# ---------------------------------------------------------------- money
def part_prices(ds: Dataset, links: dict):
    last12 = ds.months[-12:]
    q = ds.R[last12].sum(1)
    df = pd.DataFrame(dict(part=pd.Series(links), q=q, p=ds.price)).dropna(subset=["part"])
    df["v"] = df.q * df.p
    g = df.groupby("part")
    qs = g.q.sum()
    out = (g.v.sum() / qs.where(qs > 0)).fillna(g.p.mean())   # no quantities: plain average unit price
    return out.to_dict()


def product_costs(ds: Dataset, links_before: dict, links_after: dict):
    pb, pa = part_prices(ds, links_before), part_prices(ds, links_after)
    last12 = ds.months[-12:]
    rows = []
    for pid, name in zip(ds.products.product_id, ds.products.product_name):
        lines = ds.bom[ds.bom.product_id == pid]
        cb = sum(q * pb.get(j, 0) for j, q in zip(lines.eng_part_id, lines.qty_per_unit))
        ca = sum(q * pa.get(j, 0) for j, q in zip(lines.eng_part_id, lines.qty_per_unit))
        units = ds.Q.loc[pid, last12].values
        delta = ca - cb
        rows.append(dict(product_id=pid, product=name, cost_name_only=cb, cost_recon=ca,
                         unpriced_before=[j for j in lines.eng_part_id if j not in pb],
                         units_12m=float(units.sum()), attributed=boot_range(units * delta, 12)))
    return rows


def price_gaps(ds: Dataset, links: dict):
    last12 = ds.months[-12:]
    by = {}
    for s, j in links.items():
        if isinstance(j, str):
            by.setdefault(j, []).append(s)
    out = []
    for j, skus in by.items():
        if len(skus) < 2:
            continue
        pmin = ds.price.loc[skus].min()
        monthly = ((ds.price.loc[skus] - pmin).values[:, None] * ds.R.loc[skus, last12].values).sum(0)
        out.append(dict(part=j, skus=skus, suppliers=ds.spend.set_index("sku").supplier.loc[skus].tolist(),
                        prices=ds.price.loc[skus].tolist(), qty_12m=ds.R.loc[skus, last12].sum(1).astype(int).tolist(),
                        rng=boot_range(monthly, 12)))
    return out


# ---------------------------------------------------------------- Track 1
STD_ASSUMPTIONS = dict(volume_discount=(0.02, 0.04, 0.06), carrying_cost=(1500, 2500, 4000),
                       qualification=(12000, 8000, 5000), amortize_years=3)


def standardization(ds: Dataset, links: dict, sim_min=0.8):
    parts = ds.parts[ds.parts.status == "active"].reset_index(drop=True)
    txt = (parts.description + " " + parts.spec.str.replace(";", " ")).str.lower().tolist()
    S = cosine_similarity(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit_transform(txt))
    last12 = ds.months[-12:]
    prices = part_prices(ds, links)
    vol = ds.E[last12].sum(1)
    sup_spend = {}
    for s, j in links.items():
        if isinstance(j, str):
            sup = ds.spend.set_index("sku").supplier[s]
            sup_spend.setdefault(j, {}).setdefault(sup, 0.0)
            sup_spend[j][sup] += float(ds.R.loc[s, last12].sum() * ds.price[s])
    common = []
    for j in parts.eng_part_id:
        prods = [p for p in ds.B.columns if ds.B.loc[j, p] > 0]
        if len(prods) > 1:
            common.append(dict(part=j, description=parts.set_index("eng_part_id").description[j], products=prods,
                               volume_12m=float(vol[j])))
    cands = []
    for a in range(len(parts)):
        for b in range(a + 1, len(parts)):
            pa, pb = parts.iloc[a], parts.iloc[b]
            if pa.category != pb.category or S[a, b] < sim_min:
                continue
            A_, B_ = (pa, pb) if pa.std_cost <= pb.std_cost else (pb, pa)   # merge INTO the higher-spec part
            ja, jb = A_.eng_part_id, B_.eng_part_id     # merge lower-spec/cheaper A into B
            prA, prB = prices.get(ja, A_.std_cost), prices.get(jb, B_.std_cost)
            vA, vB = float(vol[ja]), float(vol[jb])
            res = {}
            if not ds.readiness.get("capabilities", {}).get("standardization_savings", True):
                res = None
            for i, lab in enumerate(("low", "base", "high")):
                if res is None:
                    break
                d, cc, ql = (STD_ASSUMPTIONS["volume_discount"][i], STD_ASSUMPTIONS["carrying_cost"][i],
                             STD_ASSUMPTIONS["qualification"][i])
                res[lab] = d * prB * (vA + vB) + cc - (prB - prA) * vA - ql / STD_ASSUMPTIONS["amortize_years"]
            def top_share(js):
                tot = {}
                for j in js:
                    for k, v in sup_spend.get(j, {}).items():
                        tot[k] = tot.get(k, 0) + v
                return max(tot.values()) / sum(tot.values()) if tot and sum(tot.values()) > 0 else None
            cands.append(dict(merge_from=ja, merge_into=jb, from_desc=A_.description, into_desc=B_.description,
                              similarity=float(S[a, b]), price_from=prA, price_into=prB, vol_from=vA, vol_into=vB,
                              products_from=[p for p in ds.B.columns if ds.B.loc[ja, p] > 0],
                              products_into=[p for p in ds.B.columns if ds.B.loc[jb, p] > 0],
                              net_saving=res, top_supplier_share_before=top_share([jb]),
                              top_supplier_share_after=top_share([ja, jb])))
    return dict(common=common, candidates=sorted(cands, key=lambda c: -(c["net_saving"] or {"base": 0})["base"]),
                assumptions=STD_ASSUMPTIONS)
