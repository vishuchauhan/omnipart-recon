"""
Core: data loading, the four quantities, name signal, behavior signal (demand signature),
backflush detector, decision cascade, signature twins, group volume check.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .config import CONFIG, ABBREVIATIONS


# ---------------------------------------------------------------- data
@dataclass
class Dataset:
    name: str
    months: list
    products: pd.DataFrame
    parts: pd.DataFrame
    bom: pd.DataFrame
    spend: pd.DataFrame
    Q: pd.DataFrame          # products x months   units built
    B: pd.DataFrame          # parts x products    BOM qty per unit
    E: pd.DataFrame          # parts x months      EXPECTED  = B @ Q
    R: pd.DataFrame          # skus x months       PURCHASED (receipts)
    POST: pd.DataFrame       # skus x months       POSTED (ERP goods issues / backflush)
    counts: pd.DataFrame     # skus x count dates  OBSERVED (physical counts, first col = opening)
    lead: pd.Series
    truth: pd.Series | None = None
    planted: pd.DataFrame | None = None
    readiness: dict = field(default_factory=dict)

    @property
    def skus(self):
        return self.spend.sku.tolist()

    @property
    def price(self):
        return self.spend.set_index("sku").unit_price


ALIASES = {
    "product_id": ["product", "product_code", "model", "model_id", "parent", "parent_id", "assembly", "fg", "finished_good"],
    "product_name": ["name", "product_description", "model_name", "description"],
    "eng_part_id": ["part_id", "part", "part_number", "part_no", "pn", "component", "component_id", "child", "child_id",
                    "material", "material_number", "item", "item_number", "engineering_part"],
    "qty_per_unit": ["qty", "quantity", "qty_per", "quantity_per", "usage", "bom_qty", "component_qty"],
    "description": ["part_description", "desc", "part_name", "component_description", "material_description"],
    "sku": ["sku_id", "supplier_part", "supplier_part_number", "vendor_part", "mpn", "purchase_item", "item_code", "material_code"],
    "supplier": ["vendor", "vendor_name", "supplier_name", "seller"],
    "sku_description": ["description", "item_description", "po_description", "short_text", "material_description", "desc"],
    "unit_price": ["price", "unit_cost", "cost", "net_price", "price_per_unit"],
    "annual_qty": ["annual_quantity", "qty_12m", "annual_volume", "volume", "quantity", "qty"],
    "month": ["period", "date", "posting_month", "fiscal_period"],
    "units_built": ["units", "quantity", "qty", "production_qty", "output", "units_produced"],
    "qty_received": ["quantity", "qty", "received_qty", "gr_qty", "receipt_qty"],
    "qty_posted": ["quantity", "qty", "issued_qty", "consumption", "goods_issue_qty"],
    "count_date": ["date", "period", "month", "count_month"],
    "counted_on_hand": ["on_hand", "count", "counted_qty", "quantity", "qty", "stock"],
    "category": ["commodity", "part_category", "class", "group"],
    "spec": ["specification", "attributes", "specs"],
    "std_cost": ["standard_cost", "cost", "unit_cost"],
    "status": ["lifecycle", "part_status", "state"],
}
TABLE_COLS = {
    "products": ["product_id", "product_name"], "production": ["product_id", "month", "units_built"],
    "parts_master": ["eng_part_id", "description", "category", "spec", "std_cost", "status"],
    "bom": ["product_id", "eng_part_id", "qty_per_unit"],
    "spend_cube": ["sku", "supplier", "sku_description", "unit_price", "annual_qty"],
    "receipts": ["sku", "month", "qty_received"], "posted_consumption": ["sku", "month", "qty_posted"],
    "physical_counts": ["sku", "count_date", "counted_on_hand"], "supplier_lead_times": ["sku", "lead_time_months"],
    "ground_truth": ["sku", "true_eng_part_id"],
}


def canon(df: pd.DataFrame, table: str):
    """Rename common alternative column names to the canonical ones. Returns (df, {old: new})."""
    df = df.copy()
    norm = {c: re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_") for c in df.columns}
    df.columns = [norm[c] for c in df.columns]
    renames = {}
    for want in TABLE_COLS.get(table, []):
        if want in df.columns:
            continue
        for alt in ALIASES.get(want, []):
            if alt in df.columns and alt not in renames:
                renames[alt] = want
                break
    df = df.rename(columns=renames)
    return df, {k: v for k, v in renames.items()}


def _month(x) -> str:
    try:
        return pd.Period(pd.to_datetime(str(x)), freq="M").strftime("%Y-%m")
    except Exception:  # noqa: BLE001
        return str(x)[:7]


def load(path: str | Path) -> Dataset:
    """Load whatever tables exist. Minimum: bom, parts_master, spend_cube. production + receipts unlock the
    behavior signal; physical_counts unlock reconciliation; posted_consumption enables the backflush check."""
    p = Path(path)
    renames = {}

    def rd(n):
        f = p / f"{n}.csv"
        if not f.exists():
            return None
        df, r = canon(pd.read_csv(f), n)
        if r:
            renames[n] = r
        return df

    bom, parts, spend = rd("bom"), rd("parts_master"), rd("spend_cube")
    prod, rec, post, cnt_raw, prods = rd("production"), rd("receipts"), rd("posted_consumption"), rd("physical_counts"), rd("products")
    spend["sku"] = spend.sku.astype(str); bom["eng_part_id"] = bom.eng_part_id.astype(str); bom["product_id"] = bom.product_id.astype(str)
    parts["eng_part_id"] = parts.eng_part_id.astype(str)
    for c, d in (("category", "Unspecified"), ("spec", ""), ("std_cost", np.nan), ("status", "active")):
        if c not in parts.columns:
            parts[c] = d
    parts["spec"] = parts.spec.fillna("").astype(str); parts["category"] = parts.category.fillna("Unspecified")
    if "supplier" not in spend.columns:
        spend["supplier"] = "Unknown supplier"
    extra = sorted(set(bom.eng_part_id) - set(parts.eng_part_id))   # BOM parts missing from the part master
    if extra:
        parts = pd.concat([parts, pd.DataFrame(dict(eng_part_id=extra, description=extra, category="Unspecified",
                                                    spec="", std_cost=np.nan, status="active"))], ignore_index=True)
    if prods is None:
        ids = sorted(bom.product_id.unique())
        prods = pd.DataFrame(dict(product_id=ids, product_name=ids))
    prods["product_id"] = prods.product_id.astype(str)
    if "product_name" not in prods.columns:
        prods["product_name"] = prods.product_id
    for df in (prod, rec, post):
        if df is not None:
            df["month"] = df.month.map(_month)
            if "sku" in df.columns:
                df["sku"] = df.sku.astype(str)
    if prod is not None:
        prod["product_id"] = prod.product_id.astype(str)

    if rec is not None:
        months = sorted(rec.month.unique())
        R = rec.pivot_table(index="sku", columns="month", values="qty_received", aggfunc="sum").reindex(index=spend.sku, columns=months).fillna(0).astype(float)
    else:   # no history: one annual column from spend_cube.annual_qty (or 0)
        months = ["annual"]
        q = spend.annual_qty.fillna(0).values if "annual_qty" in spend.columns else np.zeros(len(spend))
        R = pd.DataFrame(q, index=spend.sku, columns=months).astype(float)
    prod_ids = prods.product_id.tolist()
    if prod is not None and rec is not None:
        Q = prod.pivot_table(index="product_id", columns="month", values="units_built", aggfunc="sum").reindex(index=prod_ids, columns=months).fillna(0).astype(float)
    elif prod is not None:
        last = sorted(prod.month.unique())[-12:]
        Q = prod[prod.month.isin(last)].groupby("product_id").units_built.sum().reindex(prod_ids).fillna(0).to_frame("annual").astype(float)
    else:
        Q = pd.DataFrame(0.0, index=prod_ids, columns=months)
    B = bom.pivot_table(index="eng_part_id", columns="product_id", values="qty_per_unit", aggfunc="sum", fill_value=0)
    B = B.reindex(index=parts.eng_part_id, columns=Q.index, fill_value=0).astype(float)
    E = pd.DataFrame(B.values @ Q.values, index=B.index, columns=months)
    POST = (post.pivot_table(index="sku", columns="month", values="qty_posted", aggfunc="sum").reindex(index=spend.sku, columns=months).fillna(0).astype(float)
            if post is not None and rec is not None else pd.DataFrame(0.0, index=spend.sku, columns=months))
    if cnt_raw is not None:
        cnt_raw["count_date"] = cnt_raw.count_date.map(_month); cnt_raw["sku"] = cnt_raw.sku.astype(str)
        cnt = cnt_raw.pivot_table(index="sku", columns="count_date", values="counted_on_hand", aggfunc="sum").reindex(index=spend.sku).fillna(0).astype(float)
        cnt = cnt[sorted(cnt.columns)]
    else:
        cnt = pd.DataFrame(index=spend.sku)
    lt = rd("supplier_lead_times")
    lead = lt.assign(sku=lt.sku.astype(str)).set_index("sku").lead_time_months if lt is not None else pd.Series(0, index=spend.sku)
    gt = rd("ground_truth")
    truth = gt.assign(sku=gt.sku.astype(str)).set_index("sku").true_eng_part_id.astype(str) if gt is not None else None
    planted = pd.read_csv(p / "planted_cases.csv") if (p / "planted_cases.csv").exists() else None
    name = (p / "company.txt").read_text() if (p / "company.txt").exists() else p.name
    ds = Dataset(name, months, prods, parts, bom, spend, Q, B, E, R, POST, cnt, lead, truth, planted)
    n_hist = len(months) if rec is not None else 0
    ds.readiness = dict(
        tables={t: (p / f"{t}.csv").exists() for t in TABLE_COLS},
        renamed_columns=renames, months_of_history=n_hist,
        count_dates=len(cnt.columns), parts_added_from_bom=extra,
        capabilities=dict(
            name_matching=True,
            behavior_matching=bool(prod is not None and rec is not None and n_hist >= CONFIG["min_windows"] + CONFIG["window"] - 1),
            spend_amounts=bool(rec is not None or "annual_qty" in spend.columns),
            reconciliation=bool(rec is not None and prod is not None and len(cnt.columns) >= 2),
            backflush_check=bool(post is not None and rec is not None and prod is not None),
            standardization_savings=bool(prod is not None)),
    )
    return ds


# ---------------------------------------------------------------- name signal
def _norm(s: str) -> str:
    s = s.lower().replace("-", " ").replace("x", " x ")
    toks = re.findall(r"[a-z]+|\d+(?:\.\d+)?", s)
    return " ".join(ABBREVIATIONS.get(t, t) for t in toks)


def _nums(s: str) -> set:
    return set(re.findall(r"\d+(?:\.\d+)?", s.lower()))


def name_scores(ds: Dataset) -> pd.DataFrame:
    """name = 0.75 * char-ngram TF-IDF cosine + 0.25 * spec-number Jaccard, rescaled to [0,1]."""
    sku_txt = [_norm(d) for d in ds.spend.sku_description]
    part_raw = (ds.parts.description + " " + ds.parts.spec.str.replace(";", " ")).tolist()
    part_txt = [_norm(t) for t in part_raw]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit(sku_txt + part_txt)
    tf = cosine_similarity(vec.transform(sku_txt), vec.transform(part_txt))
    jac = np.array([[len(_nums(a) & _nums(b)) / max(len(_nums(a) | _nums(b)), 1) for b in part_raw]
                    for a in ds.spend.sku_description])
    raw = 0.75 * tf + 0.25 * jac
    return pd.DataFrame(np.clip(raw / raw.max(), 0, 1), index=ds.skus, columns=ds.parts.eng_part_id)


# ---------------------------------------------------------------- behavior signal
def windows(M: np.ndarray, w: int) -> np.ndarray:
    """Rolling w-month sums along the last axis (valid windows only)."""
    c = np.cumsum(np.pad(M, [(0, 0)] * (M.ndim - 1) + [(1, 0)]), axis=-1)
    return c[..., w:] - c[..., :-w]


def _rowcorr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pearson correlation of every row of A with every row of B; NaN where a row has zero variance."""
    Ac = A - A.mean(1, keepdims=True); Bc = B - B.mean(1, keepdims=True)
    na = np.sqrt((Ac ** 2).sum(1)); nb = np.sqrt((Bc ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (Ac @ Bc.T) / np.outer(na, nb)


def level_penalty(share, lo, hi, k):
    """1 inside [lo,hi]; exp(-k*|ln(share/clip)|) outside. Works on arrays."""
    share = np.asarray(share, float)
    clip = np.clip(share, lo, hi)
    with np.errstate(divide="ignore", invalid="ignore"):
        pen = np.exp(-k * np.abs(np.log(np.where(share > 0, share, 1e-12) / clip)))
    return np.where((share >= lo) & (share <= hi), 1.0, pen)


def behavior_from(P: np.ndarray, E: np.ndarray, cfg=CONFIG):
    """
    P: skus x windows (purchased), E: parts x windows (expected).
    shape = max(0, 0.5*corr(levels) + 0.5*corr(month-to-month changes))   (clipped BEFORE squaring)
    share = sum(P_s) / sum(E_j);  behavior = shape^2 * level_penalty(share)
    Returns (behavior, shape, share, reason) with NaN behavior where the signal is insufficient.
    """
    n_w = P.shape[1]
    r_lvl = _rowcorr(P, E)
    r_dif = _rowcorr(np.diff(P, axis=1), np.diff(E, axis=1))
    shape = np.maximum(0.0, 0.5 * np.nan_to_num(r_lvl) + 0.5 * np.nan_to_num(r_dif))
    with np.errstate(divide="ignore", invalid="ignore"):
        share = P.sum(1)[:, None] / E.sum(1)[None, :]
    lvl = level_penalty(share, cfg["share_min"], cfg["share_max"], cfg["level_penalty_k"])
    beh = shape ** 2 * lvl
    reason = np.full(beh.shape, "", dtype=object)
    bad_p = (P.std(1) == 0) | ((P > 0).sum(1) < cfg["min_nonzero_windows"])
    bad_e = (E.std(1) == 0) | (E.sum(1) == 0)
    insufficient = n_w < cfg["min_windows"]
    mask = bad_p[:, None] | bad_e[None, :] | insufficient
    beh = np.where(mask, np.nan, beh)
    reason[mask] = "insufficient signal"
    reason[bad_e[None, :].repeat(P.shape[0], 0)] = "no active BOM demand"
    return beh, shape, share, reason


def behavior_scores(ds: Dataset, R=None, cfg=CONFIG):
    R = ds.R if R is None else R
    Rv = R.values if hasattr(R, "values") else R
    if cfg["use_lead_time_shift"]:
        Rv = np.stack([np.roll(r, -int(ds.lead.get(s, 0))) for s, r in zip(ds.skus, Rv)])
    P = windows(Rv, cfg["window"]); E = windows(ds.E.values, cfg["window"])
    beh, shape, share, reason = behavior_from(P, E, cfg)
    f = lambda a: pd.DataFrame(a, index=ds.skus, columns=ds.parts.eng_part_id)
    return f(beh), f(shape), f(share), f(reason)


# ---------------------------------------------------------------- backflush detector
def backflush_check(ds: Dataset, links: pd.Series, shares: pd.DataFrame, cfg=CONFIG) -> pd.DataFrame:
    """Posted is 'derived from BOM' if it tracks alpha*Expected almost perfectly."""
    rows = []
    for s in ds.skus:
        j = links.get(s)
        post = ds.POST.loc[s].values
        if not isinstance(j, str) or ds.E.loc[j].sum() == 0 or post.sum() == 0:
            rows.append(dict(sku=s, part=j, corr=None, cv=None, circular=False,
                             label="Not assessed (no linked BOM demand)")); continue
        e = ds.E.loc[j].values
        a = post.sum() / e.sum()
        corr = float(np.corrcoef(post, a * e)[0, 1])
        ratio = post / np.where(e > 0, e, np.nan)
        cv = float(np.nanstd(ratio) / np.nanmean(ratio))
        circ = corr >= cfg["backflush_corr"] and cv <= cfg["backflush_cv"]
        rows.append(dict(sku=s, part=j, corr=round(corr, 4), cv=round(cv, 4), circular=bool(circ),
                         label="Derived from BOM - excluded as evidence" if circ else "Real goods issues - usable evidence"))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- decision cascade
def decide(ds: Dataset, NAME: pd.DataFrame, BEH: pd.DataFrame, cfg=CONFIG,
           verified: dict | None = None, rejected: set | None = None, method="recon") -> pd.DataFrame:
    """
    method = 'name' | 'behavior' | 'recon'
    recon cascade:
      1 name-led:   name_top >= 0.55 and margin >= 0.08
      2 rescue:     name_top < 0.55 and behavior_top >= 0.85 and behavior margin >= 0.05  -> hidden link
      3 combined:   0.45*name + 0.55*behavior >= 0.55   (parts without demand: name only)
      else Unresolved.  Verified links override; rejected pairs are excluded.
    """
    verified = verified or {}; rejected = rejected or set()
    rows = []
    for s in ds.skus:
        n = NAME.loc[s].copy(); b = BEH.loc[s].copy()
        for (rs, rj) in rejected:
            if rs == s:
                n[rj] = -1; b[rj] = np.nan
        if s in verified:
            j = verified[s]
            rows.append(dict(sku=s, part=j, state="Verified", route="human", score=1.0,
                             name=float(NAME.loc[s, j]), behavior=_f(BEH.loc[s, j]))); continue
        ns = n.sort_values(ascending=False)
        bs = b.dropna().sort_values(ascending=False)
        comb = (cfg["w_name"] * n + cfg["w_behavior"] * b).where(b.notna(), n * 0.9)
        cs = comb.sort_values(ascending=False)
        part, route, score = None, None, 0.0
        if method == "name":
            if ns.iloc[0] >= cfg["name_accept"]:
                part, route, score = ns.index[0], "name-only", ns.iloc[0]
        elif method == "behavior":
            if len(bs) and bs.iloc[0] >= cfg["name_accept"]:
                part, route, score = bs.index[0], "behavior-only", bs.iloc[0]
        else:
            if ns.iloc[0] >= cfg["name_accept"] and ns.iloc[0] - ns.iloc[1] >= cfg["name_margin"]:
                part, route, score = ns.index[0], "name-led", ns.iloc[0]
            elif (ns.iloc[0] < cfg["name_accept"] and len(bs) > 1 and bs.iloc[0] >= cfg["rescue_behavior"]
                  and bs.iloc[0] - bs.iloc[1] >= cfg["rescue_margin"]):
                part, route, score = bs.index[0], "hidden link (behavior rescue)", bs.iloc[0]
            elif cs.iloc[0] >= cfg["combined_accept"]:
                part, route, score = cs.index[0], "combined", cs.iloc[0]
                if np.isnan(b[part]):
                    route = "name only (no active BOM demand)"
        rows.append(dict(sku=s, part=part, state="Proposed" if part else "Unresolved", route=route or "-",
                         score=float(score), name=float(NAME.loc[s, part]) if part else float(ns.iloc[0]),
                         behavior=_f(BEH.loc[s, part]) if part else _f(bs.iloc[0] if len(bs) else np.nan),
                         best_name_part=ns.index[0], best_behavior_part=bs.index[0] if len(bs) else None))
    return pd.DataFrame(rows)


def _f(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def annotate(ds: Dataset, dec: pd.DataFrame, NAME, BEH, SHARE, cfg=CONFIG) -> pd.DataFrame:
    """Signature twins, behavior-disagrees warnings, group volume check, evidence class."""
    dec = dec.copy()
    twins, warn = [], []
    for _, r in dec.iterrows():
        tw, w = [], []
        if isinstance(r.part, str) and r.state != "Unresolved":
            b = BEH.loc[r.sku].dropna()
            if len(b) and r.part in b.index:
                top = b.max()
                cand = b[(b >= top - cfg["twin_margin"]) & (b >= cfg["twin_min_behavior"])].index
                rivals = [x for x in cand if x != r.part and NAME.loc[r.sku, x] >= cfg["twin_min_name"]]
                if r.part in cand and rivals:
                    tw = [r.part] + rivals
                if r.route == "name-led" and BEH.loc[r.sku, r.part] < 0.3 and b.max() >= 0.85:
                    w.append("behavior disagrees with name")
        twins.append(tw); warn.append(w)
    dec["twin_group"] = twins
    dec["warnings"] = warn
    dec["share"] = [float(SHARE.loc[r.sku, r.part]) if isinstance(r.part, str) and pd.notna(SHARE.loc[r.sku, r.part]) and np.isfinite(SHARE.loc[r.sku, r.part]) else None
                    for _, r in dec.iterrows()]
    # group volume check: SKUs proposed for one part should together cover its demand
    grp = dec[dec.part.notna() & dec.share.notna()].groupby("part").share.sum()
    dec["group_share"] = dec.part.map(grp)
    for i, r in dec.iterrows():
        g = r.group_share
        if pd.notna(g) and not (cfg["group_share_min"] <= g <= cfg["group_share_max"]):
            dec.at[i, "warnings"] = r.warnings + [f"supplier SKUs cover {g:.0%} of this part's demand"]
    dec["evidence_class"] = [_evidence(r) for _, r in dec.iterrows()]
    return dec


def _evidence(r):
    if r.state == "Unresolved":
        return "Unresolved"
    if r.state == "Verified":
        return "Verified"
    if r.route.startswith("hidden"):
        return "Hidden link"
    if r.twin_group:
        return "Look-alike (twin)"
    if r.route.startswith("name only"):
        return "Name only (no BOM demand)"
    if r.behavior is not None and r.behavior >= 0.7 and r.name >= 0.5:
        return "Both signals agree"
    return "One strong signal"


# ---------------------------------------------------------------- residual-demand volume check
def name_led_links(ds: Dataset, NAME: pd.DataFrame, cfg=CONFIG) -> dict:
    """Pass 1: SKUs whose name match is clear (top >= accept and margin >= name_margin)."""
    out = {}
    for s in ds.skus:
        ns = NAME.loc[s].sort_values(ascending=False)
        if ns.iloc[0] >= cfg["name_accept"] and ns.iloc[0] - ns.iloc[1] >= cfg["name_margin"]:
            out[s] = ns.index[0]
    return out


def residual_behavior(ds: Dataset, SHAPE: pd.DataFrame, SHARE: pd.DataFrame, REASON: pd.DataFrame,
                      NAME: pd.DataFrame, cfg=CONFIG, pass1: dict | None = None) -> pd.DataFrame:
    """
    Group-level volume plausibility (Velocity feedback): a SKU's share of part j's demand is judged
    against the demand NOT already covered by SKUs confidently linked to j by name.
        target(s,j) = max(0, 1 - sum of shares of OTHER name-led SKUs on j)
        r = (share + 0.05) / (target + 0.05);   level = penalty(r, 0.75, 1.33)
    So a 5% backup supplier fits a part that is 95% covered, while a SKU claiming a share of a part
    that is already fully supplied is penalised.
    """
    pass1 = name_led_links(ds, NAME, cfg) if pass1 is None else pass1
    covered = {}
    for s, j in pass1.items():
        v = SHARE.loc[s, j]
        if pd.notna(v) and np.isfinite(v):
            covered.setdefault(j, []).append((s, float(v)))
    out = SHAPE.copy() * np.nan
    for s in ds.skus:
        tgt = np.array([max(0.0, 1.0 - sum(v for (s2, v) in covered.get(j, []) if s2 != s))
                        for j in SHAPE.columns])
        sh = SHARE.loc[s].values.astype(float)
        r = (np.nan_to_num(sh, nan=0, posinf=0) + 0.05) / (tgt + 0.05)
        lvl = level_penalty(r, 0.75, 1.33, cfg["level_penalty_k"])
        out.loc[s] = SHAPE.loc[s].values ** 2 * lvl
    return out.where(REASON == "")


def cascade(ds: Dataset, NAME, SHAPE, SHARE, REASON, cfg=CONFIG, verified=None, rejected=None):
    """
    RECON decision cascade (greedy, most-certain-first):
      Pass 1  name-led links (clear name match)                     -> Proposed (name-led)
      Pass 2  repeat: recompute residual-demand behavior; among unresolved SKUs take the single best
              rescue with behavior >= rescue_behavior and margin >= rescue_margin -> Proposed (hidden link);
              its share now counts as covered demand.  Stop when no rescue qualifies.
      Pass 3  combined score 0.45*name + 0.55*behavior >= 0.55     -> Proposed (combined)
      else    Unresolved.   Verified links are fixed inputs; rejected pairs are never proposed.
    Returns (decisions DataFrame, final behavior matrix).
    """
    verified = dict(verified or {}); rejected = set(rejected or set())
    N = NAME.copy()
    for (s, j) in rejected:
        N.loc[s, j] = -1.0
    links = {**{s: j for s, j in name_led_links(ds, N, cfg).items() if s not in verified}, **verified}
    route = {s: ("human" if s in verified else "name-led") for s in links}
    while True:
        BEH = residual_behavior(ds, SHAPE, SHARE, REASON, N, cfg, pass1=links)
        for (s, j) in rejected:
            BEH.loc[s, j] = np.nan
        best = None
        for s in ds.skus:
            if s in links:
                continue
            b = BEH.loc[s].dropna().sort_values(ascending=False)
            if len(b) < 2 or N.loc[s].max() >= cfg["name_accept"]:
                continue
            m = b.iloc[0] - b.iloc[1]
            if b.iloc[0] >= cfg["rescue_behavior"] and m >= cfg["rescue_margin"] and (best is None or m > best[2]):
                best = (s, b.index[0], m)
        if best is None:
            break
        links[best[0]] = best[1]; route[best[0]] = "hidden link (behavior rescue)"
    BEH = residual_behavior(ds, SHAPE, SHARE, REASON, N, cfg, pass1=links)
    rows = []
    for s in ds.skus:
        n, b = N.loc[s], BEH.loc[s]
        if s in links:
            j = links[s]
            sc = 1.0 if route[s] == "human" else (n[j] if route[s] == "name-led" else b[j])
            rows.append(dict(sku=s, part=j, state="Verified" if route[s] == "human" else "Proposed",
                             route=route[s], score=float(sc)))
            continue
        comb = (cfg["w_name"] * n + cfg["w_behavior"] * b).where(b.notna(), n * 0.9)
        j = comb.idxmax()
        if comb[j] >= cfg["combined_accept"]:
            rows.append(dict(sku=s, part=j, state="Proposed",
                             route="combined" if pd.notna(b[j]) else "name only (no active BOM demand)", score=float(comb[j])))
        else:
            rows.append(dict(sku=s, part=None, state="Unresolved", route="-", score=float(comb[j])))
    dec = pd.DataFrame(rows)
    dec["name"] = [float(NAME.loc[r.sku, r.part]) if isinstance(r.part, str) else float(NAME.loc[r.sku].max()) for r in dec.itertuples()]
    dec["behavior"] = [_f(BEH.loc[r.sku, r.part]) if isinstance(r.part, str) else _f(BEH.loc[r.sku].max()) for r in dec.itertuples()]
    dec["best_name_part"] = [NAME.loc[s].idxmax() for s in dec.sku]
    dec["best_behavior_part"] = [BEH.loc[s].idxmax() if BEH.loc[s].notna().any() else None for s in dec.sku]
    return dec, BEH


def signal_self_check(ds: Dataset, NAME: pd.DataFrame, SHAPE: pd.DataFrame, cfg=CONFIG) -> dict:
    """How informative is purchase behavior on THIS dataset? Needs no answer key: for SKUs whose name match
    is already clear, measure how well their purchases follow the matched part's expected demand.
    strong >= 0.70, moderate >= 0.55, else weak."""
    if not ds.readiness.get("capabilities", {}).get("behavior_matching", False):
        return dict(strength=None, n=0, verdict="not available", note="Needs monthly production and receipts (12+ months).")
    vals = [float(SHAPE.loc[s, j]) for s, j in name_led_links(ds, NAME, cfg).items()
            if ds.E.loc[j].sum() > 0 and pd.notna(SHAPE.loc[s, j])]
    if len(vals) < 5:
        return dict(strength=None, n=len(vals), verdict="not enough clear name matches to judge", note="")
    m = float(np.median(vals))
    lump = float(np.median([r.std() / r.mean() for r in ds.R.values if r.mean() > 0]))
    if m >= 0.70:
        v, note = "strong", "Purchases track production closely: hidden-link rescue is reliable here."
    elif m >= 0.55:
        v, note = "moderate", "Purchases follow production with noise: treat hidden links with extra review."
    else:
        v, note = "weak", ("Purchases are too lumpy to follow production month to month: RECON relies on names and "
                           "leaves the rest unresolved. Consumption-level data (goods issues) or longer history would help.")
    return dict(strength=m, n=len(vals), verdict=v, purchase_lumpiness=lump, note=note)
