"""Benchmark (name-only vs behavior-only vs RECON), segmented results, blind test, stress grid."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import CONFIG
from .core import Dataset, name_scores, behavior_scores, decide, cascade

SEGMENTS = {"MTR-X72": "Hidden link", "NTN-6204L": "Hidden link", "ITEM-4471B": "Hidden link",
            "SLX-4040G": "Look-alike twins", "SLX-4040R": "Look-alike twins",
            "CLT-S20": "Trap (not in any BOM)", "GLV-NL100": "Trap (not in any BOM)",
            "TB-6W-L": "Obsolete part", "QNC-M6NL": "5% backup supplier"}


def score(dec: pd.DataFrame, truth: pd.Series, method: str):
    d = dec.copy()
    d["truth"] = d.sku.map(truth)
    d["linked"] = d.part.apply(lambda p: isinstance(p, str))
    d["correct"] = [(p == t) if isinstance(p, str) else (t == "NONE") for p, t in zip(d.part, d.truth)]
    linkable = d.truth != "NONE"
    tp = int((d.linked & d.correct).sum())
    prec = tp / d.linked.sum() if d.linked.sum() else 0.0
    rec = tp / linkable.sum()
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(method=method, proposed=int(d.linked.sum()), correct_links=tp,
                false_links=int((d.linked & ~d.correct).sum()), missed=int(linkable.sum() - tp),
                traps_linked=int((d.linked & ~linkable).sum()),
                precision=round(prec, 3), recall=round(rec, 3), f1=round(f1, 3)), d


def run_methods(ds: Dataset, cfg=CONFIG):
    N = name_scores(ds)
    BEH_ind, SH, SR, RS = behavior_scores(ds, cfg=cfg)
    out = {}
    out["Name only"] = decide(ds, N, BEH_ind, cfg, method="name")
    out["Behavior only"] = decide(ds, N, BEH_ind, cfg, method="behavior")
    out["RECON"], _ = cascade(ds, N, SH, SR, RS, cfg)
    return out


def segment_map(ds: Dataset) -> dict:
    """Planted-case segments: fixed for companies A/B; from the generator's segments file otherwise."""
    if ds.planted is not None and "segment" in ds.planted.columns:
        return dict(zip(ds.planted.sku, ds.planted.segment))
    return SEGMENTS


def benchmark(ds: Dataset, cfg=CONFIG):
    res = run_methods(ds, cfg)
    segs = segment_map(ds)
    rows, seg, errs = [], [], []
    for m, d in res.items():
        r, dd = score(d, ds.truth, m)
        rows.append(r)
        for x in dd[~dd.correct].itertuples():
            errs.append(dict(method=m, sku=x.sku, truth=x.truth, predicted=x.part if isinstance(x.part, str) else None,
                             kind=("trap linked" if x.truth == "NONE" else "missed (left unresolved)" if not isinstance(x.part, str) else "false link"),
                             description=ds.spend.set_index("sku").sku_description[x.sku]))
        dd["segment"] = dd.sku.map(segs).fillna("Standard")
        for g, x in dd.groupby("segment"):
            seg.append(dict(method=m, segment=g, n=len(x), correct=int(x.correct.sum())))
    hard = {}
    for s in [k for k in segs if k in set(ds.skus)][:10]:
        hard[s] = {m: (lambda p, t: ("OK - unresolved" if not isinstance(p, str) and t == "NONE" else
                                    ("OK" if p == t else ("MISS" if not isinstance(p, str) else f"WRONG ({p})"))))(
            res[m].set_index("sku").part.get(s), ds.truth[s]) for m in res}
    return rows, seg, hard, errs


def slice_ds(ds: Dataset, months: int, noise: float, seed: int) -> Dataset:
    cols = ds.months[-months:]
    rng = np.random.default_rng(seed)
    R = ds.R[cols] * (1 + rng.normal(0, noise, (len(ds.R), len(cols)))) if noise else ds.R[cols]
    return Dataset(ds.name, cols, ds.products, ds.parts, ds.bom, ds.spend, ds.Q[cols], ds.B, ds.E[cols],
                   R.clip(lower=0), ds.POST[cols], ds.counts, ds.lead, ds.truth, ds.planted)


def stress(ds: Dataset, cfg=CONFIG):
    rows = []
    for months in (6, 12, 18, 24):
        for noise in (0.0, 0.05, 0.10, 0.20):
            for seed in (range(3) if noise else [0]):
                sub = slice_ds(ds, months, noise, seed)
                for m, d in run_methods(sub, cfg).items():
                    r, _ = score(d, ds.truth, m)
                    rows.append(dict(months=months, noise=noise, seed=seed, method=m, f1=r["f1"],
                                     precision=r["precision"], recall=r["recall"], false_links=r["false_links"]))
    df = pd.DataFrame(rows)
    return df.groupby(["months", "noise", "method"])[["f1", "precision", "recall", "false_links"]].mean().reset_index()
