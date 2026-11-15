import numpy as np, pandas as pd, pytest
from pathlib import Path
from recon.core import behavior_from, level_penalty, load, name_scores, behavior_scores, cascade, backflush_check
from recon.analysis import wilson, integrity, findings, inventory_builds
from recon.benchmark import benchmark
from recon.config import CONFIG

DATA = Path(__file__).resolve().parent.parent / "data"

def test_negative_correlation_gives_zero_shape():
    t = np.arange(12, dtype=float)
    P = (100 + 10 * np.sin(t))[None, :]; E = (100 - 10 * np.sin(t))[None, :]
    beh, shape, *_ = behavior_from(P, E, {**CONFIG, "min_windows": 6})
    assert shape[0, 0] == 0 and beh[0, 0] == 0

def test_zero_variance_is_null():
    P = np.full((1, 12), 50.0); E = (100 + np.arange(12.0))[None, :]
    beh, *_ , reason = behavior_from(P, E, CONFIG)
    assert np.isnan(beh[0, 0]) and reason[0, 0] == "insufficient signal"

def test_short_history_is_null():
    P = np.arange(1, 5.0)[None, :]; E = np.arange(1, 5.0)[None, :]
    beh, *_ = behavior_from(P, E, CONFIG)
    assert np.isnan(beh[0, 0])

def test_level_penalty():
    assert level_penalty(0.05, 0.02, 1.25, 2.5) == 1.0          # 5% backup supplier accepted
    assert level_penalty(1.6, 0.85, 1.25, 2.5) < 0.6            # group share above 125% penalised

def test_wilson_known_value():
    p, lo, hi = wilson(14, 14)
    assert p == 1 and abs(lo - 0.785) < 0.01 and hi == 1

@pytest.fixture(scope="module")
def run_A():
    ds = load(DATA / "A"); N = name_scores(ds); _, SH, SR, RS = behavior_scores(ds)
    dec, _ = cascade(ds, N, SH, SR, RS)
    return ds, dec

def test_planted_cases(run_A):
    ds, dec = run_A
    m = dict(zip(dec.sku, dec.part))
    assert m["MTR-X72"] == "ASSEMBLY-204" and m["ITEM-4471B"] == "ELE-0402" and m["NTN-6204L"] == "BRG-0201"
    assert m["QNC-M6NL"] == "FST-0104"                           # 5% backup accepted
    assert not isinstance(m["CLT-S20"], str) and not isinstance(m["GLV-NL100"], str)   # traps unresolved
    F, _ = findings(ds, dec)
    types = {(f["type"], f["part"]) for f in F}
    assert ("CONSUMPTION_SHIFT", "PWR-0702") in types and ("PERSISTENT_EXCESS", "BRG-0201") in types
    assert ("NO_ACTIVE_DEMAND", "OBS-9901") in types
    assert not any(f["part"] == "PWR-0701" for f in F)           # inventory build NOT flagged
    b = inventory_builds(ds, dec)
    assert any(x["sku"] == "GR-32-15" and x["explained"] >= 0.9 for x in b)

def test_backflush_detected(run_A):
    ds, dec = run_A
    bf = backflush_check(ds, dec.set_index("sku").part, None).set_index("sku")
    assert bf.loc["SKF-6204", "circular"] and not bf.loc["ENC-1024", "circular"]

def test_buckets_mutually_exclusive(run_A):
    ds, dec = run_A
    F, _ = findings(ds, dec)
    I = integrity(ds, F, inventory_builds(ds, dec))
    assert abs(sum(b["p50"] for b in I["buckets"]) - I["gap"]["p50"]) < 1e-6

def test_blind_company_no_false_links():
    rows, *_ = benchmark(load(DATA / "B"))
    r = {x["method"]: x for x in rows}
    assert r["RECON"]["false_links"] == 0 and r["RECON"]["recall"] > r["Name only"]["recall"]
