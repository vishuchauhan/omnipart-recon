import io
from pathlib import Path
import pandas as pd
from server import services as S
from recon.core import load, name_scores, behavior_scores, signal_self_check
from recon.benchmark import benchmark

ROOT = Path(__file__).resolve().parent.parent / "data"

def _files(names, rename=None, src="A"):
    out = {}
    for n in names:
        b = (ROOT / src / f"{n}.csv").read_bytes()
        if rename and n in rename:
            b = pd.read_csv(io.BytesIO(b)).rename(columns=rename[n]).to_csv(index=False).encode()
        out[f"{n}.csv"] = b
    return out

def test_minimum_upload_names_only():
    b = S.run_upload(_files(["bom", "parts_master", "spend_cube"]), "min")
    caps = b["readiness"]["capabilities"]
    assert caps["name_matching"] and not caps["behavior_matching"] and not caps["reconciliation"]
    m = {l["sku"]: l["part"] for l in b["links"]}
    assert m["MTR-X72"] is None                       # names alone cannot find the hidden motor
    assert m["ACM-55180"] == "FST-0102"               # clear names still link

def test_behavior_unlocked_and_aliases():
    ren = {"bom": {"eng_part_id": "Part Number", "qty_per_unit": "Qty"},
           "spend_cube": {"supplier": "Vendor", "unit_price": "Unit Cost", "sku_description": "Item Description"},
           "receipts": {"qty_received": "GR Qty"}}
    b = S.run_upload(_files(["bom", "parts_master", "spend_cube", "products", "production", "receipts"], ren), "ren")
    assert b["readiness"]["capabilities"]["behavior_matching"]
    assert {l["sku"]: l["part"] for l in b["links"]}["MTR-X72"] == "ASSEMBLY-204"
    assert "bom" in b["readiness"]["renamed_columns"]

def test_unfamiliar_company_is_safe():
    r = {x["method"]: x for x in benchmark(load(ROOT / "C"))[0]}
    assert r["RECON"]["false_links"] == 0 and r["RECON"]["traps_linked"] == 0
    assert r["RECON"]["recall"] >= r["Name only"]["recall"]

def test_self_check_flags_weak_signal():
    out = {}
    for c in "AC":
        ds = load(ROOT / c); N = name_scores(ds); _, SH, _, _ = behavior_scores(ds)
        out[c] = signal_self_check(ds, N, SH)["verdict"]
    assert out == {"A": "strong", "C": "weak"}
