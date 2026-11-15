import json, os
from pathlib import Path
from server import services as S

DATA = Path(__file__).resolve().parent.parent / "data" / "A"

def _files(drop=()):
    return {f.name: f.read_bytes() for f in DATA.glob("*.csv") if f.stem not in drop}

def test_validate_ok_and_missing():
    assert S.validate(_files()) == []
    assert S.validate(_files(drop=("physical_counts", "posted_consumption"))) == []   # optional now
    probs = S.validate(_files(drop=("spend_cube",)))
    assert any("spend_cube.csv" in p for p in probs)

def test_validate_bad_column():
    f = _files(); f["bom.csv"] = b"model,widget,howmany\nP1,X,1\n"
    assert any("bom.csv is missing column" in p for p in S.validate(f))

def test_upload_without_answer_key():
    b = S.run_upload(_files(drop=("ground_truth", "planted_cases")), "Test Plant")
    assert b["company"] == "Test Plant" and b["planted"] == []
    assert "demo dataset" in b["proof_source"]
    m = {l["sku"]: l["part"] for l in b["links"]}
    assert m["MTR-X72"] == "ASSEMBLY-204"

def test_pdfs():
    b = S.demo_bundle()
    assert S.pdf_onepager(b)[:4] == b"%PDF"
    assert S.pdf_passport(b, "MTR-X72")[:4] == b"%PDF"

def test_draft_fallback_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    c = S.demo_bundle()["cards"][0]
    d = S.draft(c, "Engineer")
    assert d["text"] == c["messages"]["Engineer"] and d["source"].startswith("template")
