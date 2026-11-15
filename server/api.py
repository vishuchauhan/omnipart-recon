"""OmniPart RECON - FastAPI server.
Run:  python run.py   (or: uvicorn server.api:app --port 8000)
Then open http://localhost:8000
"""
from __future__ import annotations
import json
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel
from . import services as S

ROOT = Path(__file__).resolve().parent.parent
app = FastAPI(title="OmniPart RECON API", version="1.0",
              description="Evidence and reconciliation layer between engineering BOMs and procurement spend.")
STATE = {"bundle": None, "source": "demo"}


def bundle() -> dict:
    if STATE["bundle"] is None:
        STATE["bundle"] = S.demo_bundle()
    return STATE["bundle"]


@app.get("/", response_class=HTMLResponse)
def index():
    html = (ROOT / "app" / "template.html").read_text(encoding="utf-8")
    slides = ROOT / "app" / "slides.json"
    html = html.replace("__SLIDES__", slides.read_text(encoding="utf-8") if slides.exists() else "[]")
    return html.replace("__BUNDLE__", json.dumps(bundle()).replace("</", "<\\/"))


@app.get("/api/health")
def health():
    return {"ok": True, "source": STATE["source"], "company": bundle()["company"], "ai_drafting": bool(S.os.getenv("ANTHROPIC_API_KEY"))}


@app.get("/api/bundle")
def get_bundle():
    return JSONResponse(bundle())


@app.get("/api/links")
def links():
    return bundle()["links"]


@app.get("/api/links/{sku}")
def link(sku: str):
    l = next((x for x in bundle()["links"] if x["sku"] == sku), None)
    if not l:
        raise HTTPException(404, f"Unknown SKU {sku}")
    return l


@app.get("/api/findings")
def findings():
    return bundle()["findings"]


@app.get("/api/cards")
def cards():
    return bundle()["cards"]


@app.get("/api/integrity")
def integrity():
    return bundle()["integrity"]


@app.get("/api/standardization")
def standardization():
    return bundle()["standardization"]


@app.get("/api/benchmark")
def benchmark():
    b = bundle()
    return {"benchmark": b["benchmark"], "calibration": b["calibration"], "stress": b["stress"], "source": b.get("proof_source")}


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...), company: str = Form("Uploaded company")):
    data = {f.filename: await f.read() for f in files}
    try:
        STATE["bundle"] = S.run_upload(data, company)
    except S.UploadError as e:
        raise HTTPException(422, str(e))
    STATE["source"] = "upload"
    return {"ok": True, "company": company, "links": len(STATE["bundle"]["links"])}


@app.post("/api/reset")
def reset():
    STATE["bundle"], STATE["source"] = None, "demo"
    return {"ok": True}


@app.get("/api/template/{table}.csv")
def template(table: str):
    cols = S.REQUIRED.get(table) or S.OPTIONAL.get(table)
    if not cols:
        raise HTTPException(404, "Unknown table")
    return Response(",".join(cols) + "\n", media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{table}.csv"'})


@app.get("/api/export/onepager.pdf")
def onepager_pdf():
    return Response(S.pdf_onepager(bundle()), media_type="application/pdf",
                    headers={"Content-Disposition": 'attachment; filename="steering_update.pdf"'})


@app.get("/api/export/passport/{sku}.pdf")
def passport_pdf(sku: str):
    try:
        pdf = S.pdf_passport(bundle(), sku)
    except KeyError:
        raise HTTPException(404, f"Unknown SKU {sku}")
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="passport_{sku}.pdf"'})


class DraftIn(BaseModel):
    card_id: str
    audience: str = "Engineer"


@app.post("/api/draft")
def draft(body: DraftIn):
    c = next((x for x in bundle()["cards"] if x["id"] == body.card_id), None)
    if not c:
        raise HTTPException(404, "Unknown card")
    return S.draft(c, body.audience)
