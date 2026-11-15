"""Server-side services: CSV upload validation + engine run, PDF exports, AI message drafting.
Pure functions (no web framework) so they can be tested on their own."""
from __future__ import annotations
import io
import json
import os
import tempfile
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

REQUIRED = {   # minimum to run: name matching, cost roll-up, structural checks
    "bom": ["product_id", "eng_part_id", "qty_per_unit"],
    "parts_master": ["eng_part_id", "description"],
    "spend_cube": ["sku", "sku_description", "unit_price"],
}
OPTIONAL = {   # each one unlocks more analysis
    "products": ["product_id", "product_name"],
    "production": ["product_id", "month", "units_built"],
    "receipts": ["sku", "month", "qty_received"],
    "posted_consumption": ["sku", "month", "qty_posted"],
    "physical_counts": ["sku", "count_date", "counted_on_hand"],
    "supplier_lead_times": ["sku", "lead_time_months"],
    "ground_truth": ["sku", "true_eng_part_id"],
}
UNLOCKS = {
    "production + receipts (12+ months)": "behavior matching (hidden links), stability ranges",
    "physical_counts (2+ count dates)": "count-to-count reconciliation: usage shifts, excess usage, inventory builds",
    "posted_consumption": "backflush (circularity) check on ERP postings",
    "spend_cube.annual_qty or receipts": "spend amounts, true product cost, price gaps",
    "ground_truth": "Proof view scored on your own data",
}

LANGUAGE_RULES = """Rules: call unverified links "proposed"; findings are "investigation leads", never root causes;
show money as the given P10-P90 range; say "spend to investigate" or "spend newly attributed to BOMs", never "savings"
(price gaps are "benchmark-based, not guaranteed"); be concise and specific; do not invent numbers."""


class UploadError(ValueError):
    pass


def demo_bundle() -> dict:
    p = ROOT / "app" / "bundle.json"
    if not p.exists():
        build_demo()
    return json.loads(p.read_text())


def build_demo():
    from recon.generate import generate
    from recon.pipeline import run
    if not (ROOT / "data" / "A" / "spend_cube.csv").exists():
        generate("A", 7, ROOT / "data" / "A")
        generate("B", 2027, ROOT / "data" / "B")
    run(ROOT / "data" / "A", ROOT / "data" / "B", ROOT / "app" / "bundle.json")


def _table_name(filename: str):
    stem = Path(filename).stem.lower().replace(" ", "_").replace("-", "_")
    for t in list(REQUIRED) + list(OPTIONAL):
        if stem == t or stem.startswith(t):
            return t
    return None


def validate(files: dict[str, bytes]) -> list[str]:
    """files: {filename: bytes}. Returns human-readable problems (empty = OK). Common alternative column
    names (e.g. part_number, vendor, price) are accepted and renamed automatically."""
    from recon.core import canon
    found = {}
    for n in files:
        t = _table_name(n)
        if t:
            found[t] = n
    problems = []
    for t, cols in {**REQUIRED, **OPTIONAL}.items():
        if t not in found:
            if t in REQUIRED:
                problems.append(f"Missing {t}.csv (needs columns: {', '.join(cols)})")
            continue
        try:
            df, _ = canon(pd.read_csv(io.BytesIO(files[found[t]])), t)
        except Exception as e:  # noqa: BLE001
            problems.append(f"{t}.csv could not be read as CSV: {e}")
            continue
        miss = [c for c in cols if c not in df.columns]
        if miss:
            problems.append(f"{t}.csv is missing column(s): {', '.join(miss)} (found: {', '.join(map(str, df.columns))})")
        elif len(df) == 0:
            problems.append(f"{t}.csv has no rows")
    return problems


def run_upload(files: dict[str, bytes], company: str = "Uploaded company") -> dict:
    problems = validate(files)
    if problems:
        raise UploadError("; ".join(problems))
    from recon.pipeline import run
    ref = demo_bundle()
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        for n, b in files.items():
            t = _table_name(n)
            if t:
                (d / f"{t}.csv").write_bytes(b)
        (d / "company.txt").write_text(company)
        out = d / "bundle.json"
        run(d, ROOT / "data" / "B", out, reference=ref)
        bundle = json.loads(out.read_text())
    (ROOT / "app" / "bundle_uploaded.json").write_text(json.dumps(bundle))
    return bundle


# ---------------------------------------------------------------- PDF
def _pdf(title: str, blocks: list) -> bytes:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    ss = getSampleStyleSheet()
    H = ParagraphStyle("h", parent=ss["Title"], fontSize=16, alignment=0, spaceAfter=4)
    H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11, textColor=colors.HexColor("#2350d8"), spaceBefore=8)
    B = ParagraphStyle("b", parent=ss["Normal"], fontSize=9.5, leading=12.5)
    S = ParagraphStyle("s", parent=B, fontSize=8, textColor=colors.HexColor("#5b6678"))
    esc = lambda t: str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    story = [Paragraph(esc(title), H)]
    for kind, val in blocks:
        if kind == "h":
            story.append(Paragraph(esc(val), H2))
        elif kind == "p":
            story.append(Paragraph(esc(val), B))
        elif kind == "small":
            story.append(Paragraph(esc(val), S))
        elif kind == "table":
            t = Table([[Paragraph(esc(c), B) for c in r] for r in val], colWidths=[2.0 * inch, 5.0 * inch])
            t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dde2ea")),
                                   ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                   ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f4f6f9"))]))
            story.append(t)
        story.append(Spacer(1, 4))
    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=letter, leftMargin=0.7 * inch, rightMargin=0.7 * inch,
                      topMargin=0.6 * inch, bottomMargin=0.6 * inch, title=title).build(story)
    return buf.getvalue()


def _money(r):
    f = lambda v: f"${v/1000:,.1f}K" if abs(v) >= 1000 else f"${v:,.0f}"
    return f"{f(r['p10'])}-{f(r['p90'])}"


def pdf_onepager(b: dict) -> bytes:
    lines = b["onepager"].splitlines()
    blocks = []
    for ln in lines[1:]:
        if not ln.strip():
            continue
        if ln.isupper():
            blocks.append(("h", ln.title()))
        elif ln.startswith("All figures"):
            blocks.append(("small", ln))
        else:
            blocks.append(("p", ln))
    return _pdf(lines[0].title(), blocks)


def pdf_passport(b: dict, sku: str) -> bytes:
    l = next((x for x in b["links"] if x["sku"] == sku), None)
    if l is None:
        raise KeyError(sku)
    bp = l.get("band_precision")
    conf = (f"{l['band']}: {bp[0]:.0%} correct (95% CI {bp[1]:.0%}-{bp[2]:.0%}, n={bp[3]})" if bp else "n/a")
    rows = [["Purchasing record", f"{sku} - \"{l['sku_description']}\" - {l['supplier']} - ${l['price']:.3f}"],
            ["Proposed engineering part", f"{l['part']} - {l['part_description']}" if l["part"] else "None: unresolved"],
            ["Status", l["state"]], ["Decision route", l["route"]], ["Evidence class", l["evidence_class"]],
            ["Name signal", f"{l['name']:.2f} (best name candidate {l['best_name_part']} {l['best_name_score']:.2f})"],
            ["Behavior signal", "n/a" if l["behavior"] is None else f"{l['behavior']:.2f}"
             + (f" - volume share {l['share']:.0%}" if l.get("share") is not None else "")],
            ["Quantities used", f"Matched on Purchased + name. Posted: {l.get('backflush') or 'n/a'}."],
            ["Confidence", conf], ["Stability", f"{l['stability_txt']} bootstrap resamples keep this decision"],
            ["Spend involved (12 mo)", _money(l["spend12_rng"])],
            ["Look-alike group", ", ".join(l["twin_group"]) or "None"],
            ["Warnings", "; ".join(l["warnings"]) or "None"],
            ["Required approval", "Engineering confirms spec/drawing; Procurement updates cross-reference"
             if l["part"] else "Procurement classifies (indirect spend or manual map)"],
            ["Sign-off", "Engineering: ____________   Procurement: ____________   Date: ________"]]
    return _pdf(f"Evidence Passport - {sku}" + (f" -> {l['part']}" if l["part"] else ""),
                [("small", f"{b['company']} - OmniPart RECON"), ("table", rows),
                 ("small", "Scores are model-fit measures, not probabilities. Confidence bands are calibrated on "
                           "labeled data and must be recalibrated in a pilot.")])


# ---------------------------------------------------------------- AI drafting
def draft(card: dict, audience: str) -> dict:
    """Draft a message with Claude if ANTHROPIC_API_KEY is set; otherwise return the template."""
    template = card["messages"].get(audience) or next(iter(card["messages"].values()))
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return dict(text=template, source="template (set ANTHROPIC_API_KEY to enable AI drafting)")
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        facts = {k: card[k] for k in ("title", "subtitle", "owners", "next_action", "evidence", "confidence")}
        facts["spend_range_p10_p90"] = _money(card["rng"])
        msg = client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"), max_tokens=1200,
            system="You draft short workplace messages for a manufacturing cost/procurement team. " + LANGUAGE_RULES,
            messages=[{"role": "user", "content":
                       f"Write a message for the {audience} audience about this finding. Use only these facts:\n"
                       f"{json.dumps(facts, indent=1)}\n\nStart with a subject line if it is an email."}])
        text = "\n".join(getattr(b, "text", "") for b in (msg.content or []) if getattr(b, "type", "") == "text").strip()
        if not text:
            kinds = ", ".join(getattr(b, "type", "?") for b in (msg.content or [])) or "empty"
            return dict(text=template, source=f"template (AI reply had no text: {kinds}; stop_reason={getattr(msg, 'stop_reason', '?')})")
        return dict(text=text, source=f"AI draft by {getattr(msg, 'model', 'Claude')} (review before sending)")
    except Exception as e:  # noqa: BLE001
        return dict(text=template, source=f"template (AI drafting failed: {type(e).__name__}: {str(e)[:160]})")
