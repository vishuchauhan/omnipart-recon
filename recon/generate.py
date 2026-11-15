"""
Synthetic data generator for OmniPart RECON.

Two fictional companies share the same engineering structure but differ in random seed,
production patterns and - critically - purchasing description vocabulary:
  A = "Prairie Motion Systems"   (development set: thresholds and abbreviations tuned here)
  B = "Lakeshore Drives"         (blind test set: new vocabulary, frozen settings)

Physical model per purchasing SKU, per month:
  true consumption  = BOM qty x units built x reality multiplier (ECO, scrap) x supplier share
  receipts          = replenishment to a stock-cover target, rounded to lot sizes (lumpy)
  physical stock    = stock + receipts - true consumption
  posted (ERP)      = BACKFLUSH: BOM qty x units built x share  (most SKUs - circular!)
                      or real goods issues for a few SKUs
  counts            = physical stock at quarter ends (+ small count error)

Planted cases (see planted_cases.csv) - 11 in total.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

MONTHS = [str(p) for p in pd.period_range("2025-01", periods=24, freq="M")]
T = len(MONTHS)
ECO_MONTH = MONTHS.index("2026-03")
COUNT_MONTHS = [2, 5, 8, 11, 14, 17, 20, 23]          # quarter-end physical counts

PRODUCTS = [("P1", "Conveyor Drive CD-100", "Material handling"), ("P2", "Lift Actuator LA-40", "Motion"),
            ("P3", "Pump Module PM-7", "Fluid"), ("P4", "Gearbox GX-220", "Power transmission"),
            ("P5", "Robotic Joint RJ-5", "Motion"), ("P6", "Fan Unit FU-30", "Air")]

PARTS = [  # eng_part_id, description, category, spec, std_cost, status
    ("FST-0101", "Hex bolt M6 x 20 zinc plated steel", "Fastener", "M6;20mm;steel;zinc", 0.11, "active"),
    ("FST-0107", "Hex bolt M6 x 25 zinc plated steel", "Fastener", "M6;25mm;steel;zinc", 0.12, "active"),
    ("FST-0102", "Hex bolt M8 x 30 zinc plated steel", "Fastener", "M8;30mm;steel;zinc", 0.19, "active"),
    ("FST-0103", "Socket head cap screw M5 x 16 stainless", "Fastener", "M5;16mm;stainless", 0.14, "active"),
    ("FST-0104", "Nylon insert lock nut M6 zinc", "Fastener", "M6;nylon;zinc", 0.07, "active"),
    ("FST-0105", "Flat washer M8 steel", "Fastener", "M8;steel", 0.03, "active"),
    ("FST-0106", "Self tapping screw 4.2 x 13 pan head", "Fastener", "4.2mm;13mm;pan", 0.04, "active"),
    ("BRG-0201", "Deep groove ball bearing 6204 2RS", "Bearing", "6204;2RS", 2.40, "active"),
    ("BRG-0202", "Deep groove ball bearing 6006 ZZ", "Bearing", "6006;ZZ", 2.90, "active"),
    ("BRG-0203", "Needle roller bearing HK1012", "Bearing", "HK1012", 1.75, "active"),
    ("SEL-0301", "Gasket EPDM 40mm flange", "Seal", "EPDM;40mm;gasket", 0.55, "active"),
    ("SEL-0302", "O-ring EPDM 40mm x 3mm", "Seal", "EPDM;40mm;o-ring", 0.22, "active"),
    ("SEL-0303", "Radial shaft seal 25x40x7 NBR", "Seal", "25x40x7;NBR", 0.95, "active"),
    ("ELE-0401", "Wire harness 12 pin main", "Electrical", "12pin;harness", 6.80, "active"),
    ("ELE-0402", "Limit switch roller lever IP67", "Electrical", "IP67;limit switch", 4.10, "active"),
    ("ELE-0403", "Encoder incremental 1024 ppr", "Electrical", "1024ppr;encoder", 38.00, "active"),
    ("ELE-0404", "Power supply 24V 5A DIN rail", "Electrical", "24V;5A;PSU", 22.50, "active"),
    ("ASSEMBLY-204", "Brushless DC motor 24V 200W with integrated driver", "Motor", "24V;200W;BLDC", 74.00, "active"),
    ("MOT-0502", "Stepper motor NEMA 23 2.8A", "Motor", "NEMA23;2.8A", 31.00, "active"),
    ("MOT-0503", "AC induction motor 0.37kW 4 pole", "Motor", "0.37kW;4pole", 88.00, "active"),
    ("STR-0601", "Housing aluminium die cast GX", "Structure", "aluminium;die cast", 18.40, "active"),
    ("STR-0602", "Mounting bracket steel 3mm laser cut", "Structure", "steel;3mm;bracket", 3.20, "active"),
    ("STR-0605", "Mounting bracket steel 3mm laser cut type B", "Structure", "steel;3mm;bracket;type B", 3.45, "active"),
    ("STR-0603", "Base plate steel 8mm", "Structure", "steel;8mm;plate", 9.70, "active"),
    ("STR-0604", "Cover sheet ABS molded", "Structure", "ABS;cover", 2.60, "active"),
    ("PWR-0701", "Spur gear 32T module 1.5 steel", "Powertrain", "32T;m1.5;steel", 5.10, "active"),
    ("PWR-0702", "Helical gear 48T module 2", "Powertrain", "48T;m2", 8.90, "active"),
    ("PWR-0703", "Drive shaft 20mm hardened", "Powertrain", "20mm;shaft", 6.40, "active"),
    ("PWR-0704", "Timing belt HTD 5M 450mm", "Powertrain", "HTD5M;450mm;belt", 4.30, "active"),
    ("PWR-0705", "Coupling jaw 14mm to 20mm", "Powertrain", "14mm;20mm;coupling", 7.20, "active"),
    ("PMP-0801", "Impeller cast iron 120mm", "Pump", "120mm;cast iron;impeller", 14.50, "active"),
    ("PMP-0802", "Pump volute housing", "Pump", "volute;housing", 21.00, "active"),
    ("FAN-0901", "Fan blade 300mm polypropylene", "Fan", "300mm;PP;blade", 3.90, "active"),
    ("FAN-0902", "Fan guard wire 300mm", "Fan", "300mm;guard", 2.10, "active"),
    ("OBS-9901", "Legacy terminal block 6 way (superseded)", "Electrical", "6way;terminal", 1.30, "obsolete"),
]

BOM = [
    ("P1", "FST-0101", 8), ("P1", "FST-0104", 8), ("P1", "BRG-0201", 2), ("P1", "SEL-0303", 2),
    ("P1", "MOT-0503", 1), ("P1", "PWR-0703", 1), ("P1", "PWR-0705", 1), ("P1", "STR-0603", 1),
    ("P1", "STR-0602", 2), ("P1", "ELE-0401", 1),
    ("P2", "ASSEMBLY-204", 1), ("P2", "FST-0103", 6), ("P2", "BRG-0203", 2), ("P2", "PWR-0701", 2),
    ("P2", "ELE-0402", 2), ("P2", "STR-0604", 1), ("P2", "FST-0106", 4),
    ("P3", "PMP-0801", 1), ("P3", "PMP-0802", 1), ("P3", "SEL-0301", 1), ("P3", "SEL-0302", 1),
    ("P3", "BRG-0202", 1), ("P3", "FST-0102", 6), ("P3", "FST-0105", 6), ("P3", "ELE-0404", 1), ("P3", "STR-0605", 2),
    ("P4", "PWR-0702", 2), ("P4", "PWR-0701", 3), ("P4", "STR-0601", 1), ("P4", "BRG-0201", 4),
    ("P4", "SEL-0303", 2), ("P4", "FST-0102", 10), ("P4", "FST-0105", 10), ("P4", "FST-0107", 6),
    ("P5", "ASSEMBLY-204", 2), ("P5", "ELE-0403", 1), ("P5", "PWR-0704", 1), ("P5", "BRG-0203", 2),
    ("P5", "FST-0103", 8), ("P5", "MOT-0502", 1), ("P5", "ELE-0401", 1), ("P5", "FST-0107", 4),
    ("P6", "FAN-0901", 1), ("P6", "FAN-0902", 1), ("P6", "FST-0106", 6), ("P6", "MOT-0503", 1),
    ("P6", "STR-0602", 2), ("P6", "FST-0101", 4), ("P6", "FST-0104", 4),
]

# sku, supplier, description (company A), description (company B, new vocabulary), truth, share, price
SKUS = [
    ("ACM-55120", "Acme Fasteners", "HX BLT M6X20 ZP", "SCREW HEX HD 6X20 ZINC", "FST-0101", 0.60, 0.105),
    ("GLB-M6-20", "Global Industrial", "BOLT HEX 6MM 20L ZINC", "HHCS M6-1.0 X 20 ZN", "FST-0101", 0.40, 0.142),
    ("ACM-55125", "Acme Fasteners", "HX BLT M6X25 ZP", "SCREW HEX HD 6X25 ZINC", "FST-0107", 1.00, 0.118),
    ("ACM-55180", "Acme Fasteners", "HX BLT M8X30 ZP", "HHCS M8 X 30 ZINC PL", "FST-0102", 1.00, 0.185),
    ("PRC-3316", "Precision Supply", "SHCS M5-16 SS", "CAP SCR SKT M5X16 A2", "FST-0103", 1.00, 0.150),
    ("ACM-61006", "Acme Fasteners", "NYLOC NUT M6 ZN", "LOCKNUT NYLON M6 ZINC", "FST-0104", 0.95, 0.068),
    ("QNC-M6NL", "QuickNut Co", "LOCKNUT M6 NYL INS", "NUT M6 PREVAILING TORQUE", "FST-0104", 0.05, 0.081),
    ("GLB-W8", "Global Industrial", "WASHER FLAT 8MM STL", "FLAT WSHR M8 PLAIN", "FST-0105", 1.00, 0.031),
    ("FAS-4213", "FastTrack", "SCR ST 4.2X13 PH", "TAPPING SCREW PAN 4.2-13", "FST-0106", 1.00, 0.043),
    ("SKF-6204", "BearingWorld", "BRG 6204-2RS1", "BEARING 6204 RS SEALED", "BRG-0201", 0.70, 2.35),
    ("NTN-6204L", "Midwest Bearings", "BALL BEARING 20X47X14 SEALED", "ROLLING ELEMENT 20-47-14 LL", "BRG-0201", 0.30, 2.78),
    ("SKF-6006", "BearingWorld", "BRG 6006-ZZ", "BEARING 6006 SHIELDED", "BRG-0202", 1.00, 2.95),
    ("INA-HK10", "BearingWorld", "NEEDLE BRG HK 1012", "DRAWN CUP NEEDLE HK1012", "BRG-0203", 1.00, 1.70),
    ("SLX-4040G", "SealTech", "GSKT EPDM FLG 40", "FLANGE GASKET 40 EPDM", "SEL-0301", 1.00, 0.58),
    ("SLX-4040R", "SealTech", "ORING EPDM 40X3", "O RING 40 X 3 EPDM70", "SEL-0302", 1.00, 0.21),
    ("SLX-2547", "SealTech", "OIL SEAL 25-40-7 NBR", "LIP SEAL 25X40X7 NITRILE", "SEL-0303", 1.00, 0.92),
    ("WRX-12M", "WireRex", "HARNESS ASSY 12P MAIN", "CABLE LOOM 12 POS MAIN", "ELE-0401", 1.00, 6.95),
    ("ITEM-4471B", "Omron Dist", "ITEM 4471-B PER DWG REV C", "PN 88-2210 SEE DRAWING", "ELE-0402", 1.00, 4.05),
    ("ENC-1024", "SensorHub", "ROTARY ENCODER 1024", "INCR ENCODER 1024 PULSE", "ELE-0403", 1.00, 39.50),
    ("PSU-245D", "PowerCo", "PSU 24V 5A DIN", "POWER SUPPLY DIN 24VDC 5A", "ELE-0404", 1.00, 22.10),
    ("MTR-X72", "Dynamo Asia", "DRV UNIT SER-B KIT", "MOD KIT 7781 SERIES B", "ASSEMBLY-204", 1.00, 71.50),
    ("STP-23-28", "MotionParts", "STEPPER NEMA23 2.8A", "STEP MTR 23 FRAME 2.8 AMP", "MOT-0502", 1.00, 30.40),
    ("ACM-037-4", "Acme Motors", "MOTOR AC 0.37KW 4P", "INDUCTION MTR 370W 4 POLE", "MOT-0503", 1.00, 86.00),
    ("CST-GX-H", "CastCo", "CSTG HSG AL GX", "DIE CAST HOUSING GX ALU", "STR-0601", 1.00, 18.90),
    ("LZR-B3", "LaserFab", "BRKT LC 3MM", "BRACKET 3MM LASER", "STR-0602", 1.00, 3.10),
    ("LZR-B3B", "LaserFab", "BRKT LC 3MM TYPE B", "BRACKET 3MM LASER VAR B", "STR-0605", 1.00, 3.40),
    ("LZR-P8", "LaserFab", "PLATE 8MM STL BASE", "BASEPLATE 8 MM STEEL", "STR-0603", 1.00, 9.40),
    ("MLD-CV1", "MoldRight", "COVER ABS", "MOLDED COVER ABS GREY", "STR-0604", 1.00, 2.70),
    ("GR-32-15", "GearMasters", "GEAR SPUR 32T M1.5", "SPUR 32 TEETH MOD 1.5", "PWR-0701", 1.00, 5.00),
    ("GR-48-2H", "GearMasters", "GEAR HEL 48T M2", "HELICAL 48T MOD2", "PWR-0702", 1.00, 9.10),
    ("SH-20H", "ShaftPro", "SHAFT 20 HRD", "SHAFT DIA 20 HARDENED", "PWR-0703", 1.00, 6.20),
    ("BLT-5M450", "BeltCo", "BELT HTD 450-5M", "SYNCHRO BELT 5M 450", "PWR-0704", 1.00, 4.40),
    ("CPL-1420", "MotionParts", "JAW CPLG 14-20", "FLEX COUPLING 14X20 JAW", "PWR-0705", 1.00, 7.00),
    ("IMP-120CI", "PumpParts Inc", "IMPLR 120 CI", "IMPELLER 120 GREY IRON", "PMP-0801", 1.00, 14.20),
    ("VOL-PM7", "PumpParts Inc", "VOLUTE PM7", "PUMP CASING PM7", "PMP-0802", 1.00, 21.50),
    ("FB-300PP", "AirFlow", "BLADE FAN 300 PP", "IMPELLER FAN 300 PLASTIC", "FAN-0901", 1.00, 3.80),
    ("FG-300W", "AirFlow", "GUARD 300 WIRE", "FINGER GUARD 300", "FAN-0902", 1.00, 2.00),
    ("TB-6W-L", "Omron Dist", "TERM BLK 6W", "TERMINAL STRIP 6 POS", "OBS-9901", 1.00, 1.35),
    ("CLT-S20", "ShopSupply", "COOLANT SYNTH 20L", "CUTTING FLUID 20L PAIL", "NONE", 1.00, 48.00),
    ("GLV-NL100", "ShopSupply", "GLOVE NITRILE L BX100", "NITRILE GLOVES LARGE 100CT", "NONE", 1.00, 11.50),
]
REAL_ISSUE_POSTING = {"ENC-1024", "PSU-245D", "WRX-12M", "LZR-P8"}   # posted = real goods issues

PLANTED = [
    (1, "HIDDEN_LINK", "MTR-X72", "ASSEMBLY-204", "Description shares nothing with the engineering part; unique demand footprint (P2x1 + P5x2)"),
    (2, "HIDDEN_LINK_BACKUP", "NTN-6204L", "BRG-0201", "Backup supplier bearing with dimension-style description, ~30% share"),
    (3, "DRAWING_NUMBER_ONLY", "ITEM-4471B", "ELE-0402", "Description is only a drawing reference"),
    (4, "SIGNATURE_TWINS", "SLX-4040G / SLX-4040R", "SEL-0301 / SEL-0302", "Both 1x only in P3: identical demand signature; name/spec must break the tie"),
    (5, "ECO_CHANGE", "GR-48-2H", "PWR-0702", "BOM 2/unit in P4; physically 3/unit from 2026-03; hidden by backflush, visible in counts"),
    (6, "PERSISTENT_EXCESS", "SKF-6204 + NTN-6204L", "BRG-0201", "~7% physical over-consumption; backflush hides it, counts reveal it"),
    (7, "OBSOLETE", "TB-6W-L", "OBS-9901", "Still purchased; part in no active BOM"),
    (8, "INVENTORY_BUILD", "GR-32-15", "PWR-0701", "One-off 9,000-unit buy in 2025-10; explained by counted stock; must NOT be flagged"),
    (9, "FALSE_POSITIVE_TRAP", "CLT-S20 / GLV-NL100", "NONE", "Indirect spend (coolant tracks total output; gloves flat); must stay unresolved"),
    (10, "BACKFLUSH", "most SKUs", "-", "Posted consumption = BOM x production exactly (circular) except ENC-1024, PSU-245D, WRX-12M, LZR-P8"),
    (11, "LOW_SHARE_BACKUP", "QNC-M6NL", "FST-0104", "5% backup supplier: plausible, must not be rejected"),
    (12, "NEAR_DUPLICATES", "FST-0101/FST-0107, STR-0602/STR-0605", "-", "Track 1 standardization candidates"),
]


def _series(rng, base, trend, amp, phase, noise, launch=None, eol=None):
    t = np.arange(T)
    v = base * (1 + trend * t / T) * (1 + amp * np.sin(2 * np.pi * (t + phase) / 12))
    v = v * (1 + rng.normal(0, noise, T))
    if launch:
        v[:launch] *= np.linspace(0.1, 1, launch)
    if eol:
        v[eol:] *= np.linspace(1, 0.35, T - eol)
    return np.maximum(np.round(v), 0)


def generate(company: str = "A", seed: int = 7, out: Path | str = "data/A") -> Path:
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    name = "Prairie Motion Systems" if company == "A" else "Lakeshore Drives"
    jitter = 0 if company == "A" else 1
    vols = {
        "P1": _series(rng, 900, 0.30, 0.15, 0 + jitter, 0.06),
        "P2": _series(rng, 400, -0.20, 0.25, 3 + jitter, 0.08),
        "P3": _series(rng, 650, 0.10, 0.05, 6, 0.10, launch=6),
        "P4": _series(rng, 300, 0.50, 0.20, 9 - jitter, 0.07),
        "P5": _series(rng, 220, 0.00, 0.35, 1, 0.09),
        "P6": _series(rng, 500, 0.05, 0.10, 4 + jitter, 0.08, eol=16),
    }
    products = pd.DataFrame(PRODUCTS, columns=["product_id", "product_name", "product_family"])
    production = pd.DataFrame([(p, MONTHS[i], int(vols[p][i])) for p in vols for i in range(T)],
                              columns=["product_id", "month", "units_built"])
    parts = pd.DataFrame(PARTS, columns=["eng_part_id", "description", "category", "spec", "std_cost", "status"])
    bom = pd.DataFrame(BOM, columns=["product_id", "eng_part_id", "qty_per_unit"])
    bom["effective_from"] = MONTHS[0]; bom["effective_to"] = None

    def expected(part):
        e = np.zeros(T)
        for _, r in bom[bom.eng_part_id == part].iterrows():
            e += r.qty_per_unit * vols[r.product_id]
        return e

    def physical(part):
        c = np.zeros(T)
        for _, r in bom[bom.eng_part_id == part].iterrows():
            q = np.full(T, float(r.qty_per_unit))
            if part == "PWR-0702" and r.product_id == "P4":
                q[ECO_MONTH:] = 3.0                         # undocumented engineering change
            mult = 1.07 if part == "BRG-0201" else 1.005    # 7% excess scrap / normal handling
            c += q * mult * vols[r.product_id]
        return c

    total_units = sum(vols.values())
    spend, rec, post, cnt, lt, truth = [], [], [], [], [], []
    for sku, sup, dA, dB, part, share, price in SKUS:
        desc = dA if company == "A" else dB
        if part == "NONE":
            cons = total_units * 0.012 if sku.startswith("CLT") else np.maximum(140 + 25 * rng.normal(0, 1, T), 0)
            exp_post = cons
        elif part == "OBS-9901":
            cons = np.zeros(T); exp_post = cons
        else:
            cons = physical(part) * share
            exp_post = expected(part) * share
        cons = np.round(cons * (1 + rng.normal(0, 0.012, T)))
        lot = max(round(np.mean(cons) * 0.25 / 10) * 10, 10) if cons.sum() else 50
        stock = cons[0] * 0.6
        opening = stock
        r_arr, s_arr = np.zeros(T), np.zeros(T)
        for i in range(T):
            need = cons[i] + 0.5 * cons[min(i + 1, T - 1)] - stock
            r = max(need, 0) * (1 + rng.normal(0, 0.05))
            r = np.ceil(r / lot) * lot if r > 0 else 0
            if sku == "TB-6W-L":
                r = 400 + round(rng.normal(0, 30))
            if sku == "GR-32-15" and i == 9:
                r += 9000                                  # one-off bulk buy (price lock)
            stock = stock + r - cons[i]
            if stock < 0:
                r -= stock; stock = 0
            r_arr[i], s_arr[i] = r, stock
        if sku in REAL_ISSUE_POSTING or part in ("NONE", "OBS-9901"):
            posted = np.round(cons * (1 + rng.normal(0, 0.01, T)))
        else:
            posted = np.round(exp_post)                    # backflush: BOM x production
        spend.append((sku, sup, desc, price, "USD"))
        for i in range(T):
            rec.append((sku, MONTHS[i], int(r_arr[i])))
            post.append((sku, MONTHS[i], int(posted[i])))
        cnt.append((sku, "2024-12", int(round(opening))))
        for i in COUNT_MONTHS:
            cnt.append((sku, MONTHS[i], int(round(s_arr[i] * (1 + rng.normal(0, 0.003))))))
        lt.append((sku, int(rng.integers(1, 3))))
        truth.append((sku, part))

    tables = {
        "products": products, "production": production, "parts_master": parts, "bom": bom,
        "spend_cube": pd.DataFrame(spend, columns=["sku", "supplier", "sku_description", "unit_price", "currency"]),
        "receipts": pd.DataFrame(rec, columns=["sku", "month", "qty_received"]),
        "posted_consumption": pd.DataFrame(post, columns=["sku", "month", "qty_posted"]),
        "physical_counts": pd.DataFrame(cnt, columns=["sku", "count_date", "counted_on_hand"]),
        "supplier_lead_times": pd.DataFrame(lt, columns=["sku", "lead_time_months"]),
        "ground_truth": pd.DataFrame(truth, columns=["sku", "true_eng_part_id"]),
        "planted_cases": pd.DataFrame(PLANTED, columns=["case_no", "case", "sku", "eng_part_id", "description"]),
    }
    for k, v in tables.items():
        v.to_csv(out / f"{k}.csv", index=False)
    (out / "company.txt").write_text(name)
    return out


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent / "data"
    print(generate("A", 7, base / "A"))
    print(generate("B", 2027, base / "B"))
