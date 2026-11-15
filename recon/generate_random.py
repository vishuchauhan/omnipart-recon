"""
Company C - "Northfield Pumps & Controls": a procedurally generated, UNFAMILIAR manufacturer for blind testing.
Nothing here was seen while tuning: different products, part catalog, BOM structure, supplier naming style,
noisier production and purchasing. Thresholds stay frozen.

Description styles for purchasing records (chosen at random per SKU):
  clean     - close to the engineering text            ("BOLT HEX M8 X 40 ZINC")
  abbrev    - heavy abbreviation, some NOT in the dictionary ("HXHD CAP SCR M8-40 Z/P")
  reordered - supplier word order and units              ("40MM M8 HEX SCREW, ZINC PLATED")
  opaque    - supplier code only (needs behavior)        ("KIT 7731-C SER 4")
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

MONTHS = [str(p) for p in pd.period_range("2025-01", periods=24, freq="M")]
T = len(MONTHS)
COUNT_MONTHS = [2, 5, 8, 11, 14, 17, 20, 23]


def _catalog(rng):
    """(id, engineering description, category, spec, cost, [purchase description variants: clean, abbrev, reordered])"""
    out = []
    for d, l in [(5, 16), (6, 30), (8, 25), (8, 40), (10, 35), (12, 50)]:
        out.append((f"F-{d:02d}{l:03d}", f"Hex head cap screw M{d} x {l} zinc plated", "Fastener", f"M{d};{l}mm;zinc", 0.04 * d + 0.002 * l,
                    [f"CAP SCREW HEX M{d} X {l} ZINC", f"HXHD CAP SCR M{d}-{l} Z/P", f"{l}MM M{d} HEX SCREW, ZINC PLATED"]))
    for d in (6, 8, 10):
        out.append((f"N-{d:02d}0", f"Hex nut M{d} class 8 zinc", "Fastener", f"M{d};nut;class 8", 0.02 * d,
                    [f"HEX NUT M{d} CL8 ZINC", f"NUT HX M{d} 8.8 ZP", f"M{d} NUT HEX GRADE 8"]))
    for code, bore in [("6202", 15), ("6205", 25), ("6301", 12), ("NK2520", 25)]:
        out.append((f"B-{code}", f"Ball bearing {code} sealed bore {bore}mm", "Bearing", f"{code};{bore}mm;sealed", 1.5 + bore * 0.08,
                    [f"BEARING {code} 2RS", f"BRG {code}-2RSH", f"{bore}MM BORE BRG SEALED {code}"]))
    for d in (32, 48, 65):
        out.append((f"S-OR{d}", f"O-ring NBR {d}mm x 3.5mm", "Seal", f"NBR;{d}mm;o-ring", 0.18 + d * 0.004,
                    [f"O-RING NBR {d}X3.5", f"ORNG NITRILE {d}-3.5", f"{d} X 3.5 NITRILE RING"]))
    for name, spec, cost, alts in [
        ("Pressure sensor 0-10 bar 4-20mA", "0-10bar;4-20mA;sensor", 34.0, ["PRESSURE TRANSMITTER 10BAR", "PRESS XMTR 0-10B 4-20", "4-20MA SENSOR 10 BAR"]),
        ("Flow switch paddle 1 inch", "1in;flow switch", 27.0, ["FLOW SWITCH 1IN PADDLE", "FLW SW 1\" PDL", "PADDLE TYPE FLOW SW 1 INCH"]),
        ("Control board main PCB rev D", "PCB;rev D;controller", 61.0, ["MAIN CONTROLLER PCB D", "CTRL BRD MAIN RVD", "PCB ASSY CONTROL REV-D"]),
        ("Display module 4.3 inch touch", "4.3in;touch;display", 48.0, ["TOUCH DISPLAY 4.3IN", "DISP MOD 4.3 TCH", "4.3\" TOUCHSCREEN MODULE"]),
        ("Cable assembly 8 pin sensor", "8pin;cable", 5.6, ["SENSOR CABLE 8 PIN", "CBL ASSY 8P SNSR", "8-WAY SENSOR LEAD"]),
        ("Relay 24VDC 10A DIN", "24VDC;10A;relay", 8.9, ["RELAY 24VDC 10A DIN", "RLY 24V 10A DIN", "DIN RAIL RELAY 10 AMP 24V"]),
    ]:
        out.append((f"E-{4100 + len(out)}", name, "Electrical", spec, cost, alts))
    for kw, cost in [(0.55, 92.0), (1.1, 128.0), (2.2, 186.0)]:
        out.append((f"M-{int(kw*100):03d}", f"Induction motor {kw}kW IE3 2 pole", "Motor", f"{kw}kW;IE3;2pole", cost,
                    [f"MOTOR {kw}KW IE3 2P", f"MTR IND {kw}K IE3", f"{kw} KW 2-POLE IE3 MOTOR"]))
    for t, m in [(24, 1.5), (36, 2), (60, 2)]:
        out.append((f"G-{t}M{m}", f"Spur gear {t} teeth module {m}", "Powertrain", f"{t}T;m{m}", 3 + t * 0.12,
                    [f"SPUR GEAR {t}T MOD {m}", f"GR SPR {t}T M{m}", f"MOD {m} SPUR {t} TOOTH"]))
    for size in (40, 50, 65, 80):
        out.append((f"P-IMP{size}", f"Impeller bronze {size}mm closed", "Pump", f"{size}mm;bronze;impeller", 9 + size * 0.2,
                    [f"IMPELLER BRONZE {size}MM", f"IMPLR BRZ {size} CLSD", f"{size}MM CLOSED IMPELLER BRONZE"]))
        out.append((f"P-CAS{size}", f"Pump casing cast iron DN{size}", "Pump", f"DN{size};cast iron;casing", 18 + size * 0.3,
                    [f"PUMP CASING DN{size} CI", f"CSG PMP DN{size} GG25", f"DN{size} VOLUTE CAST IRON"]))
    for name, spec, cost in [("Mounting bracket 4mm steel powder coated", "4mm;steel;bracket", 4.1),
                             ("Base frame welded steel", "steel;frame;welded", 38.0),
                             ("Terminal box cover polyamide", "PA;cover", 3.3),
                             ("Mechanical seal 25mm carbon/ceramic", "25mm;carbon;mech seal", 22.0),
                             ("Shaft sleeve 25mm stainless", "25mm;stainless;sleeve", 7.4)]:
        w = name.upper().split()
        out.append((f"X-{7300 + len(out)}", name, "Structure" if "seal" not in name.lower() and "sleeve" not in name.lower() else "Seal",
                    spec, cost, [" ".join(w[:4]), " ".join(x[:4] for x in w[:4]), " ".join(reversed(w[:4]))]))
    return out


def generate_random(seed: int = 4242, out: Path | str = "data/C") -> Path:
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cat = _catalog(rng)
    ids = [c[0] for c in cat]
    by_id = {c[0]: c for c in cat}
    n_prod = 8
    products = pd.DataFrame(dict(product_id=[f"NP{i+1}" for i in range(n_prod)],
                                 product_name=[f"{n} {m}" for n, m in zip(
                                     ["Booster set", "Circulator", "Dosing pump", "End-suction pump", "Inline pump", "Sump pump", "Control cabinet", "Skid unit"],
                                     ["BX-2", "CR-9", "DP-1", "ES-65", "IL-40", "SP-80", "CC-3", "SK-7"])]))
    # BOMs: each product draws 9-14 parts with qty from a realistic set; overlaps create shared parts
    bom_rows = []
    for pid in products.product_id:
        k = int(rng.integers(9, 15))
        for j in rng.choice(ids, k, replace=False):
            cat_ = by_id[j][2]
            q = int(rng.choice([4, 6, 8, 12])) if cat_ == "Fastener" else int(rng.choice([1, 1, 1, 2]))
            bom_rows.append((pid, j, q))
    bom = pd.DataFrame(bom_rows, columns=["product_id", "eng_part_id", "qty_per_unit"]).drop_duplicates(["product_id", "eng_part_id"])
    used = sorted(bom.eng_part_id.unique())
    obsolete = [j for j in ids if j not in used][:2]
    parts = pd.DataFrame([(c[0], c[1], c[2], c[3], round(c[4], 3), "active" if c[0] in used else "obsolete")
                          for c in cat if c[0] in used or c[0] in obsolete],
                         columns=["eng_part_id", "description", "category", "spec", "std_cost", "status"])
    t = np.arange(T)
    vols = {}
    for pid in products.product_id:
        base = rng.uniform(80, 900); trend = rng.uniform(-0.3, 0.6); amp = rng.uniform(0.05, 0.35)
        ph = rng.uniform(0, 12); noise = rng.uniform(0.06, 0.14)
        v = base * (1 + trend * t / T) * (1 + amp * np.sin(2 * np.pi * (t + ph) / 12)) * (1 + rng.normal(0, noise, T))
        if rng.random() < 0.25:
            L = int(rng.integers(4, 9)); v[:L] *= np.linspace(0.15, 1, L)
        vols[pid] = np.maximum(np.round(v), 0)
    production = pd.DataFrame([(p, MONTHS[i], int(vols[p][i])) for p in vols for i in range(T)],
                              columns=["product_id", "month", "units_built"])
    B = bom.pivot_table(index="eng_part_id", columns="product_id", values="qty_per_unit", fill_value=0)
    expected = {j: sum(B.loc[j, p] * vols[p] for p in B.columns if B.loc[j, p] > 0) for j in used}

    suppliers = ["Rheinwerk GmbH", "Pacific Industrial", "Mercer Supply", "Hanseatic Parts", "Delta Components", "Keystone MRO"]
    style_p = [0.30, 0.35, 0.23, 0.12]   # clean, abbrev, reordered, opaque
    skus, truth, seg = [], [], {}
    n = 0
    for j in used + obsolete:
        c = by_id[j]
        shares = [1.0] if rng.random() > 0.25 else [round(float(rng.uniform(0.6, 0.85)), 2)]
        if len(shares) == 1 and shares[0] < 1:
            pass
        if shares[0] < 1:
            shares.append(round(1 - shares[0], 2))
        for k, sh in enumerate(shares):
            n += 1
            style = rng.choice(4, p=style_p)
            desc = (f"KIT {rng.integers(1000, 9999)}-{chr(65 + rng.integers(0, 6))} SER {rng.integers(1, 9)}"
                    if style == 3 else c[5][style])
            sku = f"{['RW','PI','MS','HP','DC','KM'][n % 6]}-{10000 + n * 37}"
            price = round(c[4] * float(rng.uniform(0.88, 1.18)), 3)
            skus.append((sku, suppliers[n % 6], desc, price, j, sh))
            truth.append((sku, j))
            seg[sku] = ("Opaque description" if style == 3 else "Obsolete part" if j in obsolete
                        else "Backup supplier" if k > 0 else "Standard")
    for desc, price in [("SAFETY GLASSES CLEAR", 4.2), ("SHOP TOWELS ROLL", 18.0), ("THREAD SEALANT 50ML", 9.5)]:
        n += 1
        sku = f"MRO-{5000 + n}"
        skus.append((sku, "Keystone MRO", desc, price, "NONE", 1.0)); truth.append((sku, "NONE")); seg[sku] = "Trap (not in any BOM)"

    rec, post, cnt, spend = [], [], [], []
    total_units = sum(vols.values())
    for sku, sup, desc, price, j, sh in skus:
        if j == "NONE":
            cons = (total_units * 0.01 if "SEALANT" in desc else np.maximum(60 + 15 * rng.normal(0, 1, T), 0))
            exp_post = cons
        elif j in obsolete:
            cons = np.zeros(T); exp_post = cons
        else:
            cons = expected[j] * sh * 1.005; exp_post = expected[j] * sh
        cons = np.round(cons * (1 + rng.normal(0, 0.02, T)))
        lot = max(round(np.mean(cons) * float(rng.uniform(0.2, 0.6)) / 10) * 10, 10) if cons.sum() else 50
        stock = cons[0] * 0.6; opening = stock
        r_arr, s_arr = np.zeros(T), np.zeros(T)
        for i in range(T):
            need = cons[i] + float(rng.uniform(0.3, 0.8)) * cons[min(i + 1, T - 1)] - stock
            r = max(need, 0) * (1 + rng.normal(0, 0.08))
            r = np.ceil(r / lot) * lot if r > 0 else 0
            if j in obsolete:
                r = 200 + round(rng.normal(0, 25))
            stock = stock + r - cons[i]
            if stock < 0:
                r -= stock; stock = 0
            r_arr[i], s_arr[i] = r, stock
        posted = np.round(exp_post) if j not in ("NONE",) and j not in obsolete else cons
        spend.append((sku, sup, desc, price, "USD"))
        for i in range(T):
            rec.append((sku, MONTHS[i], int(r_arr[i]))); post.append((sku, MONTHS[i], int(posted[i])))
        cnt.append((sku, "2024-12", int(round(opening))))
        for i in COUNT_MONTHS:
            cnt.append((sku, MONTHS[i], int(round(s_arr[i] * (1 + rng.normal(0, 0.004))))))
    pd.DataFrame(spend, columns=["sku", "supplier", "sku_description", "unit_price", "currency"]).to_csv(out / "spend_cube.csv", index=False)
    products.to_csv(out / "products.csv", index=False); production.to_csv(out / "production.csv", index=False)
    parts.to_csv(out / "parts_master.csv", index=False); bom.to_csv(out / "bom.csv", index=False)
    pd.DataFrame(rec, columns=["sku", "month", "qty_received"]).to_csv(out / "receipts.csv", index=False)
    pd.DataFrame(post, columns=["sku", "month", "qty_posted"]).to_csv(out / "posted_consumption.csv", index=False)
    pd.DataFrame(cnt, columns=["sku", "count_date", "counted_on_hand"]).to_csv(out / "physical_counts.csv", index=False)
    pd.DataFrame(truth, columns=["sku", "true_eng_part_id"]).to_csv(out / "ground_truth.csv", index=False)
    pd.DataFrame([(i + 1, s_, k, j, "") for i, ((k, j), s_) in enumerate(zip(truth, [seg[k] for k, _ in truth]))],
                 columns=["case_no", "segment", "sku", "eng_part_id", "description"]).to_csv(out / "planted_cases.csv", index=False)
    (out / "company.txt").write_text("Northfield Pumps & Controls")
    return out


if __name__ == "__main__":
    print(generate_random(4242, Path(__file__).resolve().parent.parent / "data" / "C"))
