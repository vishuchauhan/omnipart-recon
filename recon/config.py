"""Frozen configuration. Thresholds were tuned on company A (synthetic) and must be re-calibrated on real data."""
CONFIG = dict(
    window=3,                 # rolling window (months) for demand signatures
    min_windows=6,            # fewer usable windows -> behavior = null ("insufficient signal")
    min_nonzero_windows=3,
    use_lead_time_shift=False,  # receipts here are arrival-dated and planned against need -> lag ~0.
                                # Set True when receipts are PO-dated (shift by supplier lead time).
    share_min=0.02, share_max=1.25,           # individual SKU volume-share plausibility (backup suppliers allowed)
    group_share_min=0.85, group_share_max=1.25,  # all SKUs proposed for one part should cover its demand
    level_penalty_k=2.5,
    name_accept=0.55, name_margin=0.08,
    rescue_behavior=0.45, rescue_margin=0.15,
    w_name=0.45, w_behavior=0.55, combined_accept=0.55,
    twin_margin=0.05, twin_min_behavior=0.5, twin_min_name=0.4,
    backflush_corr=0.995, backflush_cv=0.01,
    shift_min=0.20, excess_ratio=1.04, excess_share=0.75,
    spike_multiple=3.0, build_explained=0.9,
    bootstrap_B=300, block=3, seed=11,
)

ABBREVIATIONS = {
    "hx": "hex", "blt": "bolt", "zp": "zinc", "zn": "zinc", "ss": "stainless", "shcs": "socket head cap screw",
    "nyloc": "nylon insert lock", "nyl": "nylon", "ins": "insert", "stl": "steel", "scr": "screw", "st": "self tapping",
    "ph": "pan head", "brg": "bearing", "gskt": "gasket", "flg": "flange", "oring": "o-ring", "assy": "assembly",
    "psu": "power supply", "cstg": "casting die cast", "hsg": "housing", "al": "aluminium", "brkt": "bracket",
    "lc": "laser cut", "hel": "helical", "hrd": "hardened", "cplg": "coupling", "implr": "impeller",
    "ci": "cast iron", "pp": "polypropylene", "term": "terminal", "blk": "block", "w": "way", "drv": "drive",
    "p": "pin", "4p": "4 pole", "12p": "12 pin",
}
