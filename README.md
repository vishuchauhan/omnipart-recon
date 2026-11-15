# OmniPart RECON

**Live demo:** open `index.html` from this repo's GitHub Pages site (link in the About box), or download `index.html` / `app/omnipart_recon.html` and double-click it. It runs fully offline in any browser, no install needed. Menu item **14 Presentation** holds the pitch deck; the other 13 views are the product.

Team Mavericks · UIUC × Kearney PERLab Hackathon 2026 · Corporate Track 3 (BOM Cost Linkage), with Track 1 (Standardization) built on the same data and Track 2 on the roadmap. All data is synthetic.

An evidence and reconciliation layer between engineering BOMs and procurement spend.
It **proposes** (never forces) links between purchasing SKUs and engineering parts using two
complementary signals (names/specs and purchase behavior against production-driven demand), puts a
range on every number, separates Purchased / Expected / Posted / Observed quantities, and routes every
finding to an owner with audience-specific messages.

## Run
```
python -m recon.generate     # synthetic companies A (development) and B (blind test) -> data/
python -m recon.pipeline     # full engine -> app/bundle.json   (~20 s, includes 300-resample bootstrap)
python app/build.py          # single self-contained web app -> app/omnipart_recon.html
python -m pytest -q tests    # 18 tests: formulas, planted cases, backflush, buckets, blind test, server, partial data, unfamiliar company
python run.py                # local FastAPI server + app at http://localhost:8000 (see START_HERE.md)
```
Needs numpy, pandas, scipy, scikit-learn. Open `app/omnipart_recon.html` in any browser.

## Layout
| Path | What |
|---|---|
| `recon/generate_random.py` | Company C: procedurally generated unfamiliar manufacturer for blind testing |
| `recon/generate.py` | Synthetic manufacturer (24 months, quarterly counts, backflushed postings, 12 planted cases) and blind company with new vocabulary |
| `recon/config.py` | Frozen thresholds (tuned on A) and abbreviation dictionary |
| `recon/core.py` | Four quantities, name signal, behavior signal, residual-demand volume check, greedy decision cascade, twins, backflush detector |
| `recon/analysis.py` | Wilson calibration, block-bootstrap stability, NNLS implied usage, count-to-count reconciliation, findings, inventory builds, mutually exclusive integrity buckets, true cost, price gaps, Track 1 standardization |
| `recon/workflow.py` | Resolution board cards, routing, message drafts per audience, client one-pager |
| `recon/benchmark.py` | Name-only vs behavior-only vs RECON, segments, stress grid |
| `recon/pipeline.py` | Runs everything, exports `app/bundle.json` |
| `app/template.html`, `app/build.py` | 13-view web app (vanilla JS, SVG charts, no build step) |

## Results on synthetic data
| | Name only | Behavior only | RECON |
|---|---|---|---|
| Company A (development) F1 | 0.94 | 0.41 | 1.00 |
| **Company B (blind, frozen settings)** F1 | 0.78 | 0.43 | **0.99** (0 false links, 1 left unresolved) |
| **Company C (unfamiliar: procedural products, BOMs, naming, lumpier purchasing)** F1 | 0.84 | 0.04 | **0.86** (0 false links) |

**Behavior self-check (no answer key):** median purchase-vs-demand fit on links whose names are already clear.
A 0.82 strong, B 0.78 strong, C 0.53 weak. It predicts when hidden-link rescue will help, so the app says so
on any new dataset and falls back to names when the signal is weak.

- Stress grid (6-24 months x 0-20% noise x 3 seeds): RECON matched or outperformed name-only in 15 of 16
  conditions. Exception: 12 months with 20% noise (F1 0.936 vs 0.944). With 6 months the behavior signal
  switches off and RECON falls back to names.
- Count-to-count balance catches the 7% bearing excess (ERP postings show 1.000x because they are backflushed)
  and the 2 -> 3 per-unit gear change (implied 3.02/unit, 90% CI 3.01-3.03).
- Spend newly attributed to BOMs: $815K-$958K (not savings). Spend Integrity Gap: $78K-$104K (1.9-2.5%).

## Design decisions made during the build (and why)
1. **Volume check against remaining demand.** Allowing 2-5% backup-supplier shares (Velocity feedback) removed
   the volume check that separated the motor from other parts in the same products. Shares are now judged
   against the part's demand *not already covered* by confident name matches.
2. **Greedy, most-certain-first rescue.** Two unresolved SKUs can compete for the same uncovered part; the most
   certain rescue is taken first and its demand counted as covered.
3. **Twins use the raw signal.** Look-alike detection asks whether demand patterns alone can tell candidates
   apart, so it uses behavior before the remaining-demand adjustment. It also found two real twin pairs that
   were not planted (M8 bolt/washer, fan blade/guard).
4. **Lead-time shift is configurable and off by default**: receipts here are arrival-dated and planned against
   need. Turn it on when receipts are PO-dated.
5. **Thresholds were re-tuned** for realistically lumpy purchases (lot sizes, stock cover): rescue at
   behavior >= 0.45 with margin >= 0.15.

## Local server (server/)
FastAPI backend (`server/api.py`, docs at /docs): CSV upload with validation and re-run, PDF export of
Evidence Passports and the client one-pager, AI-drafted messages via the Anthropic API when ANTHROPIC_API_KEY
is set (template fallback otherwise). Verified on Windows: 14/14 tests, all endpoints 200.

## Roadmap (vs. the full build prompt)
- Next.js frontend, SQLite persistence, Docker, multi-user board state (board state is per browser today).

## Partial data and column names
Minimum upload: bom, parts_master, spend_cube. production + receipts unlock behavior matching; physical_counts unlock
reconciliation; posted_consumption the backflush check. Common ERP column names are mapped automatically
(`recon/core.py: ALIASES`). The app shows a data-readiness table and the behavior self-check for every dataset.

## Limitations
Identifiability (same-footprint parts), dependence on production + counts, thresholds tuned on synthetic data,
wide calibration intervals at these sample sizes. A pilot with a held-back verified crosswalk is required.
