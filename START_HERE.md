# OmniPart RECON: run it on your laptop

## Easiest (Windows)
1. Install **Python 3.11 or 3.12** from https://www.python.org/downloads/ if you don't have it.
   During install, tick **"Add python.exe to PATH"**.
2. Double-click **`start_windows.bat`** in this folder.
   - First run: creates a private environment and installs packages (2-5 minutes, needs internet).
   - Then your browser opens **http://localhost:8000**.
3. To stop: close the black window (or press Ctrl+C in it).

## Manual (any OS)
```
python -m venv .venv
.venv\Scripts\activate            (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python run.py
```

## What the local version adds
| Feature | Where |
|---|---|
| FastAPI backend + live API docs | http://localhost:8000/docs |
| Upload your own CSVs, re-run the engine | **Data & backflush** view, "Upload your own data" (blank CSV templates linked there) |
| PDF export: Evidence Passport | Link map, open a row, **Download PDF** |
| PDF export: client steering one-pager | Communications, **Download PDF** |
| AI-drafted messages (optional) | Communications, **Draft with AI** |

## Turning on AI drafting
1. Open the file **`.env`** in this folder with Notepad (right-click > Open with > Notepad).
2. Replace `paste-your-key-here` with your Anthropic API key (from console.anthropic.com > Keys). Save.
3. Restart `start_windows.bat`. The black window should say **AI drafting: ON**.
4. In the app: Communications > pick a finding > pick an audience > **Draft with AI**.
The `.env` file stays on this computer and is never zipped or uploaded. Without a key the app uses its templates.
Each draft costs a fraction of a cent to a few cents of API credit.

## Uploading your own data
Data & backflush > Upload your own data. **Minimum: bom.csv, parts_master.csv, spend_cube.csv.**
Each extra file unlocks more (the page shows a readiness table):
| Add | Unlocks |
|---|---|
| receipts.csv (or an annual_qty column in spend_cube) | spend amounts, true product cost, price gaps |
| production.csv + receipts.csv, 12+ months | behavior matching: hidden links |
| physical_counts.csv (2+ count dates) | reconciliation: usage shifts, excess usage, inventory builds |
| posted_consumption.csv | backflush check |
| ground_truth.csv | Proof view scored on your data |
Common ERP column names (Part Number, Qty, Vendor, Unit Cost, Item Description, GR Qty...) are renamed automatically.
Try it: upload everything in `data/C` (an unfamiliar company the method never saw).

## Demo script (3 minutes)
1. **Overview**: $4.1M spend, 99% linked, $815K-$958K of cost that name-matching could not see.
2. **The mystery motor**: name 0.11 vs behavior chart; click **Verify**; product costs update. Click **Reject** to show the fallback.
3. **Proof**: blind test, name-only F1 0.78 vs RECON 0.99 with zero false links; then the **Unfamiliar company** tab:
   the self-check says behavior evidence is weak there, RECON falls back to names, still zero false links.
4. **Reconciliation**: bearing excess visible in physical counts, invisible in ERP postings (backflush).
5. **Resolution board**: drag a card; **Communications**: download the one-pager PDF.

## Tests
```
python -m pytest -q tests      # 18 tests
```

## Troubleshooting
- "python is not recognized": reinstall Python with "Add to PATH" ticked, then open a new window.
- Port 8000 busy: `set PORT=8010` then `python run.py`.
- Package install fails: check internet / campus Wi-Fi login, then run `start_windows.bat` again.
- To regenerate the demo from scratch: delete `app/bundle.json` and restart.
