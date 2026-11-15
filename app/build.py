"""Inject bundle.json (and slides.json, the pitch deck) into the app template -> app/omnipart_recon.html (single self-contained file)."""
from pathlib import Path
here = Path(__file__).parent
slides = here / "slides.json"
html = (here / "template.html").read_text(encoding="utf-8").replace("__BUNDLE__", (here / "bundle.json").read_text(encoding="utf-8").replace("</", "<\\/"))
html = html.replace("__SLIDES__", slides.read_text(encoding="utf-8") if slides.exists() else "[]")
(here / "omnipart_recon.html").write_text(html, encoding="utf-8")
print("wrote", here / "omnipart_recon.html", f"{len(html)/1024:.0f} KB")
