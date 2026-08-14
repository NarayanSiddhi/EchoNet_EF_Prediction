#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MD="$ROOT/ef_prediction/documents/HCL_REMOVAL_REPORT.md"
OUT="$ROOT/ef_prediction/documents/HCL_REMOVAL_REPORT"

cd "$ROOT"

pandoc "$MD" -o "${OUT}.docx" --standalone
pandoc "$MD" -o "${OUT}.html" --standalone --metadata title="HCL Ablation — Short Summary"

python3 <<'PY'
import re
from pathlib import Path
from xhtml2pdf import pisa

root = Path("ef_prediction/documents")
html_path = root / "HCL_REMOVAL_REPORT.html"
pdf_path = root / "HCL_REMOVAL_REPORT.pdf"
html = html_path.read_text(encoding="utf-8")
body = re.search(r"<body>.*</body>", html, re.S)
if not body:
    raise SystemExit("Could not find HTML body")
title = re.search(r"<title>(.*?)</title>", html)
title = title.group(1) if title else "HCL Ablation Report"
simple = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>{title}</title>
<style>
body {{ font-family: Helvetica, Arial, sans-serif; font-size: 11pt; margin: 24px; color: #111; }}
h1 {{ font-size: 18pt; }}
h2 {{ font-size: 14pt; margin-top: 1.2em; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
th, td {{ border: 1px solid #333; padding: 6px 8px; text-align: left; }}
th {{ background: #eee; }}
code {{ font-family: monospace; font-size: 10pt; }}
hr {{ border: none; border-top: 1px solid #ccc; margin: 1.5em 0; }}
</style></head>
{body.group(0)}</html>"""
with pdf_path.open("wb") as out:
    status = pisa.CreatePDF(simple, dest=out, encoding="utf-8")
if status.err:
    raise SystemExit(f"xhtml2pdf failed ({status.err} errors)")
print(f"Wrote {pdf_path}")
PY

echo "Exported:"
echo "  ${OUT}.docx"
echo "  ${OUT}.html"
echo "  ${OUT}.pdf"
