"""
draft_report_v4_to_pdf.py
-------------------------
Convert V4/DRAFT_REPORT_V4.md to a compact 2-page PDF that follows the
required template style for the project draft report.

Usage:
    python V4/draft_report_v4_to_pdf.py
"""
import os, sys, subprocess, tempfile, shutil
import markdown

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE     = os.path.dirname(os.path.abspath(__file__))
MD_PATH  = os.path.join(HERE, "DRAFT_REPORT_V4.md")
PDF_PATH = os.path.join(HERE, "DRAFT_REPORT_V4.pdf")
EDGE_EXE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

# Compact CSS — tuned to fit 2 A4 pages
CSS = """
@page { size: A4; margin: 12mm 14mm 14mm 14mm; }
* { box-sizing: border-box; }
body {
    font-family: "Sarabun", "Tahoma", "Segoe UI", sans-serif;
    font-size: 9.5pt;
    line-height: 1.35;
    color: #1a1a1a;
    margin: 0;
    padding: 0;
}
h1 {
    font-size: 14pt;
    text-align: center;
    margin: 0 0 2px 0;
    color: #0F4C75;
}
h2 {
    font-size: 11pt;
    color: #0F4C75;
    border-bottom: 1.5px solid #3282B8;
    padding-bottom: 2px;
    margin: 8px 0 4px 0;
    page-break-after: avoid;
}
h3 {
    font-size: 10pt;
    color: #1C7293;
    margin: 6px 0 2px 0;
    font-weight: bold;
    page-break-after: avoid;
}
p { margin: 2px 0; }
strong { color: #0F4C75; }
ul, ol { margin: 2px 0 4px 0; padding-left: 18px; }
li { margin: 1px 0; }
hr { border: none; border-top: 1px solid #BBB; margin: 4px 0; }
code {
    font-family: "Consolas", monospace;
    background: #F4F4F4;
    padding: 0 3px;
    border-radius: 2px;
    font-size: 8.5pt;
    color: #C7254E;
}
pre {
    background: #F4F4F4;
    color: #333;
    padding: 4px 8px;
    border-radius: 3px;
    font-size: 8pt;
    line-height: 1.25;
    margin: 3px 0;
    page-break-inside: avoid;
    white-space: pre;
    border: 1px solid #DDD;
}
pre code { background: transparent; color: #333; padding: 0; }
table {
    border-collapse: collapse;
    width: 100%;
    margin: 3px 0;
    font-size: 8.5pt;
    page-break-inside: avoid;
}
th, td {
    border: 0.7px solid #999;
    padding: 2px 5px;
    text-align: left;
    vertical-align: top;
}
th { background: #E5F0F7; color: #0F4C75; font-weight: bold; }
tr:nth-child(even) { background: #FAFAFA; }
"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
{content}
</body>
</html>
"""

print("=" * 60, flush=True)
print(f"Reading: {MD_PATH}", flush=True)
with open(MD_PATH, "r", encoding="utf-8") as f:
    md_text = f.read()
print(f"  ({len(md_text):,} chars)", flush=True)

print("Converting markdown -> HTML...", flush=True)
html_body = markdown.markdown(
    md_text,
    extensions=["tables", "fenced_code", "sane_lists"],
)
html_full = HTML_TEMPLATE.format(
    title="Draft Report V4 - Fingerprint Verification",
    css=CSS,
    content=html_body,
)

tmp_dir = tempfile.mkdtemp(prefix="draftv4_2pdf_")
html_path = os.path.join(tmp_dir, "draft_v4.html")
with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_full)
print(f"Wrote HTML: {html_path}  ({os.path.getsize(html_path):,} bytes)",
      flush=True)

if not os.path.exists(EDGE_EXE):
    print(f"ERROR: Edge not found at {EDGE_EXE}", flush=True)
    sys.exit(1)

print(f"\nRunning Edge headless...", flush=True)
url = "file:///" + html_path.replace("\\", "/")
cmd = [
    EDGE_EXE,
    "--headless=new",
    "--disable-gpu",
    "--no-pdf-header-footer",
    f"--print-to-pdf={PDF_PATH}",
    url,
]
result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
print(f"  Exit code: {result.returncode}", flush=True)

shutil.rmtree(tmp_dir, ignore_errors=True)

if os.path.exists(PDF_PATH):
    print(f"\nSaved: {PDF_PATH}", flush=True)
    print(f"  Size: {os.path.getsize(PDF_PATH):,} bytes", flush=True)
else:
    print(f"\nFAILED: {PDF_PATH} not created", flush=True)
    sys.exit(1)
