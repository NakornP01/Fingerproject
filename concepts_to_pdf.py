"""
concepts_to_pdf.py
------------------
Convert V4/GUIDE_concepts_th.md to PDF with page numbers + embedded graphs.

Usage:
    python V4/concepts_to_pdf.py
"""
import os, sys, subprocess, tempfile, shutil
import markdown

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE     = os.path.dirname(os.path.abspath(__file__))
MD_PATH  = os.path.join(HERE, "GUIDE_concepts_th.md")
PDF_PATH = os.path.join(HERE, "GUIDE_concepts_th.pdf")
EDGE_EXE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

IMAGES = [
    "confusion_matrix_v4.png",
    "score_distribution_v4.png",
    "roc_curve_v4.png",
    "det_curve_v4.png",
    "pr_curve_v4.png",
    "threshold_methods_v4.png",
]

CSS = """
@page { size: A4; margin: 18mm 12mm 20mm 12mm; }
* { box-sizing: border-box; }
body {
    font-family: "Sarabun", "Tahoma", "Segoe UI", sans-serif;
    font-size: 10.5pt;
    line-height: 1.65;
    color: #222;
    margin: 0;
    padding: 0;
}
h1 {
    font-size: 22pt;
    color: #0F4C75;
    border-bottom: 3px solid #3282B8;
    padding-bottom: 8px;
    margin-top: 0;
}
h2 {
    font-size: 16pt;
    color: #0F4C75;
    border-left: 5px solid #3282B8;
    padding-left: 10px;
    margin-top: 24px;
    margin-bottom: 12px;
    page-break-before: always;
    page-break-after: avoid;
}
h2:first-of-type { page-break-before: auto; }
h3 {
    font-size: 13pt;
    color: #1C7293;
    margin-top: 18px;
    margin-bottom: 8px;
    page-break-after: avoid;
}
h4 {
    font-size: 11pt;
    color: #444;
    margin-top: 12px;
    margin-bottom: 6px;
    page-break-after: avoid;
}
p { margin: 6px 0; }
strong { color: #0F4C75; }
em { color: #555; }
blockquote {
    border-left: 4px solid #3282B8;
    background: #EAF4FA;
    margin: 10px 0;
    padding: 8px 14px;
    color: #1C7293;
    page-break-inside: avoid;
}
ul, ol { margin: 6px 0; padding-left: 22px; }
li { margin: 3px 0; }
hr { border: none; border-top: 1px dashed #BBB; margin: 18px 0; }
code {
    font-family: "Consolas", "Courier New", monospace;
    background: #F4F4F4;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 9.5pt;
    color: #C7254E;
}
pre {
    background: #2C3E50;
    color: #ECF0F1;
    padding: 10px 14px;
    border-radius: 5px;
    overflow-x: auto;
    font-size: 9pt;
    line-height: 1.45;
    page-break-inside: avoid;
    white-space: pre;
}
pre code { background: transparent; color: #ECF0F1; padding: 0; }
table {
    border-collapse: collapse;
    width: 100%;
    margin: 10px 0;
    font-size: 9.5pt;
    page-break-inside: avoid;
}
th, td {
    border: 1px solid #BBB;
    padding: 6px 9px;
    text-align: left;
    vertical-align: top;
}
th { background: #0F4C75; color: white; font-weight: bold; }
tr:nth-child(even) { background: #F9F9F9; }
img {
    display: block;
    max-width: 88%;
    max-height: 480px;
    height: auto;
    margin: 14px auto;
    border: 1px solid #DDD;
    border-radius: 4px;
    padding: 4px;
    background: #FFF;
    page-break-inside: avoid;
}
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

FOOTER_TEMPLATE = (
    "<div style='font-size:8pt; width:100%; text-align:center; "
    "color:#666; padding: 0 12mm;'>"
    "<span class='title'></span> &nbsp;&nbsp;|&nbsp;&nbsp; "
    "หน้า <span class='pageNumber'></span> / "
    "<span class='totalPages'></span>"
    "</div>"
)
HEADER_TEMPLATE = "<div></div>"

print("=" * 60, flush=True)
print(f"Reading: {MD_PATH}", flush=True)
with open(MD_PATH, "r", encoding="utf-8") as f:
    md_text = f.read()
print(f"  ({len(md_text):,} chars)", flush=True)

print("Converting markdown -> HTML...", flush=True)
html_body = markdown.markdown(
    md_text,
    extensions=["tables", "fenced_code", "sane_lists", "toc"],
)
html_full = HTML_TEMPLATE.format(
    title="Concepts - Fingerprint Verification Metrics",
    css=CSS,
    content=html_body,
)

tmp_dir = tempfile.mkdtemp(prefix="concepts2pdf_")
html_path = os.path.join(tmp_dir, "concepts.html")
with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_full)
print(f"Wrote HTML: {html_path}  ({os.path.getsize(html_path):,} bytes)",
      flush=True)

print("Copying images to temp dir...", flush=True)
for img in IMAGES:
    src = os.path.join(HERE, img)
    dst = os.path.join(tmp_dir, img)
    if os.path.exists(src):
        shutil.copy2(src, dst)

if not os.path.exists(EDGE_EXE):
    print(f"ERROR: Edge not found at {EDGE_EXE}", flush=True)
    sys.exit(1)

print(f"\nRunning Edge headless...", flush=True)
url = "file:///" + html_path.replace("\\", "/")
cmd = [
    EDGE_EXE,
    "--headless=new",
    "--disable-gpu",
    "--allow-file-access-from-files",
    f"--print-to-pdf={PDF_PATH}",
    f"--header-template={HEADER_TEMPLATE}",
    f"--footer-template={FOOTER_TEMPLATE}",
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
