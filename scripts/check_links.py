"""Check that every relative link / image path in the Markdown files resolves:  python scripts/check_links.py"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
bad = []
for md in list(ROOT.glob("*.md")) + list((ROOT / ".github").rglob("*.md")):
    text = md.read_text(encoding="utf-8")
    links = re.findall(r"\]\(([^)\s]+)\)", text) + re.findall(r'(?:src|href)="([^"]+)"', text)
    for link in links:
        if link.startswith(("http://", "https://", "#", "mailto:")):
            continue
        p = (md.parent / link.split("#")[0]).resolve()
        if not p.exists():
            bad.append(f"{md.relative_to(ROOT)}: {link}")
print("\n".join(bad) if bad else "all relative links resolve")
sys.exit(1 if bad else 0)
