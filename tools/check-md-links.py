#!/usr/bin/env python3
"""Gate 3: every relative link in every *.md in the repo must resolve."""
import pathlib
import re
import sys

root = pathlib.Path(__file__).resolve().parent.parent
link_re = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
skip_dirs = {".build", ".git", "models", "baselines"}

failures = []
checked = 0
for md in sorted(root.rglob("*.md")):
    if skip_dirs & set(part for part in md.relative_to(root).parts):
        continue
    checked += 1
    for target in link_re.findall(md.read_text(encoding="utf-8", errors="replace")):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        path = target.split("#", 1)[0]
        if not path:
            continue
        if not (md.parent / path).exists():
            failures.append(f"{md.relative_to(root)}: broken relative link -> {target}")

for failure in failures:
    print(failure)
print(f"checked {checked} markdown files, {len(failures)} broken links")
sys.exit(1 if failures else 0)
