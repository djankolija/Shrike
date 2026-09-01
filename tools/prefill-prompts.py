#!/usr/bin/env python3
"""Write chat-completion request bodies at four prompt lengths.

Same ledger construction as tools/golden-baseline.sh's long_prompt (60 entries
is 3,756 ornith tokens; the numbers-heavy text tokenizes at ~2.6 bytes/token).
The entry numbering is offset per length so no prompt is a prefix of another
and the server's prompt cache cannot short-circuit a run.

Usage: prefill-prompts.py <outdir> [model-id]
Writes prompt-2k.json (3,756 tok), prompt-6k.json (12,285), prompt-12k.json
(25,245), prompt-2kb.json (4,305; a fresh-salt twin of 2k for warm reruns).
"""
import json
import os
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
MODEL = sys.argv[2] if len(sys.argv) > 2 else "ornith15"
LENGTHS = {"2k": (60, 0), "6k": (180, 1000), "12k": (360, 5000), "2kb": (60, 9000)}


def ledger(entries, offset):
    lines = ["You are auditing a build ledger. Entries follow."]
    for i in range(1, entries + 1):
        n = i + offset
        lines.append(
            f"Entry {n}: commit c{n * 37:04d} built target shrike-core in "
            f"{1200 + n * 13} ms with 0 warnings, ran 1108 tests in "
            f"{80000 + n * 211} ms, linked 3 artifacts, and archived bundle "
            f"b{n:03d} to shelf s{n % 7}."
        )
    lines.append("Summarize: how many entries, which shelf received the most "
                 "bundles, and the trend in build times.")
    return "\n".join(lines) + "\n"


os.makedirs(OUT, exist_ok=True)

for label, (entries, offset) in LENGTHS.items():
    prompt = ledger(entries, offset)
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8,
        "temperature": 0,
        "stream": False,
    }
    path = os.path.join(OUT, f"prompt-{label}.json")
    with open(path, "w") as f:
        json.dump(body, f)
    print(f"{path}: {len(prompt)} bytes, {entries} entries")
