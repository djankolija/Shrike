#!/usr/bin/env python3
"""Write the v13 "the turn" chapter's request-shape payloads.

Same ledger construction as tools/prefill-prompts.py (~62.6 ornith tokens per
entry). Cold/warm pairs at ~300 / 1k / 2k tokens (distinct offsets, so no
prompt is a prefix of another), a suffix chain X (32 entries) -> X+4 ->
X+16 whose cached prefix is X's entries, and a turns chain X -> Xturn2 ->
Xturn3 that replays X's own saved 8-token answer as the assistant turn.

Usage: turn-prompts.py <outdir>
Writes t300.json, t300b.json, t1k.json, t1kb.json, t2k.json, t2kb.json,
tX.json, tXp4.json, tXp16.json, tXturn2.json, tXturn3.json.
"""
import json
import os
import sys

CANNED_ANSWER = "## Build Ledger Summary\n\n**Number of"


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


def write(out, label, messages):
    body = {"model": "ornith15", "messages": messages,
            "max_tokens": 8, "temperature": 0, "stream": False}
    with open(os.path.join(out, f"{label}.json"), "w") as f:
        json.dump(body, f)


OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
SHAPES = {
    "t300": (4, 0), "t300b": (4, 300),
    "t1k": (16, 100), "t1kb": (16, 400),
    "t2k": (32, 200), "t2kb": (32, 500),
    "tX": (32, 600), "tXp4": (36, 600), "tXp16": (48, 600),
}
for label, (entries, offset) in SHAPES.items():
    prompt = ledger(entries, offset)
    write(OUT, label, [{"role": "user", "content": prompt}])
    print(f"{label}: {entries} entries, offset {offset}, {len(prompt)} bytes")

tx_entries, tx_offset = SHAPES["tX"]
turn2_messages = [
    {"role": "user", "content": ledger(tx_entries, tx_offset)},
    {"role": "assistant", "content": CANNED_ANSWER},
    {"role": "user", "content": "And which shelf received the fewest bundles?"},
]
turn3_messages = turn2_messages + [
    {"role": "assistant", "content": CANNED_ANSWER},
    {"role": "user", "content": "List the three slowest builds."},
]
for label, messages in (("tXturn2", turn2_messages), ("tXturn3", turn3_messages)):
    write(OUT, label, messages)
    print(f"{label}: {len(messages)} messages")
