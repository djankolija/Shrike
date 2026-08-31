#!/usr/bin/env python3
"""Send ONE non-streamed card request at a chosen user-turn depth (history from
the export, so prompt length is controlled): turn 1 = 48-token prompt (no
prefill tiles), turn 4 = 1900-token prompt (tile-pipelined prefill)."""
import json
import subprocess
import sys
import time

EXPORT = "/Users/davorjankolija/Developer/LLMBench/results/raw/cards-local-v1/keynes-bancor__ornith__tools-off.json"
TURN = int(sys.argv[1]) if len(sys.argv) > 1 else 1

with open(EXPORT) as f:
    export = json.load(f)

history = [{"role": "system", "content": export["system_prompt"]}]
seen = 0
for turn in export["turns"]:
    history.append({"role": turn["role"], "content": turn["text"]})
    if turn["role"] == "user":
        seen += 1
        if seen == TURN:
            break
payload = json.dumps({"model": "ornith15", "messages": history,
                      "temperature": 0, "max_tokens": 256, "stream": False})
t0 = time.time()
proc = subprocess.run(
    ["ssh", "macmini",
     "curl -s -m 300 http://localhost:8081/v1/chat/completions "
     "-H 'Content-Type: application/json' -d @-"],
    input=payload, capture_output=True, text=True, timeout=330)
resp = json.loads(proc.stdout)
u = resp.get("usage", {})
print(f"turn {TURN}: wall={time.time()-t0:.1f}s prompt={u.get('prompt_tokens')} out={u.get('completion_tokens')}")
