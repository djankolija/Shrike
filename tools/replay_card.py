#!/usr/bin/env python3
"""Replay an exported LLMBench card conversation against the live Shrike server
on the mini (via ssh + curl to localhost:8081), preserving exported assistant
turns as history so context bytes match the original session."""
import json
import subprocess
import sys
import time

EXPORT = "/Users/davorjankolija/Developer/LLMBench/results/raw/cards-local-v1/keynes-bancor__ornith__tools-off.json"

with open(EXPORT) as f:
    export = json.load(f)

history = [{"role": "system", "content": export["system_prompt"]}]
turn_no = 0
for turn in export["turns"]:
    if turn["role"] == "user":
        turn_no += 1
        history.append({"role": "user", "content": turn["text"]})
        payload = json.dumps({
            "model": "ornith15",
            "messages": history,
            "temperature": 0,
            "max_tokens": 256,
            "stream": False,
        })
        t0 = time.time()
        proc = subprocess.run(
            ["ssh", "macmini",
             "curl -s -m 300 http://localhost:8081/v1/chat/completions "
             "-H 'Content-Type: application/json' -d @-"],
            input=payload, capture_output=True, text=True, timeout=330)
        elapsed = time.time() - t0
        if proc.returncode != 0:
            print(f"turn {turn_no}: ssh/curl failed rc={proc.returncode}: {proc.stderr[:500]}")
            sys.exit(1)
        try:
            resp = json.loads(proc.stdout)
        except json.JSONDecodeError:
            print(f"turn {turn_no}: bad response: {proc.stdout[:500]}")
            sys.exit(1)
        if "error" in resp:
            print(f"turn {turn_no}: server error: {json.dumps(resp)[:500]}")
            sys.exit(1)
        usage = resp.get("usage", {})
        content = resp["choices"][0]["message"].get("content") or ""
        print(f"turn {turn_no}: wall={elapsed:.1f}s prompt_tokens={usage.get('prompt_tokens')} "
              f"completion_tokens={usage.get('completion_tokens')} "
              f"reply[:80]={content[:80]!r}")
        # the fresh generation is discarded; the exported assistant turn
        # (appended by the branch below on the next iteration) is the history
    else:
        history.append({"role": "assistant", "content": turn["text"]})
print("done")
