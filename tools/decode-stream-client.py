#!/usr/bin/env python3
"""decode-stream-client.py <payload.json> <max_tokens> <out.json> [port]

The decode rig's streaming client; tools/decode-rig.sh copies it to the mini and
runs it there. Sends the payload to the local server with stream=true and records
the arrival time of every content chunk (monotonic ns, relative to the send), so a
token's wall by answer position can be joined with the route trace's misses. Writes
{"sent_ns", "first_chunk_ms", "tokens": [[index, ms_since_send, text]...],
 "text", "finish_reason", "chunks"}. One timestamp is taken per socket read,
so chunks that arrive together share it: the mean wall per token is exact, the
median and the per-token distribution are skewed toward zero-and-double.
"""
import http.client
import json
import sys
import time

payload_path, max_tokens, out_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
port = int(sys.argv[4]) if len(sys.argv) > 4 else 8081

body = json.load(open(payload_path))
body["max_tokens"] = max_tokens
body["stream"] = True
body.setdefault("temperature", 0)
data = json.dumps(body).encode()

conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1800)
sent = time.monotonic_ns()
conn.request("POST", "/v1/chat/completions", body=data,
             headers={"Content-Type": "application/json"})
resp = conn.getresponse()
if resp.status != 200:
    print(f"HTTP {resp.status}: {resp.read()[:300]!r}", file=sys.stderr)
    sys.exit(1)

tokens = []
finish = None
first_ms = None
chunks = 0
buf = b""
while True:
    piece = resp.read1(65536) if hasattr(resp, "read1") else resp.read(1)
    if not piece:
        break
    now = time.monotonic_ns()
    buf += piece
    while b"\n" in buf:
        line, buf = buf.split(b"\n", 1)
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        raw = line[5:].strip()
        if raw == b"[DONE]":
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        chunks += 1
        for choice in obj.get("choices", []):
            delta = choice.get("delta", {})
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
            content = delta.get("content")
            if content is None:
                continue
            ms = (now - sent) / 1e6
            if first_ms is None:
                first_ms = ms
            tokens.append([len(tokens), round(ms, 3), content])

text = "".join(t[2] for t in tokens)
json.dump({"sent_ns": sent, "first_chunk_ms": first_ms, "tokens": tokens,
           "text": text, "finish_reason": finish, "chunks": chunks},
          open(out_path, "w"))
last = tokens[-1][1] if tokens else 0.0
print(f"tokens={len(tokens)} first_chunk_ms={first_ms} last_ms={last:.1f} "
      f"finish={finish} chunks={chunks}")
