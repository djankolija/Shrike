#!/bin/bash
# measure.sh <label> <runs>   -- assumes server already up on 8081
set -u
LABEL="${1:-run}"; N="${2:-10}"
P='{"model":"ornith15","messages":[{"role":"user","content":"Count from 1 to 60, one number per line."}],"max_tokens":200,"temperature":0}'
ssh macmini "until curl -s -m 3 http://127.0.0.1:8081/v1/models > /dev/null 2>&1; do :; done"
ssh macmini "curl -s -m 300 http://127.0.0.1:8081/v1/chat/completions -H 'Content-Type: application/json' -d '$P' -o /tmp/m_warm.json"
ssh macmini "rm -f /tmp/m_times.txt"
for i in $(seq 1 "$N"); do
  ssh macmini "curl -s -m 300 http://127.0.0.1:8081/v1/chat/completions -H 'Content-Type: application/json' -d '$P' -o /tmp/m_$i.json -w '%{time_total}\n' >> /tmp/m_times.txt"
done
ssh macmini "python3 -c \"
import json,hashlib,statistics
ts=[float(l) for l in open('/tmp/m_times.txt') if l.strip()]
d=json.load(open('/tmp/m_1.json')); c=d['choices'][0]['message']['content']
h=hashlib.sha256(c.encode()).hexdigest()[:12]; tok=d['usage']['completion_tokens']
import subprocess
lines=[l for l in open('/tmp/ornith.log',errors='ignore') if 'Shrike runner' in l][-len(ts):]
def field(f):
    v=[float(l.split(f+'=')[1].split()[0]) for l in lines if f+'=' in l]
    return (statistics.mean(v), statistics.stdev(v) if len(v)>1 else 0.0)
wm,ws=field('wait_ms'); bm,bs=field('body_ms')
print(f'$LABEL  n={len(ts)}')
print(f'  wall    {statistics.mean(ts):7.3f} s  sd {statistics.stdev(ts):.3f} ({100*statistics.stdev(ts)/statistics.mean(ts):.1f}%)')
print(f'  wait_ms {wm:7.2f}    sd {ws:.2f} ({100*ws/wm:.1f}%)')
print(f'  body_ms {bm:7.2f}    sd {bs:.2f}')
print(f'  out     {h}  {tok} tok')
\""
