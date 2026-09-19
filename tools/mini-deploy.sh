#!/bin/bash
# mini-deploy.sh [--restart]
# Copies the release binaries and their resource bundles from .build/release to
# the mini's ~/shrike-runtime/bin. Each binary and bundle is staged remotely as
# <name>.staging first; only after every copy succeeds does one ssh command
# rm -rf the old bundle (mv -f for a binary) and swap the staged copy into
# place, so a dropped connection mid-copy can never leave a half-written
# artifact live.
#
# By default this only copies — the mini's server is left running. The mini's
# owner approved restarts and deploy actions on 2026-09-01, but --restart is
# still an explicit act: pass it to also kill the running server, relaunch it
# with the production launch command, and poll for readiness.
#
# The env and flags in the launch below must stay byte-for-byte CLAUDE.md's
# production launch line; they drifted from it silently between v17 and v22.
#
# Run from the repo root after `swift build -c release`.
set -euo pipefail
BIN=.build/release
RESTART=0
if [ "${1:-}" = "--restart" ]; then
  RESTART=1
fi

for f in shrike; do
  [ -x "$BIN/$f" ] || { echo "missing $BIN/$f — build release first" >&2; exit 1; }
done

if [ "$RESTART" -eq 1 ]; then
  ssh macmini '
    pkill -f "bin/shrike serve --model ./models/ornith15.gturbo" || true
    # A server predating v24 is still named ShrikeServer; -fi below for the same reason.
    pkill -f "bin/ShrikeServer --model ./models/ornith15.gturbo" || true
    sleep 3
    if pgrep -fi "bin/shrike" > /dev/null; then echo "server still running" >&2; exit 1; fi
  '
fi

for f in shrike; do
  scp -q "$BIN/$f" "macmini:shrike-runtime/bin/$f.staging"
done
# The mini runs the golden from its own copy; deploy it with the binary so its
# process guard cannot go stale against a renamed binary.
scp -q tools/golden-baseline.sh "macmini:shrike-runtime/golden-baseline.sh"
for bundle in "$BIN"/*.bundle; do
  name=$(basename "$bundle")
  scp -q -r "$bundle" "macmini:shrike-runtime/bin/$name.staging"
done

ssh macmini '
  set -eu
  cd ~/shrike-runtime/bin
  for f in shrike; do
    mv -f "$f.staging" "$f"
  done
  for staged in *.bundle.staging; do
    [ -e "$staged" ] || continue
    name="${staged%.staging}"
    rm -rf "$name"
    mv -f "$staged" "$name"
  done
'
echo "copied binaries + bundles"

if [ "$RESTART" -eq 0 ]; then
  exit 0
fi

ssh macmini '
  set -eu
  cd ~/shrike-runtime
  [ -f /tmp/ornith.log ] && mv -f /tmp/ornith.log "/tmp/ornith.log.$(date +%Y%m%d-%H%M%S)"
  env SHRIKE_RUNNER_STATS=1 SHRIKE_KERNEL_STATS=1 \
    SHRIKE_EXPERT_SLOT_TABLE=256,256,246,209,191,171,171,162,171,149,164,169,155,144,137,135,133,131,132,130,145,133,141,142,137,137,130,137,137,142,137,135,157,157,162,160,166,161,178,194 \
    SHRIKE_EXPERT_POLICY=slru \
    nohup ./bin/shrike serve \
    --model ./models/ornith15.gturbo --port 8081 \
    --max-context 32768 --ram-budget 11324620800 --thinking off > /tmp/ornith.log 2>&1 &
  server_pid=$!
  # Keyed to the pid, never to a log line: the banner goes through `print`, which
  # is block-buffered to a file and may never flush while the server runs fine.
  tries=0
  until curl -sf -m 3 http://127.0.0.1:8081/v1/models > /dev/null 2>&1; do
    kill -0 "$server_pid" 2>/dev/null || {
      echo "the server this script launched exited" >&2; tail -n 5 /tmp/ornith.log; exit 1; }
    tries=$((tries + 1))
    if [ "$tries" -gt 60 ]; then echo "server never became ready" >&2; tail -n 5 /tmp/ornith.log; exit 1; fi
    sleep 3
  done
  kill -0 "$server_pid" 2>/dev/null || {
    echo "/v1/models answered but the launched server is gone: something else holds 8081" >&2
    exit 1; }
  echo "server ready after $((tries * 3))s (pid $server_pid)"
  logged_bits=""
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    logged_bits=$(grep -a -o "prefill_router_bits=[0-9]*" /tmp/ornith.log | head -n 1)
    [ -n "$logged_bits" ] && break
    sleep 2
  done
  echo "${logged_bits:-prefill_router_bits=NOT-LOGGED}"
'
