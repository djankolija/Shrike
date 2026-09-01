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
# with the production launch command (stats env, 8G budget, thinking off,
# rotating /tmp/ornith.log), and poll for readiness.
#
# Run from the repo root after `swift build -c release`.
set -euo pipefail
BIN=.build/release
RESTART=0
if [ "${1:-}" = "--restart" ]; then
  RESTART=1
fi

for f in ShrikeServer ShrikeCLI ShrikeRepack; do
  [ -x "$BIN/$f" ] || { echo "missing $BIN/$f — build release first" >&2; exit 1; }
done

if [ "$RESTART" -eq 1 ]; then
  ssh macmini '
    pkill -f "bin/ShrikeServer --model ./models/ornith15.gturbo" || true
    sleep 3
    if pgrep -f "bin/ShrikeServer" > /dev/null; then echo "server still running" >&2; exit 1; fi
  '
fi

for f in ShrikeServer ShrikeCLI ShrikeRepack; do
  scp -q "$BIN/$f" "macmini:shrike-runtime/bin/$f.staging"
done
for bundle in "$BIN"/*.bundle; do
  name=$(basename "$bundle")
  scp -q -r "$bundle" "macmini:shrike-runtime/bin/$name.staging"
done

ssh macmini '
  set -eu
  cd ~/shrike-runtime/bin
  for f in ShrikeServer ShrikeCLI ShrikeRepack; do
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
  env SHRIKE_RUNNER_STATS=1 SHRIKE_KERNEL_STATS=1 nohup ./bin/ShrikeServer \
    --model ./models/ornith15.gturbo --model-id ornith15 --port 8081 \
    --max-context 32768 --ram-budget 8G --thinking off > /tmp/ornith.log 2>&1 &
  tries=0
  until curl -sf -m 3 http://127.0.0.1:8081/v1/models > /dev/null 2>&1; do
    tries=$((tries + 1))
    if [ "$tries" -gt 200 ]; then echo "server never became ready" >&2; tail -n 5 /tmp/ornith.log; exit 1; fi
    sleep 3
  done
  echo "server ready after $((tries * 3))s"
  path=""
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    path=$(grep -a -o "prefill_projection_path=[a-z0-9-]*" /tmp/ornith.log | head -n 1)
    [ -n "$path" ] && break
    sleep 2
  done
  echo "${path:-prefill_projection_path=NOT-LOGGED}"
'
