#!/usr/bin/env bash
# Capture a deterministic generation baseline, so a refactor of the runtime
# can be checked against byte-identical output rather than "the tests still
# pass".
#
#   tools/golden-baseline.sh [profile ...]     # capture; default: all four
#   tools/golden-baseline.sh --check [profile ...]
#
# Profiles: short and long on the CLI's fused greedy head; short-lh and long-lh
# on the server's head path (--logits-head: the logits head and the GPU greedy
# sampler), the files named ...-<profile>.<tag>.txt. turns-lh is the two-turn
# continuation gate (v20 T3.3): a chat turn answered to its stop token on the
# server's head path, then a follow-up generated from the state the stop left,
# so the pass committed ahead of the stop and drained must leave that state
# exactly as a run without it. The default set is all five, so `--check` alone
# covers both heads and the continuation (v19 Task 1, v20 T3.3).
#
# Determinism comes from greedy decoding: --temperature 0 with a fixed seed and
# a fixed prompt. Greedy means the sampler never draws, so the only inputs are
# the weights and the kernels — exactly what a runtime refactor must not change.
# The `long` profile pins a ~2k-token context: long-context near-tie picks are
# where reduction-order drift between binaries shows first (v6 numerics note).
#
# SCOPE: a baseline is valid for one (machine, build, model) triple. Metal
# reduction order is not guaranteed across GPU families, so a file captured on
# an M1 is not a reference for an M4; files carry a machine tag in their name.
# Re-capture after a deliberate, signed-off numerics change; a diff at any
# other time is a regression.
#
# Env overrides (the mini has no repo checkout: run with all five):
#   CLI=~/shrike-runtime/bin/ShrikeCLI
#   MODEL=~/shrike-runtime/models/ornith15.gturbo
#   OUT_DIR=~/shrike-runtime/baselines
#   MACHINE_TAG=mini
#   CLI_EXTRA_ARGS="--expert-cache-slots 160"
# The fifth is not optional on the mini: the CLI's default 64 slots is one arena
# chunk there, while production serves 160 slots across two, so without it the
# golden never crosses a chunk boundary (v22 Task 2).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CLI="${CLI:-$ROOT/.build/arm64-apple-macosx/release/ShrikeCLI}"
OUT_DIR="${OUT_DIR:-$ROOT/baselines}"
MACHINE_TAG="${MACHINE_TAG:-$(sysctl -n hw.model | tr -cd '[:alnum:]')}"
MAX_NEW="${MAX_NEW:-96}"
SEED="${SEED:-1234}"
# CLI_EXTRA_ARGS (optional): appended to every CLI run, e.g. "--expert-cache-slots 160"
# to run the golden at a pool the box splits across arena chunks (v22).
CLI_EXTRA_ARGS="${CLI_EXTRA_ARGS:-}"

if [ -z "${MODEL:-}" ]; then
  for candidate in /Volumes/BuildSSD/shrike/ornith15.gturbo \
                   "$HOME/shrike-runtime/models/ornith15.gturbo"; do
    if [ -f "$candidate/verified-install.json" ]; then MODEL="$candidate"; break; fi
  done
fi
if [ -z "${MODEL:-}" ] || [ ! -f "$MODEL/verified-install.json" ]; then
  echo "no verified ornith15.gturbo install found; set MODEL=" >&2
  exit 2
fi

SHORT_PROMPT="Explain what a mutex is and when you would use one."
# ~2k tokens of deterministic context: a fixed ledger the model is asked to
# summarize. Built from literals only — never touch this construction, the
# stored baselines depend on its exact bytes.
long_prompt() {
  printf 'You are auditing a build ledger. Entries follow.\n'
  for i in $(seq 1 60); do
    printf 'Entry %d: commit c%04d built target shrike-core in %d ms with 0 warnings, ran 1108 tests in %d ms, linked 3 artifacts, and archived bundle b%03d to shelf s%d.\n' \
      "$i" $((i * 37)) $((1200 + i * 13)) $((80000 + i * 211)) "$i" $((i % 7))
  done
  printf 'Summarize: how many entries, which shelf received the most bundles, and the trend in build times.\n'
}

# The continuation gate's turns. The question must end at the model's stop
# token within max-new, or the gate never exercises the drained pass; the
# follow-up is encoded verbatim behind that stop token, as a chat template
# would render the next user turn.
TURNS_MESSAGES='[{"role":"user","content":"In one sentence, what is a mutex?"}]'
TURNS_FOLLOW_UP=$'\n<|im_start|>user\nAnd a semaphore, in one sentence?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

mode=capture
if [ "${1:-}" = "--check" ]; then mode=check; shift; fi
profiles=("$@"); [ ${#profiles[@]} -eq 0 ] && profiles=(short long short-lh long-lh turns-lh)

if [ ! -x "$CLI" ]; then
  echo "missing $CLI — run: swift build -c release (or set CLI=)" >&2
  exit 2
fi

# CLAUDE.md: never run alongside another model process, and never terminate one
# we did not start. Refuse rather than race.
if pgrep -f 'ShrikeServer|ShrikeCLI|ShrikePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm' >/dev/null 2>&1; then
  echo "a model process is already running; stop it yourself, then re-run" >&2
  exit 3
fi

mkdir -p "$OUT_DIR"
status=0

for profile in "${profiles[@]}"; do
  prompt_args=()
  case "$profile" in
    short|short-lh) prompt="$SHORT_PROMPT"; max_new="$MAX_NEW"
                    prompt_args=(--prompt "$prompt") ;;
    long|long-lh)   prompt="$(long_prompt)"; max_new=128
                    prompt_args=(--prompt "$prompt") ;;
    turns-lh)       prompt="$TURNS_MESSAGES"; max_new=128
                    messages="$(mktemp "${TMPDIR:-/tmp}/golden-turns.XXXXXX")"
                    printf '%s' "$TURNS_MESSAGES" > "$messages"
                    prompt_args=(--messages-file "$messages" --follow-up "$TURNS_FOLLOW_UP") ;;
    *) echo "unknown profile: $profile (expected short, long, short-lh, long-lh or turns-lh)" >&2
       status=1; continue ;;
  esac
  case "$profile" in
    *-lh) head=logits; head_flag="--logits-head" ;;
    *)    head=fused;  head_flag="" ;;
  esac
  file="$OUT_DIR/ornith15-int4-${profile}.${MACHINE_TAG}.txt"
  work="$(mktemp "${TMPDIR:-/tmp}/golden-baseline.XXXXXX")"

  echo "== $profile =="
  # --quiet keeps the timing footer out of the compared text; only the
  # generated tokens are the contract. Timings vary run to run by design.
  "$CLI" --model "$MODEL" "${prompt_args[@]}" --max-new "$max_new" \
         --temperature 0 --seed "$SEED" --quiet $head_flag $CLI_EXTRA_ARGS > "$work" 2>"$work.err"
  rc=$?
  [ "$profile" = turns-lh ] && rm -f "$messages"
  if [ $rc -ne 0 ]; then
    echo "  FAILED (exit $rc)"; sed 's/^/  /' "$work.err" | head -20
    rm -f "$work" "$work.err"; status=1; continue
  fi

  if [ "$mode" = capture ]; then
    {
      echo "# profile:     $profile (prompt $(printf '%s' "$prompt" | wc -c | tr -d ' ') bytes)"
      echo "# max-new:     $max_new"
      echo "# temperature: 0 (greedy)"
      echo "# head:        $head"
      echo "# seed:        $SEED"
      echo "# model:       $(basename "$MODEL")"
      echo "# captured-on: $(sysctl -n hw.model), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, macOS $(sw_vers -productVersion)"
      echo "# commit:      $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)$(git -C "$ROOT" diff --quiet 2>/dev/null || echo '-dirty')"
      echo "---"
      cat "$work"
    } > "$file"
    echo "  captured -> $file ($(wc -c < "$work" | tr -d ' ') bytes)"
  else
    if [ ! -f "$file" ]; then
      echo "  no baseline at $file; run without --check first"; status=1
    elif diff -q <(sed '1,/^---$/d' "$file") "$work" >/dev/null; then
      echo "  ok — output identical to baseline"
    else
      echo "  MISMATCH against $file:"
      diff <(sed '1,/^---$/d' "$file") "$work" | head -30 | sed 's/^/    /'
      status=1
    fi
  fi
  rm -f "$work" "$work.err"
done

exit $status
