# v21 implementation plan: lossless compression

Companion to [v21-compression.md](v21-compression.md). Checkboxes here are the
only status tracking. Branch: `perf/v21-compression` off `main` at `17babb9`.

Protocol per task: the four gates per commit (release build with zero warnings,
`swiftlint lint --strict`, `tools/check-md-links.py`, `swift test --no-parallel`);
the golden byte-identical on all five profiles on both boxes for every commit, on
the plain build always and on the coded build once it exists (the chapter is
class 1 throughout); deploy to the mini with `tools/mini-deploy.sh` (binary plus
the `*.bundle` directories); the arms rig, two production lifetimes per shape on
four shapes, read against the previous task's arms; the rows the task
pre-registered, moved or not, recorded in the design document's task record;
ThreadSanitizer once at the close. Every model run on the mini stops production
and runs under the session's deploy leave (given 2026-09-18); production is
restored and verified against the golden before the session ends.

Order: step zero's offline pricing first (S0.1, S0.2), the drive probe (S0.3), the
gate (S0.4), the ruling (S0.5); the tasks only if the gate opens.

## Step zero: the board priced on the current tree

- [x] **S0.1 The format candidates priced offline.** DONE 2026-09-18: the palette
      (B) closed, every group uses all sixteen levels on layers 5 to 39 (layer 0's
      zero spike aside, 0.7 % of the model); the aux table (C) holds at 11 bits,
      3.5 % of the stride at no decoder cost; the entropy code (A) is the only
      path to the 12 %, its per-lane length header 3.6 % of a row, A with C about
      11 %. The head's rows and the per-layer worst expert deferred to the coder's
      first run at Task 1. Script, JSON and log at
      `~/.claude/handoffs/archive/shrike-v21-step0/`.
- [x] **S0.2 The pool's prize.** DONE 2026-09-18: the replay on the eight v19
      and the four Q3 traces; at the coded stride (0.88) the production pool's
      decode misses cut 22 to 29 % (145 slots uniform, the table scaled to
      5,820), about 2.5 to 3.5 ms per token modelled, the largest row; C alone
      (0.96) 8 to 9 %. The prize belongs to the in-lane design only. Script,
      JSON and log at `~/.claude/handoffs/archive/shrike-v21-step0/`.
- [x] **S0.3 The read-size probe on the mini's drive.** DONE 2026-09-18: serial
      `pread`, three seeds; the drive scales a single read with its size (0.16 ms
      fixed, 0.35 ms per MB); the coded read at 0.89 of the stride 0.69 to 0.71 ms
      against 0.76 to 0.77, 0.9 to 1.2 ms per token at today's misses. Probe and
      runs at `~/.claude/handoffs/archive/shrike-v21-step0/`.
- [x] **S0.4 The in-lane decoder microbench on the mini.** DONE 2026-09-18, the
      gate failed: `ShrikeExpertBench` (a new executable target, kept) runs the
      production phase-1 kernel on eight real experts of a layer against a coded
      kernel with the same arithmetic in the same order (bit-identical on every
      arm, layer and box); on the mini the plain kernel runs at the roof (60.5 to
      63.0 GB/s) and the coded at 9 to 10 GB/s, 5.3 to 6.9× slower, 6.1 to 7.7×
      with the aux table; the bytes 0.986 (indices) and 0.958 (with aux) of plain
      on the typical layer. Runs on both boxes at
      `~/.claude/handoffs/archive/shrike-v21-step0/`.
- [x] **S0.5 The record and the ruling.** DONE 2026-09-18: Davor's ruling, closed
      at step zero; the three venues to be tried in Claude's order (the RAM
      ledger, speculation revisited, B3/B4).

## Tasks

None: the gate at S0.4 failed, and the recommendation at S0.5 is to close the
chapter at step zero. The close items below are the ones that apply to a
chapter that changed no runtime code.

## The close

- [x] Davor's ruling on S0.5 recorded in the design document (2026-09-18).
- [x] The bench, the documents and the ledger's H2 entry committed on the
      four gates (release build with zero warnings, lint, links, the full suite
      1,293 tests in 176 suites), main fast-forwarded (2026-09-18).
- [x] `docs/architecture.md`'s instruments: `--dump-hidden` (the Q3 close) and
      `ShrikeExpertBench` added (2026-09-18).
- [x] Production on the mini untouched (no runtime change); the bench binary
      and its bundle removed from the mini's `bin/` (2026-09-18).
