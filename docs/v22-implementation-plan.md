# v22 implementation plan: the pool's capacity

Companion to [v22-pool-capacity.md](v22-pool-capacity.md). Checkboxes here are
the only status tracking. Branch: `perf/v22-pool-capacity` off `main` at
`57c7f2f`.

Protocol per task: the four gates per commit (release build with zero warnings,
`swiftlint lint --strict`, `tools/check-md-links.py`, `swift test --no-parallel`);
the golden byte-identical on all five profiles on both boxes for every commit;
deploy to the mini with `tools/mini-deploy.sh` (binary plus the `*.bundle`
directories); the arms rig, two production lifetimes per shape on four shapes,
read against today's production configuration, every arm carrying misses per
token, the io, the token and the box's memory pressure; ThreadSanitizer once
at the close.

## Step zero

- [x] **S0.1 The RAM ledger** on the mini under load. DONE 2026-09-18.
- [x] **S0.2 The prize per slot** by replay. DONE 2026-09-18.
- [x] **S0.3 The wall**: the single-buffer arena against the device's 8.88 GiB.
      READ 2026-09-18 from v16's record.

## Tasks

- [x] **Task 1 The prefill scratch released.** DONE 2026-09-18, `9e02f3b`: the
      four gates green (1,294 tests in 176 suites).
- [x] **Task 2 The arena in chunks.** DONE 2026-09-18: the arena over chunks
      under the device's limit with `PoolBases` for the kernels; the two
      speculative kernels and their encoders; the streamer's identity and
      binding offsets; the allowed slot counts to 256; the tests; the four
      gates green (1,297 tests in 176 suites); the golden byte-identical on
      this box. The mini's golden at two chunks is Task 3's.
- [ ] **Task 3 The mini's configuration.** oMLX restarted; the budget raised
      to the headroom's slot count with the table scaled; the arms against
      today's production; production relaunched at the winning configuration;
      the working instructions' launch line updated.

## The close

- [ ] ThreadSanitizer over the whole suite.
- [ ] The whole-branch review, its findings folded into the owning commits.
- [ ] `docs/architecture.md` at the close's tree (the arena's chunks, the
      allowed counts, the scratch's lifetime, the production launch).
- [ ] Production on the mini at the close's build and configuration,
      golden-verified on both boxes.
