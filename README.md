# Shrike

An inference engine for mixture-of-experts models that are larger than the machine's
RAM. Routed experts stay on SSD and are read per token into a bounded cache, so the
memory ceiling is a number you declare rather than one you discover.

It serves an OpenAI-compatible HTTP API on loopback, and also ships a CLI, a native
Mac app, and a repacking tool.

## Requirements

- Apple Silicon Mac
- macOS 26 or later
- Swift 6.3 or later
- Disk space for the models you install — these are 35B-class checkpoints

## Build

```bash
swift build -c release
```

Products land in `.build/release/`: `ShrikeServer`, `ShrikeCLI`, `ShrikeRepack`,
`ShrikeMac`, `ShrikeBench`, and `ShrikeDecodeService` (an out-of-process decode
helper the Mac app spawns; not run directly).

## Install a model

Models are converted into the `.gturbo` format, which stores routed experts in a
layout that can be read a single expert at a time.

```bash
swift run -c release ShrikeRepack --help
```

An install writes a `verified-install.json` receipt bound to the absolute path it was
installed to. Moving or renaming an installed model therefore makes it fail to load;
re-issue the receipt in place rather than editing it:

```bash
swift run -c release ShrikeRepack --verify-install --input-gturbo <model.gturbo>
```

## Serve

Config mode is the default. With no `--model`, the server scans a models directory
and serves everything it finds, reading `~/.shrike/server.json` unless given
`--config`:

```bash
.build/release/ShrikeServer
```

To serve exactly one model and ignore any config or roster:

```bash
.build/release/ShrikeServer --model models/<name>.gturbo
```

`--help` lists the full flag set. The two worth knowing first:

- `--ram-budget <size>` — bytes the routed-expert cache may use (default `8G`).
  This is the knob; slot count is derived from it and the model's expert stride.
  Smaller budgets are markedly slower, because expert reads bypass the page cache
  and have no fallback.
- `--kv-bits <4|8|16>` — KV-cache precision, independent of model quantization
  (default 8).

Any OpenAI-compatible client can point at the loopback endpoint. The repository no
longer ships launcher scripts; a server has no business shipping its own launcher,
and client configuration belongs on the client.

## Supported models

Three architecture families, in 4-bit and 8-bit:

| Family | Models | Chat dialect |
| --- | --- | --- |
| Qwen3.5/3.6 MoE | Ornith 1.5 35B-A3B, Qwen 3.6 35B-A3B | ChatML |
| gpt-oss | gpt-oss 20B | Harmony |
| Kimi Linear | Kimi Linear 48B | Kimi |

Other tensor-compatible Qwen-based MoE checkpoints are usually straightforward to
add, since the parser, repacker, and inference path are shared.

## Notable behaviour

- **Long context.** Native RoPE to 262K tokens; optional YaRN extends to 512K or 1M.
- **Compressed KV cache.** 16-, 8-, or 4-bit, independent of model quantization.
- **Thinking mode.** Off/on/adaptive for the Qwen-family templates. gpt-oss
  cannot disable thinking; its knob is `--reasoning-effort low|medium|high`
  (per-request via the OpenAI `reasoning_effort` field), and `--thinking off`
  on a Harmony model warns and maps to effort `low`.
- **MTP is off by default.** Speculative decoding is experimental; measured runs
  showed no benefit, and it requires greedy decoding, native RoPE, and prompt-cache
  reuse disabled.
- **ANE prefill is off by default.** Opt-in via `SHRIKE_PREFILL_ANE=on`, worth a
  measured 2.31x on prefill for one qualified prompt. See
  [docs/ane-prefill.md](docs/ane-prefill.md).

## Performance

There are no published numbers in this repository, deliberately. Throughput depends
on the machine, the model, the quantization, and the RAM budget, and figures measured
on someone else's hardware do not transfer.

To measure your own:

```bash
tools/golden-baseline.sh --check 4
```

A baseline is valid for one (machine, build, model) triple and must be captured on
the machine it will be checked against. It is the only check in the repository that
exercises real inference — the unit tests never load a model.

## Documentation

- [docs/architecture.md](docs/architecture.md) — how the engine works and why
- [docs/ane-prefill.md](docs/ane-prefill.md) — Neural Engine prefill, opt-in
- [docs/multi-model-serving.md](docs/multi-model-serving.md) — serving several models
- [docs/v5-second-architecture-objective.md](docs/v5-second-architecture-objective.md)
- [docs/v6-dialect-normalized-cache.md](docs/v6-dialect-normalized-cache.md)

## License

See [LICENSE](LICENSE).
