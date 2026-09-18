#!/usr/bin/env python3
"""A9's Q3: the MTP drafter's hidden state through the forty main routers as a
predictor of the next pass's routes (docs/v18-avenues.md, A9; the premise in
the v18 chapter's record).

Reads one CLI run of a shape: the hidden dump (`--dump-hidden`, every
position's residual before the final norm), the logits dump (`--dump-logits`,
the answer's ids and the validation of the head), the prompt's ids
(`--tokenize`) and the route trace (`SHRIKE_ROUTE_TRACE`). Replays the
drafter, one block, over the whole sequence in fp32 from the sidecar's own
`.gturbo`, then scores six predictors of each decode position's top-8 per
layer against the trace:

  drafter-out  the drafter's output vector at q-1 through layer L's post
               norm and router
  drafter-in   the drafter's input (fc of the previous final state and the
               next id's embedding) through the same
  prev-final   the main model's own final residual at q-1 through the same
  embed        the embedding of the id fed at q through the same
  id-table     the route the same id took at its last earlier occurrence
  previous     the route at q-1

and three diagnostics that are not predictors: drafter-out-normed (the
drafter's output after its own final norm, the head's view of it),
final-same (the main model's final residual at q itself, which exists only
once the pass has run: whether a router can read an end-of-network vector at
all) and popular (the eight most frequent experts of the layer over the
prompt: the baseline a predictor must beat).

Two validations gate the replica: the main head over the dumped residuals
must reproduce the dumped logits' argmax, and the drafter's own token guess
must land in the band v12's P17 measured (20.6 / 31.7 / 75.6 % by shape).

Writes <out>/q3-<shape>.json with every number, and per arm a prefetch-trace
JSONL (<out>/q3-fills-<shape>-<arm>.jsonl) that tools/expert-pool-replay.py
takes as `--speculative-fills`, so a recall becomes misses saved per token.

Needs numpy only:
  <venv>/bin/python tools/q3-drafter-routes.py --main <ornith15.gturbo> \
      --mtp <ornith15-mtp.gturbo> --hidden hidden.bin --logits logits.bin \
      --tokens tokenize.json --trace route.trace --shape card --out q3/
"""
from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys
import time

import numpy as np

D = 2048
N_Q_HEADS = 16
N_KV_HEADS = 2
HEAD_DIM = 256
ROTARY = 64
THETA = 10_000_000.0
SCALE = 0.0625
EPS = 1e-6
N_EXPERTS = 256
TOP_K = 8
N_LAYERS = 40
GROUP = 64
GROUPS = [("0-3", range(0, 4)), ("0-9", range(0, 10)), ("10-19", range(10, 20)),
          ("20-29", range(20, 30)), ("30-39", range(30, 40)), ("all", range(0, 40))]
ARMS = ["drafter-out", "drafter-out-normed", "drafter-in", "prev-final", "final-same", "embed",
        "id-table", "previous", "popular"]


# ---------------------------------------------------------------- weights

def read_index(path):
    with open(path, "rb") as handle:
        index_size, _resident, entry_count = struct.unpack("<QQQ", handle.read(24))
        handle.seek(0)
        region = handle.read(index_size)
    entries = {}
    for i in range(entry_count):
        off = 24 + i * 72
        name_off, name_len = struct.unpack_from("<IH", region, off)
        name = region[name_off:name_off + name_len].decode()
        file_off, size = struct.unpack_from("<QQ", region, off + 8)
        shape = struct.unpack_from("<4I", region, off + 24)
        scale_off, scale_size, bias_off, bias_size = struct.unpack_from(
            "<QQQQ", region, off + 40)
        entries[name] = dict(dtype=region[off + 6], offset=file_off, size=size,
                             shape=[d for d in shape if d],
                             scale=(scale_off, scale_size), bias=(bias_off, bias_size))
    return entries


def bf16_to_f32(raw):
    u16 = np.frombuffer(raw, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32)


def unpack_affine(packed, rows, cols, scales, biases):
    """4-bit (low nibble first) or 8-bit codes, one bf16 scale and bias per
    group of 64 along the row: the `.gturbo` affine format."""
    if packed.size == rows * cols:
        q = packed.reshape(rows, cols).astype(np.float32)
    elif packed.size * 2 == rows * cols:
        packed = packed.reshape(rows, cols // 2)
        q = np.empty((rows, cols), dtype=np.float32)
        q[:, 0::2] = (packed & 0x0F).astype(np.float32)
        q[:, 1::2] = (packed >> 4).astype(np.float32)
    else:
        raise SystemExit(f"affine tensor of {packed.size} bytes does not fit {rows}x{cols}")
    groups = cols // GROUP
    scales = scales.reshape(rows, groups)
    biases = biases.reshape(rows, groups)
    return q * np.repeat(scales, GROUP, axis=1) + np.repeat(biases, GROUP, axis=1)


class WeightFile:
    def __init__(self, gturbo):
        self.path = pathlib.Path(gturbo) / "model_weights.bin"
        self.entries = read_index(self.path)
        self.handle = open(self.path, "rb")

    def raw(self, offset, size):
        self.handle.seek(offset)
        return self.handle.read(size)

    def vector(self, name):
        entry = self.entries[name]
        assert entry["dtype"] == 1, name
        return bf16_to_f32(self.raw(entry["offset"], entry["size"])).reshape(entry["shape"])

    def matrix(self, name, rows=None):
        """The dequantized [rows, cols] tensor, or only the given rows (the
        embedding and the head are 248,320 rows each)."""
        entry = self.entries[name]
        if entry["dtype"] == 1:
            return self.vector(name)
        n_rows, cols = entry["shape"]
        groups = cols // GROUP
        if rows is None:
            packed = np.frombuffer(self.raw(entry["offset"], entry["size"]), dtype=np.uint8)
            scales = bf16_to_f32(self.raw(*entry["scale"]))
            biases = bf16_to_f32(self.raw(*entry["bias"]))
            return unpack_affine(packed, n_rows, cols, scales, biases)
        row_bytes = entry["size"] // n_rows
        out = np.empty((len(rows), cols), dtype=np.float32)
        for i, row in enumerate(rows):
            packed = np.frombuffer(self.raw(entry["offset"] + row * row_bytes, row_bytes),
                                   dtype=np.uint8)
            scales = bf16_to_f32(self.raw(entry["scale"][0] + row * groups * 2, groups * 2))
            biases = bf16_to_f32(self.raw(entry["bias"][0] + row * groups * 2, groups * 2))
            out[i] = unpack_affine(packed, 1, cols, scales, biases)[0]
        return out


class PackedExperts:
    """The sidecar's 256 private experts from `packed_experts/layer_00.bin`,
    dequantized on demand and kept."""

    def __init__(self, gturbo):
        layout = json.load(open(pathlib.Path(gturbo) / "packed_experts" / "layout.json"))
        layer = layout["layers"][0]
        self.file = open(pathlib.Path(gturbo) / "packed_experts" / layer["file"], "rb")
        self.experts = {e["expert"]: e for e in layer["experts"]}
        self.cache = {}

    def tensor(self, expert, role):
        base = self.experts[expert]["offset"]
        tensors = self.experts[expert]["tensors"]
        w = tensors[role]
        s = tensors[role + "_scales"]
        b = tensors[role + "_biases"]
        rows, cols = w["shape"]
        self.file.seek(base + w["offset"])
        packed = np.frombuffer(self.file.read(w["size"]), dtype=np.uint8)
        self.file.seek(base + s["offset"])
        scales = bf16_to_f32(self.file.read(s["size"]))
        self.file.seek(base + b["offset"])
        biases = bf16_to_f32(self.file.read(b["size"]))
        return unpack_affine(packed, rows, cols, scales, biases)

    def get(self, expert):
        if expert not in self.cache:
            self.cache[expert] = tuple(self.tensor(expert, role)
                                       for role in ("gate", "up", "down"))
        return self.cache[expert]


# ---------------------------------------------------------------- math

def rmsnorm(x, weight):
    x = x.astype(np.float32)
    denom = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS)
    return x / denom * weight


def silu(x):
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def rope(x, positions):
    """NeoX pairing on the first ROTARY dims of every head: (i, ROTARY/2 + i)."""
    half = ROTARY // 2
    inv = THETA ** (-np.arange(half, dtype=np.float64) * 2 / ROTARY)
    angles = positions[:, None].astype(np.float64) * inv[None, :]
    cos = np.cos(angles).astype(np.float32)[None, :, :]
    sin = np.sin(angles).astype(np.float32)[None, :, :]
    r1 = x[..., :half]
    r2 = x[..., half:ROTARY]
    out = x.copy()
    out[..., :half] = r1 * cos - r2 * sin
    out[..., half:ROTARY] = r2 * cos + r1 * sin
    return out


def top8(logits):
    """The select kernel's order: the largest logit first, the lower index on a tie."""
    idx = np.argpartition(-logits, TOP_K - 1)[:TOP_K]
    order = np.lexsort((idx, -logits[idx]))
    return idx[order]


# ---------------------------------------------------------------- the drafter

class Drafter:
    def __init__(self, main: WeightFile, mtp: WeightFile, experts: PackedExperts):
        p = "language_model.model.layers.0."
        self.w_e = mtp.vector("pre_fc_norm_embedding.weight")
        self.w_h = mtp.vector("pre_fc_norm_hidden.weight")
        self.fc = mtp.matrix("fc.weight")                       # [2048, 4096]
        self.w_in = mtp.vector(p + "input_layernorm.weight")
        self.w_post = mtp.vector(p + "post_attention_layernorm.weight")
        self.wq = mtp.matrix(p + "self_attn.q_proj.weight")     # [8192, 2048]
        self.wk = mtp.matrix(p + "self_attn.k_proj.weight")     # [512, 2048]
        self.wv = mtp.matrix(p + "self_attn.v_proj.weight")
        self.wo = mtp.matrix(p + "self_attn.o_proj.weight")     # [2048, 4096]
        self.q_norm = mtp.vector(p + "self_attn.q_norm.weight")
        self.k_norm = mtp.vector(p + "self_attn.k_norm.weight")
        self.router = mtp.matrix(p + "mlp.gate.weight")         # [256, 2048]
        self.shared_gate = mtp.matrix(p + "mlp.shared_expert_gate.weight")[0]
        self.s_gate = mtp.matrix(p + "mlp.shared_expert.gate_proj.weight")
        self.s_up = mtp.matrix(p + "mlp.shared_expert.up_proj.weight")
        self.s_down = mtp.matrix(p + "mlp.shared_expert.down_proj.weight")
        self.w_final = mtp.vector("language_model.model.norm.weight")
        self.experts = experts

    def inputs(self, hidden, next_embed):
        cat = np.concatenate([rmsnorm(next_embed, self.w_e), rmsnorm(hidden, self.w_h)], axis=1)
        return cat @ self.fc.T

    def attention(self, x, positions, block=256):
        n = x.shape[0]
        normed = rmsnorm(x, self.w_in)
        packed = (normed @ self.wq.T).reshape(n, N_Q_HEADS, 2 * HEAD_DIM)
        q = rmsnorm(packed[:, :, :HEAD_DIM], self.q_norm).transpose(1, 0, 2)
        gate = packed[:, :, HEAD_DIM:]
        k = rmsnorm((normed @ self.wk.T).reshape(n, N_KV_HEADS, HEAD_DIM),
                    self.k_norm).transpose(1, 0, 2)
        v = (normed @ self.wv.T).reshape(n, N_KV_HEADS, HEAD_DIM).transpose(1, 0, 2)
        q = rope(q, positions)
        k = rope(k, positions)
        rep = N_Q_HEADS // N_KV_HEADS
        k_g = np.repeat(k, rep, axis=0)
        v_g = np.repeat(v, rep, axis=0)
        out = np.empty((n, N_Q_HEADS, HEAD_DIM), dtype=np.float32)
        for start in range(0, n, block):
            stop = min(n, start + block)
            scores = np.einsum("hqd,hkd->hqk", q[:, start:stop], k_g[:, :stop]) * SCALE
            rows = np.arange(start, stop)[:, None]
            scores[:, rows[:, 0] - start, :] = np.where(
                np.arange(stop)[None, :] <= rows, scores[:, :, :], -np.inf)
            scores -= scores.max(axis=-1, keepdims=True)
            probs = np.exp(scores)
            probs /= probs.sum(axis=-1, keepdims=True)
            out[start:stop] = np.einsum("hqk,hkd->qhd", probs, v_g[:, :stop])
        gated = out * sigmoid(gate)
        return gated.reshape(n, N_Q_HEADS * HEAD_DIM) @ self.wo.T

    def moe(self, a):
        m = rmsnorm(a, self.w_post)
        shared = (silu(m @ self.s_gate.T) * (m @ self.s_up.T)) @ self.s_down.T
        shared *= sigmoid(m @ self.shared_gate)[:, None]
        logits = m @ self.router.T
        routed = np.zeros_like(a)
        for i in range(a.shape[0]):
            chosen = top8(logits[i])
            weights = np.exp(logits[i][chosen] - logits[i][chosen].max())
            weights /= weights.sum()
            for expert, weight in zip(chosen, weights):
                g, u, d = self.experts.get(int(expert))
                routed[i] += weight * ((silu(m[i] @ g.T) * (m[i] @ u.T)) @ d.T)
        return a + shared + routed

    def forward(self, hidden, next_embed, positions):
        x = self.inputs(hidden, next_embed)
        a = x + self.attention(x, positions)
        return x, self.moe(a)


# ---------------------------------------------------------------- inputs

def load_rows(path, width):
    raw = np.fromfile(path, dtype=np.float16)
    return raw.reshape(-1, width).astype(np.float32)


def parse_trace(path):
    """(position, layer) -> experts for the decode plans (the bare lines) and
    the prefill rows (`q`); the request line; the decode ids by position. The
    `p` lines are prefill tile plans, not routes."""
    decode, prefill, ids, request = {}, {}, {}, None
    with open(path) as handle:
        for raw in handle:
            parts = raw.split()
            if not parts:
                continue
            kind = parts[0]
            if kind == "r":
                request = (int(parts[1]), int(parts[2]))
            elif kind == "t":
                ids[int(parts[1])] = int(parts[2])
            elif kind == "q":
                prefill[(int(parts[1]), int(parts[2]))] = [int(e) for e in parts[3:3 + TOP_K]]
            elif kind[0].isdigit():
                decode[(int(parts[0]), int(parts[1]))] = [int(e) for e in parts[2:2 + TOP_K]]
    return decode, prefill, ids, request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", required=True, help="the served model's .gturbo")
    parser.add_argument("--mtp", required=True, help="the MTP sidecar's .gturbo")
    parser.add_argument("--hidden", required=True, help="--dump-hidden path (sidecar beside it)")
    parser.add_argument("--logits", required=True, help="--dump-logits path (sidecar beside it)")
    parser.add_argument("--tokens", required=True, help="--tokenize JSON of the same prompt")
    parser.add_argument("--trace", required=True, help="SHRIKE_ROUTE_TRACE of the same run")
    parser.add_argument("--shape", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--head-check", type=int, default=64,
                        help="decode positions to validate the main head on (0: skip)")
    args = parser.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    prompt = json.load(open(args.tokens))["ids"]
    logits_meta = json.load(open(args.hidden + ".json"))
    hidden_positions = logits_meta["positions"]
    hidden = load_rows(args.hidden, logits_meta["hidden"])
    chosen = json.load(open(args.logits + ".json"))["chosen"]
    n_prompt = len(prompt)
    tokens = list(prompt) + list(chosen)
    n_rows = hidden.shape[0]
    if hidden_positions != list(range(n_rows)) or n_rows != n_prompt + len(chosen) - 1:
        raise SystemExit(f"hidden rows {n_rows} at {hidden_positions[:3]}.. do not match "
                         f"prompt {n_prompt} + answer {len(chosen)} - 1")
    decode, prefill, trace_ids, request = parse_trace(args.trace)
    if request and request[1] != n_prompt:
        raise SystemExit(f"trace prompt {request[1]} != tokenize {n_prompt}")
    for position, ident in trace_ids.items():
        if tokens[position] != ident:
            raise SystemExit(f"trace id at {position} is {ident}, the run fed {tokens[position]}")
    decode_positions = sorted({p for (p, _l) in decode})
    print(f"{args.shape}: prompt {n_prompt} ids, answer {len(chosen)}, hidden rows {n_rows}, "
          f"decode plans at {len(decode_positions)} positions "
          f"({decode_positions[0]}..{decode_positions[-1]})", flush=True)

    main_w = WeightFile(args.main)
    mtp_w = WeightFile(args.mtp)
    lm_head = main_w.matrix("language_model.lm_head.weight")
    w_norm = main_w.vector("language_model.model.norm.weight")
    print(f"  head loaded ({time.time() - t0:.0f} s)", flush=True)

    # Validation A: the main head over the dumped residuals.
    logits_rows = load_rows(args.logits, json.load(open(args.logits + ".json"))["vocab"])
    head_agree, head_total = 0, 0
    for g in range(min(args.head_check, len(chosen))):
        row = n_prompt - 1 + g
        if row >= n_rows:
            break
        predicted = int(np.argmax(rmsnorm(hidden[row], w_norm) @ lm_head.T))
        head_agree += int(predicted == int(np.argmax(logits_rows[g])))
        head_total += 1
    print(f"  head check: {head_agree}/{head_total} argmax agree", flush=True)

    # The embeddings of every id the sequence feeds.
    unique_ids = sorted(set(tokens))
    embed_rows = main_w.matrix("language_model.model.embed_tokens.weight", rows=unique_ids)
    embed_of = {ident: embed_rows[i] for i, ident in enumerate(unique_ids)}
    next_embed = np.stack([embed_of[tokens[p + 1]] for p in range(n_rows)])
    embed_at = np.stack([embed_of[tokens[p]] for p in range(len(tokens))])

    # The drafter over the whole sequence.
    drafter = Drafter(main_w, mtp_w, PackedExperts(args.mtp))
    print(f"  drafter loaded ({time.time() - t0:.0f} s)", flush=True)
    x_in, x_out = drafter.forward(hidden, next_embed, np.arange(n_rows))
    print(f"  drafter run ({time.time() - t0:.0f} s)", flush=True)

    # Validation B: the drafter's token guess at p is token p+2.
    accept, total = 0, 0
    for p in range(max(0, n_prompt - 2), n_rows):
        if p + 2 >= len(tokens):
            break
        guess = int(np.argmax(rmsnorm(x_out[p], drafter.w_final) @ lm_head.T))
        accept += int(guess == tokens[p + 2])
        total += 1
    acceptance = accept / total if total else float("nan")
    print(f"  drafter acceptance on the answer: {accept}/{total} = {acceptance:.3f}", flush=True)
    del lm_head

    # The forty routers and their post-attention norms.
    routers, post_norms = [], []
    for layer in range(N_LAYERS):
        p = f"language_model.model.layers.{layer}."
        routers.append(main_w.matrix(p + "mlp.gate.weight"))
        post_norms.append(main_w.vector(p + "post_attention_layernorm.weight"))

    # The arms.
    vectors = {
        "drafter-out": lambda q: x_out[q - 1],
        "drafter-out-normed": lambda q: rmsnorm(x_out[q - 1], drafter.w_final),
        "drafter-in": lambda q: x_in[q - 1],
        "prev-final": lambda q: hidden[q - 1],
        "final-same": lambda q: hidden[q],
        "embed": lambda q: embed_at[q],
    }
    counts = np.zeros((N_LAYERS, N_EXPERTS), dtype=np.int64)
    for (_position, layer), experts in prefill.items():
        counts[layer, experts] += 1
    popular = [[int(e) for e in np.argsort(-counts[layer], kind="stable")[:TOP_K]]
               for layer in range(N_LAYERS)]
    overlaps = {arm: np.zeros((N_LAYERS, len(decode_positions))) for arm in ARMS}
    fills = {arm: [] for arm in ARMS}
    last_route = {}
    for (position, layer), experts in prefill.items():
        last_route[(tokens[position], layer)] = (position, experts)
    for column, q in enumerate(decode_positions):
        for layer in range(N_LAYERS):
            actual = set(decode[(q, layer)])
            predictions = {}
            for arm, vector in vectors.items():
                logits = rmsnorm(vector(q), post_norms[layer]) @ routers[layer].T
                predictions[arm] = [int(e) for e in top8(logits)]
            seen = last_route.get((tokens[q], layer))
            predictions["id-table"] = list(seen[1]) if seen and seen[0] < q else []
            earlier = decode.get((q - 1, layer)) or prefill.get((q - 1, layer)) or []
            predictions["previous"] = list(earlier)
            predictions["popular"] = popular[layer]
            for arm, predicted in predictions.items():
                overlaps[arm][layer, column] = len(actual & set(predicted)) / TOP_K
                if predicted:
                    fills[arm].append({"layer": layer, "probe_distance": 0, "position": q,
                                       "next_layer_prediction": predicted})
        for layer in range(N_LAYERS):
            last_route[(tokens[q], layer)] = (q, decode[(q, layer)])
    print(f"  arms scored ({time.time() - t0:.0f} s)", flush=True)

    summary = {arm: {name: float(overlaps[arm][list(members)].mean())
                     for name, members in GROUPS} for arm in ARMS}
    per_layer = {arm: [float(v) for v in overlaps[arm].mean(axis=1)] for arm in ARMS}
    print(f"\n  top-8 overlap with the actual route, {len(decode_positions)} decode positions")
    print("  arm          " + "".join(f"{name:>8}" for name, _ in GROUPS))
    for arm in ARMS:
        print(f"  {arm:<12} " + "".join(f"{summary[arm][name]:>8.3f}" for name, _ in GROUPS))

    for arm in ARMS:
        with open(out / f"q3-fills-{args.shape}-{arm}.jsonl", "w") as handle:
            for row in fills[arm]:
                handle.write(json.dumps(row) + "\n")
    result = {
        "shape": args.shape, "prompt_tokens": n_prompt, "answer_tokens": len(chosen),
        "decode_positions": len(decode_positions),
        "head_check": {"agree": head_agree, "total": head_total},
        "drafter_acceptance": {"accepted": accept, "total": total, "rate": acceptance},
        "overlap": summary, "overlap_per_layer": per_layer,
        "seconds": time.time() - t0,
    }
    with open(out / f"q3-{args.shape}.json", "w") as handle:
        json.dump(result, handle, indent=1)
    print(f"  wrote {out / f'q3-{args.shape}.json'} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
