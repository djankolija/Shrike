#!/usr/bin/env python3
"""Convert Kimi-Linear's tiktoken vocabulary into a Hugging Face
`tokenizer.json` with a ByteLevel decoder (the streaming-decoder contract
NVMAI requires).

The upstream checkpoint ships no `tokenizer.json` -- only `tiktoken.model`
plus a custom `TikTokenTokenizer` class. This tool reproduces that class's
construction (byte-level BPE from the tiktoken ranks, its split pattern, and
all 258 special slots above the base vocabulary) as a plain tokenizers file.

Usage:
    python3 convert_kimi_tokenizer.py --snapshot <checkpoint-dir> \
        --output <tokenizer.json> [--verify]

Requires `tokenizers`; `--verify` additionally requires `tiktoken` and
compares both implementations over a battery of texts, e.g.:
    uv run --with tokenizers --with tiktoken python3 \
        tools/convert_kimi_tokenizer.py --snapshot <dir> --output <file> --verify
"""

import argparse
import base64
import json
import sys
from pathlib import Path

from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, pre_tokenizers

# The split pattern of `tokenization_kimi.py` (class attribute `pat_str`),
# verbatim. Oniguruma (the tokenizers regex engine) supports the character
# class intersections and inline case-insensitive groups it uses.
PAT_STR = "|".join(
    [
        r"""[\p{Han}]+""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ]
)

NUM_SPECIAL_SLOTS = 256 + 2

VERIFY_TEXTS = [
    "The capital of France is Paris.",
    "I'll say they're right; we've done it. It's fine.",
    "12345 678 9 1000000",
    "你好，世界。今天天气很好。",
    "mixed 中文 and English words",
    "a  b\n\n  c\t d   \n",
    "café naïve 🚀🎉 über",
    "def f(x):\n    return x*2  # comment\n",
    "<|im_user|>user<|im_middle|>Hello<|im_end|>",
    "<|im_assistant|>assistant<|im_middle|>Sure — here: [1, 2]<|im_end|>[EOS]",
    "<|tool_calls_section_begin|><|tool_call_begin|>f:0<|tool_call_argument_begin|>"
    '{"a":1}<|tool_call_end|><|tool_calls_section_end|>',
    "",
]


def bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def load_tiktoken_ranks(path):
    ranks = {}
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            token_b64, rank = line.split()
            ranks[base64.b64decode(token_b64)] = int(rank)
    if sorted(ranks.values()) != list(range(len(ranks))):
        raise SystemExit("tiktoken ranks are not dense from 0")
    return ranks


def special_slot_contents(snapshot, num_base):
    config_path = snapshot / "tokenizer_config.json"
    named = {}
    flags = {}
    with open(config_path) as f:
        config = json.load(f)
    for token_id, entry in config["added_tokens_decoder"].items():
        named[int(token_id)] = entry["content"]
        flags[int(token_id)] = bool(entry["special"])
    for token_id in named:
        if not num_base <= token_id < num_base + NUM_SPECIAL_SLOTS:
            raise SystemExit(f"added token {token_id} outside the special slot range")
    return [
        (named.get(i, f"<|reserved_token_{i}|>"), flags.get(i, True))
        for i in range(num_base, num_base + NUM_SPECIAL_SLOTS)
    ]


def build_tokenizer(ranks, specials):
    byte_encoder = bytes_to_unicode()

    def to_string(token_bytes):
        return "".join(byte_encoder[b] for b in token_bytes)

    vocab = {to_string(token): rank for token, rank in ranks.items()}
    merges = []
    for token, rank in ranks.items():
        if len(token) == 1:
            continue
        local = []
        for split in range(1, len(token)):
            left, right = token[:split], token[split:]
            if left in ranks and right in ranks:
                local.append((left, right))
        local.sort(key=lambda pair: (ranks[pair[0]], ranks[pair[1]]))
        merges.extend((to_string(l), to_string(r), rank) for l, r in local)
    merges.sort(key=lambda entry: entry[2])
    merge_pairs = [(l, r) for l, r, _ in merges]

    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=merge_pairs,
                                     fuse_unk=False, byte_fallback=False))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(PAT_STR), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tokenizer.decoder = decoders.ByteLevel()
    for content, special in specials:
        added = [AddedToken(content, normalized=False, special=special)]
        if special:
            tokenizer.add_special_tokens(added)
        else:
            tokenizer.add_tokens(added)
    return tokenizer


def check_assigned_ids(tokenizer, specials, num_base):
    for offset, (content, _) in enumerate(specials):
        expected = num_base + offset
        actual = tokenizer.token_to_id(content)
        if actual != expected:
            raise SystemExit(
                f"added token {content!r} landed at {actual}, expected {expected}")


def verify(tokenizer, ranks, specials, num_base):
    import tiktoken

    reference = tiktoken.Encoding(
        name="kimi",
        pat_str=PAT_STR,
        mergeable_ranks=ranks,
        special_tokens={content: num_base + offset
                        for offset, (content, _) in enumerate(specials)},
    )
    failures = 0
    for text in VERIFY_TEXTS:
        expected = reference.encode(text, allowed_special="all")
        actual = tokenizer.encode(text, add_special_tokens=False).ids
        if actual != expected:
            failures += 1
            print(f"MISMATCH on {text!r}:")
            print(f"  tiktoken : {expected}")
            print(f"  converted: {actual}")
        if reference.decode(expected) != text:
            failures += 1
            print(f"tiktoken decode round-trip failed on {text!r}")
        round_trip = tokenizer.decode(actual, skip_special_tokens=False)
        if round_trip != text:
            failures += 1
            print(f"converted decode round-trip failed on {text!r}: {round_trip!r}")
    if failures:
        raise SystemExit(f"verification failed: {failures} mismatches")
    print(f"verified: {len(VERIFY_TEXTS)} texts agree with tiktoken")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True,
                        help="checkpoint directory holding tiktoken.model "
                             "and tokenizer_config.json")
    parser.add_argument("--output", required=True, help="tokenizer.json path")
    parser.add_argument("--verify", action="store_true",
                        help="compare against tiktoken over a text battery")
    args = parser.parse_args()

    snapshot = Path(args.snapshot)
    ranks = load_tiktoken_ranks(snapshot / "tiktoken.model")
    num_base = len(ranks)
    specials = special_slot_contents(snapshot, num_base)
    tokenizer = build_tokenizer(ranks, specials)
    check_assigned_ids(tokenizer, specials, num_base)
    if args.verify:
        verify(tokenizer, ranks, specials, num_base)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output))
    print(f"wrote {output} (base vocab {num_base}, "
          f"{NUM_SPECIAL_SLOTS} special slots)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
