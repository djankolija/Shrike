#!/usr/bin/env python3
"""The per-position logit comparison (docs/v19-scan-rewrite.md, Task 1).

Two dumps from `shrike generate --dump-logits`, the same prompt on two builds,
compared position by position: the KL divergence old to new, the largest logit
difference, and every argmax flip with the old build's top-2 margin. Give both
runs `--temperature 0 --seed <n>` so they decode one sequence for as long as they
agree; v24 retired `--force-tokens`, which held them on one sequence by
construction, so `forced` is null in every sidecar written since, and the
comparison stops at the first position whose chosen ids differ: every row after
it is conditioned on a different history. The band is a multiple of the median over positions of the largest
logit difference (the median, so one bad position cannot widen the band and
hide the rest); a flip inside the band is variance, a flip outside it is a
defect, a position whose largest difference is far above the median is a defect
on its own, a non-finite logit anywhere is a defect, and the exit status says
which. Pure Python: neither box has numpy.

A dump is raw little-endian float16 rows, vocab values each, beside a
`<dump>.json` sidecar with vocab, positions, chosen and (null since v24) forced
ids; the file's length must be vocab * positions * 2 bytes or the run is refused.
"""
import argparse
import heapq
import json
import math
import os
import struct
import sys

OUTLIER_FACTOR = 10.0


def read_sidecar(path):
    with open(path + ".json") as f:
        return json.load(f)


def rows(path, vocab, positions):
    size = vocab * 2
    with open(path, "rb") as f:
        for _ in range(positions):
            chunk = f.read(size)
            if len(chunk) < size:
                raise EOFError("short read in %s" % path)
            yield struct.unpack("<%de" % vocab, chunk)


def log_sum_exp(row, m):
    return m + math.log(math.fsum(math.exp(x - m) for x in row))


def compare_rows(old, new):
    """(kl, max_delta, argmax_old, argmax_new, margin_old) for one position."""
    m_old = max(old)
    m_new = max(new)
    top2 = heapq.nlargest(2, old)
    margin = top2[0] - top2[1] if len(top2) == 2 else float("inf")
    lse_old = log_sum_exp(old, m_old)
    lse_new = log_sum_exp(new, m_new)
    kl = 0.0
    max_delta = 0.0
    for a, b in zip(old, new):
        p = math.exp(a - lse_old)
        kl += p * ((a - lse_old) - (b - lse_new))
        d = abs(a - b)
        if not (d <= max_delta):
            max_delta = d
    return kl, max_delta, old.index(m_old), new.index(m_new), margin


def first_divergence(old_chosen, new_chosen):
    for i, (a, b) in enumerate(zip(old_chosen or [], new_chosen or [])):
        if a != b:
            return i
    return None


def compare(old_rows, new_rows, band_factor, show_all):
    records = []
    for i, (old, new) in enumerate(zip(old_rows, new_rows)):
        records.append((i,) + compare_rows(old, new))
    if not records:
        print("no positions to compare")
        return 2
    finite = sorted(r[2] for r in records if math.isfinite(r[2]))
    median_delta = finite[len(finite) // 2] if finite else float("nan")
    band = band_factor * median_delta
    outlier = OUTLIER_FACTOR * median_delta
    print("position  kl            max|d|     argmax_old argmax_new margin_old  verdict")
    defects = 0
    flips = 0
    for i, kl, max_delta, am_old, am_new, margin in records:
        flip = am_old != am_new
        notes = []
        if not (math.isfinite(kl) and math.isfinite(max_delta)):
            notes.append("DEFECT: non-finite logits")
            defects += 1
        elif median_delta > 0 and max_delta > outlier:
            notes.append("DEFECT: |d| %.1fx the median" % (max_delta / median_delta))
            defects += 1
        if flip:
            flips += 1
            if math.isfinite(margin) and margin <= band:
                notes.append("flip: variance (margin inside the band)")
            else:
                notes.append("flip: DEFECT (margin outside the band)")
                defects += 1
        if notes or show_all:
            print("%8d  %-12.4e  %-9.4e  %10d %10d  %-10.4e  %s"
                  % (i, kl, max_delta, am_old, am_new, margin, "; ".join(notes)))
    finite_kl = [r[1] for r in records if math.isfinite(r[1])]
    mean_kl = sum(finite_kl) / len(finite_kl) if finite_kl else float("nan")
    worst_kl = max(records, key=lambda r: r[1] if math.isfinite(r[1]) else float("inf"))
    worst_delta = max(records, key=lambda r: r[2] if math.isfinite(r[2]) else float("inf"))
    print("positions %d; mean KL %.4e; max KL %.4e at %d; max |d| %.4e at %d; "
          "median |d| %.4e; band %.4e (%gx the median); outlier above %.4e; "
          "flips %d; defects %d"
          % (len(records), mean_kl, worst_kl[1], worst_kl[0], worst_delta[2],
             worst_delta[0], median_delta, band, band_factor, outlier, flips, defects))
    return 1 if defects else 0


def self_test():
    vocab = 1000
    base = [math.sin(j * 0.37) * 3.0 for j in range(vocab)]
    old_rows = []
    new_rows = []
    for i in range(6):
        old = [x + 0.01 * i for x in base]
        new = [x + 1e-4 * math.cos(j) for j, x in enumerate(old)]
        old_rows.append(tuple(old))
        new_rows.append(tuple(new))
    top = max(range(vocab), key=lambda j: old_rows[0][j])
    near = list(old_rows[1])
    second = 5 if top != 5 else 6
    near[second] = near[top] - 1e-5
    old_rows[1] = tuple(near)
    tied = list(new_rows[1])
    tied[second] = tied[top] + 1e-4
    new_rows[1] = tuple(tied)
    wide = list(new_rows[3])
    wide[second] = wide[top] + 5.0
    new_rows[3] = tuple(wide)
    poisoned = list(new_rows[5])
    poisoned[500] = float("nan")
    new_rows[5] = tuple(poisoned)
    print("self-test: expect a variance flip at 1, an outlier defect at 3 and a non-finite defect at 5")
    code = compare(old_rows, new_rows, 3.0, True)
    ok = code == 1
    print("self-test %s" % ("passed" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("old", nargs="?")
    parser.add_argument("new", nargs="?")
    parser.add_argument("--band-factor", type=float, default=3.0,
                        help="the band as a multiple of the median max |d| (default 3)")
    parser.add_argument("--all", action="store_true", help="print every position, not only the flagged ones")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    if not (args.old and args.new):
        parser.print_usage()
        return 2
    old_meta = read_sidecar(args.old)
    new_meta = read_sidecar(args.new)
    if old_meta["vocab"] != new_meta["vocab"]:
        print("vocab differs: %d vs %d" % (old_meta["vocab"], new_meta["vocab"]))
        return 2
    vocab = old_meta["vocab"]
    for path, meta in ((args.old, old_meta), (args.new, new_meta)):
        expected = vocab * meta["positions"] * 2
        actual = os.path.getsize(path)
        if actual != expected:
            print("%s: %d bytes, the sidecar says %d positions of %d values (%d bytes); refusing"
                  % (path, actual, meta["positions"], vocab, expected))
            return 2
        forced = meta.get("forced")
        if forced and len(forced) != meta["positions"]:
            print("warning: %s forced %d ids but dumped %d positions" % (path, len(forced), meta["positions"]))
    if old_meta.get("forced") != new_meta.get("forced"):
        print("warning: the two dumps were not forced with the same tokens")
    positions = min(old_meta["positions"], new_meta["positions"])
    if old_meta["positions"] != new_meta["positions"]:
        print("warning: position counts differ (%d vs %d); comparing the first %d"
              % (old_meta["positions"], new_meta["positions"], positions))
    diverged = first_divergence(old_meta.get("chosen"), new_meta.get("chosen"))
    if diverged is not None and diverged + 1 < positions:
        positions = diverged + 1
        print("the chosen tokens diverge at position %d; comparing positions 0..%d, since every "
              "later one is conditioned on a different history" % (diverged, diverged))
    print("old %s (%s)\nnew %s (%s)" % (args.old, old_meta.get("binary_sha256", "?")[:16],
                                        args.new, new_meta.get("binary_sha256", "?")[:16]))
    return compare(rows(args.old, vocab, positions), rows(args.new, vocab, positions),
                   args.band_factor, args.all)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
