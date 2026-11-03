#!/usr/bin/env python3
"""Print the README performance table from two benches/report.sh outputs.

    python3 benches/readme_table.py benches/results/aarch64-apple-DATE.md \
        benches/results/x86_64-linux-DATE.md

Each cell is a Criterion median copied from the reports. The reference
column is the fastest consumer fallback (`local_*`) for that shape on the
same machine, so every number in the table is traceable to a report row.
"""

import re
import sys
from pathlib import Path

# (report section, shape, label) for the README; a short, representative set.
SHAPES = [
    ("pair_f32_dot", "16", "dot, d=16"),
    ("pair_f32_dot", "768", "dot, d=768"),
    ("pair_f32_cosine", "768", "cosine, d=768"),
    ("pair_f32_l2sq", "768", "squared L2, d=768"),
    ("hamming_bits", "96", "bit Hamming, 96 B"),
    ("matvec_qntz_rabitq_rotation", "768x768", "matrix-vector, 768x768"),
    ("scan_vicinity_brute_force", "100000x128", "scan, 100k x 128"),
    ("maxsim_per_doc", "32x128x128", "MaxSim, 32x128 tokens, d=128"),
]


def parse(path):
    """Map (section, shape) to {column: cell} for one report."""
    text = Path(path).read_text(encoding="utf-8")
    rows = {}
    for sec in re.split(r"\n## ", text)[1:]:
        name = sec.split("\n", 1)[0].strip()
        lines = [ln for ln in sec.split("\n") if ln.startswith("|")]
        if len(lines) < 3:
            continue
        header = [h.strip() for h in lines[0].strip("|").split("|")]
        for ln in lines[2:]:
            cells = [c.strip() for c in ln.strip("|").split("|")]
            rows[(name, cells[0])] = dict(zip(header, cells))
    return rows


def ns(cell):
    """Criterion cell ('12.3 ns', '1.44 µs') to nanoseconds."""
    value, unit = cell.split()
    return float(value) * {"ns": 1.0, "µs": 1e3, "ms": 1e6}[unit]


def best_local(row):
    locals_ = [(k, ns(v)) for k, v in row.items() if k.startswith("local_") and v]
    if not locals_:
        return "", None
    name, t = min(locals_, key=lambda kv: kv[1])
    return name.removeprefix("local_"), t


def fmt(t):
    if t >= 1e6:
        return f"{t / 1e6:.2f} ms"
    if t >= 1e3:
        return f"{t / 1e3:.1f} µs"
    return f"{t:.1f} ns"


def main():
    mac, x86 = parse(sys.argv[1]), parse(sys.argv[2])
    print(
        "| Operation | M3 Max `simd` | M3 Max portable | Ryzen 5600X `simd` | Ryzen 5600X portable | Consumer fallback (M3 / Ryzen) |"
    )
    print("|---|---:|---:|---:|---:|---|")
    for sec, shape, label in SHAPES:
        m, x = mac[(sec, shape)], x86[(sec, shape)]
        mname, mt = best_local(m)
        xname, xt = best_local(x)
        ref = f"{mname} {fmt(mt)} / {xname} {fmt(xt)}" if mt is not None else "none"
        print(
            f"| {label} | {fmt(ns(m['innr_simd']))} | {fmt(ns(m['innr_portable']))} "
            f"| {fmt(ns(x['innr_simd']))} | {fmt(ns(x['innr_portable']))} | {ref} |"
        )


if __name__ == "__main__":
    main()
