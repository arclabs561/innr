#!/usr/bin/env python3
"""Turn a Criterion output directory into innr's consumer-bench markdown table.

Usage: report.py CRITERION_HOME > out.md   (normally called by report.sh)

Ratios are speedups (other median / this median): above 1 means the named
innr path is faster. The verdict compares innr_simd with the fastest local
fallback, using the A/A noise floor (at least 5%) as the tie band.
"""

import json
import os
import sys
from collections import defaultdict

MIN_FLOOR = 0.05


def load(root):
    rows = []
    for dirpath, _dirs, files in os.walk(root):
        if os.path.basename(dirpath) != "new" or "benchmark.json" not in files:
            continue
        with open(os.path.join(dirpath, "benchmark.json")) as f:
            bench = json.load(f)
        with open(os.path.join(dirpath, "estimates.json")) as f:
            est = json.load(f)
        med = est["median"]
        rows.append(
            {
                "group": bench["group_id"],
                "variant": bench["function_id"],
                "param": bench.get("value_str") or "",
                "median": med["point_estimate"],
                "lo": med["confidence_interval"]["lower_bound"],
                "hi": med["confidence_interval"]["upper_bound"],
            }
        )
    return rows


def fmt_time(ns):
    if ns < 1e3:
        return f"{ns:.1f} ns"
    if ns < 1e6:
        return f"{ns / 1e3:.2f} µs"
    if ns < 1e9:
        return f"{ns / 1e6:.2f} ms"
    return f"{ns / 1e9:.2f} s"


def param_key(p):
    # Sort "128", "1000x64", "Q8_10000x768" numerically by their numbers.
    nums = []
    cur = ""
    for ch in p:
        if ch.isdigit():
            cur += ch
        elif cur:
            nums.append(int(cur))
            cur = ""
    if cur:
        nums.append(int(cur))
    return (nums, p)


def variant_key(v):
    order = ["innr_portable", "innr_simd", "local", "numkong"]
    for i, prefix in enumerate(order):
        if v.startswith(prefix):
            return (i, v)
    return (len(order), v)


def aa_floor(rows):
    pairs = defaultdict(dict)
    for r in rows:
        if r["group"] == "aa_noise":
            build = r["variant"].rsplit("_aa_", 1)[0]
            pairs[(build, r["param"])][r["variant"]] = r["median"]
    spreads = []
    for (build, param), meds in sorted(
        pairs.items(), key=lambda kv: (kv[0][0], param_key(kv[0][1]))
    ):
        if len(meds) == 2:
            a, b = sorted(meds.values())
            spreads.append((build, param, a, b, b / a - 1.0))
    return spreads


def main():
    root = sys.argv[1]
    rows = load(root)
    if not rows:
        sys.exit(f"no Criterion results under {root}")

    env = os.environ.get
    out = []
    out.append(f"# innr consumer-shape benchmarks: {env('INNR_REPORT_ARCH', '?')}")
    out.append("")
    out.append(f"- CPU: {env('INNR_REPORT_CPU', '?')}")
    out.append(f"- Toolchain: {env('INNR_REPORT_RUSTC', '?')}")
    out.append(f"- innr revision: {env('INNR_REPORT_REV', '?')}")
    out.append(f"- Settings: {env('INNR_REPORT_SETTINGS', '?')}")
    out.append(f"- Benchmark wall time: {env('INNR_REPORT_ELAPSED', '?')} s")
    out.append("- Command: `benches/report.sh OUT.md` (see the script header)")
    out.append("")
    out.append("Cells are Criterion medians. `innr_portable` is innr built with")
    out.append("`--no-default-features`; `innr_simd` is the default build. `local_*`")
    out.append(
        "columns are the consumer's own no-innr fallback, copied verbatim (sources"
    )
    out.append("cited in the bench files). Speedups are other / innr, so above 1 means")
    out.append("innr is faster.")
    out.append("")

    spreads = aa_floor(rows)
    floor = max([MIN_FLOOR] + [s[4] for s in spreads])
    out.append("## A/A noise floor")
    out.append("")
    out.append(
        "The same kernel (`innr::dot`) and inputs benchmarked twice under two names."
    )
    out.append("")
    out.append("| build | dim | run 1 | run 2 | spread |")
    out.append("|---|---:|---:|---:|---:|")
    for build, param, a, b, spread in spreads:
        out.append(
            f"| {build} | {param} | {fmt_time(a)} | {fmt_time(b)} | {spread * 100:.1f}% |"
        )
    out.append("")
    out.append(
        f"Tie band for verdicts: ±{floor * 100:.1f}% (the largest A/A spread, at least {MIN_FLOOR * 100:.0f}%)."
    )
    out.append("")

    groups = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if r["group"] != "aa_noise":
            groups[r["group"]][r["param"]][r["variant"]] = r["median"]

    group_order = [
        "pair_",
        "hamming",
        "slot",
        "mixed",
        "matvec",
        "scan",
        "maxsim",
        "topk",
    ]

    def group_key(g):
        for i, prefix in enumerate(group_order):
            if g.startswith(prefix):
                return (i, g)
        return (len(group_order), g)

    verdicts = []
    for group in sorted(groups, key=group_key):
        params = groups[group]
        variants = sorted({v for p in params.values() for v in p}, key=variant_key)
        locals_ = [v for v in variants if v.startswith("local")]
        has_pair = "innr_simd" in variants and "innr_portable" in variants
        header = ["shape"] + variants
        if has_pair:
            header.append("simd vs portable")
        if locals_:
            header += ["best local / simd", "best local / portable"]
        has_nk = "numkong" in variants
        if has_nk:
            header.append("numkong / simd")
        out.append(f"## {group}")
        out.append("")
        out.append("| " + " | ".join(header) + " |")
        out.append("|---|" + "---:|" * (len(header) - 1))
        for param in sorted(params, key=param_key):
            meds = params[param]
            cells = [param] + [
                fmt_time(meds[v]) if v in meds else "—" for v in variants
            ]
            simd, port = meds.get("innr_simd"), meds.get("innr_portable")
            if has_pair:
                cells.append(f"{port / simd:.2f}x" if simd and port else "—")
            if locals_:
                present = [meds[v] for v in locals_ if v in meds]
                best = min(present) if present else None
                cells.append(f"{best / simd:.2f}x" if best and simd else "—")
                cells.append(f"{best / port:.2f}x" if best and port else "—")
                if best and simd:
                    ratio = best / simd
                    verdict = (
                        "win"
                        if ratio > 1 + floor
                        else "lose"
                        if ratio < 1 - floor
                        else "tie"
                    )
                    verdicts.append((group, param, ratio, verdict))
            if has_nk:
                nk = meds.get("numkong")
                cells.append(f"{nk / simd:.2f}x" if nk and simd else "—")
            out.append("| " + " | ".join(cells) + " |")
        out.append("")

    if verdicts:
        out.append("## innr_simd against the fastest local fallback")
        out.append("")
        for kind in ("win", "tie", "lose"):
            items = [f"{g} {p} ({r:.2f}x)" for g, p, r, v in verdicts if v == kind]
            out.append(
                f"- {kind} ({len(items)}): " + ("; ".join(items) if items else "none")
            )
        out.append("")

    print("\n".join(out))


if __name__ == "__main__":
    main()
