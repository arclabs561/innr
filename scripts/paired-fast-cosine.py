#!/usr/bin/env python3
"""Non-merge diagnostic: compare exact revisions on one native x86 runner."""

import hashlib
import os
import re
import subprocess
import tempfile
from pathlib import Path

BASELINE = "917749d678fcc50f0a3cbb5c292677528bdb6bb0"
CANDIDATE = "daa4c71fece97a8aade862f54cb8d81c5bcbe9f1"
LENGTHS = {15, 16, 17, 31, 32, 33}


def run(*args, **kwargs):
    return subprocess.run(args, check=True, text=True, **kwargs)


def main():
    run("git", "fetch", "--no-tags", "origin", BASELINE, CANDIDATE)
    pins = run(
        "git", "show", f"{CANDIDATE}:src/bit_pins.rs", capture_output=True
    ).stdout
    # Copy the actual generator verbatim; n=17 must remain the first case so
    # its inputs exactly match the failed pinned fixture's RNG position.
    generator = pins.split("    let mut state =", 1)[1].split("    let mut out =", 1)[0]
    generator = "    let mut state =" + generator
    arrays = pins.split("        let a: Vec<f32> =", 1)[1].split("        let u:", 1)[0]
    arrays = "        let a: Vec<f32> =" + arrays
    probe = r"""
#[cfg(test)]
mod paired_cosine_probe {
    #[test]
    fn paired_fast_cosine() {
        assert!(std::is_x86_feature_detected!("avx2"));
        assert!(std::is_x86_feature_detected!("fma"));
        // Require every Pulp 0.22.3 V3 feature so neither side silently falls back.
        assert!(std::is_x86_feature_detected!("sse"));
        assert!(std::is_x86_feature_detected!("sse2"));
        assert!(std::is_x86_feature_detected!("fxsr"));
        assert!(std::is_x86_feature_detected!("sse3"));
        assert!(std::is_x86_feature_detected!("ssse3"));
        assert!(std::is_x86_feature_detected!("sse4.1"));
        assert!(std::is_x86_feature_detected!("sse4.2"));
        assert!(std::is_x86_feature_detected!("popcnt"));
        assert!(std::is_x86_feature_detected!("avx"));
        assert!(std::is_x86_feature_detected!("bmi1"));
        assert!(std::is_x86_feature_detected!("bmi2"));
        assert!(std::is_x86_feature_detected!("lzcnt"));
        println!("PAIRED_CPU avx2={} fma={} bmi1={} bmi2={} lzcnt={} avx512f={}",
            std::is_x86_feature_detected!("avx2"), std::is_x86_feature_detected!("fma"),
            std::is_x86_feature_detected!("bmi1"), std::is_x86_feature_detected!("bmi2"),
            std::is_x86_feature_detected!("lzcnt"), std::is_x86_feature_detected!("avx512f"));
GENERATOR
        for n in [17usize, 15, 16, 31, 32, 33] {
ARRAYS
            let inputs: Vec<u32> = a.iter().chain(&b).map(|x| x.to_bits()).collect();
            println!("PAIRED_INPUT {n} {inputs:?}");
            let got = crate::fast_math::fast_cosine_dispatch(&a, &b);
            println!("PAIRED_RESULT {n} {:08x}", got.to_bits());
        }
    }
}
""".replace("GENERATOR", generator).replace("ARRAYS", arrays)
    print("PAIRED_PROBE_SHA256", hashlib.sha256(probe.encode()).hexdigest(), flush=True)
    outputs = {}
    # Directories live only for this disposable CI job; retained until job ends.
    root = Path(tempfile.mkdtemp(prefix="innr-paired-", dir=os.environ["RUNNER_TEMP"]))
    for label, revision in [("baseline", BASELINE), ("candidate", CANDIDATE)]:
        dest = root / label
        dest.mkdir()
        archive = root / f"{label}.tar"
        run("git", "archive", "--format=tar", f"--output={archive}", revision)
        run("tar", "-xf", str(archive), "-C", str(dest))
        with (dest / "src/lib.rs").open("a") as file:
            file.write(probe)
        env = os.environ.copy()  # Preserve RUSTC_WRAPPER and all compiler flags.
        env["CARGO_TARGET_DIR"] = str(root / f"target-{label}")
        command = [
            "cargo",
            "test",
            "--lib",
            "paired_cosine_probe::paired_fast_cosine",
            "--",
            "--exact",
            "--nocapture",
            "--test-threads=1",
        ]
        print("PAIRED_RUN", label, revision, command, flush=True)
        proc = subprocess.run(
            command,
            cwd=dest,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        print(proc.stdout, end="", flush=True)
        proc.check_returncode()
        rows = re.findall(
            r"^PAIRED_RESULT (\d+) ([0-9a-f]{8})$", proc.stdout, re.MULTILINE
        )
        inputs = re.findall(r"^PAIRED_INPUT (\d+) (\[.*\])$", proc.stdout, re.MULTILINE)
        assert len(rows) == len(LENGTHS) and {int(n) for n, _ in rows} == LENGTHS
        assert len(inputs) == len(LENGTHS)
        outputs[label] = (dict(inputs), dict(rows))
    assert outputs["baseline"][0] == outputs["candidate"][0], "paired input mismatch"
    for n in sorted(LENGTHS):
        old = outputs["baseline"][1][str(n)]
        new = outputs["candidate"][1][str(n)]
        print(f"PAIRED_COMPARE {n} baseline={old} candidate={new}", flush=True)
    assert outputs["baseline"][1] == outputs["candidate"][1], (
        "same-machine output mismatch"
    )
    print("PAIRED_MATCH all six lengths", flush=True)


if __name__ == "__main__":
    main()
