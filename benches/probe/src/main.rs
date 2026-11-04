//! Per-call cost of innr kernels against consumer fallbacks, outside
//! Criterion, plus stable symbols for reading their machine code.
//!
//! Each `probe_*` function is `#[inline(never)]`, so
//! `objdump -d target/release/innr-probe` shows its code as one symbol
//! (`benches/probe/codegen.sh` does that and checks the portable loops).
//! Timings rotate inputs through a pool of eight so results are not hoisted,
//! interleave the variants so drift hits each equally, and report the
//! minimum of three rounds.
//!
//! The `prev` column is a published release (pinned in Cargo.toml) linked
//! into the same binary: in-version columns compare SIMD with portable, and
//! missed that 0.7.0 to 0.7.3 ran Hamming up to 1.6x slower than 0.6.3 on
//! Apple silicon.
use std::hint::black_box;
use std::time::Instant;

use innr_bench_support::{data, rankops};

#[inline(never)]
pub fn probe_dot_simd(a: &[f32], b: &[f32]) -> f32 {
    innr::dot(a, b)
}
#[inline(never)]
pub fn probe_dot_prev(a: &[f32], b: &[f32]) -> f32 {
    innr_prev::dot(a, b)
}
#[inline(never)]
pub fn probe_dot_portable(a: &[f32], b: &[f32]) -> f32 {
    innr::dense::dot_portable(a, b)
}
#[inline(never)]
pub fn probe_dot_rankops(a: &[f32], b: &[f32]) -> f32 {
    rankops::dot(a, b)
}

#[inline(never)]
pub fn probe_cosine_simd(a: &[f32], b: &[f32]) -> f32 {
    innr::cosine(a, b)
}
#[inline(never)]
pub fn probe_cosine_prev(a: &[f32], b: &[f32]) -> f32 {
    innr_prev::cosine(a, b)
}
#[inline(never)]
pub fn probe_cosine_portable(a: &[f32], b: &[f32]) -> f32 {
    innr::dense::cosine_portable(a, b)
}
/// clump's scalar cosine distance (see `benches/consumer_pair.rs`).
#[inline(never)]
pub fn probe_cosine_clump(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = (na * nb).sqrt();
    if denom < 1e-9 {
        return 0.0;
    }
    1.0 - (dot / denom)
}

#[inline(never)]
pub fn probe_slot_u64(a: &[u64], b: &[u64]) -> u64 {
    innr::slot_hamming_u64(a, b)
}
/// sketchir's slot Hamming loop (see `benches/consumer_pair.rs`).
#[inline(never)]
pub fn probe_slot_sketchir(a: &[u64], b: &[u64]) -> u64 {
    a.iter().zip(b.iter()).filter(|(x, y)| x != y).count() as u64
}

#[inline(never)]
pub fn probe_hamming(a: &[u8], b: &[u8]) -> u32 {
    innr::hamming_distance(a, b)
}
#[inline(never)]
pub fn probe_hamming_prev(a: &[u8], b: &[u8]) -> u32 {
    innr_prev::hamming_distance(a, b)
}
#[inline(never)]
pub fn probe_hamming_portable(a: &[u8], b: &[u8]) -> u32 {
    innr::quant::hamming_portable(a, b)
}
/// The byte loop most consumers write (and innr 0.6.3's portable kernel).
#[inline(never)]
pub fn probe_hamming_bytes(a: &[u8], b: &[u8]) -> u32 {
    a.iter().zip(b).map(|(&x, &y)| (x ^ y).count_ones()).sum()
}

/// A packed byte-corpus Hamming scan, as one symbol to disassemble.
#[inline(never)]
pub fn probe_packed_hamming(c: &innr::packed::PackedRows<u8>, q: &[u8], out: &mut Vec<u32>) {
    c.scores_into::<innr::metric::Hamming>(q, out)
}

#[inline(never)]
pub fn probe_sign_dot(q: &[f32], code: &[u8]) -> f32 {
    innr::binary::sign_dot(q, code)
}
/// qntz 0.3's `asymmetric_distance_prerotated` inner loop: one accumulator,
/// one branch per bit.
#[inline(never)]
pub fn probe_sign_dot_qntz(q: &[f32], code: &[u8]) -> f32 {
    let mut ip = 0.0f32;
    for (j, &byte) in code.iter().enumerate().take(q.len() / 8) {
        for t in 0..8 {
            let x = q[j * 8 + t];
            ip += if byte & (1 << t) != 0 { x } else { -x };
        }
    }
    ip
}

/// A named variant: label and kernel.
type Variant<'a, T, O> = (&'a str, fn(&[T], &[T]) -> O);

fn time<T, O: std::ops::AddAssign + Default>(
    f: fn(&[T], &[T]) -> O,
    a: &[Vec<T>],
    b: &[Vec<T>],
    iters: usize,
) -> f64 {
    let mut s = O::default();
    let t = Instant::now();
    for i in 0..iters {
        let k = i & 7;
        s += f(black_box(&a[k]), black_box(&b[k]));
    }
    black_box(s);
    t.elapsed().as_nanos() as f64 / iters as f64
}

fn row<T, O: std::ops::AddAssign + Default>(
    label: &str,
    fs: &[Variant<'_, T, O>],
    a: &[Vec<T>],
    b: &[Vec<T>],
    iters: usize,
) {
    let mut best = vec![f64::MAX; fs.len()];
    for _ in 0..3 {
        for (j, (_, f)) in fs.iter().enumerate() {
            best[j] = best[j].min(time(*f, a, b, iters));
        }
    }
    let cells: Vec<String> = fs
        .iter()
        .zip(&best)
        .map(|((name, _), t)| format!("{name} {t:7.2} ns"))
        .collect();
    println!("{label:>14}  {}", cells.join("  "));
}

fn pool<T>(gen: impl Fn(u64) -> Vec<T>, seed: u64) -> Vec<Vec<T>> {
    (0..8).map(|s| gen(seed + s)).collect()
}

fn main() {
    println!("{}", innr::capabilities());
    let dot: [Variant<'_, f32, f32>; 4] = [
        ("simd", probe_dot_simd),
        ("prev", probe_dot_prev),
        ("portable", probe_dot_portable),
        ("rankops", probe_dot_rankops),
    ];
    for d in [8usize, 16, 24, 32, 48, 64, 128, 768] {
        let (a, b) = (
            pool(|s| data::f32s(d, s), 0),
            pool(|s| data::f32s(d, s), 100),
        );
        row(&format!("dot d={d}"), &dot, &a, &b, 20_000_000);
    }
    let cosine: [Variant<'_, f32, f32>; 4] = [
        ("simd", probe_cosine_simd),
        ("prev", probe_cosine_prev),
        ("portable", probe_cosine_portable),
        ("clump", probe_cosine_clump),
    ];
    for d in [16usize, 32, 64, 128, 768] {
        let (a, b) = (
            pool(|s| data::f32s(d, s), 0),
            pool(|s| data::f32s(d, s), 100),
        );
        row(&format!("cosine d={d}"), &cosine, &a, &b, 20_000_000);
    }
    let hamming: [Variant<'_, u8, u32>; 4] = [
        ("innr", probe_hamming),
        ("prev", probe_hamming_prev),
        ("portable", probe_hamming_portable),
        ("bytes", probe_hamming_bytes),
    ];
    for n in [16usize, 32, 96, 128, 256, 768, 2048] {
        let (a, b) = (
            pool(|s| data::bytes(n, s), 0),
            pool(|s| data::bytes(n, s), 100),
        );
        row(&format!("hamming {n} B"), &hamming, &a, &b, 10_000_000);
    }
    // One query against many byte codes: per-row calls (qntz's
    // `codes.iter().map(|c| hamming_distance(query, c))`) against one
    // packed-corpus scan. Reported per row.
    for (rows, len) in [(256usize, 16usize), (256, 96), (10_000, 96), (10_000, 256)] {
        let codes: Vec<Vec<u8>> = (0..rows).map(|s| data::bytes(len, s as u64)).collect();
        let corpus =
            innr::packed::PackedRows::from_rows(codes.iter().map(Vec::as_slice), len).unwrap();
        let query = data::bytes(len, 9_999);
        let mut out: Vec<u32> = Vec::with_capacity(rows);
        let reps = (20_000_000 / (rows * len)).max(3);
        let (mut per_row, mut packed) = (f64::MAX, f64::MAX);
        for _ in 0..5 {
            let t = Instant::now();
            for _ in 0..reps {
                out.clear();
                out.extend(codes.iter().map(|c| probe_hamming(black_box(&query), c)));
                black_box(&out);
            }
            per_row = per_row.min(t.elapsed().as_nanos() as f64 / (reps * rows) as f64);
            let t = Instant::now();
            for _ in 0..reps {
                probe_packed_hamming(&corpus, black_box(&query), &mut out);
                black_box(&out);
            }
            packed = packed.min(t.elapsed().as_nanos() as f64 / (reps * rows) as f64);
        }
        println!(
            "{:>14}  per-row {per_row:7.2} ns  corpus {packed:7.2} ns  ({:.2}x)",
            format!("scan {rows}x{len} B"),
            per_row / packed
        );
    }
    // One query against many f32 rows under cosine: per-row calls, a packed
    // corpus (stored row norms, one dot per row), and a corpus dot scan as
    // the floor. Reported per row.
    for (rows, dim) in [(10_000usize, 128usize), (10_000, 768), (2_000, 1536)] {
        let data: Vec<Vec<f32>> = (0..rows).map(|s| data::f32s(dim, s as u64)).collect();
        let corpus =
            innr::packed::PackedRows::from_rows(data.iter().map(Vec::as_slice), dim).unwrap();
        let query = data::f32s(dim, 9_999);
        let mut out: Vec<f32> = Vec::with_capacity(rows);
        let reps = (40_000_000 / (rows * dim)).max(3);
        let mut best = [f64::MAX; 3];
        for _ in 0..5 {
            for (j, b) in best.iter_mut().enumerate() {
                let t = Instant::now();
                for _ in 0..reps {
                    match j {
                        0 => {
                            out.clear();
                            out.extend(
                                data.iter().map(|r| probe_cosine_simd(black_box(&query), r)),
                            );
                        }
                        1 => {
                            corpus.scores_into::<innr::metric::Cosine>(black_box(&query), &mut out)
                        }
                        _ => corpus.scores_into::<innr::metric::Dot>(black_box(&query), &mut out),
                    }
                    black_box(&out);
                }
                *b = b.min(t.elapsed().as_nanos() as f64 / (reps * rows) as f64);
            }
        }
        println!(
            "{:>14}  per-row cosine {:7.2} ns  corpus cosine {:7.2} ns  corpus dot {:7.2} ns",
            format!("scan {rows}x{dim}"),
            best[0],
            best[1],
            best[2]
        );
    }
    // f32 query against sign-bit codes: per call (innr, qntz's serial
    // loop), and a packed scan of many codes, per code.
    for bits in [128usize, 768, 1536] {
        let q = data::f32s(bits, 5);
        let codes: Vec<Vec<u8>> = (0..8).map(|s| data::bytes(bits / 8, s)).collect();
        let qs: Vec<Vec<f32>> = (0..8).map(|s| data::f32s(bits, s + 50)).collect();
        let mut best = [f64::MAX; 2];
        let iters = (8_000_000 / bits).max(10_000);
        for _ in 0..5 {
            for (j, f) in [probe_sign_dot, probe_sign_dot_qntz].iter().enumerate() {
                let mut acc = 0.0f32;
                let t = Instant::now();
                for i in 0..iters {
                    acc += f(black_box(&qs[i & 7]), black_box(&codes[i & 7]));
                }
                black_box(acc);
                best[j] = best[j].min(t.elapsed().as_nanos() as f64 / iters as f64);
            }
        }
        let rows = 10_000;
        let many: Vec<Vec<u8>> = (0..rows).map(|s| data::bytes(bits / 8, s as u64)).collect();
        let packed =
            innr::packed::PackedRows::from_rows(many.iter().map(Vec::as_slice), bits / 8).unwrap();
        let mut out = Vec::with_capacity(rows);
        let (mut per, mut scan) = (f64::MAX, f64::MAX);
        let reps = (20_000_000 / (rows * bits)).max(2);
        for _ in 0..5 {
            let t = Instant::now();
            for _ in 0..reps {
                out.clear();
                out.extend(many.iter().map(|c| probe_sign_dot(black_box(&q), c)));
                black_box(&out);
            }
            per = per.min(t.elapsed().as_nanos() as f64 / (reps * rows) as f64);
            let t = Instant::now();
            for _ in 0..reps {
                packed.sign_dot_into(black_box(&q), &mut out);
                black_box(&out);
            }
            scan = scan.min(t.elapsed().as_nanos() as f64 / (reps * rows) as f64);
        }
        println!(
            "{:>14}  innr {:7.2} ns  qntz {:7.2} ns ({:.1}x)  scan/code {per:7.2} -> {scan:7.2} ns",
            format!("sign_dot {bits}b"),
            best[0],
            best[1],
            best[1] / best[0]
        );
    }
    let slot: [Variant<'_, u64, u64>; 2] =
        [("innr", probe_slot_u64), ("sketchir", probe_slot_sketchir)];
    for n in [128usize, 256] {
        let (a, b) = (
            pool(|s| data::slots_u64(n, s), 0),
            pool(|s| data::slots_u64(n, s), 100),
        );
        row(&format!("slot_u64 n={n}"), &slot, &a, &b, 5_000_000);
    }
}
