//! Per-call cost of innr kernels against consumer fallbacks, outside
//! Criterion, plus stable symbols for reading their machine code.
//!
//! Each `probe_*` function is `#[inline(never)]`, so
//! `objdump -d target/release/innr-probe` shows its code as one symbol
//! (`benches/probe/codegen.sh` does that and checks the portable loops).
//! Timings rotate inputs through a pool of eight so results are not hoisted,
//! interleave the variants so drift hits each equally, and report the
//! minimum of three rounds.
use std::hint::black_box;
use std::time::Instant;

use innr_bench_support::{data, rankops};

#[inline(never)]
pub fn probe_dot_simd(a: &[f32], b: &[f32]) -> f32 {
    innr::dot(a, b)
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
pub fn probe_hamming_portable(a: &[u8], b: &[u8]) -> u32 {
    innr::quant::hamming_portable(a, b)
}
/// The byte loop most consumers write (and innr 0.6.3's portable kernel).
#[inline(never)]
pub fn probe_hamming_bytes(a: &[u8], b: &[u8]) -> u32 {
    a.iter().zip(b).map(|(&x, &y)| (x ^ y).count_ones()).sum()
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
    let dot: [Variant<'_, f32, f32>; 3] = [
        ("simd", probe_dot_simd),
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
    let cosine: [Variant<'_, f32, f32>; 3] = [
        ("simd", probe_cosine_simd),
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
    let hamming: [Variant<'_, u8, u32>; 3] = [
        ("innr", probe_hamming),
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
