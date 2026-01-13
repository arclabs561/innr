//! SIMD Benchmark Demo
//!
//! Compares SIMD-accelerated operations against portable implementations.
//!
//! ```bash
//! cargo run --example simd_benchmark --release
//! ```

use innr::{cosine, dot, dot_portable, l2_distance, norm};
use std::time::Instant;

fn main() {
    println!("innr SIMD Benchmark");
    println!("===================\n");

    // Print detected architecture
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            println!("Architecture: x86_64 with AVX2 + FMA");
        } else if is_x86_feature_detected!("avx") {
            println!("Architecture: x86_64 with AVX");
        } else if is_x86_feature_detected!("sse4.1") {
            println!("Architecture: x86_64 with SSE4.1");
        } else {
            println!("Architecture: x86_64 (portable)");
        }
    }
    #[cfg(target_arch = "aarch64")]
    println!("Architecture: aarch64 with NEON");
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    println!("Architecture: portable");
    println!();

    // Test different vector sizes
    let sizes = [32, 128, 384, 768, 1536]; // Common embedding dimensions

    for &dim in &sizes {
        benchmark_size(dim);
    }

    // Demonstrate operations
    println!("\nOperation Examples:");
    println!("-------------------");

    let a: Vec<f32> = (0..128).map(|i| (i as f32 * 0.01).sin()).collect();
    let b: Vec<f32> = (0..128).map(|i| (i as f32 * 0.02).cos()).collect();

    println!("Vectors: a[128], b[128] (sin/cos pattern)");
    println!("  dot(a, b)          = {:.6}", dot(&a, &b));
    println!("  cosine(a, b)       = {:.6}", cosine(&a, &b));
    println!("  l2_distance(a, b)  = {:.6}", l2_distance(&a, &b));
    println!("  norm(a)            = {:.6}", norm(&a));
    println!("  norm(b)            = {:.6}", norm(&b));

    println!("\nDone!");
}

fn benchmark_size(dim: usize) {
    println!("Dimension: {}", dim);

    // Generate random vectors (deterministic for reproducibility)
    let a: Vec<f32> = (0..dim).map(|i| ((i * 17) % 100) as f32 / 100.0).collect();
    let b: Vec<f32> = (0..dim).map(|i| ((i * 31) % 100) as f32 / 100.0).collect();

    let iterations = 100_000;

    // Benchmark SIMD dot
    let start = Instant::now();
    let mut simd_sum = 0.0f32;
    for _ in 0..iterations {
        simd_sum += dot(&a, &b);
    }
    let simd_time = start.elapsed();

    // Benchmark portable dot
    let start = Instant::now();
    let mut portable_sum = 0.0f32;
    for _ in 0..iterations {
        portable_sum += dot_portable(&a, &b);
    }
    let portable_time = start.elapsed();

    // Prevent optimization from removing the loops
    if simd_sum.abs() < f32::EPSILON && portable_sum.abs() < f32::EPSILON {
        println!("  (both zero)"); // This should never print
    }

    let speedup = portable_time.as_nanos() as f64 / simd_time.as_nanos() as f64;
    let simd_ns = simd_time.as_nanos() as f64 / iterations as f64;
    let portable_ns = portable_time.as_nanos() as f64 / iterations as f64;

    println!(
        "  SIMD dot:     {:>7.1} ns/op ({:>10.0} ops/sec)",
        simd_ns,
        1e9 / simd_ns
    );
    println!(
        "  Portable dot: {:>7.1} ns/op ({:>10.0} ops/sec)",
        portable_ns,
        1e9 / portable_ns
    );
    println!("  Speedup: {:.1}x\n", speedup);
}
