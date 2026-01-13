//! Fast Math Demo
//!
//! Demonstrates Newton-Raphson rsqrt approximation for 3-10x faster cosine similarity.
//!
//! ```bash
//! cargo run --example fast_math_demo --release
//! ```

use innr::cosine;
use innr::fast_math::{fast_cosine, fast_cosine_dispatch, fast_rsqrt, fast_rsqrt_precise};
use std::time::Instant;

fn main() {
    println!("Fast Math Demo: Newton-Raphson rsqrt Approximation");
    println!("===================================================\n");

    // 1. Accuracy analysis
    demo_rsqrt_accuracy();

    // 2. Cosine similarity accuracy
    demo_cosine_accuracy();

    // 3. Performance comparison
    demo_performance();

    // 4. Real-world search scenario
    demo_search_scenario();

    println!("Done!");
}

fn demo_rsqrt_accuracy() {
    println!("1. Inverse Square Root Accuracy");
    println!("   -----------------------------\n");

    println!("   The classic Quake III rsqrt bit-hack + Newton-Raphson iteration:\n");
    println!("   y' = y * (1.5 - 0.5 * x * y * y)\n");

    println!(
        "   {:>12}  {:>12}  {:>12}  {:>12}  {:>12}",
        "Input", "Standard", "1 NR iter", "2 NR iter", "Rel Error"
    );
    println!(
        "   {:->12}  {:->12}  {:->12}  {:->12}  {:->12}",
        "", "", "", "", ""
    );

    for &x in &[0.001f32, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0] {
        let standard = 1.0 / x.sqrt();
        let fast_1nr = fast_rsqrt(x);
        let fast_2nr = fast_rsqrt_precise(x);
        let rel_error = (fast_1nr - standard).abs() / standard;

        println!(
            "   {:>12.4}  {:>12.6}  {:>12.6}  {:>12.6}  {:>11.2e}",
            x, standard, fast_1nr, fast_2nr, rel_error
        );
    }
    println!();
    println!("   1 NR iteration: ~0.2% relative error (sufficient for similarity)");
    println!("   2 NR iterations: ~0.001% relative error (overkill for f32)");
    println!();
}

fn demo_cosine_accuracy() {
    println!("2. Cosine Similarity Accuracy");
    println!("   ---------------------------\n");

    println!("   fast_cosine uses rsqrt to avoid sqrt and division:");
    println!("   cosine = ab * rsqrt(aa) * rsqrt(bb)\n");

    let dims = [32, 128, 384, 768, 1536];

    println!(
        "   {:>8}  {:>12}  {:>12}  {:>12}",
        "Dim", "Standard", "Fast", "Abs Error"
    );
    println!("   {:->8}  {:->12}  {:->12}  {:->12}", "", "", "", "");

    for &dim in &dims {
        let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.17).cos()).collect();

        let standard = cosine(&a, &b);
        let fast = fast_cosine(&a, &b);
        let error = (standard - fast).abs();

        println!(
            "   {:>8}  {:>12.6}  {:>12.6}  {:>11.2e}",
            dim, standard, fast, error
        );
    }
    println!();
    println!("   Error < 1e-4 for all tested dimensions (more than enough for ranking).");
    println!();
}

fn demo_performance() {
    println!("3. Performance Comparison");
    println!("   -----------------------\n");

    // Print architecture
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            println!("   Architecture: x86_64 with AVX-512F");
        } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            println!("   Architecture: x86_64 with AVX2 + FMA");
        } else {
            println!("   Architecture: x86_64 (portable)");
        }
    }
    #[cfg(target_arch = "aarch64")]
    println!("   Architecture: aarch64 with NEON");
    println!();

    let iterations = 100_000;
    let dims = [128, 384, 768, 1536];

    println!(
        "   {:>8}  {:>12}  {:>12}  {:>12}  {:>8}",
        "Dim", "Standard", "Fast", "Dispatch", "Speedup"
    );
    println!(
        "   {:->8}  {:->12}  {:->12}  {:->12}  {:->8}",
        "", "", "", "", ""
    );

    for &dim in &dims {
        let a: Vec<f32> = (0..dim).map(|i| ((i * 17) % 100) as f32 / 100.0).collect();
        let b: Vec<f32> = (0..dim).map(|i| ((i * 31) % 100) as f32 / 100.0).collect();

        // Standard cosine
        let start = Instant::now();
        let mut sum = 0.0f32;
        for _ in 0..iterations {
            sum += cosine(&a, &b);
        }
        let standard_time = start.elapsed();
        std::hint::black_box(sum);

        // Fast cosine (portable)
        let start = Instant::now();
        let mut sum = 0.0f32;
        for _ in 0..iterations {
            sum += fast_cosine(&a, &b);
        }
        let fast_time = start.elapsed();
        std::hint::black_box(sum);

        // Fast cosine (dispatched to best SIMD)
        let start = Instant::now();
        let mut sum = 0.0f32;
        for _ in 0..iterations {
            sum += fast_cosine_dispatch(&a, &b);
        }
        let dispatch_time = start.elapsed();
        std::hint::black_box(sum);

        let standard_ns = standard_time.as_nanos() as f64 / iterations as f64;
        let fast_ns = fast_time.as_nanos() as f64 / iterations as f64;
        let dispatch_ns = dispatch_time.as_nanos() as f64 / iterations as f64;
        let speedup = standard_ns / dispatch_ns;

        println!(
            "   {:>8}  {:>10.1}ns  {:>10.1}ns  {:>10.1}ns  {:>7.1}x",
            dim, standard_ns, fast_ns, dispatch_ns, speedup
        );
    }
    println!();
}

fn demo_search_scenario() {
    println!("4. Real-World Search Scenario");
    println!("   ---------------------------\n");

    let dim = 768;
    let n_docs = 10_000;
    let iterations = 100;

    println!(
        "   Scenario: {} queries against {} {}-dim documents\n",
        iterations, n_docs, dim
    );

    // Generate data
    let docs: Vec<Vec<f32>> = (0..n_docs)
        .map(|i| normalize(&generate_embedding(dim, i as u64)))
        .collect();

    let queries: Vec<Vec<f32>> = (0..iterations)
        .map(|i| normalize(&generate_embedding(dim, (i + 100000) as u64)))
        .collect();

    // Standard search
    let start = Instant::now();
    let mut total_matches = 0;
    for query in &queries {
        let scores: Vec<f32> = docs.iter().map(|d| cosine(query, d)).collect();
        total_matches += scores.iter().filter(|&&s| s > 0.5).count();
    }
    let standard_time = start.elapsed();
    std::hint::black_box(total_matches);

    // Fast search (dispatch)
    let start = Instant::now();
    let mut total_matches = 0;
    for query in &queries {
        let scores: Vec<f32> = docs
            .iter()
            .map(|d| fast_cosine_dispatch(query, d))
            .collect();
        total_matches += scores.iter().filter(|&&s| s > 0.5).count();
    }
    let fast_time = start.elapsed();
    std::hint::black_box(total_matches);

    let standard_qps = iterations as f64 / standard_time.as_secs_f64();
    let fast_qps = iterations as f64 / fast_time.as_secs_f64();
    let speedup = fast_qps / standard_qps;

    println!("   Method      Total Time       QPS");
    println!("   ----------------------------------");
    println!(
        "   Standard:   {:>10.2}s   {:>7.1}",
        standard_time.as_secs_f64(),
        standard_qps
    );
    println!(
        "   Fast math:  {:>10.2}s   {:>7.1}",
        fast_time.as_secs_f64(),
        fast_qps
    );
    println!("   Speedup:    {:>10.1}x", speedup);
    println!();

    // Why this matters
    println!("   Why this matters:");
    println!("   -----------------");
    println!("   Traditional: cosine = dot(a,b) / sqrt(dot(a,a) * dot(b,b))");
    println!("     - 2 sqrt operations (~20 cycles each)");
    println!("     - 1 division (~15 cycles)");
    println!("     - Total: ~55 cycles for normalization");
    println!();
    println!("   Fast rsqrt: cosine = dot(a,b) * rsqrt(aa) * rsqrt(bb)");
    println!("     - 2 rsqrt estimates (~5 cycles each with SIMD)");
    println!("     - 2 NR iterations (~4 cycles each)");
    println!("     - 2 multiplies (~2 cycles)");
    println!("     - Total: ~20 cycles for normalization");
    println!();
}

// --- Helpers ---

fn generate_embedding(dim: usize, seed: u64) -> Vec<f32> {
    (0..dim)
        .map(|i| ((seed as f64 * 0.618033988 + i as f64 * 0.414213562).fract() * 2.0 - 1.0) as f32)
        .collect()
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}
