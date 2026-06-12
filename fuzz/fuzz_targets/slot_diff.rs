//! Differential fuzz: SIMD slot_hamming_u32/u64 vs the scalar count, over
//! arbitrary byte-decoded integer slots. Exercises the mask/movemask tail
//! handling on adversarial bit patterns.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let u32s: Vec<u32> = data
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    if !u32s.is_empty() {
        let n = u32s.len() / 2;
        let (a, b) = (&u32s[..n], &u32s[n..2 * n]);
        let simd = innr::slot_hamming_u32(a, b);
        let scalar = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count() as u32;
        assert_eq!(simd, scalar, "slot_hamming_u32 n={n}");
    }

    let u16s: Vec<u16> = data
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    if !u16s.is_empty() {
        let n = u16s.len() / 2;
        let (a, b) = (&u16s[..n], &u16s[n..2 * n]);
        let simd = innr::slot_hamming_u16(a, b);
        let scalar = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count() as u32;
        assert_eq!(simd, scalar, "slot_hamming_u16 n={n}");
    }

    let u64s: Vec<u64> = data
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect();
    if !u64s.is_empty() {
        let n = u64s.len() / 2;
        let (a, b) = (&u64s[..n], &u64s[n..2 * n]);
        let simd = innr::slot_hamming_u64(a, b);
        let scalar = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count() as u64;
        assert_eq!(simd, scalar, "slot_hamming_u64 n={n}");
    }
});
