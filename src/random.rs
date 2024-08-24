use std::f64;
use std::time::{SystemTime, UNIX_EPOCH};

/// Simple Pseudo Random Number Generator
fn linear_congruential_generator(seed: &mut u64) -> u64 {
    const A: u64 = 1664525;
    const C: u64 = 1013904223;
    const M: u64 = 1 << 32;

    *seed = (*seed).wrapping_mul(A).wrapping_add(C) % M;
    *seed
}

/// Generate a (pseudo)random `Vec<T>`
pub fn gen_rand_vec<T: Random>(n: usize) -> Vec<T> {
    // Seed is current time (secs) since the unix epoch
    let start = SystemTime::now();
    let duration = start
        .duration_since(UNIX_EPOCH)
        .expect("Oh shit broo, time went backwards!");
    let mut seed = duration.as_secs();

    (0..n).map(|_| T::random(&mut seed)).collect()
}

/// Trait for generating random values
pub trait Random {
    fn random(seed: &mut u64) -> Self;
}
/// Implements Random trait for `f64`
impl Random for f64 {
    fn random(seed: &mut u64) -> Self {
        linear_congruential_generator(seed) as f64 / (u64::MAX as f64)
    }
}
/// Implements Random trait for `i64`
impl Random for i64 {
    fn random(seed: &mut u64) -> Self {
        linear_congruential_generator(seed) as i64
    }
}
/// Implements Random trait for `i32`
impl Random for i32 {
    fn random(seed: &mut u64) -> Self {
        linear_congruential_generator(seed) as i32
    }
}
/// Implements Random trait for `u8`
impl Random for u8 {
    fn random(seed: &mut u64) -> Self {
        linear_congruential_generator(seed) as u8
    }
}
