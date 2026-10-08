//! Deterministic synthetic noise for reproducible Monte Carlo tests.

/// Deterministic SplitMix64 PRNG (Steele, Lea & Flood 2014) for reproducible test noise.
///
/// ```rust
/// let mut r = eye_geometry::synth::SplitMix64::new(7);
/// let g = r.gaussian();
/// assert!(g.is_finite());
/// ```
#[derive(Debug, Clone)]
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1) with 53 random mantissa bits.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Standard normal, Box-Muller (cosine branch only).
    pub fn gaussian(&mut self) -> f64 {
        let u1 = 1.0 - self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::SplitMix64;

    #[test]
    fn test_splitmix_gaussian_moments() {
        let mut rng = SplitMix64::new(1);
        let n = 100_000;
        let samples: Vec<f64> = (0..n).map(|_| rng.gaussian()).collect();
        let mean = samples.iter().sum::<f64>() / n as f64;
        let var = samples.iter().map(|g| (g - mean).powi(2)).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.01, "mean = {mean}");
        assert!((var - 1.0).abs() < 0.02, "var = {var}");
    }
}
