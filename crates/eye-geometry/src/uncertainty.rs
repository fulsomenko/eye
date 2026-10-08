//! First-order ("delta method") covariance propagation.
//!
//! For `y = f(x)` with `x ~ (x0, Σx)`: `Σy ≈ J Σx Jᵀ`, with `J` the Jacobian of `f` at `x0`,
//! numerically approximated by central differences.

use eye_core::Measured;
use nalgebra::{Matrix2, Matrix3, Point2, SMatrix, SVector};

pub type Cov2 = Matrix2<f64>;
pub type Cov3 = Matrix3<f64>;

const STEP: f64 = 6.055_454_452_393_343e-6;

pub fn isotropic2(sigma: f64) -> Cov2 {
    Cov2::identity() * (sigma * sigma)
}

pub fn measured_point_cov(m: &Measured<Point2<f64>>) -> Cov2 {
    isotropic2(m.sigma())
}

pub fn propagate<const M: usize, const N: usize>(
    j: &SMatrix<f64, M, N>,
    cov: &SMatrix<f64, N, N>,
) -> SMatrix<f64, M, M> {
    let out = j * cov * j.transpose();
    (out + out.transpose()) * 0.5
}

pub fn numeric_jacobian<const M: usize, const N: usize>(
    f: impl Fn(&SVector<f64, N>) -> Option<SVector<f64, M>>,
    x: &SVector<f64, N>,
) -> Option<SMatrix<f64, M, N>> {
    let mut j = SMatrix::<f64, M, N>::zeros();
    for i in 0..N {
        let h = STEP * x[i].abs().max(1.0);
        let mut xp = *x;
        let mut xm = *x;
        xp[i] += h;
        xm[i] -= h;
        j.set_column(i, &((f(&xp)? - f(&xm)?) / (2.0 * h)));
    }
    Some(j)
}

/// Value and first-order covariance of `f` at `x`.
pub fn propagate_fn<const M: usize, const N: usize>(
    f: impl Fn(&SVector<f64, N>) -> Option<SVector<f64, M>>,
    x: &SVector<f64, N>,
    cov: &SMatrix<f64, N, N>,
) -> Option<(SVector<f64, M>, SMatrix<f64, M, M>)> {
    let y = f(x)?;
    let j = numeric_jacobian(&f, x)?;
    Some((y, propagate(&j, cov)))
}

pub fn block_diag<const A: usize, const B: usize, const N: usize>(
    a: &SMatrix<f64, A, A>,
    b: &SMatrix<f64, B, B>,
) -> SMatrix<f64, N, N> {
    const { assert!(A + B == N) };
    let mut out = SMatrix::<f64, N, N>::zeros();
    out.fixed_view_mut::<A, A>(0, 0).copy_from(a);
    out.fixed_view_mut::<B, B>(A, A).copy_from(b);
    out
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use nalgebra::{Matrix2, Matrix5, SMatrix, SVector, Vector2, matrix, vector};

    use super::{block_diag, numeric_jacobian, propagate, propagate_fn};
    use crate::synth::SplitMix64;

    #[test]
    fn test_propagate_linear_map_matches_closed_form() {
        let j = matrix![2.0, 0.0; 1.0, 1.0];
        let cov = Matrix2::from_diagonal(&vector![1.0, 4.0]);
        let out = propagate(&j, &cov);
        assert_abs_diff_eq!(out, matrix![4.0, 2.0; 2.0, 5.0], epsilon = 1e-15);
    }

    #[test]
    fn test_numeric_jacobian_matches_analytic_polar() {
        let f = |x: &SVector<f64, 2>| -> Option<SVector<f64, 2>> {
            let (r, theta) = (x[0], x[1]);
            Some(vector![r * theta.cos(), r * theta.sin()])
        };
        let x = vector![2.0_f64, 0.3];
        let j = numeric_jacobian(f, &x).unwrap();
        let (r, theta) = (x[0], x[1]);
        let expected = matrix![theta.cos(), -r * theta.sin(); theta.sin(), r * theta.cos()];
        assert_abs_diff_eq!(j, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_propagate_fn_matches_monte_carlo() {
        let f = |x: &SVector<f64, 2>| -> Option<SVector<f64, 2>> {
            Some(vector![x[0] * x[1], x[0] + x[1].sin()])
        };
        let x0 = vector![3.0_f64, 0.5];
        let sigma = vector![0.01_f64, 0.02];
        let cov0 = Matrix2::from_diagonal(&sigma.component_mul(&sigma));

        let (_, predicted) = propagate_fn(f, &x0, &cov0).unwrap();

        let n = 20_000;
        let mut rng = SplitMix64::new(7);
        let mut sum = Vector2::zeros();
        let mut samples = Vec::with_capacity(n);
        for _ in 0..n {
            let noise = vector![rng.gaussian(), rng.gaussian()];
            let x = x0 + sigma.component_mul(&noise);
            let y = f(&x).unwrap();
            sum += y;
            samples.push(y);
        }
        let mean = sum / n as f64;
        let mut empirical = Matrix2::zeros();
        for y in &samples {
            let d = y - mean;
            empirical += d * d.transpose();
        }
        empirical /= (n - 1) as f64;

        for i in 0..2 {
            assert!(
                (empirical[(i, i)] - predicted[(i, i)]).abs() <= 0.05 * predicted[(i, i)].abs(),
                "diag {i}: empirical {} predicted {}",
                empirical[(i, i)],
                predicted[(i, i)]
            );
        }
        for i in 0..2 {
            for j in 0..2 {
                if i == j {
                    continue;
                }
                let tol = 0.05 * (predicted[(i, i)] * predicted[(j, j)]).sqrt();
                assert!(
                    (empirical[(i, j)] - predicted[(i, j)]).abs() <= tol,
                    "offdiag ({i},{j}): empirical {} predicted {}",
                    empirical[(i, j)],
                    predicted[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_block_diag_places_blocks() {
        let a: SMatrix<f64, 3, 3> = SMatrix::<f64, 3, 3>::identity() * 2.0;
        let b: SMatrix<f64, 2, 2> = SMatrix::<f64, 2, 2>::identity() * 3.0;
        let out: Matrix5<f64> = block_diag::<3, 2, 5>(&a, &b);
        for i in 0..5 {
            for j in 0..5 {
                let expected = if i != j {
                    0.0
                } else if i < 3 {
                    2.0
                } else {
                    3.0
                };
                assert_abs_diff_eq!(out[(i, j)], expected, epsilon = 1e-15);
            }
        }
    }
}
