use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};
use nalgebra::{Cholesky, DMatrix, DVector, Dyn, U1, VecStorage};

use crate::GeometryError;

/// Finite stand-in residual for a trial step outside the model's domain: LM sees a
/// norm far above the current one and rejects the step instead of aborting.
const OUT_OF_DOMAIN_RESIDUAL: f64 = 1e100;
const STEP: f64 = 6.055_454_452_393_343e-6;

pub trait ResidualModel {
    fn num_residuals(&self) -> usize;
    /// Whitened residuals; `None` when `x` is outside the model's domain (e.g. a point behind a camera).
    fn residuals(&self, x: &DVector<f64>) -> Option<DVector<f64>>;
}

#[derive(Debug, Clone, Copy)]
pub struct SolveOptions {
    pub tol: f64,
    pub patience: usize,
    pub inflate_by_chi2: bool,
}

impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            tol: 1e-10,
            patience: 100,
            inflate_by_chi2: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Fit {
    pub x: DVector<f64>,
    pub cov: DMatrix<f64>,
    pub chi2: f64,
    pub dof: usize,
    pub evaluations: usize,
}

impl Fit {
    pub fn chi2_reduced(&self) -> f64 {
        if self.dof == 0 {
            0.0
        } else {
            self.chi2 / self.dof as f64
        }
    }

    pub fn rms_whitened(&self) -> f64 {
        (self.chi2 / (self.dof + self.x.len()) as f64).sqrt()
    }
}

struct Adapter<'m, R> {
    model: &'m R,
    x: DVector<f64>,
    r: Option<DVector<f64>>,
}

impl<R: ResidualModel> LeastSquaresProblem<f64, Dyn, Dyn> for Adapter<'_, R> {
    type ResidualStorage = VecStorage<f64, Dyn, U1>;
    type JacobianStorage = VecStorage<f64, Dyn, Dyn>;
    type ParameterStorage = VecStorage<f64, Dyn, U1>;

    fn set_params(&mut self, x: &DVector<f64>) {
        self.x.copy_from(x);
        self.r = Some(self.model.residuals(x).unwrap_or_else(|| {
            DVector::from_element(self.model.num_residuals(), OUT_OF_DOMAIN_RESIDUAL)
        }));
    }

    fn params(&self) -> DVector<f64> {
        self.x.clone()
    }

    fn residuals(&self) -> Option<DVector<f64>> {
        self.r.clone()
    }

    fn jacobian(&self) -> Option<DMatrix<f64>> {
        jacobian(self.model, &self.x)
    }
}

pub fn solve<R: ResidualModel>(
    model: &R,
    x0: DVector<f64>,
    opts: &SolveOptions,
) -> Result<Fit, GeometryError> {
    let (m, n) = (model.num_residuals(), x0.len());
    if m < n {
        return Err(GeometryError::TooFewObservations { need: n, got: m });
    }
    let r0 = model
        .residuals(&x0)
        .ok_or_else(|| GeometryError::NotConverged("User(\"residuals\") at x0".into()))?;
    let adapter = Adapter {
        model,
        x: x0,
        r: Some(r0),
    };
    let (done, report) = LevenbergMarquardt::new()
        .with_tol(opts.tol)
        .with_patience(opts.patience)
        .minimize(adapter);
    if !report.termination.was_successful() {
        return Err(GeometryError::NotConverged(format!(
            "{:?}",
            report.termination
        )));
    }
    let chi2 = 2.0 * report.objective_function;
    let j = jacobian(model, &done.x).ok_or(GeometryError::Degenerate("jacobian at optimum"))?;
    let inflate_chi2 = if opts.inflate_by_chi2 { chi2 } else { 0.0 };
    let cov = covariance_from_jacobian(&j, inflate_chi2, m - n)?;
    Ok(Fit {
        x: done.x,
        cov,
        chi2,
        dof: m - n,
        evaluations: report.number_of_evaluations,
    })
}

/// Central-difference Jacobian of the whitened residuals; `None` if any probe leaves the domain.
pub fn jacobian<R: ResidualModel>(model: &R, x: &DVector<f64>) -> Option<DMatrix<f64>> {
    let mut j = DMatrix::zeros(model.num_residuals(), x.len());
    for i in 0..x.len() {
        let h = STEP * x[i].abs().max(1.0);
        let mut xp = x.clone();
        let mut xm = x.clone();
        xp[i] += h;
        xm[i] -= h;
        j.set_column(
            i,
            &((model.residuals(&xp)? - model.residuals(&xm)?) / (2.0 * h)),
        );
    }
    Some(j)
}

pub fn covariance_from_jacobian(
    j: &DMatrix<f64>,
    chi2: f64,
    dof: usize,
) -> Result<DMatrix<f64>, GeometryError> {
    let jtj = j.transpose() * j;
    let scale = if dof == 0 {
        1.0
    } else {
        (chi2 / dof as f64).max(1.0)
    };
    let chol = Cholesky::new(jtj).ok_or(GeometryError::Unobservable)?;
    Ok(chol.inverse() * scale)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::{DMatrix, DVector};

    use super::{ResidualModel, SolveOptions, solve};
    use crate::GeometryError;
    use crate::synth::SplitMix64;

    struct Line {
        x: Vec<f64>,
        y: Vec<f64>,
        s: f64,
    }

    impl ResidualModel for Line {
        fn num_residuals(&self) -> usize {
            self.x.len()
        }

        fn residuals(&self, p: &DVector<f64>) -> Option<DVector<f64>> {
            let (a, b) = (p[0], p[1]);
            Some(DVector::from_iterator(
                self.x.len(),
                self.x
                    .iter()
                    .zip(&self.y)
                    .map(|(&xi, &yi)| (a * xi + b - yi) / self.s),
            ))
        }
    }

    fn line_data(seed: u64, sigma: f64) -> (Vec<f64>, Vec<f64>) {
        let mut rng = SplitMix64::new(seed);
        let mut x = Vec::with_capacity(50);
        let mut y = Vec::with_capacity(50);
        for i in 0..50 {
            let xi = 10.0 * i as f64 / 49.0;
            let yi = 1.5 * xi - 2.0 + sigma * rng.gaussian();
            x.push(xi);
            y.push(yi);
        }
        (x, y)
    }

    fn closed_form(x: &[f64], s: f64) -> DMatrix<f64> {
        let mut xtx = DMatrix::<f64>::zeros(2, 2);
        for &xi in x {
            xtx += DMatrix::from_row_slice(2, 2, &[xi * xi, xi, xi, 1.0]);
        }
        let inv = xtx.try_inverse().unwrap();
        inv * (s * s)
    }

    #[test]
    fn test_lsq_line_fit_recovers_parameters_and_covariance() {
        let (x, y) = line_data(1, 0.1);
        let model = Line {
            x: x.clone(),
            y,
            s: 0.1,
        };
        let fit = solve(
            &model,
            DVector::from_vec(vec![0.0, 0.0]),
            &SolveOptions::default(),
        )
        .expect("should converge");

        let expected_cov = closed_form(&x, 0.1) * fit.chi2_reduced().max(1.0);
        assert_relative_eq!(fit.cov, expected_cov, max_relative = 1e-6);
        assert!((fit.x[0] - 1.5).abs() < 3.0 * fit.cov[(0, 0)].sqrt());
        assert!((fit.x[1] + 2.0).abs() < 3.0 * fit.cov[(1, 1)].sqrt());
    }

    #[test]
    fn test_lsq_covariance_inflates_when_sigma_underreported() {
        let (x, y) = line_data(1, 0.3);
        let model = Line {
            x: x.clone(),
            y,
            s: 0.1,
        };
        let fit = solve(
            &model,
            DVector::from_vec(vec![0.0, 0.0]),
            &SolveOptions::default(),
        )
        .expect("should converge");

        let chi2_red = fit.chi2_reduced();
        assert!((6.0..=12.0).contains(&chi2_red), "chi2_red = {chi2_red}");

        let expected_cov = closed_form(&x, 0.1) * chi2_red;
        assert_relative_eq!(fit.cov, expected_cov, max_relative = 1e-6);
    }

    #[test]
    fn test_lsq_inflation_disabled_keeps_raw_covariance() {
        let (x, y) = line_data(1, 0.3);
        let model = Line {
            x: x.clone(),
            y,
            s: 0.1,
        };
        let opts = SolveOptions {
            inflate_by_chi2: false,
            ..SolveOptions::default()
        };
        let fit = solve(&model, DVector::from_vec(vec![0.0, 0.0]), &opts).expect("should converge");

        let expected_cov = closed_form(&x, 0.1);
        assert_relative_eq!(fit.cov, expected_cov, max_relative = 1e-6);
    }

    #[test]
    fn test_lsq_too_few_residuals_errors() {
        struct OneResidual;
        impl ResidualModel for OneResidual {
            fn num_residuals(&self) -> usize {
                1
            }
            fn residuals(&self, p: &DVector<f64>) -> Option<DVector<f64>> {
                Some(DVector::from_vec(vec![p[0] + p[1]]))
            }
        }

        let err = solve(
            &OneResidual,
            DVector::from_vec(vec![0.0, 0.0]),
            &SolveOptions::default(),
        )
        .unwrap_err();
        assert_eq!(err, GeometryError::TooFewObservations { need: 2, got: 1 });
    }

    #[test]
    fn test_lsq_unobservable_parameter_errors() {
        struct Unobservable;
        impl ResidualModel for Unobservable {
            fn num_residuals(&self) -> usize {
                5
            }
            fn residuals(&self, p: &DVector<f64>) -> Option<DVector<f64>> {
                Some(DVector::from_iterator(5, (0..5).map(|i| p[0] - i as f64)))
            }
        }

        let err = solve(
            &Unobservable,
            DVector::from_vec(vec![0.3, 0.7]),
            &SolveOptions::default(),
        )
        .unwrap_err();
        assert_eq!(err, GeometryError::Unobservable);
    }

    #[test]
    fn test_lsq_residuals_none_reports_not_converged() {
        struct AlwaysNone;
        impl ResidualModel for AlwaysNone {
            fn num_residuals(&self) -> usize {
                3
            }
            fn residuals(&self, _p: &DVector<f64>) -> Option<DVector<f64>> {
                None
            }
        }

        let err = solve(
            &AlwaysNone,
            DVector::from_vec(vec![0.0]),
            &SolveOptions::default(),
        )
        .unwrap_err();
        match err {
            GeometryError::NotConverged(s) => assert!(s.contains("User"), "message: {s}"),
            other => panic!("expected NotConverged, got {other:?}"),
        }
    }

    #[test]
    fn test_lsq_step_into_invalid_domain_is_rejected_not_fatal() {
        struct Clamp;
        impl ResidualModel for Clamp {
            fn num_residuals(&self) -> usize {
                2
            }
            fn residuals(&self, p: &DVector<f64>) -> Option<DVector<f64>> {
                let x = p[0];
                if x > 5.0 {
                    return None;
                }
                Some(DVector::from_vec(vec![
                    x.exp() - 4.9_f64.exp(),
                    0.1 * (x - 4.9),
                ]))
            }
        }

        let fit = solve(
            &Clamp,
            DVector::from_vec(vec![0.0]),
            &SolveOptions::default(),
        )
        .expect("should converge despite the domain edge");
        assert!((fit.x[0] - 4.9).abs() < 1e-6, "x = {}", fit.x[0]);
    }
}
