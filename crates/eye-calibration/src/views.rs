//! Shared joint-solve-and-reject-outliers policy used by both `intrinsics::calibrate_intrinsics`
//! and `stereo::calibrate_stereo`: solve over all views, drop views whose RMS exceeds
//! `factor * median`, error below `min_views`, and refit over the kept subset.

use eye_core::log::field;
use eye_geometry::lsq::Fit;
use nalgebra::DVector;

use crate::error::CalibrationError;

/// Outcome of `solve_with_view_rejection`: the all-views fit, the optional refit over the kept
/// views, and the per-view RMS of the first fit that drove the rejection.
#[derive(Debug)]
pub(crate) struct ViewFit<F> {
    pub first: F,
    pub refit: Option<F>,
    pub kept: Vec<usize>,
    pub per_view_rms: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct RejectionConfig {
    pub min_views: usize,
    pub factor: f64,
}

/// Solves over all `n_views`, rejects views whose RMS exceeds `factor * median`, errors below
/// `min_views`, and refits over the kept subset from the warm start `subset_params(head, ..)`.
/// `solve(kept, x0)` must build its problem over exactly the views in `kept`, in that order;
/// `view_rms(fit, pos, view)` is the RMS of original view `view` sitting at position `pos` in `fit`.
pub(crate) fn solve_with_view_rejection<F>(
    n_views: usize,
    head: usize,
    x0: DVector<f64>,
    cfg: RejectionConfig,
    mut solve: impl FnMut(&[usize], DVector<f64>) -> Result<(F, DVector<f64>), CalibrationError>,
    view_rms: impl Fn(&F, usize, usize) -> f64,
) -> Result<ViewFit<F>, CalibrationError> {
    let all: Vec<usize> = (0..n_views).collect();
    let (first, x1) = solve(&all, x0)?;
    let per_view_rms: Vec<f64> = (0..n_views).map(|v| view_rms(&first, v, v)).collect();
    let (kept, threshold_px) = reject_views(&per_view_rms, cfg.factor);
    for (view, &rms_px) in per_view_rms.iter().enumerate() {
        if !kept.contains(&view) {
            tracing::debug!(
                view = view as u64,
                rms_px,
                threshold_px,
                { field::REASON } = "rms_above_threshold",
                "view rejected"
            );
        }
    }
    if kept.len() < cfg.min_views {
        return Err(CalibrationError::InsufficientData {
            what: "views",
            need: cfg.min_views,
            got: kept.len(),
        });
    }
    let refit = if kept.len() == n_views {
        None
    } else {
        Some(solve(&kept, subset_params(head, &x1, &kept))?.0)
    };
    Ok(ViewFit {
        first,
        refit,
        kept,
        per_view_rms,
    })
}

pub(crate) fn subset_params(head: usize, x: &DVector<f64>, kept: &[usize]) -> DVector<f64> {
    let mut x0 = DVector::zeros(head + 6 * kept.len());
    x0.rows_mut(0, head).copy_from(&x.rows(0, head));
    for (new_v, &orig_v) in kept.iter().enumerate() {
        let src = head + 6 * orig_v;
        let dst = head + 6 * new_v;
        for off in 0..6 {
            x0[dst + off] = x[src + off];
        }
    }
    x0
}

/// Drops views whose RMS exceeds `factor * median`, returns (kept indices, threshold).
pub(crate) fn reject_views(per_view_rms: &[f64], factor: f64) -> (Vec<usize>, f64) {
    let mut sorted = per_view_rms.to_vec();
    let median_val = median(&mut sorted);
    let threshold = factor * median_val;
    let kept = per_view_rms
        .iter()
        .enumerate()
        .filter(|&(_, &r)| r <= threshold)
        .map(|(i, _)| i)
        .collect();
    (kept, threshold)
}

pub(crate) fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    let n = values.len();
    if n == 0 {
        0.0
    } else if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

pub(crate) fn trace_lm_solve(views: usize, residuals: usize, fit: &Fit) {
    tracing::trace!(
        views = views as u64,
        params = fit.x.len() as u64,
        residuals = residuals as u64,
        chi2 = fit.chi2,
        dof = fit.dof as u64,
        evaluations = fit.evaluations as u64,
        chi2_reduced = fit.chi2_reduced(),
        "lm solve"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subset_params_keeps_head_and_reorders_views() {
        let head = 3;
        let mut x = DVector::zeros(head + 6 * 3);
        x[0] = 1.0;
        x[1] = 2.0;
        x[2] = 3.0;
        for v in 0..3 {
            for off in 0..6 {
                x[head + 6 * v + off] = 100.0 * (v as f64) + off as f64;
            }
        }
        let kept = [2, 0];
        let x0 = subset_params(head, &x, &kept);
        assert_eq!(x0.len(), head + 6 * kept.len());
        assert_eq!(x0[0], 1.0);
        assert_eq!(x0[1], 2.0);
        assert_eq!(x0[2], 3.0);
        for off in 0..6 {
            assert_eq!(x0[head + off], 200.0 + off as f64);
            assert_eq!(x0[head + 6 + off], 0.0 + off as f64);
        }
    }

    #[test]
    fn test_reject_views_median_factor() {
        let rms = [1.0, 1.1, 0.9, 5.0];
        let (kept, threshold) = reject_views(&rms, 2.0);
        assert_eq!(kept, vec![0, 1, 2]);
        assert!((threshold - 2.1).abs() < 1e-12, "threshold={threshold}");
    }

    #[test]
    fn test_median_odd_even_empty() {
        let mut odd = [3.0, 1.0, 2.0];
        assert_eq!(median(&mut odd), 2.0);
        let mut even = [4.0, 1.0, 3.0, 2.0];
        assert_eq!(median(&mut even), 2.5);
        let mut empty: [f64; 0] = [];
        assert_eq!(median(&mut empty), 0.0);
    }

    type FakeSolveResult = Result<(Vec<f64>, DVector<f64>), CalibrationError>;

    fn fake_solve(
        head: usize,
        rms_for_view: impl Fn(usize) -> f64,
        calls: std::rc::Rc<std::cell::RefCell<Vec<Vec<usize>>>>,
    ) -> impl FnMut(&[usize], DVector<f64>) -> FakeSolveResult {
        move |kept: &[usize], _x0: DVector<f64>| {
            calls.borrow_mut().push(kept.to_vec());
            let fit: Vec<f64> = kept.iter().map(|&v| rms_for_view(v)).collect();
            Ok((fit, DVector::zeros(head + 6 * kept.len())))
        }
    }

    #[test]
    fn test_solve_with_view_rejection_refits_only_when_a_view_is_dropped() {
        let calls = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let rms_for_view = |v: usize| if v == 3 { 10.0 } else { 1.0 };
        let solve = fake_solve(0, rms_for_view, calls.clone());
        let view_rms = |fit: &Vec<f64>, pos: usize, _view: usize| fit[pos];

        let result = solve_with_view_rejection(
            4,
            0,
            DVector::zeros(0),
            RejectionConfig {
                min_views: 1,
                factor: 2.0,
            },
            solve,
            view_rms,
        )
        .unwrap();

        assert_eq!(*calls.borrow(), vec![vec![0, 1, 2, 3], vec![0, 1, 2]]);
        assert!(result.refit.is_some());
        assert_eq!(result.kept, vec![0, 1, 2]);

        let calls2 = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let all_equal_rms = |_v: usize| 1.0;
        let solve2 = fake_solve(0, all_equal_rms, calls2.clone());
        let result2 = solve_with_view_rejection(
            4,
            0,
            DVector::zeros(0),
            RejectionConfig {
                min_views: 1,
                factor: 2.0,
            },
            solve2,
            view_rms,
        )
        .unwrap();
        assert_eq!(calls2.borrow().len(), 1);
        assert!(result2.refit.is_none());
    }

    #[test]
    fn test_solve_with_view_rejection_errors_below_min_views() {
        let calls = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let rms_for_view = |v: usize| [1.0, 2.0, 3.0][v];
        let solve = fake_solve(0, rms_for_view, calls.clone());
        let view_rms = |fit: &Vec<f64>, pos: usize, _view: usize| fit[pos];

        let err = solve_with_view_rejection(
            3,
            0,
            DVector::zeros(0),
            RejectionConfig {
                min_views: 3,
                factor: 0.5,
            },
            solve,
            view_rms,
        )
        .unwrap_err();

        assert!(matches!(
            err,
            CalibrationError::InsufficientData {
                what: "views",
                need: 3,
                got: 1,
            }
        ));
        assert_eq!(calls.borrow().len(), 1);
    }

    #[test]
    fn test_logs_view_rejected_at_debug_from_views() {
        let calls = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let rms_for_view = |v: usize| if v == 3 { 10.0 } else { 1.0 };
        let solve = fake_solve(0, rms_for_view, calls.clone());
        let view_rms = |fit: &Vec<f64>, pos: usize, _view: usize| fit[pos];

        let (_result, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            solve_with_view_rejection(
                4,
                0,
                DVector::zeros(0),
                RejectionConfig {
                    min_views: 1,
                    factor: 2.0,
                },
                solve,
                view_rms,
            )
            .unwrap()
        });

        let rejected: Vec<_> = records
            .iter()
            .filter(|r| r.message == "view rejected")
            .collect();
        assert_eq!(rejected.len(), 1);
        let rec = rejected[0];
        assert_eq!(rec.level, eye_log::Level::Debug);
        assert_eq!(rec.target, "eye_calibration::views");
        assert_eq!(rec.fields["view"], eye_log::Value::U64(3));
        assert!(rec.fields.contains_key("rms_px"));
        assert!(rec.fields.contains_key("threshold_px"));
        assert_eq!(
            rec.fields[field::REASON],
            eye_log::Value::Str("rms_above_threshold".to_string())
        );
    }
}
