//! Leave-one-target-out (LOTO) calibration: fit a `UserProfile` on every target but one,
//! evaluate only on the held-out target, for every target in turn.

use eye::pipeline::RayBatch;
use eye_calibration::correction::UserProfile;
use eye_calibration::protocol::FixationWindow;
use eye_calibration::user_fit::{DotSessionFit, FitConfig, FitSample, ProfileMeta};
use eye_core::log::{field, span};
use eye_core::{GazePoint, Rig};

use crate::metrics::EvalInput;
use crate::runner::{Replayed, eval_input, window_at};

pub type Fitter<'a> = &'a dyn Fn(&[FitSample], &Rig) -> Result<UserProfile, String>;

/// `(target_mm.x.to_bits(), target_mm.y.to_bits())`: the same key `DotSessionFit` groups targets by.
pub fn position_key(window: &FixationWindow) -> (u64, u64) {
    (window.target_mm.x.to_bits(), window.target_mm.y.to_bits())
}

/// Every uncorrected ray of every batch whose timestamp lies in a window selected by `include`,
/// with that window's `target_mm`.
pub fn fit_samples<'a>(
    windows: &[FixationWindow],
    batches: impl IntoIterator<Item = &'a RayBatch>,
    include: impl Fn(&FixationWindow) -> bool,
) -> Vec<FitSample> {
    let mut out = Vec::new();
    for batch in batches {
        let Some(w) = window_at(windows, batch.timestamp)
            .map(|i| &windows[i])
            .filter(|w| include(w))
        else {
            continue;
        };
        out.extend(batch.rays.iter().map(|ray| FitSample {
            ray: ray.clone(),
            target_mm: w.target_mm,
        }));
    }
    out
}

#[derive(Debug, Clone, PartialEq)]
pub struct LotoOutcome {
    pub input: EvalInput,
    pub warnings: Vec<String>,
}

fn fold_keys(windows: &[FixationWindow]) -> Vec<(u64, u64)> {
    let mut keys = Vec::new();
    for w in windows {
        let key = position_key(w);
        if !keys.contains(&key) {
            keys.push(key);
        }
    }
    keys
}

/// Err when there is no window or every fold failed. Leaves the pipeline uncorrected.
pub fn loto(replayed: &mut Replayed, fitter: Fitter<'_>) -> Result<LotoOutcome, String> {
    if replayed.run.windows.is_empty() {
        return Err("no fixation windows".to_string());
    }
    let _stage = tracing::debug_span!(
        span::STAGE,
        { field::STAGE_KIND } = "calibrate",
        { field::STAGE_NAME } = "loto",
    )
    .entered();

    let mut warnings = Vec::new();
    let mut eval_windows = Vec::new();
    let mut first_error: Option<String> = None;
    let mut any_ok = false;

    for key in fold_keys(&replayed.run.windows) {
        let train = fit_samples(
            &replayed.run.windows,
            replayed.run.steps.iter().filter_map(|s| s.batch.as_ref()),
            |w| position_key(w) != key,
        );
        let profile = match fitter(&train, &replayed.run.rig) {
            Ok(profile) => profile,
            Err(e) => {
                let held_out_window = replayed
                    .run
                    .windows
                    .iter()
                    .find(|w| position_key(w) == key)
                    .expect("key was derived from these windows");
                tracing::warn!(
                    target_px_x = held_out_window.target_px_logical.x,
                    target_px_y = held_out_window.target_px_logical.y,
                    train_samples = train.len() as u64,
                    { field::REASON } = %e,
                    "loto fold fit failed"
                );
                warnings.push(format!(
                    "fold ({:.0}, {:.0}): fit failed: {e}",
                    held_out_window.target_px_logical.x, held_out_window.target_px_logical.y
                ));
                first_error.get_or_insert(e);
                continue;
            }
        };
        any_ok = true;
        replayed.pipeline.set_correction(Some(Box::new(profile)));

        let held_out: Vec<FixationWindow> = replayed
            .run
            .windows
            .iter()
            .filter(|w| position_key(w) == key)
            .cloned()
            .collect();

        let mut kept: Vec<(&RayBatch, GazePoint)> = Vec::new();
        for step in &replayed.run.steps {
            let Some(batch) = &step.batch else { continue };
            let Some(point) = replayed.pipeline.finish(batch) else {
                continue;
            };
            if held_out
                .iter()
                .any(|w| w.start <= point.timestamp && point.timestamp < w.end)
            {
                kept.push((batch, point));
            }
        }

        let fold_input = eval_input(
            &replayed.run.rig,
            &held_out,
            kept.iter().map(|(batch, point)| (*batch, point)),
            Vec::new(),
        );
        tracing::debug!(
            target_mm_x = held_out[0].target_mm.x,
            target_mm_y = held_out[0].target_mm.y,
            train_samples = train.len() as u64,
            held_out_windows = held_out.len() as u64,
            kept_samples = kept.len() as u64,
            "loto fold evaluated"
        );
        eval_windows.extend(fold_input.windows);
    }

    replayed.pipeline.set_correction(None);

    if !any_ok {
        return Err(format!(
            "every leave-one-target-out fold failed: {}",
            first_error.unwrap_or_default()
        ));
    }

    eval_windows.sort_by_key(|w| w.start);

    Ok(LotoOutcome {
        input: EvalInput {
            windows: eval_windows,
            processing: replayed.run.processing(),
        },
        warnings,
    })
}

pub fn dot_session_fitter(samples: &[FitSample], rig: &Rig) -> Result<UserProfile, String> {
    DotSessionFit::fit(samples, rig).map_err(|e| e.to_string())
}

pub fn dot_session_fitter_with(
    cfg: FitConfig,
) -> impl Fn(&[FitSample], &Rig) -> Result<UserProfile, String> {
    move |samples, rig| {
        DotSessionFit::fit_with(samples, rig, &cfg, ProfileMeta::default())
            .map(|outcome| outcome.profile)
            .map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::BTreeSet;

    use approx::assert_relative_eq;
    use eye_calibration::protocol::ProtocolConfig;
    use eye_core::log::field;
    use eye_geometry::angles::{direction_from_yaw_pitch, yaw_pitch_from_direction};
    use eye_log::testing::capture_logs;
    use nalgebra::{Point2, Point3, Unit, UnitQuaternion, Vector2};

    use super::*;
    use crate::row::RowOutcome;
    use crate::runner::replay_session;
    use crate::testing::{
        FOUR_BY_FOUR_CENTRES, SyntheticSession, fake_registry, kappa_ray_config, kappa_ray_toml,
        synthetic_screen, write_synthetic_session,
    };

    fn eye() -> Point3<f64> {
        Point3::new(155.0, 85.0, -500.0)
    }

    #[test]
    fn test_fit_samples_excludes_held_out_target() {
        let dir = tempfile::tempdir().unwrap();
        let spec = SyntheticSession {
            targets: FOUR_BY_FOUR_CENTRES[..4].to_vec(),
            code_frames: true,
            ..Default::default()
        };
        let session_dir = write_synthetic_session(dir.path(), "s1", &spec).unwrap();
        let config = kappa_ray_config(&FOUR_BY_FOUR_CENTRES[..4], [0.0, 0.0], None);
        let replayed = replay_session(
            &session_dir,
            &config,
            &fake_registry(),
            &ProtocolConfig::default(),
        )
        .unwrap();
        let held_out_mm = replayed.run.windows[2].target_mm;
        let batches = replayed.run.steps.iter().filter_map(|s| s.batch.as_ref());
        let samples = fit_samples(&replayed.run.windows, batches, |w| w.index != 2);
        assert!(samples.iter().all(|s| s.target_mm != held_out_mm));
        let expected: usize = replayed
            .run
            .steps
            .iter()
            .filter_map(|s| s.batch.as_ref())
            .filter(|b| {
                window_at(&replayed.run.windows, b.timestamp)
                    .map(|i| i != 2)
                    .unwrap_or(false)
            })
            .map(|b| b.rays.len())
            .sum();
        assert_eq!(samples.len(), expected);
    }

    #[test]
    fn test_fit_samples_ignores_batches_outside_windows() {
        let dir = tempfile::tempdir().unwrap();
        let spec = SyntheticSession {
            targets: FOUR_BY_FOUR_CENTRES[..4].to_vec(),
            code_frames: true,
            ..Default::default()
        };
        let session_dir = write_synthetic_session(dir.path(), "s1", &spec).unwrap();
        let config = kappa_ray_config(&FOUR_BY_FOUR_CENTRES[..4], [0.0, 0.0], None);
        let replayed = replay_session(
            &session_dir,
            &config,
            &fake_registry(),
            &ProtocolConfig::default(),
        )
        .unwrap();
        let batches = replayed.run.steps.iter().filter_map(|s| s.batch.as_ref());
        let samples = fit_samples(&replayed.run.windows, batches, |_| true);
        let in_window: usize = replayed
            .run
            .steps
            .iter()
            .filter_map(|s| s.batch.as_ref())
            .filter(|b| window_at(&replayed.run.windows, b.timestamp).is_some())
            .map(|b| b.rays.len())
            .sum();
        assert_eq!(samples.len(), in_window);
        assert!(in_window < replayed.run.steps.len());
    }

    #[test]
    fn test_fit_samples_keeps_every_ray_of_a_batch() {
        let window = FixationWindow {
            index: 0,
            cell: (0, 0),
            onset: eye_core::Timestamp::from_nanos(0),
            start: eye_core::Timestamp::from_nanos(0),
            end: eye_core::Timestamp::from_nanos(1_000_000_000),
            target_mm: Point2::new(155.0, 85.0),
            target_px_logical: Point2::new(960.0, 540.0),
        };
        let ray = eye_core::GazeRay {
            side: None,
            origin: eye(),
            direction: Unit::new_normalize(nalgebra::Vector3::new(0.0, 0.0, 1.0)),
            origin_cov: nalgebra::Matrix3::zeros(),
            angular_cov: nalgebra::Matrix2::identity() * 1e-6,
            head_rotation: UnitQuaternion::identity(),
        };
        let batch = RayBatch {
            timestamp: eye_core::Timestamp::from_nanos(500_000_000),
            rays: vec![ray.clone(), ray],
        };
        let samples = fit_samples(std::slice::from_ref(&window), [&batch], |_| true);
        assert_eq!(samples.len(), 2);
        assert!(samples.iter().all(|s| s.target_mm == window.target_mm));
    }

    type SpyResult = (Result<LotoOutcome, String>, Vec<BTreeSet<(u64, u64)>>);

    fn spy_loto(replayed: &mut Replayed) -> SpyResult {
        let calls: RefCell<Vec<BTreeSet<(u64, u64)>>> = RefCell::new(Vec::new());
        let fitter = |samples: &[FitSample], rig: &Rig| {
            let keys: BTreeSet<(u64, u64)> = samples
                .iter()
                .map(|s| (s.target_mm.x.to_bits(), s.target_mm.y.to_bits()))
                .collect();
            calls.borrow_mut().push(keys);
            dot_session_fitter(samples, rig)
        };
        let result = loto(replayed, &fitter);
        (result, calls.into_inner())
    }

    #[test]
    fn test_loto_never_trains_on_held_out_position() {
        let dir = tempfile::tempdir().unwrap();
        let spec = SyntheticSession {
            targets: FOUR_BY_FOUR_CENTRES.to_vec(),
            code_frames: true,
            ..Default::default()
        };
        let session_dir = write_synthetic_session(dir.path(), "s1", &spec).unwrap();
        let config = kappa_ray_config(&FOUR_BY_FOUR_CENTRES, [3.0, -1.0], None);
        let mut replayed = replay_session(
            &session_dir,
            &config,
            &fake_registry(),
            &ProtocolConfig::default(),
        )
        .unwrap();

        let (result, calls) = spy_loto(&mut replayed);
        result.unwrap();
        assert_eq!(calls.len(), 16);
        for (i, keys) in calls.iter().enumerate() {
            assert_eq!(keys.len(), 15, "fold {i} trained on {} targets", keys.len());
            let missing_key = position_key(&replayed.run.windows[i]);
            assert!(
                !keys.contains(&missing_key),
                "fold {i} trained on its own held-out target"
            );
        }
    }

    #[test]
    fn test_loto_removes_constant_kappa_offset() {
        let dir = tempfile::tempdir().unwrap();
        let spec = SyntheticSession {
            targets: FOUR_BY_FOUR_CENTRES.to_vec(),
            code_frames: true,
            ..Default::default()
        };
        let session_dir = write_synthetic_session(dir.path(), "s1", &spec).unwrap();
        let toml_path = dir.path().join("kappa.toml");
        std::fs::write(
            &toml_path,
            kappa_ray_toml(&FOUR_BY_FOUR_CENTRES, [3.0, -1.0], None),
        )
        .unwrap();
        let matrix = crate::matrix::BenchMatrix {
            recordings: vec![session_dir],
            pipelines: vec![crate::matrix::PipelineSpec {
                name: "kappa".to_string(),
                config: Some(toml_path),
            }],
            evaluation: crate::matrix::Evaluation {
                calibration: vec![
                    crate::row::CalibrationMode::None,
                    crate::row::CalibrationMode::Loto,
                ],
                ..Default::default()
            },
        };
        let report = crate::runner::run_matrix(&matrix, &fake_registry());

        let none_row = report
            .rows
            .iter()
            .find(|r| {
                r.kind == crate::row::RowKind::Session
                    && r.calibration == crate::row::CalibrationMode::None
            })
            .unwrap();
        let RowOutcome::Ok {
            metrics: none_metrics,
        } = &none_row.outcome
        else {
            panic!("expected ok row: {none_row:?}");
        };

        let screen = synthetic_screen();
        let expected_none: f64 = FOUR_BY_FOUR_CENTRES
            .iter()
            .map(|&(x, y)| {
                let target_mm = eye_geometry::screen::px_logical_to_mm(&screen, &Point2::new(x, y));
                let base = yaw_pitch_from_direction(&Unit::new_normalize(
                    Point3::new(target_mm.x, target_mm.y, 0.0) - eye(),
                ));
                let shifted = direction_from_yaw_pitch(
                    &(base + Vector2::new(3.0_f64.to_radians(), (-1.0_f64).to_radians())),
                );
                let base_dir = direction_from_yaw_pitch(&base);
                crate::metrics::vector_angle_deg(&shifted.into_inner(), &base_dir.into_inner())
            })
            .sum::<f64>()
            / FOUR_BY_FOUR_CENTRES.len() as f64;
        assert_relative_eq!(
            none_metrics.angular_error_deg.as_ref().unwrap().mean,
            expected_none,
            epsilon = 1e-6
        );

        let loto_row = report
            .rows
            .iter()
            .find(|r| {
                r.kind == crate::row::RowKind::Session
                    && r.calibration == crate::row::CalibrationMode::Loto
            })
            .unwrap();
        let RowOutcome::Ok {
            metrics: loto_metrics,
        } = &loto_row.outcome
        else {
            panic!("expected ok row: {loto_row:?}");
        };
        assert!(loto_metrics.angular_error_deg.as_ref().unwrap().mean < 0.1);
        assert!(loto_row.warnings.is_empty());
    }

    #[test]
    fn test_loto_held_out_outlier_is_not_absorbed() {
        let dir = tempfile::tempdir().unwrap();
        let spec = SyntheticSession {
            targets: FOUR_BY_FOUR_CENTRES.to_vec(),
            code_frames: true,
            ..Default::default()
        };
        let session_dir = write_synthetic_session(dir.path(), "s1", &spec).unwrap();
        let config = kappa_ray_config(&FOUR_BY_FOUR_CENTRES, [3.0, -1.0], Some((5, [6.0, 0.0])));
        let mut replayed = replay_session(
            &session_dir,
            &config,
            &fake_registry(),
            &ProtocolConfig::default(),
        )
        .unwrap();
        let outcome = loto(&mut replayed, &dot_session_fitter).unwrap();

        let window = outcome
            .input
            .windows
            .iter()
            .find(|w| w.target_px_logical == Point2::new(720.0, 405.0))
            .unwrap();
        let mean_error: f64 = window
            .points
            .iter()
            .map(|p| crate::metrics::angle_deg(&p.eye_mm, &p.gaze_mm, &window.target_mm))
            .sum::<f64>()
            / window.points.len() as f64;
        assert!(
            (2.5..4.0).contains(&mean_error),
            "mean error {mean_error} outside expected range"
        );
    }

    #[test]
    fn test_failed_fold_becomes_warning() {
        let dir = tempfile::tempdir().unwrap();
        let spec = SyntheticSession {
            targets: FOUR_BY_FOUR_CENTRES.to_vec(),
            code_frames: true,
            ..Default::default()
        };
        let session_dir = write_synthetic_session(dir.path(), "s1", &spec).unwrap();
        let config = kappa_ray_config(&FOUR_BY_FOUR_CENTRES, [3.0, -1.0], None);
        let mut replayed = replay_session(
            &session_dir,
            &config,
            &fake_registry(),
            &ProtocolConfig::default(),
        )
        .unwrap();
        let screen = synthetic_screen();
        let failing_mm =
            eye_geometry::screen::px_logical_to_mm(&screen, &Point2::new(240.0, 135.0));
        let fitter = |samples: &[FitSample], rig: &Rig| {
            if samples.iter().all(|s| s.target_mm != failing_mm) {
                Err("too few samples".to_string())
            } else {
                dot_session_fitter(samples, rig)
            }
        };
        let outcome = loto(&mut replayed, &fitter).unwrap();
        assert_eq!(
            outcome.warnings,
            vec!["fold (240, 135): fit failed: too few samples"]
        );
        assert_eq!(outcome.input.windows.len(), 15);
    }

    #[test]
    fn test_repeated_position_is_held_out_together() {
        let dir = tempfile::tempdir().unwrap();
        let nine: Vec<(f64, f64)> = [320.0, 960.0, 1600.0]
            .iter()
            .flat_map(|&x| [180.0, 540.0, 900.0].iter().map(move |&y| (x, y)))
            .collect();
        let mut targets = nine.clone();
        targets.push((320.0, 180.0));
        let spec = SyntheticSession {
            targets: targets.clone(),
            code_frames: true,
            ..Default::default()
        };
        let session_dir = write_synthetic_session(dir.path(), "s1", &spec).unwrap();
        let config = kappa_ray_config(&targets, [3.0, -1.0], None);
        let mut replayed = replay_session(
            &session_dir,
            &config,
            &fake_registry(),
            &ProtocolConfig::default(),
        )
        .unwrap();

        let (result, calls) = spy_loto(&mut replayed);
        let outcome = result.unwrap();
        assert_eq!(calls.len(), 9);

        let screen = synthetic_screen();
        let repeated_key = {
            let mm = eye_geometry::screen::px_logical_to_mm(&screen, &Point2::new(320.0, 180.0));
            (mm.x.to_bits(), mm.y.to_bits())
        };
        let missing_count = calls.iter().filter(|k| !k.contains(&repeated_key)).count();
        assert_eq!(missing_count, 1);

        let repeated_windows = outcome
            .input
            .windows
            .iter()
            .filter(|w| w.target_px_logical == Point2::new(320.0, 180.0))
            .count();
        assert_eq!(repeated_windows, 2);
    }

    #[test]
    fn test_all_folds_failed_is_error() {
        let dir = tempfile::tempdir().unwrap();
        let spec = SyntheticSession {
            targets: FOUR_BY_FOUR_CENTRES.to_vec(),
            code_frames: true,
            ..Default::default()
        };
        let session_dir = write_synthetic_session(dir.path(), "s1", &spec).unwrap();
        let config = kappa_ray_config(&FOUR_BY_FOUR_CENTRES, [3.0, -1.0], None);
        let mut replayed = replay_session(
            &session_dir,
            &config,
            &fake_registry(),
            &ProtocolConfig::default(),
        )
        .unwrap();
        let fitter = |_: &[FitSample], _: &Rig| Err("x".to_string());
        let err = loto(&mut replayed, &fitter).unwrap_err();
        assert!(err.contains("every leave-one-target-out fold failed"));
    }

    #[test]
    fn test_loto_restores_no_correction() {
        let dir = tempfile::tempdir().unwrap();
        let spec = SyntheticSession {
            targets: FOUR_BY_FOUR_CENTRES.to_vec(),
            code_frames: true,
            ..Default::default()
        };
        let session_dir = write_synthetic_session(dir.path(), "s1", &spec).unwrap();
        let config = kappa_ray_config(&FOUR_BY_FOUR_CENTRES, [3.0, -1.0], None);
        let mut replayed = replay_session(
            &session_dir,
            &config,
            &fake_registry(),
            &ProtocolConfig::default(),
        )
        .unwrap();
        let first_batch = replayed
            .run
            .steps
            .iter()
            .find_map(|s| s.batch.clone())
            .unwrap();
        let first_point = replayed
            .run
            .steps
            .iter()
            .find(|s| s.batch.is_some())
            .unwrap()
            .point
            .clone();

        loto(&mut replayed, &dot_session_fitter).unwrap();

        let after = replayed.pipeline.finish(&first_batch);
        assert_eq!(after, first_point);
    }

    fn fit_section_toml(offset_prior_sigma_deg: Option<f64>) -> String {
        match offset_prior_sigma_deg {
            Some(sigma) => format!("\n[evaluation.fit]\noffset_prior_sigma_deg = {sigma:?}\n"),
            None => String::new(),
        }
    }

    #[test]
    fn test_matrix_fit_config_is_passed_to_fitter() {
        let dir = tempfile::tempdir().unwrap();
        let spec = SyntheticSession {
            targets: FOUR_BY_FOUR_CENTRES.to_vec(),
            code_frames: true,
            ..Default::default()
        };
        let session_dir = write_synthetic_session(dir.path(), "s1", &spec).unwrap();
        let eye_toml = dir.path().join("kappa.toml");
        std::fs::write(
            &eye_toml,
            kappa_ray_toml(&FOUR_BY_FOUR_CENTRES, [5.0, 0.0], None),
        )
        .unwrap();

        let run = |offset_prior_sigma_deg: Option<f64>| {
            let bench_toml_path = dir.path().join("bench.toml");
            std::fs::write(
                &bench_toml_path,
                format!(
                    "recordings = [{:?}]\n\n[[pipeline]]\nname = \"kappa\"\nconfig = {:?}\n\n[evaluation]\ncalibration = [\"loto\"]\n{}",
                    session_dir.to_str().unwrap(),
                    eye_toml.to_str().unwrap(),
                    fit_section_toml(offset_prior_sigma_deg),
                ),
            )
            .unwrap();
            let matrix = crate::matrix::BenchMatrix::from_path(&bench_toml_path).unwrap();
            let report = crate::runner::run_matrix(&matrix, &fake_registry());
            let loto_row = report
                .rows
                .iter()
                .find(|r| {
                    r.kind == crate::row::RowKind::Session
                        && r.calibration == crate::row::CalibrationMode::Loto
                })
                .unwrap();
            let RowOutcome::Ok { metrics } = &loto_row.outcome else {
                panic!("expected ok row: {loto_row:?}");
            };
            metrics.angular_error_deg.as_ref().unwrap().mean
        };

        let default_err = run(None);
        let tight_prior_err = run(Some(0.001));

        assert!(
            default_err < 0.5,
            "default offset prior should recover the bias: {default_err} deg"
        );
        assert!(
            tight_prior_err > 3.0,
            "a 0.001 deg offset prior should leave most of the 5 deg bias uncorrected: {tight_prior_err} deg"
        );
    }

    #[test]
    fn test_logs_loto_fold_evaluated_at_debug() {
        let dir = tempfile::tempdir().unwrap();
        let spec = SyntheticSession {
            targets: FOUR_BY_FOUR_CENTRES.to_vec(),
            code_frames: true,
            ..Default::default()
        };
        let session_dir = write_synthetic_session(dir.path(), "s1", &spec).unwrap();
        let config = kappa_ray_config(&FOUR_BY_FOUR_CENTRES, [3.0, -1.0], None);
        let mut replayed = replay_session(
            &session_dir,
            &config,
            &fake_registry(),
            &ProtocolConfig::default(),
        )
        .unwrap();

        let (result, records) = capture_logs(tracing::Level::TRACE, || {
            loto(&mut replayed, &dot_session_fitter)
        });
        result.unwrap();
        let matches: Vec<_> = records
            .iter()
            .filter(|r| r.message == "loto fold evaluated")
            .collect();
        assert_eq!(matches.len(), 16);
        for rec in &matches {
            assert_eq!(rec.level, eye_log::Level::Debug);
            assert_eq!(
                rec.context[field::STAGE_KIND],
                eye_log::Value::Str("calibrate".to_string())
            );
            assert_eq!(
                rec.context[field::STAGE_NAME],
                eye_log::Value::Str("loto".to_string())
            );
            assert_eq!(rec.fields["held_out_windows"], eye_log::Value::U64(1));
        }
    }

    #[test]
    fn test_logs_loto_fold_fit_failed_at_warn() {
        let dir = tempfile::tempdir().unwrap();
        let spec = SyntheticSession {
            targets: FOUR_BY_FOUR_CENTRES.to_vec(),
            code_frames: true,
            ..Default::default()
        };
        let session_dir = write_synthetic_session(dir.path(), "s1", &spec).unwrap();
        let config = kappa_ray_config(&FOUR_BY_FOUR_CENTRES, [3.0, -1.0], None);
        let mut replayed = replay_session(
            &session_dir,
            &config,
            &fake_registry(),
            &ProtocolConfig::default(),
        )
        .unwrap();
        let screen = synthetic_screen();
        let failing_mm =
            eye_geometry::screen::px_logical_to_mm(&screen, &Point2::new(240.0, 135.0));
        let fitter = |samples: &[FitSample], rig: &Rig| {
            if samples.iter().all(|s| s.target_mm != failing_mm) {
                Err("too few samples".to_string())
            } else {
                dot_session_fitter(samples, rig)
            }
        };

        let (result, records) =
            capture_logs(tracing::Level::TRACE, || loto(&mut replayed, &fitter));
        result.unwrap();
        let matches: Vec<_> = records
            .iter()
            .filter(|r| r.message == "loto fold fit failed")
            .collect();
        assert_eq!(matches.len(), 1);
        let rec = matches[0];
        assert_eq!(rec.level, eye_log::Level::Warn);
        assert_eq!(rec.fields["target_px_x"], eye_log::Value::F64(240.0));
        assert_eq!(
            rec.fields[field::REASON],
            eye_log::Value::Str("too few samples".to_string())
        );
    }
}
