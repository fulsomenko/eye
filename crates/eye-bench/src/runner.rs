//! Replays recordings through pipelines and scores the result.

use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};

use eye::config::{Config, EnvOverrides};
use eye::pipeline::{Pipeline, PipelineError, RayBatch, RayStep};
use eye::registry::Registry;
use eye_calibration::protocol::{FixationWindow, ProtocolConfig, TargetProtocol};
use eye_calibration::store::rig_from_table;
use eye_capture::session::{RecordedCamera, Recording};
use eye_core::{CameraInfo, GazePoint, Rig, Timestamp};
use nalgebra::{Point3, Vector3};

use crate::error::BenchError;
use crate::matrix::{BenchMatrix, PipelineSpec};
use crate::metrics::{EvalInput, EvalPoint, EvalWindow, compute};
use crate::report::BenchReport;
use crate::row::{BenchRow, CalibrationMode, RowKind, RowOutcome};

#[derive(Debug)]
pub struct Step {
    pub batch: Option<RayBatch>,
    pub point: Option<GazePoint>,
    /// `rays` returned `RayStep::Skipped` (a tolerated stage error).
    pub skipped: bool,
    pub processing: Duration,
}

#[derive(Debug)]
pub struct SessionRun {
    /// `meta.session_id`.
    pub session: String,
    pub rig: Rig,
    pub windows: Vec<FixationWindow>,
    pub steps: Vec<Step>,
}

/// The pipeline used for the replay, so `finish` can be re-run on the cached
/// batches without rebuilding the pipeline.
#[derive(Debug)]
pub struct Replayed {
    pub run: SessionRun,
    pub pipeline: Pipeline,
}

impl SessionRun {
    /// Steps with `skipped`.
    pub fn step_errors(&self) -> usize {
        self.steps.iter().filter(|s| s.skipped).count()
    }

    /// Every step, in order.
    pub fn processing(&self) -> Vec<Duration> {
        self.steps.iter().map(|s| s.processing).collect()
    }

    /// Steps with both batch and point.
    pub fn samples(&self) -> impl Iterator<Item = (&RayBatch, &GazePoint)> {
        self.steps
            .iter()
            .filter_map(|s| Some((s.batch.as_ref()?, s.point.as_ref()?)))
    }
}

fn pipeline_error(e: PipelineError) -> BenchError {
    BenchError::Pipeline(Box::new(e))
}

#[allow(clippy::result_large_err)]
fn step(pipeline: &mut Pipeline, set: &eye_core::FrameSet) -> Result<Step, BenchError> {
    let started = Instant::now();
    Ok(match pipeline.rays(set).map_err(pipeline_error)? {
        RayStep::Rays(batch) => {
            let point = pipeline.finish(&batch);
            Step {
                batch: Some(batch),
                point,
                skipped: false,
                processing: started.elapsed(),
            }
        }
        RayStep::NoGaze => Step {
            batch: None,
            point: None,
            skipped: false,
            processing: started.elapsed(),
        },
        RayStep::Skipped(_) => Step {
            batch: None,
            point: None,
            skipped: true,
            processing: started.elapsed(),
        },
    })
}

/// Replays one recording through a fresh pipeline without correction.
#[allow(clippy::result_large_err)]
pub fn replay_session(
    dir: &Path,
    config: &Config,
    registry: &Registry,
    protocol: &ProtocolConfig,
) -> Result<Replayed, BenchError> {
    let capture = |source| BenchError::Capture {
        session: dir.to_path_buf(),
        source,
    };
    let recording = Recording::open(dir).map_err(capture)?;
    let meta = recording.meta();
    let session = meta.session_id.clone();
    let rig_table = meta.rig.as_ref().ok_or_else(|| BenchError::NoRig {
        session: session.clone(),
    })?;
    let rig = rig_from_table(rig_table).map_err(|e| BenchError::Calibration(Box::new(e)))?;
    let cameras = config
        .cameras
        .iter()
        .map(|cam| {
            recording
                .camera(cam.id.as_str())
                .map(RecordedCamera::to_info)
                .ok_or_else(|| BenchError::MissingCamera {
                    camera: cam.id.to_string(),
                    session: session.clone(),
                })
        })
        .collect::<Result<Vec<CameraInfo>, _>>()?;
    let windows = TargetProtocol::new(*protocol)
        .map_err(|e| BenchError::Calibration(Box::new(e)))?
        .fixation_windows(recording.targets(), rig.screen())
        .map_err(|e| BenchError::Calibration(Box::new(e)))?;
    let mut pipeline = Pipeline::from_config(registry, config, rig.clone(), &cameras, None)
        .map_err(pipeline_error)?;
    let mut steps = Vec::new();
    for record in recording
        .merged_index()
        .into_iter()
        .filter(|r| config.camera(&r.camera).is_some())
    {
        let frame = recording.read_frame(record).map_err(capture)?;
        if let Some(set) = pipeline.pair(frame) {
            steps.push(step(&mut pipeline, &set)?);
        }
    }
    while let Some(set) = pipeline.flush() {
        steps.push(step(&mut pipeline, &set)?);
    }
    Ok(Replayed {
        run: SessionRun {
            session,
            rig,
            windows,
            steps,
        },
        pipeline,
    })
}

fn mean_origin(rays: &[eye_core::GazeRay]) -> Option<Point3<f64>> {
    (!rays.is_empty()).then(|| {
        Point3::from(rays.iter().map(|r| r.origin.coords).sum::<Vector3<f64>>() / rays.len() as f64)
    })
}

/// Index of the window containing `t` (windows sorted, disjoint, end exclusive).
pub fn window_at(windows: &[FixationWindow], t: Timestamp) -> Option<usize> {
    let i = windows.partition_point(|w| w.end <= t);
    windows.get(i).filter(|w| w.start <= t).map(|_| i)
}

/// Maps gaze points into fixation windows (points outside every window are dropped).
pub fn eval_input<'a>(
    rig: &Rig,
    windows: &[FixationWindow],
    samples: impl IntoIterator<Item = (&'a RayBatch, &'a GazePoint)>,
    processing: Vec<Duration>,
) -> EvalInput {
    let screen = rig.screen();
    let logical = nalgebra::Vector2::new(f64::from(screen.size_px.0), f64::from(screen.size_px.1))
        / screen.scale;
    let mut out: Vec<EvalWindow> = windows
        .iter()
        .map(|w| EvalWindow {
            target_index: w.index as usize,
            start: w.start,
            end: w.end,
            target_mm: w.target_mm,
            target_px_logical: w.target_px_logical,
            screen_logical: logical,
            points: Vec::new(),
        })
        .collect();
    for (batch, point) in samples {
        let Some(eye_mm) = mean_origin(&batch.rays) else {
            continue;
        };
        if let Some(i) = window_at(windows, point.timestamp) {
            out[i].points.push(EvalPoint {
                timestamp: point.timestamp,
                eye_mm,
                gaze_mm: point.mm,
                gaze_px_logical: point.px_logical,
            });
        }
    }
    EvalInput {
        windows: out,
        processing,
    }
}

/// `Err(message)` becomes that mode's `RowOutcome::Error`; the `None` arm always returns `Ok`.
fn evaluate(
    mode: CalibrationMode,
    replayed: &mut Replayed,
    fit: eye_calibration::user_fit::FitConfig,
) -> Result<(EvalInput, Vec<String>), String> {
    match mode {
        CalibrationMode::None => Ok((
            eval_input(
                &replayed.run.rig,
                &replayed.run.windows,
                replayed.run.samples(),
                replayed.run.processing(),
            ),
            Vec::new(),
        )),
        CalibrationMode::Loto => {
            let fitter = crate::calibration::dot_session_fitter_with(fit);
            crate::calibration::loto(replayed, &fitter).map(|o| (o.input, o.warnings))
        }
    }
}

fn session_name(dir: &Path) -> String {
    dir.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| dir.display().to_string())
}

#[allow(clippy::result_large_err)]
fn load_pipeline_config(spec: &PipelineSpec) -> Result<Config, BenchError> {
    match &spec.config {
        Some(path) => Config::load_with(Some(path), &EnvOverrides::default())
            .map(|loaded| loaded.config)
            .map_err(|source| BenchError::Config {
                pipeline: spec.name.clone(),
                source,
            }),
        None => Ok(Config::builtin_default()),
    }
}

fn error_row(
    pipeline: &str,
    mode: CalibrationMode,
    kind: RowKind,
    session: &str,
    step_errors: usize,
    message: String,
) -> BenchRow {
    BenchRow {
        pipeline: pipeline.to_string(),
        calibration: mode,
        kind,
        session: session.to_string(),
        step_errors,
        warnings: Vec::new(),
        outcome: RowOutcome::Error { message },
    }
}

/// A broken config gives error rows for every session of that pipeline (and its aggregates).
fn config_error_rows(spec: &PipelineSpec, matrix: &BenchMatrix, message: &str) -> Vec<BenchRow> {
    let mut rows = Vec::new();
    for dir in &matrix.recordings {
        let session = session_name(dir);
        for &mode in &matrix.evaluation.calibration {
            rows.push(error_row(
                &spec.name,
                mode,
                RowKind::Session,
                &session,
                0,
                message.to_string(),
            ));
        }
    }
    for &mode in &matrix.evaluation.calibration {
        rows.push(error_row(
            &spec.name,
            mode,
            RowKind::Aggregate,
            "all",
            0,
            "no session produced a result".to_string(),
        ));
    }
    rows
}

fn rows_for_pipeline(
    spec: &PipelineSpec,
    matrix: &BenchMatrix,
    registry: &Registry,
) -> Vec<BenchRow> {
    let config = match load_pipeline_config(spec) {
        Ok(config) => config,
        Err(e) => return config_error_rows(spec, matrix, &e.to_string()),
    };

    let mut rows = Vec::new();
    let mut per_mode: HashMap<CalibrationMode, (Vec<EvalInput>, usize)> = HashMap::new();

    for dir in &matrix.recordings {
        let session = session_name(dir);
        let started = Instant::now();
        match replay_session(dir, &config, registry, &matrix.evaluation.protocol) {
            Ok(mut replayed) => {
                let step_errors = replayed.run.step_errors();
                tracing::info!(
                    pipeline = %spec.name,
                    session = %session,
                    steps = replayed.run.steps.len(),
                    elapsed_ms = started.elapsed().as_millis(),
                    "session replayed"
                );
                for &mode in &matrix.evaluation.calibration {
                    match evaluate(mode, &mut replayed, matrix.evaluation.fit) {
                        Ok((input, warnings)) => {
                            let metrics = compute(&input, &matrix.evaluation.metrics);
                            rows.push(BenchRow {
                                pipeline: spec.name.clone(),
                                calibration: mode,
                                kind: RowKind::Session,
                                session: session.clone(),
                                step_errors,
                                warnings,
                                outcome: RowOutcome::Ok { metrics },
                            });
                            let entry = per_mode.entry(mode).or_default();
                            entry.0.push(input);
                            entry.1 += step_errors;
                        }
                        Err(message) => {
                            rows.push(error_row(
                                &spec.name,
                                mode,
                                RowKind::Session,
                                &session,
                                step_errors,
                                message,
                            ));
                        }
                    }
                }
            }
            Err(e) => {
                for &mode in &matrix.evaluation.calibration {
                    rows.push(error_row(
                        &spec.name,
                        mode,
                        RowKind::Session,
                        &session,
                        0,
                        e.to_string(),
                    ));
                }
            }
        }
    }

    for &mode in &matrix.evaluation.calibration {
        match per_mode.get(&mode) {
            Some((inputs, step_errors)) if !inputs.is_empty() => {
                let pooled = EvalInput::concat(inputs);
                let metrics = compute(&pooled, &matrix.evaluation.metrics);
                rows.push(BenchRow {
                    pipeline: spec.name.clone(),
                    calibration: mode,
                    kind: RowKind::Aggregate,
                    session: "all".to_string(),
                    step_errors: *step_errors,
                    warnings: Vec::new(),
                    outcome: RowOutcome::Ok { metrics },
                });
            }
            _ => {
                rows.push(error_row(
                    &spec.name,
                    mode,
                    RowKind::Aggregate,
                    "all",
                    0,
                    "no session produced a result".to_string(),
                ));
            }
        }
    }

    rows
}

pub fn run_matrix(matrix: &BenchMatrix, registry: &Registry) -> BenchReport {
    let mut rows = Vec::new();
    for spec in &matrix.pipelines {
        rows.extend(rows_for_pipeline(spec, matrix, registry));
    }
    BenchReport::new(matrix.evaluation.metrics.clone(), rows)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use eye_core::{GazeRay, OutputId};
    use nalgebra::{Matrix2, Matrix3, Point2, Unit, UnitQuaternion};

    use super::*;
    use crate::metrics;
    use crate::row::RowOutcome;
    use crate::testing::{
        FOUR_BY_FOUR_CENTRES, SyntheticSession, fake_registry, fixed_ray_config, fixed_ray_toml,
        kappa_ray_toml, write_synthetic_session,
    };

    fn eye() -> Point3<f64> {
        Point3::new(155.0, 85.0, -500.0)
    }

    fn target(seq: u32, start_ms: u64, end_ms: u64, mm: Point2<f64>) -> FixationWindow {
        FixationWindow {
            index: seq,
            cell: (0, 0),
            onset: Timestamp::from_nanos(start_ms * 1_000_000),
            start: Timestamp::from_nanos(start_ms * 1_000_000),
            end: Timestamp::from_nanos(end_ms * 1_000_000),
            target_mm: mm,
            target_px_logical: Point2::new(0.0, 0.0),
        }
    }

    #[test]
    fn test_window_at_is_half_open() {
        let windows = vec![
            target(0, 1000, 2000, Point2::new(0.0, 0.0)),
            target(1, 3000, 4000, Point2::new(0.0, 0.0)),
        ];
        assert_eq!(
            window_at(&windows, Timestamp::from_nanos(500_000_000)),
            None
        );
        assert_eq!(
            window_at(&windows, Timestamp::from_nanos(1_000_000_000)),
            Some(0)
        );
        assert_eq!(
            window_at(&windows, Timestamp::from_nanos(1_990_000_000)),
            Some(0)
        );
        assert_eq!(
            window_at(&windows, Timestamp::from_nanos(2_000_000_000)),
            None
        );
        assert_eq!(
            window_at(&windows, Timestamp::from_nanos(3_500_000_000)),
            Some(1)
        );
    }

    #[test]
    fn test_eval_input_eye_is_mean_of_ray_origins() {
        let rig = crate::testing::synthetic_rig();
        let window = FixationWindow {
            index: 0,
            cell: (0, 0),
            onset: Timestamp::from_nanos(0),
            start: Timestamp::from_nanos(0),
            end: Timestamp::from_nanos(1_000_000_000),
            target_mm: Point2::new(155.0, 85.0),
            target_px_logical: Point2::new(960.0, 540.0),
        };
        let windows = vec![window];
        let batch = RayBatch {
            timestamp: Timestamp::from_nanos(500_000_000),
            rays: vec![
                GazeRay {
                    side: None,
                    origin: Point3::new(120.0, 85.0, -500.0),
                    direction: Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0)),
                    origin_cov: Matrix3::zeros(),
                    angular_cov: Matrix2::identity() * 1e-6,
                    head_rotation: UnitQuaternion::identity(),
                },
                GazeRay {
                    side: None,
                    origin: Point3::new(190.0, 85.0, -500.0),
                    direction: Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0)),
                    origin_cov: Matrix3::zeros(),
                    angular_cov: Matrix2::identity() * 1e-6,
                    head_rotation: UnitQuaternion::identity(),
                },
            ],
        };
        let point = GazePoint {
            timestamp: Timestamp::from_nanos(500_000_000),
            output: OutputId::from("eDP-1"),
            mm: Point2::new(155.0, 85.0),
            px_physical: Point2::new(0.0, 0.0),
            px_logical: Point2::new(960.0, 540.0),
            cov_mm: Matrix2::zeros(),
            confidence: 1.0,
        };
        let input = eval_input(&rig, &windows, [(&batch, &point)], Vec::new());
        assert_eq!(input.windows[0].points.len(), 1);
        let eye_mm = input.windows[0].points[0].eye_mm;
        assert_relative_eq!(eye_mm.x, eye().x, epsilon = 1e-9);
        assert_relative_eq!(eye_mm.y, eye().y, epsilon = 1e-9);
        assert_relative_eq!(eye_mm.z, eye().z, epsilon = 1e-9);
    }

    #[test]
    fn test_replay_feeds_only_configured_cameras() {
        let dir = tempfile::tempdir().unwrap();
        let spec = SyntheticSession {
            cameras: vec!["ir", "rgb"],
            ..Default::default()
        };
        let session_dir = write_synthetic_session(dir.path(), "s1", &spec).unwrap();
        let config = fixed_ray_config([155.0, 85.0, -500.0], [161.458333, 94.444444]);
        let replayed = replay_session(
            &session_dir,
            &config,
            &fake_registry(),
            &ProtocolConfig::default(),
        )
        .unwrap();
        assert_eq!(replayed.run.steps.len(), 61);
    }

    #[test]
    fn test_fixed_ray_gives_known_region_hits() {
        let dir = tempfile::tempdir().unwrap();
        let spec = SyntheticSession {
            targets: FOUR_BY_FOUR_CENTRES.to_vec(),
            ..Default::default()
        };
        let session_dir = write_synthetic_session(dir.path(), "s1", &spec).unwrap();
        let toml_path = dir.path().join("fixed.toml");
        std::fs::write(
            &toml_path,
            fixed_ray_toml([155.0, 85.0, -500.0], [161.458333, 94.444444]),
        )
        .unwrap();
        let matrix = BenchMatrix {
            recordings: vec![session_dir],
            pipelines: vec![crate::matrix::PipelineSpec {
                name: "fixed".to_string(),
                config: Some(toml_path),
            }],
            evaluation: crate::matrix::Evaluation {
                calibration: vec![CalibrationMode::None],
                ..Default::default()
            },
        };
        let report = run_matrix(&matrix, &fake_registry());
        let session_row = report
            .rows
            .iter()
            .find(|r| r.kind == RowKind::Session)
            .unwrap();
        let RowOutcome::Ok { metrics } = &session_row.outcome else {
            panic!("expected ok row: {session_row:?}");
        };
        let region_3x3 = metrics.regions.iter().find(|r| r.cols == 3).unwrap();
        assert_eq!(region_3x3.windows, 16);
        assert_eq!(region_3x3.excluded, 0);
        assert_eq!(region_3x3.hits, 4);
        assert_eq!(region_3x3.hit_rate, Some(0.25));

        let region_4x4 = metrics.regions.iter().find(|r| r.cols == 4).unwrap();
        assert_eq!(region_4x4.hits, 1);
    }

    #[test]
    fn test_fixed_ray_angular_error_matches_geometry() {
        let dir = tempfile::tempdir().unwrap();
        let spec = SyntheticSession {
            targets: FOUR_BY_FOUR_CENTRES.to_vec(),
            ..Default::default()
        };
        let session_dir = write_synthetic_session(dir.path(), "s1", &spec).unwrap();
        let toml_path = dir.path().join("fixed.toml");
        std::fs::write(
            &toml_path,
            fixed_ray_toml([155.0, 85.0, -500.0], [161.458333, 94.444444]),
        )
        .unwrap();
        let matrix = BenchMatrix {
            recordings: vec![session_dir],
            pipelines: vec![crate::matrix::PipelineSpec {
                name: "fixed".to_string(),
                config: Some(toml_path),
            }],
            evaluation: crate::matrix::Evaluation {
                calibration: vec![CalibrationMode::None],
                ..Default::default()
            },
        };
        let report = run_matrix(&matrix, &fake_registry());
        let session_row = report
            .rows
            .iter()
            .find(|r| r.kind == RowKind::Session)
            .unwrap();
        let RowOutcome::Ok { metrics } = &session_row.outcome else {
            panic!("expected ok row: {session_row:?}");
        };
        assert_eq!(metrics.samples, 384);
        let screen = crate::testing::synthetic_screen();
        let expected: f64 = FOUR_BY_FOUR_CENTRES
            .iter()
            .map(|&(x, y)| {
                let target_mm = eye_geometry::screen::px_logical_to_mm(&screen, &Point2::new(x, y));
                metrics::angle_deg(&eye(), &Point2::new(161.458333, 94.444444), &target_mm)
            })
            .sum::<f64>()
            / FOUR_BY_FOUR_CENTRES.len() as f64;
        let summary = metrics.angular_error_deg.as_ref().unwrap();
        assert_relative_eq!(summary.mean, expected, epsilon = 1e-6);
        assert_eq!(metrics.dropout_rate, Some(0.0));
    }

    #[test]
    fn test_run_is_deterministic_except_latency() {
        let dir = tempfile::tempdir().unwrap();
        let spec = SyntheticSession {
            targets: FOUR_BY_FOUR_CENTRES[..4].to_vec(),
            ..Default::default()
        };
        let session_dir = write_synthetic_session(dir.path(), "s1", &spec).unwrap();
        let toml_path = dir.path().join("fixed.toml");
        std::fs::write(
            &toml_path,
            fixed_ray_toml([155.0, 85.0, -500.0], [161.458333, 94.444444]),
        )
        .unwrap();
        let matrix = BenchMatrix::single(Some(toml_path), vec![session_dir]);

        let a = run_matrix(&matrix, &fake_registry()).to_json().unwrap();
        let b = run_matrix(&matrix, &fake_registry()).to_json().unwrap();

        let strip = |s: &str| -> serde_json::Value {
            let mut v: serde_json::Value = serde_json::from_str(s).unwrap();
            strip_processing(&mut v);
            v
        };
        assert_eq!(strip(&a), strip(&b));
    }

    fn strip_processing(v: &mut serde_json::Value) {
        match v {
            serde_json::Value::Object(map) => {
                for (k, value) in map.iter_mut() {
                    if k == "processing_ms" {
                        *value = serde_json::Value::Null;
                    } else {
                        strip_processing(value);
                    }
                }
            }
            serde_json::Value::Array(items) => {
                for item in items {
                    strip_processing(item);
                }
            }
            _ => {}
        }
    }

    #[test]
    fn test_missing_camera_yields_error_row_and_continues() {
        let dir = tempfile::tempdir().unwrap();
        let spec = SyntheticSession::default();
        let session_dir = write_synthetic_session(dir.path(), "s1", &spec).unwrap();

        let bad_toml = fixed_ray_toml([155.0, 85.0, -500.0], [161.458333, 94.444444])
            .replace("\"ir\"", "\"rgb\"")
            .replace("ir = \"test-null\"", "rgb = \"test-null\"");
        let bad_path = dir.path().join("bad.toml");
        std::fs::write(&bad_path, bad_toml).unwrap();

        let good_path = dir.path().join("good.toml");
        std::fs::write(
            &good_path,
            fixed_ray_toml([155.0, 85.0, -500.0], [161.458333, 94.444444]),
        )
        .unwrap();

        let matrix = BenchMatrix {
            recordings: vec![session_dir],
            pipelines: vec![
                crate::matrix::PipelineSpec {
                    name: "a".to_string(),
                    config: Some(bad_path),
                },
                crate::matrix::PipelineSpec {
                    name: "b".to_string(),
                    config: Some(good_path),
                },
            ],
            evaluation: crate::matrix::Evaluation::default(),
        };
        let report = run_matrix(&matrix, &fake_registry());

        let a_row = report
            .rows
            .iter()
            .find(|r| r.pipeline == "a" && r.kind == RowKind::Session)
            .unwrap();
        let RowOutcome::Error { message } = &a_row.outcome else {
            panic!("expected error row: {a_row:?}");
        };
        assert!(message.contains("not in recording") && message.contains("rgb"));

        let b_row = report
            .rows
            .iter()
            .find(|r| r.pipeline == "b" && r.kind == RowKind::Session)
            .unwrap();
        assert!(matches!(b_row.outcome, RowOutcome::Ok { .. }));
    }

    #[test]
    fn test_recording_without_rig_is_error_row() {
        let dir = tempfile::tempdir().unwrap();
        let spec = SyntheticSession {
            with_rig: false,
            ..Default::default()
        };
        let session_dir = write_synthetic_session(dir.path(), "s1", &spec).unwrap();
        let toml_path = dir.path().join("fixed.toml");
        std::fs::write(
            &toml_path,
            fixed_ray_toml([155.0, 85.0, -500.0], [161.458333, 94.444444]),
        )
        .unwrap();
        let matrix = BenchMatrix::single(Some(toml_path), vec![session_dir]);
        let report = run_matrix(&matrix, &fake_registry());
        let row = report
            .rows
            .iter()
            .find(|r| r.kind == RowKind::Session)
            .unwrap();
        let RowOutcome::Error { message } = &row.outcome else {
            panic!("expected error row: {row:?}");
        };
        assert!(message.contains("no [rig] snapshot"));
    }

    #[test]
    fn test_bad_config_yields_error_rows_for_every_session() {
        let dir = tempfile::tempdir().unwrap();
        let spec = SyntheticSession::default();
        let session_a = write_synthetic_session(dir.path(), "s1", &spec).unwrap();
        let session_b = write_synthetic_session(dir.path(), "s2", &spec).unwrap();
        let mut matrix = BenchMatrix::single(
            Some(dir.path().join("missing.toml")),
            vec![session_a, session_b],
        );
        matrix.evaluation.calibration = vec![CalibrationMode::None];
        let report = run_matrix(&matrix, &fake_registry());
        let session_errors = report
            .rows
            .iter()
            .filter(|r| r.kind == RowKind::Session && matches!(r.outcome, RowOutcome::Error { .. }))
            .count();
        assert_eq!(session_errors, 2);
        let aggregate_errors = report
            .rows
            .iter()
            .filter(|r| {
                r.kind == RowKind::Aggregate && matches!(r.outcome, RowOutcome::Error { .. })
            })
            .count();
        assert_eq!(aggregate_errors, 1);
    }

    #[test]
    fn test_aggregate_pools_samples_across_sessions() {
        let dir = tempfile::tempdir().unwrap();
        let spec = SyntheticSession {
            targets: FOUR_BY_FOUR_CENTRES.to_vec(),
            ..Default::default()
        };
        let session_a = write_synthetic_session(dir.path(), "s1", &spec).unwrap();
        let session_b = write_synthetic_session(dir.path(), "s2", &spec).unwrap();
        let toml_path = dir.path().join("fixed.toml");
        std::fs::write(
            &toml_path,
            fixed_ray_toml([155.0, 85.0, -500.0], [161.458333, 94.444444]),
        )
        .unwrap();
        let matrix = BenchMatrix::single(Some(toml_path), vec![session_a, session_b]);
        let report = run_matrix(&matrix, &fake_registry());

        let session_row = report
            .rows
            .iter()
            .find(|r| r.kind == RowKind::Session)
            .unwrap();
        let RowOutcome::Ok {
            metrics: session_metrics,
        } = &session_row.outcome
        else {
            panic!("expected ok row: {session_row:?}");
        };

        let aggregate_row = report
            .rows
            .iter()
            .find(|r| r.kind == RowKind::Aggregate)
            .unwrap();
        let RowOutcome::Ok {
            metrics: aggregate_metrics,
        } = &aggregate_row.outcome
        else {
            panic!("expected ok row: {aggregate_row:?}");
        };

        assert_eq!(aggregate_metrics.samples, 768);
        assert_relative_eq!(
            aggregate_metrics.angular_error_deg.as_ref().unwrap().mean,
            session_metrics.angular_error_deg.as_ref().unwrap().mean,
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_fixed_ray_estimator_rejects_origin_behind_screen() {
        let config = fixed_ray_config([155.0, 85.0, 500.0], [161.458333, 94.444444]);
        let registry = fake_registry();
        let rig = crate::testing::synthetic_rig();
        let err = match registry.estimator(&config.estimate, &rig) {
            Err(e) => e,
            Ok(_) => panic!("expected the registry to reject an origin behind the screen"),
        };
        assert!(err.to_string().contains("z < 0"));
    }

    #[test]
    fn test_kappa_ray_estimator_rejects_eye_behind_screen() {
        let bad_toml = kappa_ray_toml(&FOUR_BY_FOUR_CENTRES, [0.0, 0.0], None)
            .replace("eye = [155.0, 85.0, -500.0]", "eye = [155.0, 85.0, 500.0]");
        let bad_config = Config::from_toml_str(&bad_toml).expect("parses");
        let registry = fake_registry();
        let rig = crate::testing::synthetic_rig();
        let err = match registry.estimator(&bad_config.estimate, &rig) {
            Err(e) => e,
            Ok(_) => panic!("expected the registry to reject an eye behind the screen"),
        };
        assert!(err.to_string().contains("z < 0"));
    }

    #[test]
    fn test_default_evaluation_reports_none_and_loto() {
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
        let matrix = BenchMatrix::single(Some(toml_path), vec![session_dir]);

        let report = run_matrix(&matrix, &fake_registry());
        let session_rows: Vec<_> = report
            .rows
            .iter()
            .filter(|r| r.kind == RowKind::Session)
            .collect();
        assert_eq!(session_rows.len(), 2);
        assert_eq!(session_rows[0].calibration, CalibrationMode::None);
        assert_eq!(session_rows[1].calibration, CalibrationMode::Loto);
        for row in session_rows {
            assert!(matches!(row.outcome, RowOutcome::Ok { .. }), "{row:?}");
        }
    }
}
