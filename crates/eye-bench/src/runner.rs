//! Replays recordings through pipelines and scores the result.

use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};

use eye::config::{Config, EnvOverrides};
use eye::pipeline::{Pipeline, PipelineError, RayBatch, RayStep};
use eye::registry::Registry;
use eye_calibration::correction::rig_fingerprint;
use eye_calibration::error::CalibrationError;
use eye_calibration::protocol::{FixationWindow, ProtocolConfig, TargetProtocol};
use eye_calibration::store::{read_rig, rig_from_table};
use eye_capture::session::{RecordedCamera, Recording, SessionMeta};
use eye_core::log::{field, span};
use eye_core::{CameraInfo, Frame, FrameSet, GazePoint, OutputId, Rig, Timestamp};
use nalgebra::{Point3, Vector3};

use crate::error::BenchError;
use crate::matrix::{BenchMatrix, PipelineSpec, RigSource};
use crate::metrics::{EvalInput, EvalPoint, EvalWindow, Summary, compute};
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
    /// The protocol the windows were built from (the recording's, else the fallback).
    pub protocol: ProtocolConfig,
    /// `"session"`, `"stored"` or `"file:<path>"`.
    pub rig_source: String,
    pub rig_fingerprint: String,
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

fn step_outcome(s: &Step) -> &'static str {
    if s.skipped {
        "skipped"
    } else if s.batch.is_some() {
        "rays"
    } else {
        "no_gaze"
    }
}

#[allow(clippy::result_large_err)]
fn step(pipeline: &mut Pipeline, set: &FrameSet) -> Result<Step, BenchError> {
    let h = set
        .frames()
        .iter()
        .map(Frame::header)
        .max_by_key(|h| h.timestamp)
        .expect("a FrameSet is non-empty");
    let _frame = eye_core::log::frame_span(
        h.camera.as_str(),
        h.seq,
        h.timestamp.as_nanos(),
        h.illumination.as_str(),
        set.frames().len() as u64,
    )
    .entered();
    let started = Instant::now();
    let step = match pipeline.rays(set).map_err(pipeline_error)? {
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
    };
    tracing::trace!(
        outcome = step_outcome(&step),
        rays = step.batch.as_ref().map_or(0, |b| b.rays.len()) as u64,
        point = step.point.is_some(),
        { field::ELAPSED_US } = step.processing.as_micros() as u64,
        "frame replayed"
    );
    Ok(step)
}

/// The session's own rig, parsed from its `[rig]` snapshot.
#[allow(clippy::result_large_err)]
fn session_rig(meta: &SessionMeta) -> Result<Rig, BenchError> {
    let rig_table = meta.rig.as_ref().ok_or_else(|| BenchError::NoRig {
        session: meta.session_id.to_string(),
    })?;
    rig_from_table(rig_table).map_err(|e| BenchError::Calibration(Box::new(e)))
}

/// `candidate` must carry every camera `recorded` has (same id, width, height) and `recorded`'s
/// screen output.
#[allow(clippy::result_large_err)]
fn check_rig_compatible(recorded: &Rig, candidate: &Rig, session: &str) -> Result<(), BenchError> {
    if candidate.screen().output != recorded.screen().output {
        return Err(BenchError::RigMismatch {
            session: session.to_string(),
            reason: format!(
                "rig output {:?} does not match recording output {:?}",
                candidate.screen().output,
                recorded.screen().output
            ),
        });
    }
    for cam in recorded.cameras() {
        match candidate.camera(cam.id.as_str()) {
            Some(c) if c.width == cam.width && c.height == cam.height => {}
            Some(c) => {
                return Err(BenchError::RigMismatch {
                    session: session.to_string(),
                    reason: format!(
                        "camera {:?} is {}x{} in the override rig but {}x{} in the recording",
                        cam.id, c.width, c.height, cam.width, cam.height
                    ),
                });
            }
            None => {
                return Err(BenchError::RigMismatch {
                    session: session.to_string(),
                    reason: format!("override rig has no camera {:?}", cam.id),
                });
            }
        }
    }
    Ok(())
}

/// Resolves the rig a replay runs with, and the label its rows are recorded under.
#[allow(clippy::result_large_err)]
pub fn resolve_rig(
    meta: &SessionMeta,
    source: &RigSource,
    store: &dyn Fn(&OutputId) -> Result<Option<Rig>, CalibrationError>,
) -> Result<(Rig, String), BenchError> {
    match source {
        RigSource::Session => Ok((session_rig(meta)?, source.label())),
        RigSource::Stored => {
            let recorded = session_rig(meta)?;
            let output = recorded.screen().output.clone();
            let stored = store(&output)
                .map_err(|e| BenchError::RigSource(e.to_string()))?
                .ok_or_else(|| {
                    BenchError::RigSource(format!("no stored rig for output {output}"))
                })?;
            check_rig_compatible(&recorded, &stored, meta.session_id.as_str())?;
            Ok((stored, source.label()))
        }
        RigSource::File(path) => {
            let recorded = session_rig(meta)?;
            let file_rig = read_rig(path).map_err(|e| BenchError::RigSource(e.to_string()))?;
            check_rig_compatible(&recorded, &file_rig, meta.session_id.as_str())?;
            Ok((file_rig, source.label()))
        }
    }
}

/// Replays one recording through a fresh pipeline without correction.
///
/// `fallback` is used only when the recording carries no protocol.
#[allow(clippy::result_large_err)]
pub fn replay_session(
    dir: &Path,
    config: &Config,
    registry: &Registry,
    fallback: &ProtocolConfig,
) -> Result<Replayed, BenchError> {
    replay_session_with_rig(
        dir,
        config,
        registry,
        fallback,
        &RigSource::Session,
        &|_| Ok(None),
    )
}

/// Replays one recording through a fresh pipeline without correction, scoring it with the rig
/// `rig_source` resolves to (`store` is consulted only for `RigSource::Stored`).
///
/// `fallback` is used only when the recording carries no protocol.
#[allow(clippy::result_large_err)]
pub fn replay_session_with_rig(
    dir: &Path,
    config: &Config,
    registry: &Registry,
    fallback: &ProtocolConfig,
    rig_source: &RigSource,
    store: &dyn Fn(&OutputId) -> Result<Option<Rig>, CalibrationError>,
) -> Result<Replayed, BenchError> {
    let capture = |source| BenchError::Capture {
        session: dir.to_path_buf(),
        source,
    };
    let recording = Recording::open(dir).map_err(capture)?;
    let meta = recording.meta();
    let session = meta.session_id.to_string();
    let (rig, rig_source_label) = resolve_rig(meta, rig_source, store)?;
    let rig_fingerprint_value = rig_fingerprint(&rig);
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
    let (protocol, source) = match meta.protocol {
        Some(recorded) => (recorded, "recording"),
        None => match ProtocolConfig::from_target_records(recording.targets(), fallback) {
            Some(derived) => (derived, "derived"),
            None => (*fallback, "fallback"),
        },
    };
    tracing::info!(
        source,
        dwell_ms = protocol.dwell_ms,
        settle_ms = protocol.settle_ms,
        window_ms = protocol.window_ms,
        grid_cols = u64::from(protocol.grid[0]),
        grid_rows = u64::from(protocol.grid[1]),
        "protocol"
    );
    let windows = TargetProtocol::new(protocol)
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
            protocol,
            rig_source: rig_source_label,
            rig_fingerprint: rig_fingerprint_value,
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
        let ts_ns = point.timestamp.as_nanos();
        let Some(eye_mm) = mean_origin(&batch.rays) else {
            tracing::debug!(
                { field::TS_NS } = ts_ns,
                { field::REASON } = "no_rays",
                "sample dropped"
            );
            continue;
        };
        let Some(i) = window_at(windows, point.timestamp) else {
            tracing::debug!(
                { field::TS_NS } = ts_ns,
                { field::REASON } = "outside_window",
                "sample dropped"
            );
            continue;
        };
        let w = &out[i];
        tracing::trace!(
            { field::TS_NS } = ts_ns,
            target_index = w.target_index as u64,
            target_mm_x = w.target_mm.x,
            target_mm_y = w.target_mm.y,
            gaze_mm_x = point.mm.x,
            gaze_mm_y = point.mm.y,
            error_deg = crate::metrics::angle_deg(&eye_mm, &point.mm, &w.target_mm),
            "sample"
        );
        out[i].points.push(EvalPoint {
            timestamp: point.timestamp,
            eye_mm,
            gaze_mm: point.mm,
            gaze_px_logical: point.px_logical,
        });
    }
    EvalInput {
        windows: out,
        processing,
    }
}

type SessionSlot = (String, Result<Replayed, BenchError>);

/// `Err(message)` becomes that mode's `RowOutcome::Error`; the `None` arm always returns `Ok`.
/// `index` must name a slot whose replay is `Ok`.
fn evaluate(
    mode: CalibrationMode,
    index: usize,
    replayed: &mut [SessionSlot],
    matrix: &BenchMatrix,
) -> Result<(EvalInput, Vec<String>), String> {
    match mode {
        CalibrationMode::None => {
            let r = replayed[index]
                .1
                .as_mut()
                .expect("replay checked ok by caller");
            Ok((
                eval_input(
                    &r.run.rig,
                    &r.run.windows,
                    r.run.samples(),
                    r.run.processing(),
                ),
                Vec::new(),
            ))
        }
        CalibrationMode::Loto => {
            let fitter = crate::calibration::dot_session_fitter_with(matrix.evaluation.fit);
            let r = replayed[index]
                .1
                .as_mut()
                .expect("replay checked ok by caller");
            crate::calibration::loto(r, &fitter).map(|o| (o.input, o.warnings))
        }
        CalibrationMode::Profile => {
            let path = matrix
                .evaluation
                .profile
                .as_ref()
                .expect("evaluation.profile is required for profile mode; validated on load");
            let profile =
                eye_calibration::profiles::read_profile(path).map_err(|e| e.to_string())?;
            let r = replayed[index]
                .1
                .as_mut()
                .expect("replay checked ok by caller");
            let fingerprint = eye_calibration::correction::rig_fingerprint(&r.run.rig);
            let mut warnings = Vec::new();
            if profile.rig_fingerprint != fingerprint {
                warnings.push(format!(
                    "profile rig fingerprint {:?} does not match session rig fingerprint {:?}",
                    profile.rig_fingerprint, fingerprint
                ));
            }
            let windows = r.run.windows.clone();
            let input = crate::calibration::evaluate_with(r, &profile, &windows);
            Ok((input, warnings))
        }
        CalibrationMode::Cross => {
            let fitter = crate::calibration::dot_session_fitter_with(matrix.evaluation.fit);
            let (left, right) = replayed.split_at_mut(index);
            let (held, rest) = right.split_first_mut().expect("index in range");
            let others: Vec<&SessionRun> = left
                .iter()
                .chain(rest.iter())
                .filter_map(|(_, r)| r.as_ref().ok())
                .map(|r| &r.run)
                .collect();
            let held_replayed = held.1.as_mut().expect("replay checked ok by caller");
            crate::calibration::cross(held_replayed, &others, &fitter)
                .map(|input| (input, Vec::new()))
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

/// `rig_source` is the row's `rig_source` label; error rows never carry a `rig_fingerprint`.
fn error_row(
    pipeline: &str,
    mode: CalibrationMode,
    kind: RowKind,
    session: &str,
    step_errors: usize,
    rig_source: &str,
    message: String,
) -> BenchRow {
    BenchRow {
        pipeline: pipeline.to_string(),
        calibration: mode,
        kind,
        session: session.to_string(),
        rig_source: rig_source.to_string(),
        rig_fingerprint: String::new(),
        step_errors,
        warnings: Vec::new(),
        outcome: RowOutcome::Error { message },
    }
}

/// A broken config gives error rows for every session of that pipeline (and its aggregates).
fn config_error_rows(spec: &PipelineSpec, matrix: &BenchMatrix, message: &str) -> Vec<BenchRow> {
    let rig_source = matrix.evaluation.rig.label();
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
                &rig_source,
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
            &rig_source,
            "no session produced a result".to_string(),
        ));
    }
    rows
}

fn rows_for_pipeline(
    spec: &PipelineSpec,
    matrix: &BenchMatrix,
    registry: &Registry,
    store: &dyn Fn(&OutputId) -> Result<Option<Rig>, CalibrationError>,
) -> Vec<BenchRow> {
    let config = match load_pipeline_config(spec) {
        Ok(config) => config,
        Err(e) => {
            tracing::warn!(
                pipeline = %spec.name,
                { field::REASON } = %e,
                "pipeline config failed"
            );
            return config_error_rows(spec, matrix, &e.to_string());
        }
    };

    let mut replayed: Vec<SessionSlot> = Vec::new();
    for dir in &matrix.recordings {
        let session = session_name(dir);
        let _session = tracing::info_span!(
            span::SESSION,
            { field::SESSION_ID } = %session,
            pipeline = %spec.name,
        )
        .entered();
        let started = Instant::now();
        let result = replay_session_with_rig(
            dir,
            &config,
            registry,
            &matrix.evaluation.protocol,
            &matrix.evaluation.rig,
            store,
        );
        match &result {
            Ok(r) => {
                tracing::info!(
                    steps = r.run.steps.len() as u64,
                    step_errors = r.run.step_errors() as u64,
                    { field::ELAPSED_US } = started.elapsed().as_micros() as u64,
                    "session replayed"
                );
            }
            Err(e) => {
                tracing::warn!({ field::REASON } = %e, "session replay failed");
            }
        }
        replayed.push((session, result));
    }

    let mut rows = Vec::new();
    let mut per_mode: HashMap<CalibrationMode, (Vec<EvalInput>, usize)> = HashMap::new();

    for index in 0..replayed.len() {
        let session = replayed[index].0.clone();
        let _session = tracing::info_span!(
            span::SESSION,
            { field::SESSION_ID } = %session,
            pipeline = %spec.name,
        )
        .entered();
        let (step_errors, replay_error, rig_source, rig_fingerprint_value) =
            match &replayed[index].1 {
                Ok(r) => (
                    r.run.step_errors(),
                    None,
                    r.run.rig_source.clone(),
                    r.run.rig_fingerprint.clone(),
                ),
                Err(e) => (
                    0,
                    Some(e.to_string()),
                    matrix.evaluation.rig.label(),
                    String::new(),
                ),
            };

        for &mode in &matrix.evaluation.calibration {
            if let Some(message) = &replay_error {
                rows.push(error_row(
                    &spec.name,
                    mode,
                    RowKind::Session,
                    &session,
                    0,
                    &rig_source,
                    message.clone(),
                ));
                continue;
            }
            match evaluate(mode, index, &mut replayed, matrix) {
                Ok((input, warnings)) => {
                    let metrics = compute(&input, &matrix.evaluation.metrics);
                    log_session_scored(mode, &metrics, warnings.len() as u64);
                    rows.push(BenchRow {
                        pipeline: spec.name.clone(),
                        calibration: mode,
                        kind: RowKind::Session,
                        session: session.clone(),
                        rig_source: rig_source.clone(),
                        rig_fingerprint: rig_fingerprint_value.clone(),
                        step_errors,
                        warnings,
                        outcome: RowOutcome::Ok { metrics },
                    });
                    let entry = per_mode.entry(mode).or_default();
                    entry.0.push(input);
                    entry.1 += step_errors;
                }
                Err(message) => {
                    tracing::warn!(
                        calibration = mode.as_str(),
                        { field::REASON } = %message,
                        "session evaluation failed"
                    );
                    rows.push(error_row(
                        &spec.name,
                        mode,
                        RowKind::Session,
                        &session,
                        step_errors,
                        &rig_source,
                        message,
                    ));
                }
            }
        }
    }

    let rig_source = matrix.evaluation.rig.label();
    for &mode in &matrix.evaluation.calibration {
        match per_mode.get(&mode) {
            Some((inputs, step_errors)) if !inputs.is_empty() => {
                let pooled = EvalInput::concat(inputs);
                let metrics = compute(&pooled, &matrix.evaluation.metrics);
                log_aggregate_scored(
                    &spec.name,
                    mode,
                    inputs.len() as u64,
                    *step_errors as u64,
                    &metrics,
                );
                rows.push(BenchRow {
                    pipeline: spec.name.clone(),
                    calibration: mode,
                    kind: RowKind::Aggregate,
                    session: "all".to_string(),
                    rig_source: rig_source.clone(),
                    rig_fingerprint: String::new(),
                    step_errors: *step_errors,
                    warnings: Vec::new(),
                    outcome: RowOutcome::Ok { metrics },
                });
            }
            _ => {
                tracing::warn!(
                    pipeline = %spec.name,
                    calibration = mode.as_str(),
                    { field::REASON } = "no session produced a result",
                    "aggregate has no result"
                );
                rows.push(error_row(
                    &spec.name,
                    mode,
                    RowKind::Aggregate,
                    "all",
                    0,
                    &rig_source,
                    "no session produced a result".to_string(),
                ));
            }
        }
    }

    rows
}

fn summarize(x: &Option<Summary>, f: fn(&Summary) -> f64) -> Option<f64> {
    x.as_ref().map(f)
}

fn log_session_scored(
    mode: CalibrationMode,
    metrics: &crate::metrics::SessionMetrics,
    warnings: u64,
) {
    tracing::info!(
        calibration = mode.as_str(),
        samples = metrics.samples as u64,
        windows = metrics.windows as u64,
        err_mean_deg = summarize(&metrics.angular_error_deg, |s| s.mean),
        err_p95_deg = summarize(&metrics.angular_error_deg, |s| s.p95),
        accuracy_deg = metrics.accuracy_deg,
        precision_deg = metrics.precision_rms_s2s_deg,
        proc_p50_ms = summarize(&metrics.processing_ms, |s| s.p50),
        proc_p95_ms = summarize(&metrics.processing_ms, |s| s.p95),
        dropout_rate = metrics.dropout_rate,
        output_rate_hz = metrics.output_rate_hz,
        warnings,
        "session scored"
    );
    for region in &metrics.regions {
        tracing::debug!(
            calibration = mode.as_str(),
            cols = region.cols as u64,
            rows = region.rows as u64,
            windows = region.windows as u64,
            excluded = region.excluded as u64,
            hits = region.hits as u64,
            hit_rate = region.hit_rate,
            sample_hit_rate = region.sample_hit_rate,
            "region hit rate"
        );
    }
}

fn log_aggregate_scored(
    pipeline: &str,
    mode: CalibrationMode,
    sessions: u64,
    step_errors: u64,
    metrics: &crate::metrics::SessionMetrics,
) {
    tracing::info!(
        pipeline = %pipeline,
        calibration = mode.as_str(),
        sessions,
        step_errors,
        samples = metrics.samples as u64,
        windows = metrics.windows as u64,
        err_mean_deg = summarize(&metrics.angular_error_deg, |s| s.mean),
        err_p95_deg = summarize(&metrics.angular_error_deg, |s| s.p95),
        accuracy_deg = metrics.accuracy_deg,
        precision_deg = metrics.precision_rms_s2s_deg,
        proc_p50_ms = summarize(&metrics.processing_ms, |s| s.p50),
        proc_p95_ms = summarize(&metrics.processing_ms, |s| s.p95),
        dropout_rate = metrics.dropout_rate,
        output_rate_hz = metrics.output_rate_hz,
        "aggregate scored"
    );
}

/// Delegates to `run_matrix_with_store` with a store that never has a rig (`RigSource::Stored`
/// then always fails with `BenchError::RigSource`).
pub fn run_matrix(matrix: &BenchMatrix, registry: &Registry) -> BenchReport {
    run_matrix_with_store(matrix, registry, &|_| Ok(None))
}

pub fn run_matrix_with_store(
    matrix: &BenchMatrix,
    registry: &Registry,
    store: &dyn Fn(&OutputId) -> Result<Option<Rig>, CalibrationError>,
) -> BenchReport {
    tracing::info!(
        pipelines = matrix.pipelines.len() as u64,
        recordings = matrix.recordings.len() as u64,
        modes = matrix.evaluation.calibration.len() as u64,
        "bench run started"
    );
    let mut rows = Vec::new();
    for spec in &matrix.pipelines {
        rows.extend(rows_for_pipeline(spec, matrix, registry, store));
    }
    BenchReport::new(matrix.evaluation.metrics.clone(), rows)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use approx::assert_relative_eq;
    use eye_core::log::field;
    use eye_core::{GazeRay, OutputId};
    use eye_log::testing::capture_logs;
    use nalgebra::{Matrix2, Matrix3, Point2, Unit};

    use super::*;
    use crate::metrics;
    use crate::row::RowOutcome;
    use crate::testing::{
        FOUR_BY_FOUR_CENTRES, SyntheticSession, fake_registry, fixed_ray_config, fixed_ray_toml,
        kappa_ray_toml, synthetic_rig, write_synthetic_session,
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
                    timestamp: Timestamp::from_nanos(500_000_000),
                    origin: Point3::new(120.0, 85.0, -500.0),
                    direction: Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0)),
                    origin_cov: Matrix3::zeros(),
                    angular_cov: Matrix2::identity() * 1e-6,
                    head_rotation: None,
                },
                GazeRay {
                    side: None,
                    timestamp: Timestamp::from_nanos(500_000_000),
                    origin: Point3::new(190.0, 85.0, -500.0),
                    direction: Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0)),
                    origin_cov: Matrix3::zeros(),
                    angular_cov: Matrix2::identity() * 1e-6,
                    head_rotation: None,
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
    fn test_replay_uses_recorded_protocol_over_fallback() {
        let dir = tempfile::tempdir().unwrap();
        let recorded = ProtocolConfig {
            grid: [3, 3],
            lead_in_ms: 1000,
            dwell_ms: 2500,
            settle_ms: 1000,
            window_ms: 1200,
        };
        let spec = SyntheticSession {
            targets: vec![(960.0, 540.0)],
            protocol: Some(recorded),
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
        let window = &replayed.run.windows[0];
        assert_eq!(
            window.start,
            Timestamp(window.onset.0 + Duration::from_millis(1000))
        );
        assert_eq!(replayed.run.protocol.settle_ms, 1000);
    }

    #[test]
    fn test_replay_without_recorded_protocol_uses_fallback() {
        let dir = tempfile::tempdir().unwrap();
        let spec = SyntheticSession {
            targets: vec![(960.0, 540.0)],
            protocol: None,
            ..Default::default()
        };
        let session_dir = write_synthetic_session(dir.path(), "s1", &spec).unwrap();
        let config = fixed_ray_config([155.0, 85.0, -500.0], [161.458333, 94.444444]);
        let fallback = ProtocolConfig::default();
        let replayed = replay_session(&session_dir, &config, &fake_registry(), &fallback).unwrap();
        assert_eq!(replayed.run.protocol, fallback);
    }

    #[test]
    fn test_logs_protocol_source_at_info() {
        let dir = tempfile::tempdir().unwrap();
        let recorded = ProtocolConfig {
            grid: [3, 3],
            lead_in_ms: 1000,
            dwell_ms: 2500,
            settle_ms: 1000,
            window_ms: 1200,
        };
        let spec_with = SyntheticSession {
            targets: vec![(960.0, 540.0)],
            protocol: Some(recorded),
            ..Default::default()
        };
        let session_with = write_synthetic_session(dir.path(), "with", &spec_with).unwrap();
        let spec_without = SyntheticSession {
            targets: vec![(960.0, 540.0)],
            protocol: None,
            ..Default::default()
        };
        let session_without =
            write_synthetic_session(dir.path(), "without", &spec_without).unwrap();
        let config = fixed_ray_config([155.0, 85.0, -500.0], [161.458333, 94.444444]);
        let fallback = ProtocolConfig::default();

        let (_, records) = eye_log::testing::capture_logs(tracing::Level::INFO, || {
            replay_session(&session_with, &config, &fake_registry(), &fallback).unwrap();
        });
        let rec = records
            .iter()
            .find(|r| r.message == "protocol")
            .expect("no 'protocol' record for the recorded case");
        assert_eq!(rec.level, eye_log::Level::Info);
        assert_eq!(rec.target, "eye_bench::runner");
        assert_eq!(
            rec.fields.get("source"),
            Some(&eye_log::Value::Str("recording".to_string()))
        );

        let (_, records) = eye_log::testing::capture_logs(tracing::Level::INFO, || {
            replay_session(&session_without, &config, &fake_registry(), &fallback).unwrap();
        });
        let rec = records
            .iter()
            .find(|r| r.message == "protocol")
            .expect("no 'protocol' record for the fallback case");
        assert_eq!(rec.level, eye_log::Level::Info);
        assert_eq!(rec.target, "eye_bench::runner");
        assert_eq!(
            rec.fields.get("source"),
            Some(&eye_log::Value::Str("fallback".to_string()))
        );
    }

    #[test]
    fn test_replay_session_derives_protocol_from_targets() {
        let dir = tempfile::tempdir().unwrap();
        let spec = SyntheticSession {
            targets: FOUR_BY_FOUR_CENTRES.to_vec(),
            protocol: None,
            ..Default::default()
        };
        let session_dir = write_synthetic_session(dir.path(), "s1", &spec).unwrap();
        let config = fixed_ray_config([155.0, 85.0, -500.0], [161.458333, 94.444444]);
        let wrong_fallback = ProtocolConfig {
            grid: [4, 4],
            lead_in_ms: 1000,
            dwell_ms: 1500,
            settle_ms: 900,
            window_ms: 500,
        };
        let replayed =
            replay_session(&session_dir, &config, &fake_registry(), &wrong_fallback).unwrap();
        assert_eq!(replayed.run.protocol.settle_ms, 600);
        assert_eq!(replayed.run.protocol.window_ms, 800);
        let window = &replayed.run.windows[0];
        assert_eq!(
            window.start,
            Timestamp(window.onset.0 + Duration::from_millis(600))
        );
        assert_eq!(
            window.end,
            Timestamp(window.start.0 + Duration::from_millis(800))
        );
    }

    #[test]
    fn test_logs_protocol_source_derived_at_info() {
        let dir = tempfile::tempdir().unwrap();
        let spec = SyntheticSession {
            targets: FOUR_BY_FOUR_CENTRES.to_vec(),
            protocol: None,
            ..Default::default()
        };
        let session_dir = write_synthetic_session(dir.path(), "s1", &spec).unwrap();
        let config = fixed_ray_config([155.0, 85.0, -500.0], [161.458333, 94.444444]);
        let fallback = ProtocolConfig::default();

        let (_, records) = eye_log::testing::capture_logs(tracing::Level::INFO, || {
            replay_session(&session_dir, &config, &fake_registry(), &fallback).unwrap();
        });
        let rec = records
            .iter()
            .find(|r| r.message == "protocol")
            .expect("no 'protocol' record for the derived case");
        assert_eq!(rec.level, eye_log::Level::Info);
        assert_eq!(
            rec.fields.get("source"),
            Some(&eye_log::Value::Str("derived".to_string()))
        );
        assert_eq!(rec.fields.get("dwell_ms"), Some(&eye_log::Value::U64(1500)));
        assert_eq!(rec.fields.get("settle_ms"), Some(&eye_log::Value::U64(600)));
        assert_eq!(rec.fields.get("window_ms"), Some(&eye_log::Value::U64(800)));
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

    #[test]
    fn test_cross_with_one_recording_is_error_row() {
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
        let matrix = BenchMatrix {
            recordings: vec![session_dir],
            pipelines: vec![PipelineSpec {
                name: "kappa".to_string(),
                config: Some(toml_path),
            }],
            evaluation: crate::matrix::Evaluation {
                calibration: vec![CalibrationMode::Cross],
                ..Default::default()
            },
        };
        let report = run_matrix(&matrix, &fake_registry());
        let row = report
            .rows
            .iter()
            .find(|r| r.kind == RowKind::Session && r.calibration == CalibrationMode::Cross)
            .unwrap();
        let RowOutcome::Error { message } = &row.outcome else {
            panic!("expected error row: {row:?}");
        };
        assert!(message.contains("at least two"));
    }

    fn fitted_profile_for(session_dir: &Path) -> eye_calibration::correction::UserProfile {
        let config =
            Config::from_toml_str(&kappa_ray_toml(&FOUR_BY_FOUR_CENTRES, [3.0, -1.0], None))
                .unwrap();
        let replayed = replay_session(
            session_dir,
            &config,
            &fake_registry(),
            &ProtocolConfig::default(),
        )
        .unwrap();
        let train = crate::calibration::fit_samples(
            &replayed.run.windows,
            replayed.run.steps.iter().filter_map(|s| s.batch.as_ref()),
            |_| true,
        );
        crate::calibration::dot_session_fitter(&train, &replayed.run.rig).unwrap()
    }

    #[test]
    fn test_profile_mode_applies_given_profile() {
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
        let profile = fitted_profile_for(&session_dir);
        let profile_path = dir.path().join("profile.toml");
        eye_calibration::profiles::write_profile(&profile_path, &profile).unwrap();

        let matrix = BenchMatrix {
            recordings: vec![session_dir],
            pipelines: vec![PipelineSpec {
                name: "kappa".to_string(),
                config: Some(toml_path),
            }],
            evaluation: crate::matrix::Evaluation {
                calibration: vec![CalibrationMode::Profile],
                profile: Some(profile_path),
                ..Default::default()
            },
        };
        let report = run_matrix(&matrix, &fake_registry());
        let row = report
            .rows
            .iter()
            .find(|r| r.kind == RowKind::Session && r.calibration == CalibrationMode::Profile)
            .unwrap();
        let RowOutcome::Ok { metrics } = &row.outcome else {
            panic!("expected ok row: {row:?}");
        };
        assert!(metrics.angular_error_deg.as_ref().unwrap().mean < 0.1);
        assert!(row.warnings.is_empty());
    }

    #[test]
    fn test_profile_mode_warns_on_rig_fingerprint_mismatch() {
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
        let mut profile = fitted_profile_for(&session_dir);
        profile.rig_fingerprint = "stale".to_string();
        let profile_path = dir.path().join("profile.toml");
        eye_calibration::profiles::write_profile(&profile_path, &profile).unwrap();

        let matrix = BenchMatrix {
            recordings: vec![session_dir],
            pipelines: vec![PipelineSpec {
                name: "kappa".to_string(),
                config: Some(toml_path),
            }],
            evaluation: crate::matrix::Evaluation {
                calibration: vec![CalibrationMode::Profile],
                profile: Some(profile_path),
                ..Default::default()
            },
        };
        let report = run_matrix(&matrix, &fake_registry());
        let row = report
            .rows
            .iter()
            .find(|r| r.kind == RowKind::Session && r.calibration == CalibrationMode::Profile)
            .unwrap();
        assert!(row.warnings.iter().any(|w| w.contains("rig")));
    }

    fn single_mode_matrix(dir: &Path, session_dir: PathBuf) -> BenchMatrix {
        let toml_path = dir.join("fixed.toml");
        std::fs::write(
            &toml_path,
            fixed_ray_toml([155.0, 85.0, -500.0], [161.458333, 94.444444]),
        )
        .unwrap();
        BenchMatrix {
            recordings: vec![session_dir],
            pipelines: vec![crate::matrix::PipelineSpec {
                name: "fixed".to_string(),
                config: Some(toml_path),
            }],
            evaluation: crate::matrix::Evaluation {
                calibration: vec![CalibrationMode::None],
                ..Default::default()
            },
        }
    }

    #[test]
    fn test_logs_session_scored_at_info() {
        let dir = tempfile::tempdir().unwrap();
        let spec = SyntheticSession {
            targets: FOUR_BY_FOUR_CENTRES.to_vec(),
            ..Default::default()
        };
        let session_dir = write_synthetic_session(dir.path(), "s1", &spec).unwrap();
        let matrix = single_mode_matrix(dir.path(), session_dir);

        let (_, records) = capture_logs(tracing::Level::TRACE, || {
            run_matrix(&matrix, &fake_registry())
        });
        let matches: Vec<_> = records
            .iter()
            .filter(|r| r.message == "session scored")
            .collect();
        assert_eq!(matches.len(), 1);
        let rec = matches[0];
        assert_eq!(rec.level, eye_log::Level::Info);
        assert_eq!(rec.target, "eye_bench::runner");
        assert_eq!(rec.fields["samples"], eye_log::Value::U64(384));
        assert_eq!(
            rec.fields["calibration"],
            eye_log::Value::Str("none".to_string())
        );
        assert!(matches!(rec.fields["err_mean_deg"], eye_log::Value::F64(_)));
        assert_eq!(
            rec.context[field::SESSION_ID],
            eye_log::Value::Str("s1".to_string())
        );
        assert_eq!(
            rec.context["pipeline"],
            eye_log::Value::Str("fixed".to_string())
        );
        let mut keys: Vec<&str> = rec.context.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(keys, vec!["pipeline", field::SESSION_ID]);
    }

    #[test]
    #[allow(clippy::result_large_err)]
    fn test_logs_frame_replayed_at_trace() {
        let dir = tempfile::tempdir().unwrap();
        let spec = SyntheticSession {
            cameras: vec!["ir", "rgb"],
            ..Default::default()
        };
        let session_dir = write_synthetic_session(dir.path(), "s1", &spec).unwrap();
        let config = fixed_ray_config([155.0, 85.0, -500.0], [161.458333, 94.444444]);
        let (replayed, records) = capture_logs(tracing::Level::TRACE, || {
            replay_session(
                &session_dir,
                &config,
                &fake_registry(),
                &ProtocolConfig::default(),
            )
        });
        let replayed = replayed.unwrap();
        let matches: Vec<_> = records
            .iter()
            .filter(|r| r.message == "frame replayed")
            .collect();
        assert_eq!(matches.len(), replayed.run.steps.len());
        assert_eq!(matches.len(), 61);
        for rec in &matches {
            assert_eq!(
                rec.context[field::CAMERA],
                eye_log::Value::Str("ir".to_string())
            );
            assert!(matches!(rec.context[field::SEQ], eye_log::Value::U64(_)));
            assert!(matches!(rec.context[field::TS_NS], eye_log::Value::U64(_)));
            assert_eq!(
                rec.context[field::ILLUMINATION],
                eye_log::Value::Str("unknown".to_string())
            );
            assert_eq!(rec.context[field::SET_CAMERAS], eye_log::Value::U64(1));
            assert_eq!(
                rec.fields["outcome"],
                eye_log::Value::Str("rays".to_string())
            );
            assert!(!rec.fields.contains_key("set.cameras"));
            assert!(matches!(
                rec.fields[field::ELAPSED_US],
                eye_log::Value::U64(_)
            ));
            assert!(!rec.context.contains_key(field::SESSION_ID));
        }
    }

    #[test]
    fn test_logs_sample_at_trace() {
        let dir = tempfile::tempdir().unwrap();
        let spec = SyntheticSession {
            targets: FOUR_BY_FOUR_CENTRES.to_vec(),
            ..Default::default()
        };
        let session_dir = write_synthetic_session(dir.path(), "s1", &spec).unwrap();
        let matrix = single_mode_matrix(dir.path(), session_dir);

        let (_, records) = capture_logs(tracing::Level::TRACE, || {
            run_matrix(&matrix, &fake_registry())
        });
        let matches: Vec<_> = records.iter().filter(|r| r.message == "sample").collect();
        assert_eq!(matches.len(), 384);
        for rec in &matches {
            let error_deg = match rec.fields["error_deg"] {
                eye_log::Value::F64(v) => v,
                ref other => panic!("expected F64, got {other:?}"),
            };
            assert!(error_deg >= 0.0);
            let target_index = match rec.fields["target_index"] {
                eye_log::Value::U64(v) => v,
                ref other => panic!("expected U64, got {other:?}"),
            };
            assert!(target_index < 16);
            assert!(rec.fields.contains_key(field::TS_NS));
            assert!(!rec.fields.contains_key("data"));
            assert!(!rec.fields.contains_key("frame"));
        }
    }

    #[test]
    fn test_logs_sample_dropped_at_debug() {
        let dir = tempfile::tempdir().unwrap();
        let spec = SyntheticSession {
            targets: FOUR_BY_FOUR_CENTRES.to_vec(),
            ..Default::default()
        };
        let session_dir = write_synthetic_session(dir.path(), "s1", &spec).unwrap();
        let matrix = single_mode_matrix(dir.path(), session_dir);

        let (_, records) = capture_logs(tracing::Level::TRACE, || {
            run_matrix(&matrix, &fake_registry())
        });
        let dropped: Vec<_> = records
            .iter()
            .filter(|r| r.message == "sample dropped")
            .collect();
        assert!(
            dropped
                .iter()
                .any(|r| r.fields[field::REASON] == eye_log::Value::Str("outside_window".into()))
        );
        assert!(
            !dropped
                .iter()
                .any(|r| r.fields[field::REASON] == eye_log::Value::Str("no_rays".into()))
        );
        for rec in &dropped {
            assert_eq!(rec.level, eye_log::Level::Debug);
        }
    }

    #[test]
    fn test_logs_session_replay_failed_at_warn() {
        let dir = tempfile::tempdir().unwrap();
        let spec = SyntheticSession {
            with_rig: false,
            ..Default::default()
        };
        let session_dir = write_synthetic_session(dir.path(), "s1", &spec).unwrap();
        let matrix = single_mode_matrix(dir.path(), session_dir);

        let (report, records) = capture_logs(tracing::Level::TRACE, || {
            run_matrix(&matrix, &fake_registry())
        });
        let matches: Vec<_> = records
            .iter()
            .filter(|r| r.message == "session replay failed")
            .collect();
        assert_eq!(matches.len(), 1);
        let rec = matches[0];
        assert_eq!(rec.level, eye_log::Level::Warn);
        let reason = match &rec.fields[field::REASON] {
            eye_log::Value::Str(s) => s,
            other => panic!("expected Str, got {other:?}"),
        };
        assert!(reason.contains("no [rig] snapshot"));
        let row = report
            .rows
            .iter()
            .find(|r| r.kind == RowKind::Session)
            .unwrap();
        assert!(matches!(row.outcome, RowOutcome::Error { .. }));
    }

    #[test]
    fn test_logs_pipeline_config_failed_at_warn() {
        let dir = tempfile::tempdir().unwrap();
        let spec = SyntheticSession::default();
        let session_a = write_synthetic_session(dir.path(), "s1", &spec).unwrap();
        let mut matrix =
            BenchMatrix::single(Some(dir.path().join("missing.toml")), vec![session_a]);
        matrix.evaluation.calibration = vec![CalibrationMode::None];

        let (_, records) = capture_logs(tracing::Level::TRACE, || {
            run_matrix(&matrix, &fake_registry())
        });
        let matches: Vec<_> = records
            .iter()
            .filter(|r| r.message == "pipeline config failed")
            .collect();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].level, eye_log::Level::Warn);
        assert!(matches[0].fields.contains_key("pipeline"));
        assert!(!records.iter().any(|r| r.message == "session scored"));
    }

    #[test]
    #[allow(clippy::result_large_err)]
    fn test_logs_report_written_at_info() {
        let dir = tempfile::tempdir().unwrap();
        let out_dir = dir.path().join("out");
        let report = BenchReport::new(
            crate::metrics::MetricParams::default(),
            vec![BenchRow {
                pipeline: "p".to_string(),
                calibration: CalibrationMode::None,
                kind: RowKind::Session,
                session: "s".to_string(),
                rig_source: "session".to_string(),
                rig_fingerprint: String::new(),
                step_errors: 0,
                warnings: Vec::new(),
                outcome: RowOutcome::Error {
                    message: "boom".to_string(),
                },
            }],
        );

        let (result, records) = capture_logs(tracing::Level::TRACE, || report.write_to(&out_dir));
        result.unwrap();
        let matches: Vec<_> = records
            .iter()
            .filter(|r| r.message == "report written")
            .collect();
        assert_eq!(matches.len(), 1);
        let rec = matches[0];
        assert_eq!(rec.level, eye_log::Level::Info);
        assert_eq!(rec.fields["rows"], eye_log::Value::U64(1));
        let json = match &rec.fields["json"] {
            eye_log::Value::Str(s) => s,
            other => panic!("expected Str, got {other:?}"),
        };
        assert!(json.ends_with("report.json"));
    }

    #[test]
    fn test_logs_aggregate_scored_at_info() {
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
        let matrix = BenchMatrix {
            recordings: vec![session_a, session_b],
            pipelines: vec![crate::matrix::PipelineSpec {
                name: "fixed".to_string(),
                config: Some(toml_path),
            }],
            evaluation: crate::matrix::Evaluation {
                calibration: vec![CalibrationMode::None],
                ..Default::default()
            },
        };

        let (_, records) = capture_logs(tracing::Level::TRACE, || {
            run_matrix(&matrix, &fake_registry())
        });
        let matches: Vec<_> = records
            .iter()
            .filter(|r| r.message == "aggregate scored")
            .collect();
        assert_eq!(matches.len(), 1);
        let rec = matches[0];
        assert_eq!(rec.level, eye_log::Level::Info);
        assert_eq!(rec.fields["sessions"], eye_log::Value::U64(2));
        assert_eq!(rec.fields["samples"], eye_log::Value::U64(768));
        assert!(!rec.context.contains_key(field::SESSION_ID));
    }

    fn file_rig_with_doubled_screen_x() -> Rig {
        let rig = synthetic_rig();
        let mut screen = rig.screen().clone();
        screen.size_mm.x *= 2.0;
        Rig::new(rig.cameras().to_vec(), screen).unwrap()
    }

    /// Pinning: a `file:` rig with the recording's camera and output, but a doubled screen width,
    /// is used and its provenance recorded. The synthetic estimator ignores the rig entirely
    /// (ignores its `_rig` parameter; see `crate::testing::KappaRayEstimator::estimate`), so the
    /// `none`-mode mean is unaffected by the override; both runs score the single right-column
    /// target at mean 0.0 deg.
    #[test]
    fn test_rig_override_file_is_used_and_recorded() {
        let dir = tempfile::tempdir().unwrap();
        let targets = vec![(1680.0, 405.0)];
        let spec = SyntheticSession {
            targets: targets.clone(),
            code_frames: true,
            ..Default::default()
        };
        let session_dir = write_synthetic_session(dir.path(), "s1", &spec).unwrap();
        let toml_path = dir.path().join("kappa.toml");
        std::fs::write(&toml_path, kappa_ray_toml(&targets, [0.0, 0.0], None)).unwrap();

        let base_matrix = BenchMatrix {
            recordings: vec![session_dir],
            pipelines: vec![PipelineSpec {
                name: "kappa".to_string(),
                config: Some(toml_path),
            }],
            evaluation: crate::matrix::Evaluation {
                calibration: vec![CalibrationMode::None],
                ..Default::default()
            },
        };
        let base_report = run_matrix(&base_matrix, &fake_registry());
        let base_row = base_report
            .rows
            .iter()
            .find(|r| r.kind == RowKind::Session)
            .unwrap();
        assert_eq!(base_row.rig_source, "session");
        let RowOutcome::Ok {
            metrics: base_metrics,
        } = &base_row.outcome
        else {
            panic!("expected ok row: {base_row:?}");
        };
        let base_mean = base_metrics.angular_error_deg.as_ref().unwrap().mean;

        let file_rig = file_rig_with_doubled_screen_x();
        let rig_path = dir.path().join("file_rig.toml");
        eye_calibration::store::write_rig(&rig_path, &file_rig).unwrap();

        let mut override_matrix = base_matrix;
        override_matrix.evaluation.rig = RigSource::File(rig_path.clone());
        let override_report = run_matrix(&override_matrix, &fake_registry());
        let override_row = override_report
            .rows
            .iter()
            .find(|r| r.kind == RowKind::Session)
            .unwrap();
        assert_eq!(
            override_row.rig_source,
            format!("file:{}", rig_path.display())
        );
        assert_eq!(override_row.rig_fingerprint, rig_fingerprint(&file_rig));
        let RowOutcome::Ok {
            metrics: override_metrics,
        } = &override_row.outcome
        else {
            panic!("expected ok row: {override_row:?}");
        };
        let override_mean = override_metrics.angular_error_deg.as_ref().unwrap().mean;

        assert_relative_eq!(base_mean, 0.0, epsilon = 1e-6);
        assert_relative_eq!(override_mean, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_rig_override_rejects_camera_size_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let spec = SyntheticSession::default();
        let session_dir = write_synthetic_session(dir.path(), "s1", &spec).unwrap();

        let rig = synthetic_rig();
        let mut cameras = rig.cameras().to_vec();
        cameras[0].width = 320;
        cameras[0].height = 180;
        let bad_rig = Rig::new(cameras, rig.screen().clone()).unwrap();
        let rig_path = dir.path().join("bad_rig.toml");
        eye_calibration::store::write_rig(&rig_path, &bad_rig).unwrap();

        let recording = Recording::open(&session_dir).unwrap();
        let err =
            resolve_rig(recording.meta(), &RigSource::File(rig_path), &|_| Ok(None)).unwrap_err();
        let BenchError::RigMismatch { reason, .. } = err else {
            panic!("expected RigMismatch: {err:?}");
        };
        assert!(reason.contains("ir"), "{reason}");
    }

    #[test]
    fn test_rig_override_stored_uses_recording_output() {
        let dir = tempfile::tempdir().unwrap();
        let spec = SyntheticSession::default();
        let session_dir = write_synthetic_session(dir.path(), "s1", &spec).unwrap();
        let recording = Recording::open(&session_dir).unwrap();

        let stored_rig = synthetic_rig();
        let lookup = stored_rig.clone();
        let store = move |output: &OutputId| -> Result<Option<Rig>, CalibrationError> {
            if output.as_str() == "eDP-1" {
                Ok(Some(lookup.clone()))
            } else {
                Ok(None)
            }
        };
        let (rig, label) = resolve_rig(recording.meta(), &RigSource::Stored, &store).unwrap();
        assert_eq!(label, "stored");
        assert_eq!(rig, stored_rig);
    }

    #[test]
    fn test_rig_override_stored_errors_without_a_stored_rig() {
        let dir = tempfile::tempdir().unwrap();
        let spec = SyntheticSession::default();
        let session_dir = write_synthetic_session(dir.path(), "s1", &spec).unwrap();
        let recording = Recording::open(&session_dir).unwrap();

        let err = resolve_rig(recording.meta(), &RigSource::Stored, &|_| Ok(None)).unwrap_err();
        assert!(matches!(err, BenchError::RigSource(_)), "{err:?}");
    }

    #[test]
    fn test_rig_stored_with_no_store_entry_is_error_row_not_panic() {
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
        let matrix = BenchMatrix {
            recordings: vec![session_dir],
            pipelines: vec![PipelineSpec {
                name: "kappa".to_string(),
                config: Some(toml_path),
            }],
            evaluation: crate::matrix::Evaluation {
                calibration: vec![CalibrationMode::None],
                rig: RigSource::Stored,
                ..Default::default()
            },
        };
        let report = run_matrix_with_store(&matrix, &fake_registry(), &|_| Ok(None));
        let row = report
            .rows
            .iter()
            .find(|r| r.kind == RowKind::Session)
            .unwrap();
        assert_eq!(row.rig_source, "stored");
        assert!(matches!(row.outcome, RowOutcome::Error { .. }), "{row:?}");
    }
}
