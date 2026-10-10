use std::collections::{BTreeSet, HashMap};
use std::fs::Permissions;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use eye::config::Config;
use eye::pipeline::{Pipeline, RayBatch, RayStep};
use eye::registry::Registry;
use eye_bench::calibration::{
    dot_session_fitter, fit_samples, latest_presentation, loto_with, position_key,
};
use eye_bench::metrics::{MetricParams, SessionMetrics, compute};
use eye_bench::runner::replay_session;
use eye_calibration::correction::{Provenance, UserProfile, legacy_source};
#[cfg(test)]
use eye_calibration::protocol::TargetTiming;
use eye_calibration::protocol::{FixationWindow, ProtocolConfig, TargetProtocol};
use eye_calibration::store::ProfileStore;
use eye_calibration::user_fit::{
    DotSessionFit, EyeFitReport, FitConfig, FitSample, ProfileMeta, TargetReject, TargetVerdict,
};
use eye_core::log::{field, span};
use eye_core::{CameraInfo, Frame, Rig, Timestamp};
use eye_geometry::screen::px_logical_to_mm;
use eye_overlay::ellipse::{cov_mm_to_logical_px, logical_px_per_mm};
use eye_overlay::targets::{AppendSender, Feedback, FeedbackSender, TargetShown, TargetSpec};
use nalgebra::Point2;

use crate::cli::GIT_REV;
use crate::commands::record::{
    PumpObserver, RecordOptions, RecordSummary, SessionLocation, record_session,
};
use crate::commands::run::estimator_fingerprint;
use crate::ctx::Ctx;
use crate::shutdown;

#[derive(Debug, clap::Args)]
pub struct Args {
    /// Dot targets: 9 (3x3 cell centres) or 16 (4x4)
    #[arg(long, default_value_t = 9, value_parser = parse_calibration_targets, conflicts_with = "from")]
    pub targets: u32,
    #[arg(long, value_name = "MS", conflicts_with = "from")]
    pub dwell_ms: Option<u64>,
    /// Profile name in the profile store
    #[arg(long, default_value = "default")]
    pub profile: String,
    /// Keep the recording in ./recordings/<session-id> (it contains face images)
    #[arg(long, conflicts_with = "from")]
    pub keep: bool,
    /// Fit from an existing recording instead of recording a new session
    #[arg(long, value_name = "DIR")]
    pub from: Option<PathBuf>,
    /// Skip the live gaze feedback dot during recording (for weak machines)
    #[arg(long)]
    pub no_feedback: bool,
}

/// Whether `run` should drive a live `LiveFeedback` observer for this invocation: feedback makes
/// no sense when fitting from an existing recording, since nothing is shown.
pub fn wants_feedback(args: &Args) -> bool {
    !args.no_feedback && args.from.is_none()
}

/// Target frame period at 30 fps; above this, `rays()` is falling behind and sets start getting skipped.
const FRAME_PERIOD_30HZ: Duration = Duration::from_nanos(1_000_000_000 / 30);

/// At most this many retries per target position before an online rejection is left standing.
const MAX_RETRIES: u8 = 2;

/// Summary of the live preview fit: distinct target positions shown, samples selected (by the
/// same selection the offline fit will make on the same events), and the worst-eye residual, so
/// the live number and the saved profile's LOTO can be told apart.
#[derive(Debug, Clone, PartialEq)]
pub struct PreviewFit {
    pub targets: usize,
    pub samples: usize,
    pub rms_after_deg: f64,
}

/// Drives a synchronous `Pipeline` alongside `pump`'s frame loop, refitting the profile after
/// every completed target and feeding the result back to the target overlay as a `Feedback` point.
#[derive(Debug)]
pub struct LiveFeedback {
    registry: Registry,
    config: Config,
    pipeline: Option<Pipeline>,
    feedback: Option<FeedbackSender>,
    append: Option<AppendSender>,
    protocol: TargetProtocol,
    base_len: u32,
    windows: Vec<FixationWindow>,
    batches: Vec<RayBatch>,
    last_preview: Option<PreviewFit>,
    fit_cfg: FitConfig,
    meta: ProfileMeta,
    retries: HashMap<(u64, u64), u8>,
    completed: usize,
    calibrated: bool,
    frame_period: Duration,
    last_rays_duration: Duration,
    skip_next_set: bool,
    #[cfg(test)]
    forced_rays_duration: Option<Duration>,
}

impl LiveFeedback {
    pub fn new(
        registry: Registry,
        config: Config,
        protocol: TargetProtocol,
        fit_cfg: FitConfig,
        meta: ProfileMeta,
    ) -> Self {
        let (cols, rows) = protocol.grid();
        Self {
            registry,
            config,
            pipeline: None,
            feedback: None,
            append: None,
            protocol,
            base_len: cols * rows,
            windows: Vec::new(),
            batches: Vec::new(),
            last_preview: None,
            fit_cfg,
            meta,
            retries: HashMap::new(),
            completed: 0,
            calibrated: false,
            frame_period: FRAME_PERIOD_30HZ,
            last_rays_duration: Duration::ZERO,
            skip_next_set: false,
            #[cfg(test)]
            forced_rays_duration: None,
        }
    }

    #[cfg(test)]
    pub fn with_pipeline(
        pipeline: Pipeline,
        protocol: TargetProtocol,
        fit_cfg: FitConfig,
        meta: ProfileMeta,
    ) -> Self {
        let (cols, rows) = protocol.grid();
        Self {
            registry: Registry::with_defaults(),
            config: Config::builtin_default(),
            pipeline: Some(pipeline),
            feedback: None,
            append: None,
            protocol,
            base_len: cols * rows,
            windows: Vec::new(),
            batches: Vec::new(),
            last_preview: None,
            fit_cfg,
            meta,
            retries: HashMap::new(),
            completed: 0,
            calibrated: false,
            frame_period: FRAME_PERIOD_30HZ,
            last_rays_duration: Duration::ZERO,
            skip_next_set: false,
            forced_rays_duration: None,
        }
    }

    /// The selection `fit_recording` will make on the same cached events: every ray whose own
    /// timestamp falls in the latest presentation's window for its target.
    pub fn fit_samples(&self) -> Vec<FitSample> {
        fit_samples(
            &self.windows,
            self.batches.iter(),
            latest_presentation(&self.windows, self.base_len),
            legacy_source(&self.meta.estimator),
        )
    }

    pub fn last_preview(&self) -> Option<&PreviewFit> {
        self.last_preview.as_ref()
    }

    pub fn is_calibrated(&self) -> bool {
        self.calibrated
    }

    #[cfg(test)]
    pub fn force_rays_duration(&mut self, duration: Duration) {
        self.forced_rays_duration = Some(duration);
    }

    /// Alternates skipping once `last_rays_duration` exceeds `frame_period`; resets to never-skip
    /// as soon as processing is caught up again.
    fn should_skip_set(&mut self) -> bool {
        if self.last_rays_duration <= self.frame_period {
            self.skip_next_set = false;
            return false;
        }
        let skip = self.skip_next_set;
        self.skip_next_set = !skip;
        skip
    }

    fn record_rays_duration(&mut self, started: Instant) {
        #[cfg(test)]
        if let Some(forced) = self.forced_rays_duration {
            self.last_rays_duration = forced;
            return;
        }
        self.last_rays_duration = started.elapsed();
    }

    fn process(&mut self, frame: &Frame) -> anyhow::Result<()> {
        let pipeline = self
            .pipeline
            .as_mut()
            .expect("prepare() builds the pipeline before any frame is pumped");
        let Some(set) = pipeline.pair(frame.clone()) else {
            return Ok(());
        };
        if self.should_skip_set() {
            return Ok(());
        }
        let pipeline = self
            .pipeline
            .as_mut()
            .expect("prepare() builds the pipeline before any frame is pumped");
        let started = Instant::now();
        let ray_step = pipeline.rays(&set)?;
        self.record_rays_duration(started);
        let RayStep::Rays(batch) = ray_step else {
            return Ok(());
        };
        self.batches.push(batch.clone());
        let pipeline = self
            .pipeline
            .as_mut()
            .expect("prepare() builds the pipeline before any frame is pumped");
        let screen = pipeline.rig().screen().clone();
        let Some(point) = pipeline.finish(&batch) else {
            return Ok(());
        };
        if let Some(sender) = &self.feedback {
            let px_per_mm = logical_px_per_mm(&screen);
            sender.send(Feedback {
                px_logical: point.px_logical,
                cov_px: cov_mm_to_logical_px(&point.cov_mm, &px_per_mm),
                calibrated: self.calibrated,
                at: point.timestamp,
            });
        }
        Ok(())
    }

    fn refit(&mut self, target_index: usize, last_target: Option<(Point2<f64>, Point2<f64>)>) {
        if self.completed < self.fit_cfg.min_targets_offset {
            return;
        }
        let samples = self.fit_samples();
        let pipeline = self
            .pipeline
            .as_mut()
            .expect("prepare() builds the pipeline before any target completes");
        match DotSessionFit::fit_with(&samples, pipeline.rig(), &self.fit_cfg, self.meta.clone()) {
            Ok(outcome) => {
                for report in &outcome.reports {
                    tracing::info!(
                        target_index,
                        eye = ?report.key,
                        residual_deg = report.rms_after_deg,
                        "live calibration refit"
                    );
                }
                let rms_after_deg = outcome
                    .reports
                    .iter()
                    .map(|r| r.rms_after_deg)
                    .fold(f64::MIN, f64::max);
                let targets = self
                    .windows
                    .iter()
                    .map(position_key)
                    .collect::<BTreeSet<_>>()
                    .len();
                self.last_preview = Some(PreviewFit {
                    targets,
                    samples: samples.len(),
                    rms_after_deg,
                });
                let verdicts = outcome.verdicts();
                pipeline.set_correction(Some(Box::new(outcome.profile)));
                self.calibrated = true;
                if let Some((target_mm, px_logical)) = last_target {
                    self.handle_verdict(target_mm, px_logical, &samples, &verdicts);
                }
            }
            Err(error) => {
                tracing::warn!(
                    target_index,
                    %error,
                    "live calibration refit failed; keeping the previous correction"
                );
            }
        }
    }

    /// Decides whether the just-completed target (at `target_mm`/`px_logical`) should be
    /// re-presented, looking up only its own verdict among `verdicts`.
    fn handle_verdict(
        &mut self,
        target_mm: Point2<f64>,
        px_logical: Point2<f64>,
        samples: &[FitSample],
        verdicts: &[TargetVerdict],
    ) {
        let key = target_mm_key(target_mm);
        let Some(index) = target_index_in_samples(samples, target_mm) else {
            return;
        };
        let verdict = verdicts.iter().find(|v| v.index == index);
        let (rejected, reason, residual_deg) = match verdict {
            Some(v) => (v.rejected, v.reason, v.residual_deg),
            None => (
                true,
                Some(TargetReject::TooFewSamples {
                    have: 0,
                    need: self.fit_cfg.min_samples_per_target,
                }),
                0.0,
            ),
        };
        if !rejected {
            return;
        }
        let retries = *self.retries.get(&key).unwrap_or(&0);
        if retries >= MAX_RETRIES {
            return;
        }
        self.retries.insert(key, retries + 1);
        match reason {
            Some(TargetReject::TooFewSamples { have, need }) => {
                tracing::warn!(reason = ?reason, residual_deg, have, need, "target rejected");
            }
            Some(TargetReject::Jitter { limit, .. })
            | Some(TargetReject::Residual { limit, .. }) => {
                tracing::warn!(reason = ?reason, residual_deg, limit, "target rejected");
            }
            None => {
                tracing::warn!(reason = ?reason, residual_deg, "target rejected");
            }
        }
        if let Some(sender) = &self.append {
            let spec = TargetSpec {
                px_logical,
                timing: self.protocol.timing(),
                retry: true,
            };
            let _ = sender.append(spec);
        }
    }
}

fn target_mm_key(target_mm: Point2<f64>) -> (u64, u64) {
    (target_mm.x.to_bits(), target_mm.y.to_bits())
}

/// The index `DotSessionFit::fit_with` would assign `target_mm` when fitting `samples`: targets
/// are numbered by the order their first sample appears, so this must be recomputed from the
/// current `samples` rather than cached, since dropping a retried target's stale samples can
/// change which target's samples appear first.
fn target_index_in_samples(samples: &[FitSample], target_mm: Point2<f64>) -> Option<u32> {
    let key = target_mm_key(target_mm);
    let mut next_index = 0u32;
    let mut seen: HashMap<(u64, u64), u32> = HashMap::new();
    for s in samples {
        let k = target_mm_key(s.target_mm);
        seen.entry(k).or_insert_with(|| {
            let idx = next_index;
            next_index += 1;
            idx
        });
    }
    seen.get(&key).copied()
}

impl PumpObserver for LiveFeedback {
    fn wants_feedback(&self) -> bool {
        true
    }

    fn attach_feedback(&mut self, sender: FeedbackSender) {
        self.feedback = Some(sender);
    }

    fn wants_append(&self) -> bool {
        true
    }

    fn attach_append(&mut self, sender: AppendSender) {
        self.append = Some(sender);
    }

    fn prepare(&mut self, rig: &Rig, cameras: &[CameraInfo]) -> anyhow::Result<()> {
        if self.pipeline.is_none() {
            self.pipeline = Some(Pipeline::from_config(
                &self.registry,
                &self.config,
                rig.clone(),
                cameras,
                None,
            )?);
        }
        Ok(())
    }

    fn on_frame(&mut self, frame: &Frame) -> anyhow::Result<()> {
        self.process(frame)
    }

    fn on_shown(&mut self, shown: &TargetShown) {
        let screen = self
            .pipeline
            .as_ref()
            .expect("prepare() builds the pipeline before any target is shown")
            .rig()
            .screen();
        let target_mm = px_logical_to_mm(screen, &shown.px_logical);
        let cell = self
            .protocol
            .cell_of_mm(&target_mm, screen)
            .unwrap_or((0, 0));
        let timing = self.protocol.timing();
        let onset = shown.shown_at;
        let start = Timestamp(onset.0 + timing.settle);
        let end = Timestamp(start.0 + timing.window);
        self.windows.push(FixationWindow {
            index: shown.index as u32,
            cell,
            onset,
            start,
            end,
            target_mm,
            target_px_logical: shown.px_logical,
        });
    }

    fn on_hidden(&mut self, index: usize, _at: Timestamp) {
        let last_target = self
            .windows
            .last()
            .map(|w| (w.target_mm, w.target_px_logical));
        self.completed += 1;
        self.refit(index, last_target);
    }
}

pub fn parse_calibration_targets(s: &str) -> Result<u32, String> {
    match s {
        "9" => Ok(9),
        "16" => Ok(16),
        _ => Err("allowed: 9, 16".to_string()),
    }
}

/// Where the dot session is recorded. `Temp` owns the private parent directory and deletes it on drop.
#[derive(Debug)]
pub enum RecordingDir {
    Kept(SessionLocation),
    Temp {
        _guard: tempfile::TempDir,
        location: SessionLocation,
    },
    Existing(PathBuf),
}

impl RecordingDir {
    pub fn location(&self) -> Option<&SessionLocation> {
        match self {
            RecordingDir::Kept(location) | RecordingDir::Temp { location, .. } => Some(location),
            RecordingDir::Existing(_) => None,
        }
    }

    pub fn session_dir(&self) -> PathBuf {
        match self {
            RecordingDir::Kept(location) | RecordingDir::Temp { location, .. } => location.dir(),
            RecordingDir::Existing(dir) => dir.clone(),
        }
    }
}

pub fn recording_dir(
    from: Option<PathBuf>,
    keep: bool,
    runtime_dir: Option<PathBuf>,
) -> anyhow::Result<RecordingDir> {
    if let Some(dir) = from {
        return Ok(RecordingDir::Existing(dir));
    }
    if keep {
        return Ok(RecordingDir::Kept(SessionLocation {
            root: PathBuf::from("recordings"),
            id: eye_capture::session::SessionId::now(),
        }));
    }
    let parent = runtime_dir.unwrap_or_else(|| {
        tracing::warn!("XDG_RUNTIME_DIR is not set; using the system temp directory instead");
        std::env::temp_dir()
    });
    let guard = tempfile::Builder::new()
        .prefix("eye-calibrate-")
        .permissions(Permissions::from_mode(0o700))
        .tempdir_in(&parent)?;
    let location = SessionLocation {
        root: guard.path().to_path_buf(),
        id: eye_capture::session::SessionId::now(),
    };
    Ok(RecordingDir::Temp {
        _guard: guard,
        location,
    })
}

#[derive(Debug)]
pub struct FitResult {
    pub profile: UserProfile,
    pub samples: usize,
    pub targets: usize,
    /// Leave-one-target-out estimate of the accuracy this profile gives on this pipeline; `None` if every fold failed.
    pub expected: Option<SessionMetrics>,
    /// Targets the primary source's fit excluded, merged across eyes and deduplicated.
    pub rejected_targets: Vec<u32>,
    /// The live run's own preview fit over the same kind of selection, for the printed summary
    /// to show both numbers; `None` for `--from` or `--no-feedback`, where nothing ran live.
    pub preview: Option<PreviewFit>,
}

/// Targets rejected by the primary source's reports (the lowest `source` among `reports`),
/// merged across its eyes and deduplicated: a non-primary source's rejections (for example an
/// IR-only glint dropout) must not appear as targets the offline fit excluded.
fn primary_rejected_targets(reports: &[EyeFitReport]) -> Vec<u32> {
    let Some(primary) = reports.iter().map(|r| r.source).min() else {
        return Vec::new();
    };
    reports
        .iter()
        .filter(|r| r.source == primary)
        .flat_map(|r| r.targets_rejected.iter().copied())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

pub fn fit_recording(
    dir: &Path,
    config: &Config,
    registry: &Registry,
    protocol: &ProtocolConfig,
    meta: ProfileMeta,
) -> anyhow::Result<FitResult> {
    let mut replayed = replay_session(dir, config, registry, protocol)?;
    let source = legacy_source(replayed.pipeline.estimator_name());
    let run = &replayed.run;
    let base_len = run.protocol.grid[0] * run.protocol.grid[1];
    let windows = run.windows.clone();
    let include = latest_presentation(&windows, base_len);
    let samples = fit_samples(
        &run.windows,
        run.steps.iter().filter_map(|s| s.batch.as_ref()),
        &include,
        source,
    );
    anyhow::ensure!(
        !samples.is_empty(),
        "no gaze samples in fixation windows (is the face visible and the pipeline producing rays?)"
    );
    let targets = run
        .windows
        .iter()
        .map(position_key)
        .collect::<BTreeSet<_>>()
        .len();
    let outcome = DotSessionFit::fit_with(&samples, &run.rig, &FitConfig::default(), meta)?;
    let mut profile = outcome.profile;
    let rejected_targets = primary_rejected_targets(&outcome.reports);
    profile.provenance.protocol = Some(run.protocol);
    let expected = match loto_with(&mut replayed, &dot_session_fitter, &include) {
        Ok(estimate) => {
            for warning in &estimate.warnings {
                tracing::warn!(%warning, "leave-one-target-out estimate");
            }
            Some(compute(&estimate.input, &MetricParams::default()))
        }
        Err(reason) => {
            tracing::warn!(%reason, "no leave-one-target-out estimate");
            None
        }
    };
    profile.provenance.expected_loto_mean_deg = expected
        .as_ref()
        .and_then(|m| m.angular_error_deg.as_ref())
        .map(|s| s.mean);
    Ok(FitResult {
        profile,
        samples: samples.len(),
        targets,
        expected,
        rejected_targets,
        preview: None,
    })
}

const INTERACTIVE_DWELL_MS: u64 = 2500;
const INTERACTIVE_SETTLE_MS: u64 = 1000;
const INTERACTIVE_WINDOW_MS: u64 = 1200;

fn protocol_for(targets: u32, dwell_ms: Option<u64>) -> ProtocolConfig {
    let dwell_ms = dwell_ms.unwrap_or(INTERACTIVE_DWELL_MS);
    let scale = dwell_ms as f64 / INTERACTIVE_DWELL_MS as f64;
    let settle_ms = ((INTERACTIVE_SETTLE_MS as f64 * scale).round() as u64).max(1);
    let window_ms = ((INTERACTIVE_WINDOW_MS as f64 * scale).round() as u64).max(1);
    ProtocolConfig {
        grid: if targets == 16 { [4, 4] } else { [3, 3] },
        dwell_ms,
        settle_ms,
        window_ms,
        ..ProtocolConfig::default()
    }
}

fn fit_protocol(args: &Args) -> ProtocolConfig {
    if args.from.is_some() {
        ProtocolConfig::default()
    } else {
        protocol_for(args.targets, args.dwell_ms)
    }
}

pub fn summary_line(result: &FitResult) -> String {
    let accuracy = match &result.expected {
        Some(metrics) => {
            let hit = metrics
                .regions
                .iter()
                .find(|r| r.cols == 3 && r.rows == 3)
                .and_then(|r| r.hit_rate)
                .map(|rate| format!("{:.1} %", rate * 100.0))
                .unwrap_or_else(|| "n/a".to_string());
            match &metrics.angular_error_deg {
                Some(summary) => format!(
                    "mean {:.2} deg, p95 {:.2} deg, 3x3 hit {hit}",
                    summary.mean, summary.p95
                ),
                None => "n/a".to_string(),
            }
        }
        None => "n/a".to_string(),
    };
    let rejected = if result.rejected_targets.is_empty() {
        String::new()
    } else {
        format!(
            "; {} targets rejected: {:?}",
            result.rejected_targets.len(),
            result.rejected_targets
        )
    };
    let preview = match &result.preview {
        Some(p) => format!("; live preview residual: {:.2} deg", p.rms_after_deg),
        None => String::new(),
    };
    format!(
        "{} targets, {} samples; expected accuracy (bench leave-one-target-out): {accuracy}{preview}{rejected}",
        result.targets, result.samples
    )
}

pub fn run(ctx: &Ctx, args: Args) -> anyhow::Result<()> {
    let config = Config::load(ctx.config_path.as_deref())?;
    let recording = recording_dir(
        args.from.clone(),
        args.keep,
        std::env::var_os("XDG_RUNTIME_DIR").map(PathBuf::from),
    )?;
    let session_id = recording
        .location()
        .map(|l| l.id.as_str().to_owned())
        .unwrap_or_else(|| {
            recording
                .session_dir()
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default()
        });
    let meta = ProfileMeta {
        name: args.profile.clone(),
        created_unix_s: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        estimator: config.estimate.kind.clone(),
        provenance: Provenance {
            session_id: Some(session_id.clone()),
            git_rev: Some(GIT_REV.to_string()),
            estimator_fingerprint: Some(estimator_fingerprint(&config.estimate)),
            protocol: None,
            fit: Some(FitConfig::default()),
            expected_loto_mean_deg: None,
        },
    };
    let mut preview = None;
    let mut shutdown_guard = None;
    if let Some(location) = recording.location() {
        let shutdown = shutdown_guard.insert(shutdown::install()?);
        let live_protocol = protocol_for(args.targets, args.dwell_ms);
        let opts = RecordOptions {
            protocol: Some(live_protocol),
            duration: None,
        };
        let summary: RecordSummary = if wants_feedback(&args) {
            let protocol = TargetProtocol::new(live_protocol)?;
            let mut live = LiveFeedback::new(
                Registry::with_defaults(),
                config.clone(),
                protocol,
                FitConfig::default(),
                meta.clone(),
            );
            let summary = record_session(&config, location, &opts, shutdown.receiver(), &mut live)?;
            preview = live.last_preview().cloned();
            summary
        } else {
            record_session(
                &config,
                location,
                &opts,
                shutdown.receiver(),
                &mut crate::commands::record::NoopObserver,
            )?
        };
        anyhow::ensure!(
            !summary.interrupted,
            "calibration interrupted; nothing saved"
        );
    }
    let _session =
        tracing::info_span!(span::SESSION, { field::SESSION_ID } = session_id.as_str()).entered();

    tracing::info!("fitting...");
    let protocol = fit_protocol(&args);
    let mut result = fit_recording(
        &recording.session_dir(),
        &config,
        &Registry::with_defaults(),
        &protocol,
        meta,
    )?;
    result.preview = preview;
    let saved = match &ctx.output {
        Some(path) => {
            eye_calibration::profiles::write_profile(path, &result.profile)?;
            path.clone()
        }
        None => ProfileStore::open_default()?.save_profile(&args.profile, &result.profile)?,
    };
    println!("{}  saved to {}", summary_line(&result), saved.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use clap::{CommandFactory, Parser, error::ErrorKind};
    use eye_bench::testing::{
        FOUR_BY_FOUR_CENTRES, SyntheticSession, fake_registry, kappa_ray_config, synthetic_rig,
        write_synthetic_session,
    };
    use eye_core::session::TargetClock;
    use eye_core::{CameraId, FrameHeader, Illumination, OutputId, PixelFormat};
    use eye_overlay::targets::TargetEvent;
    use nalgebra::{Matrix2, Matrix3, Point3, Vector3};

    use super::*;
    use crate::capture::CaptureMsg;
    use crate::cli::Cli;
    use crate::commands::record::{RecordSink, pump, target_record};

    #[test]
    fn test_args_defaults() {
        let cli = Cli::try_parse_from(["eye", "calibrate"]).expect("parses");
        let crate::cli::Command::Calibrate(args) = cli.command else {
            panic!("expected calibrate subcommand");
        };
        assert_eq!(args.targets, 9);
        assert_eq!(args.profile, "default");
        assert!(!args.keep);
        assert_eq!(args.from, None);
    }

    #[test]
    fn test_calibrate_default_timing_is_2500_1000_1200() {
        let protocol = protocol_for(9, None);
        assert_eq!(protocol.dwell_ms, 2500);
        assert_eq!(protocol.settle_ms, 1000);
        assert_eq!(protocol.window_ms, 1200);
    }

    #[test]
    fn test_calibrate_dwell_override_scales_settle_and_window() {
        let protocol = protocol_for(9, Some(5000));
        assert_eq!(protocol.dwell_ms, 5000);
        assert_eq!(protocol.settle_ms, 2000);
        assert_eq!(protocol.window_ms, 2400);
    }

    #[test]
    fn test_from_conflicts_with_keep_and_targets() {
        Cli::command().debug_assert();

        let err = Cli::try_parse_from(["eye", "calibrate", "--from", "d", "--keep"]).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::ArgumentConflict);

        let err = Cli::try_parse_from(["eye", "calibrate", "--from", "d", "--targets", "16"])
            .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::ArgumentConflict);
    }

    #[test]
    fn test_calibrate_fits_with_the_recorded_protocol() {
        let args = Args {
            targets: 16,
            dwell_ms: Some(5000),
            profile: "default".to_string(),
            keep: false,
            from: None,
            no_feedback: false,
        };
        assert_eq!(fit_protocol(&args), protocol_for(16, Some(5000)));

        let from_args = Args {
            targets: 9,
            dwell_ms: None,
            profile: "default".to_string(),
            keep: false,
            from: Some(PathBuf::from("d")),
            no_feedback: false,
        };
        assert_eq!(fit_protocol(&from_args), ProtocolConfig::default());
    }

    #[test]
    fn test_calibration_targets_accepts_9_and_16_only() {
        assert_eq!(parse_calibration_targets("9"), Ok(9));
        assert_eq!(parse_calibration_targets("16"), Ok(16));
        assert_eq!(
            parse_calibration_targets("4"),
            Err("allowed: 9, 16".to_string())
        );
    }

    #[test]
    fn test_recording_dir_temp_is_private_and_removed() {
        let tmp = tempfile::tempdir().unwrap();
        let recording = recording_dir(None, false, Some(tmp.path().to_path_buf())).unwrap();
        let session_dir = recording.session_dir();
        assert!(session_dir.starts_with(tmp.path()));
        let RecordingDir::Temp { location, .. } = &recording else {
            panic!("expected Temp");
        };
        let mode = std::fs::metadata(&location.root)
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o700);
        let parent = location.root.clone();
        drop(recording);
        assert!(!parent.exists());
    }

    #[test]
    fn test_recording_dir_keep_uses_recordings_root() {
        let recording = recording_dir(None, true, None).unwrap();
        let RecordingDir::Kept(location) = &recording else {
            panic!("expected Kept");
        };
        assert_eq!(location.root, PathBuf::from("recordings"));
        assert!(!location.root.exists() || std::fs::read_dir("recordings").is_ok());
    }

    #[test]
    fn test_recording_dir_from_is_existing() {
        let recording = recording_dir(Some(PathBuf::from("d")), false, None).unwrap();
        let RecordingDir::Existing(dir) = &recording else {
            panic!("expected Existing");
        };
        assert_eq!(dir, &PathBuf::from("d"));
        assert!(recording.location().is_none());
    }

    #[test]
    fn test_fit_recording_recovers_kappa_offset() {
        let dir = tempfile::tempdir().unwrap();
        let spec = SyntheticSession {
            targets: FOUR_BY_FOUR_CENTRES.to_vec(),
            code_frames: true,
            ..Default::default()
        };
        let session_dir = write_synthetic_session(dir.path(), "s1", &spec).unwrap();
        let config = kappa_ray_config(&FOUR_BY_FOUR_CENTRES, [3.0, -1.0], None);
        let meta = ProfileMeta::default();
        let result = fit_recording(
            &session_dir,
            &config,
            &fake_registry(),
            &ProtocolConfig::default(),
            meta,
        )
        .unwrap();

        assert_eq!(result.targets, 16);
        assert_eq!(result.samples, 384);
        let mean = result.expected.unwrap().angular_error_deg.unwrap().mean;
        assert!(mean < 0.1, "mean angular error {mean} not < 0.1");
    }

    #[test]
    fn test_fit_recording_uses_retry_window_not_the_rejected_one() {
        let base: Vec<(f64, f64)> = FOUR_BY_FOUR_CENTRES[..9].to_vec();
        let bad_index = 4;
        let mut targets = base.clone();
        targets.push(base[bad_index]);

        let dir = tempfile::tempdir().unwrap();
        let spec = SyntheticSession {
            targets: targets.clone(),
            code_frames: true,
            ..Default::default()
        };
        let session_dir = write_synthetic_session(dir.path(), "s1", &spec).unwrap();
        let config = kappa_ray_config(&targets, [0.0, 0.0], Some((bad_index, [10.0, 0.0])));
        let protocol = ProtocolConfig::default();
        let result = fit_recording(
            &session_dir,
            &config,
            &fake_registry(),
            &protocol,
            ProfileMeta::default(),
        )
        .unwrap();

        assert_eq!(
            result.targets, 9,
            "the retry shares a position, not a new one"
        );
        let frames_per_target = 24;
        assert_eq!(
            result.samples,
            (targets.len() - 1) * frames_per_target,
            "the rejected presentation's window must be dropped, not merged with the retry's"
        );
        assert!(
            result.rejected_targets.is_empty(),
            "the retry's unbiased samples replaced the rejected ones: {:?}",
            result.rejected_targets
        );
        assert!(
            result.expected.unwrap().angular_error_deg.unwrap().p95 < 0.1,
            "the loto estimate must not be scored on the rejected presentation"
        );
    }

    #[test]
    fn test_fit_recording_sets_profile_meta() {
        let dir = tempfile::tempdir().unwrap();
        let spec = SyntheticSession {
            targets: FOUR_BY_FOUR_CENTRES.to_vec(),
            code_frames: true,
            ..Default::default()
        };
        let session_dir = write_synthetic_session(dir.path(), "s1", &spec).unwrap();
        let config = kappa_ray_config(&FOUR_BY_FOUR_CENTRES, [3.0, -1.0], None);
        let meta = ProfileMeta {
            name: "alice".to_string(),
            created_unix_s: 1_791_409_623,
            estimator: "test-kappa-ray".to_string(),
            provenance: Provenance::default(),
        };
        let result = fit_recording(
            &session_dir,
            &config,
            &fake_registry(),
            &ProtocolConfig::default(),
            meta,
        )
        .unwrap();

        assert_eq!(result.profile.name, "alice");
        assert_eq!(result.profile.created_unix_s, 1_791_409_623);
        assert_eq!(result.profile.estimator, "test-kappa-ray");
        assert!(!result.profile.rig_fingerprint.is_empty());
    }

    #[test]
    fn test_fit_recording_fills_provenance() {
        let dir = tempfile::tempdir().unwrap();
        let protocol = ProtocolConfig::default();
        let spec = SyntheticSession {
            targets: FOUR_BY_FOUR_CENTRES.to_vec(),
            code_frames: true,
            protocol: Some(protocol),
            ..Default::default()
        };
        let session_dir = write_synthetic_session(dir.path(), "s1", &spec).unwrap();
        let config = kappa_ray_config(&FOUR_BY_FOUR_CENTRES, [3.0, -1.0], None);
        let result = fit_recording(
            &session_dir,
            &config,
            &fake_registry(),
            &protocol,
            ProfileMeta::default(),
        )
        .unwrap();

        assert_eq!(result.profile.provenance.protocol, Some(protocol));
        assert!(result.profile.provenance.expected_loto_mean_deg.is_some());
    }

    #[test]
    fn test_fit_recording_without_samples_errors() {
        let dir = tempfile::tempdir().unwrap();
        let spec = SyntheticSession {
            targets: FOUR_BY_FOUR_CENTRES.to_vec(),
            code_frames: false,
            ..Default::default()
        };
        let session_dir = write_synthetic_session(dir.path(), "s1", &spec).unwrap();
        let config = kappa_ray_config(&FOUR_BY_FOUR_CENTRES, [3.0, -1.0], None);
        let err = fit_recording(
            &session_dir,
            &config,
            &fake_registry(),
            &ProtocolConfig::default(),
            ProfileMeta::default(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("no gaze samples"));
    }

    fn fit_result(
        targets: usize,
        samples: usize,
        mean: Option<f64>,
        p95: f64,
        hit_rate: Option<f64>,
    ) -> FitResult {
        fit_result_rejecting(targets, samples, mean, p95, hit_rate, Vec::new())
    }

    fn fit_result_rejecting(
        targets: usize,
        samples: usize,
        mean: Option<f64>,
        p95: f64,
        hit_rate: Option<f64>,
        rejected_targets: Vec<u32>,
    ) -> FitResult {
        use eye_bench::metrics::{RegionHit, Summary};

        let expected = mean.map(|mean| SessionMetrics {
            windows: targets,
            samples,
            angular_error_deg: Some(Summary {
                mean,
                p50: mean,
                p95,
            }),
            px_error_logical: None,
            accuracy_deg: None,
            precision_rms_s2s_deg: None,
            precision_pooled_rms_s2s_deg: None,
            nees: None,
            regions: vec![RegionHit {
                cols: 3,
                rows: 3,
                windows: targets,
                excluded: 0,
                hits: targets,
                hit_rate,
                sample_hit_rate: None,
            }],
            processing_ms: None,
            dropout_rate: None,
            output_rate_hz: None,
        });
        FitResult {
            profile: fixture_profile(),
            samples,
            targets,
            expected,
            rejected_targets,
            preview: None,
        }
    }

    fn fixture_profile() -> UserProfile {
        UserProfile {
            version: eye_calibration::correction::PROFILE_VERSION,
            name: "default".into(),
            created_unix_s: 0,
            rig_fingerprint: String::new(),
            estimator: String::new(),
            corrections: std::collections::BTreeMap::new(),
            calibration_pose: None,
            provenance: Provenance::default(),
        }
    }

    #[test]
    fn test_summary_line_format() {
        let result = fit_result(9, 2140, Some(1.84), 3.90, Some(1.0));
        assert_eq!(
            summary_line(&result),
            "9 targets, 2140 samples; expected accuracy (bench leave-one-target-out): mean 1.84 deg, p95 3.90 deg, 3x3 hit 100.0 %"
        );

        let no_expected = FitResult {
            profile: fixture_profile(),
            samples: 2140,
            targets: 9,
            expected: None,
            rejected_targets: Vec::new(),
            preview: None,
        };
        assert_eq!(
            summary_line(&no_expected),
            "9 targets, 2140 samples; expected accuracy (bench leave-one-target-out): n/a"
        );
    }

    #[test]
    fn test_fit_recording_rejected_targets_come_from_primary_source() {
        use eye_calibration::correction::{CorrectionModel, EyeKey};
        use eye_core::RaySource;

        let report = |source, targets_rejected: Vec<u32>| EyeFitReport {
            source,
            key: EyeKey::Right,
            model: Some(CorrectionModel::Affine),
            targets_used: Vec::new(),
            targets_rejected,
            rms_before_deg: 0.0,
            rms_after_deg: 0.0,
            loo_target_mean_deg: 0.0,
            samples_without_head_pose: 0,
            diagnostics: Vec::new(),
        };
        let reports = vec![
            report(RaySource::RgbOnly, vec![1]),
            report(RaySource::IrOnly, vec![3]),
        ];
        assert_eq!(primary_rejected_targets(&reports), vec![1]);
    }

    #[test]
    fn test_summary_lists_twice_rejected_targets() {
        let result = fit_result_rejecting(9, 2140, Some(1.84), 3.90, Some(1.0), vec![4, 7]);
        assert_eq!(
            summary_line(&result),
            "9 targets, 2140 samples; expected accuracy (bench leave-one-target-out): mean 1.84 deg, p95 3.90 deg, 3x3 hit 100.0 %; 2 targets rejected: [4, 7]"
        );
    }

    #[test]
    fn test_summary_line_prints_bench_loto_and_live_preview() {
        let mut result = fit_result(9, 2140, Some(1.84), 3.90, Some(1.0));
        result.preview = Some(PreviewFit {
            targets: 9,
            samples: 2140,
            rms_after_deg: 2.37,
        });
        assert_eq!(
            summary_line(&result),
            "9 targets, 2140 samples; expected accuracy (bench leave-one-target-out): mean 1.84 deg, p95 3.90 deg, 3x3 hit 100.0 %; live preview residual: 2.37 deg"
        );
    }

    #[test]
    fn test_no_feedback_flag_skips_pipeline() {
        let cli = Cli::try_parse_from(["eye", "calibrate", "--no-feedback"]).expect("parses");
        let crate::cli::Command::Calibrate(args) = cli.command else {
            panic!("expected calibrate subcommand");
        };
        assert!(args.no_feedback);
        assert!(!wants_feedback(&args));

        let cli = Cli::try_parse_from(["eye", "calibrate"]).expect("parses");
        let crate::cli::Command::Calibrate(args) = cli.command else {
            panic!("expected calibrate subcommand");
        };
        assert!(!args.no_feedback);
        assert!(wants_feedback(&args));

        let args = Args {
            targets: 9,
            dwell_ms: None,
            profile: "default".to_string(),
            keep: false,
            from: Some(PathBuf::from("d")),
            no_feedback: false,
        };
        assert!(!wants_feedback(&args), "a --from fit shows nothing live");
    }

    struct NullSink;

    impl RecordSink for NullSink {
        fn frame(&mut self, _frame: &Frame) -> anyhow::Result<()> {
            Ok(())
        }

        fn target(&mut self, _record: &eye_core::session::TargetRecord) -> anyhow::Result<()> {
            Ok(())
        }
    }

    fn frame_at(seq: u64, t_ns: u64, code: u8) -> Frame {
        Frame::new(
            FrameHeader {
                camera: CameraId::from("ir"),
                seq,
                timestamp: Timestamp::from_nanos(t_ns),
                width: 8,
                height: 8,
                format: PixelFormat::Gray8,
                illumination: Illumination::Unknown,
            },
            Arc::from(vec![code; 64]),
        )
        .expect("valid frame")
    }

    fn shown_event(index: usize, at_ns: u64, px: (f64, f64)) -> TargetEvent {
        TargetEvent::Shown(TargetShown {
            index,
            output: OutputId::from("eDP-1"),
            px_logical: Point2::new(px.0, px.1),
            shown_at: Timestamp::from_nanos(at_ns),
            clock: TargetClock::Commit,
        })
    }

    fn hidden_event(index: usize, at_ns: u64) -> TargetEvent {
        TargetEvent::Hidden {
            index,
            at: Timestamp::from_nanos(at_ns),
            clock: TargetClock::Commit,
        }
    }

    enum LiveMsg {
        Frame(CaptureMsg),
        Target(TargetEvent),
    }

    /// A `Shown`/frames.../`Hidden` sequence for each target in order: `frames_per_target` frames
    /// spaced through the window, coded for `test-target-code` (pixel = index + 1), in a single
    /// chronological stream (`run_live_session` delivers it to `pump`'s two channels one message
    /// at a time, so `select!` cannot reorder a target event ahead of or behind its frames).
    fn live_session(
        targets_px: &[(f64, f64)],
        timing: TargetTiming,
        frames_per_target: u64,
    ) -> Vec<LiveMsg> {
        let dwell_ns = timing.dwell.as_nanos() as u64;
        let settle_ns = timing.settle.as_nanos() as u64;
        let window_ns = timing.window.as_nanos() as u64;
        let step_ns = (window_ns / frames_per_target.max(1)).max(1);
        let mut out = Vec::new();
        let mut seq = 0u64;
        for (k, &px) in targets_px.iter().enumerate() {
            let onset_ns = k as u64 * dwell_ns;
            out.push(LiveMsg::Target(shown_event(k, onset_ns, px)));
            for i in 0..frames_per_target {
                let t = onset_ns + settle_ns + i * step_ns;
                out.push(LiveMsg::Frame(CaptureMsg::Frame(frame_at(
                    seq,
                    t,
                    (k + 1) as u8,
                ))));
                seq += 1;
            }
            out.push(LiveMsg::Target(hidden_event(k, onset_ns + dwell_ns)));
        }
        out.push(LiveMsg::Target(TargetEvent::Finished));
        out
    }

    fn run_live_session(
        live: &mut LiveFeedback,
        messages: Vec<LiveMsg>,
    ) -> (
        crossbeam_channel::Receiver<Feedback>,
        crossbeam_channel::Receiver<eye_overlay::targets::AppendMsg>,
    ) {
        let mut sink = NullSink;
        run_live_session_with_sink(live, messages, &mut sink)
    }

    fn run_live_session_with_sink(
        live: &mut LiveFeedback,
        messages: Vec<LiveMsg>,
        sink: &mut dyn RecordSink,
    ) -> (
        crossbeam_channel::Receiver<Feedback>,
        crossbeam_channel::Receiver<eye_overlay::targets::AppendMsg>,
    ) {
        let (fb_tx, fb_rx) = crossbeam_channel::unbounded();
        live.attach_feedback(FeedbackSender::new(fb_tx));
        let (append_tx, append_rx) = crossbeam_channel::unbounded();
        live.attach_append(AppendSender::new(append_tx.clone()));
        let settle = AppendSender::new(append_tx);

        let (frames_tx, frames_rx) = crossbeam_channel::bounded::<CaptureMsg>(0);
        let (targets_tx, targets_rx) = crossbeam_channel::bounded::<TargetEvent>(0);
        let sender = std::thread::spawn(move || {
            for msg in messages {
                match msg {
                    LiveMsg::Frame(frame) => frames_tx.send(frame).unwrap(),
                    LiveMsg::Target(event) => targets_tx.send(event).unwrap(),
                }
            }
        });

        let screen = synthetic_rig().screen().clone();
        let to_record =
            move |s: &TargetShown, hidden: Option<Timestamp>| target_record(s, hidden, &screen);
        let end = pump(
            &frames_rx,
            &targets_rx,
            Some(&settle),
            &crossbeam_channel::never(),
            &crossbeam_channel::never(),
            &to_record,
            sink,
            live,
        )
        .expect("pump finishes");
        sender.join().unwrap();
        assert_eq!(end, crate::commands::record::PumpEnd::Finished);
        (fb_rx, append_rx)
    }

    /// Drains `rx` and keeps only the `TargetSpec`s it was told to re-present, discarding the
    /// `Settled` acks the pump sends after every `Hidden`.
    fn drain_appended(
        rx: &crossbeam_channel::Receiver<eye_overlay::targets::AppendMsg>,
    ) -> Vec<TargetSpec> {
        rx.try_iter()
            .filter_map(|msg| match msg {
                eye_overlay::targets::AppendMsg::Append(spec) => Some(spec),
                eye_overlay::targets::AppendMsg::Settled => None,
            })
            .collect()
    }

    fn px_error(fb: &Feedback, target_px: Point2<f64>) -> f64 {
        (fb.px_logical - target_px).norm()
    }

    #[derive(Debug)]
    struct AnyFrameDetector;

    impl eye_core::stage::Detector for AnyFrameDetector {
        fn name(&self) -> &'static str {
            "any-frame"
        }

        fn accepts(&self, _format: PixelFormat, _illumination: Illumination) -> bool {
            true
        }

        fn detect(
            &mut self,
            frames: &eye_core::FrameSet,
        ) -> Result<Vec<eye_core::Observations>, eye_core::stage::StageError> {
            Ok(frames
                .frames()
                .iter()
                .map(|f| {
                    eye_core::Observations::empty(f.header().camera.clone(), f.header().timestamp)
                })
                .collect())
        }
    }

    #[derive(Debug)]
    struct RgbTimestampEstimator;

    impl eye_core::stage::GazeEstimator for RgbTimestampEstimator {
        fn name(&self) -> &'static str {
            "rgb-timestamp"
        }

        fn estimate(
            &mut self,
            obs: &[eye_core::Observations],
            _rig: &Rig,
        ) -> Result<Vec<eye_core::GazeRay>, eye_core::stage::StageError> {
            let timestamp = obs
                .iter()
                .find(|o| o.camera.as_str() == "rgb")
                .or_else(|| obs.first())
                .map_or(Timestamp::from_nanos(0), |o| o.timestamp);
            Ok(vec![eye_core::GazeRay {
                side: None,
                timestamp,
                origin: Point3::new(155.0, 85.0, -500.0),
                direction: Vector3::z_axis(),
                angular_cov: Matrix2::identity() * 1e-6,
                origin_cov: Matrix3::zeros(),
                head_rotation: None,
            }])
        }
    }

    #[derive(Debug)]
    struct NoOpFilter;

    impl eye_core::stage::GazeFilter for NoOpFilter {
        fn name(&self) -> &'static str {
            "noop"
        }

        fn apply(&mut self, point: eye_core::GazePoint) -> eye_core::GazePoint {
            point
        }

        fn reset(&mut self) {}
    }

    fn dual_camera_frame(
        camera: &str,
        seq: u64,
        t_ms: u64,
        format: PixelFormat,
        illumination: Illumination,
    ) -> Frame {
        let (width, height) = (4u32, 2u32);
        let bpp = format
            .bytes_per_pixel()
            .expect("test format has a byte size");
        let data: Arc<[u8]> = vec![0u8; width as usize * height as usize * bpp].into();
        Frame::new(
            FrameHeader {
                camera: CameraId::from(camera),
                seq,
                timestamp: Timestamp::from_nanos(t_ms * 1_000_000),
                width,
                height,
                format,
                illumination,
            },
            data,
        )
        .expect("test frame is valid")
    }

    #[test]
    fn test_live_feedback_window_test_uses_ray_timestamp() {
        let ir_info = CameraInfo {
            id: CameraId::from("ir"),
            width: 4,
            height: 2,
            format: PixelFormat::Gray8,
            frame_interval: Duration::from_millis(33),
        };
        let rgb_info = CameraInfo {
            id: CameraId::from("rgb"),
            width: 4,
            height: 2,
            format: PixelFormat::Rgb8,
            frame_interval: Duration::from_millis(33),
        };
        let pairer = eye_capture::pairing::Pairer::with_config(
            &[ir_info, rgb_info],
            eye_capture::pairing::PairingConfig {
                offset_secondary_ns: 0,
                bracket_window: Duration::from_millis(200),
            },
        )
        .expect("ir/rgb pairer is valid");
        let pipeline = Pipeline::new(
            synthetic_rig(),
            pairer,
            vec![
                (CameraId::from("ir"), Box::new(AnyFrameDetector)),
                (CameraId::from("rgb"), Box::new(AnyFrameDetector)),
            ],
            Box::new(RgbTimestampEstimator),
            Box::new(NoOpFilter),
            None,
            3,
        );
        let protocol = TargetProtocol::new(ProtocolConfig {
            grid: [3, 3],
            lead_in_ms: 0,
            dwell_ms: 1000,
            settle_ms: 500,
            window_ms: 200,
        })
        .unwrap();
        let mut live = LiveFeedback::with_pipeline(
            pipeline,
            protocol,
            FitConfig::default(),
            ProfileMeta::default(),
        );
        live.on_shown(&TargetShown {
            index: 0,
            output: OutputId::from("eDP-1"),
            px_logical: Point2::new(0.0, 0.0),
            shown_at: Timestamp::from_nanos(0),
            clock: TargetClock::Commit,
        });

        let rgb_frame = dual_camera_frame("rgb", 0, 600, PixelFormat::Rgb8, Illumination::Ambient);
        let ir_frame = dual_camera_frame("ir", 1, 750, PixelFormat::Gray8, Illumination::IrLit);
        live.process(&rgb_frame)
            .expect("rgb frame held, no set yet");
        assert!(live.batches.is_empty(), "no set should have completed yet");
        live.process(&ir_frame)
            .expect("ir frame completes the pair");

        assert_eq!(
            live.fit_samples().len(),
            1,
            "the ray's own timestamp (600 ms) falls in the settle..settle+window \
             [500, 700) ms window even though the batch's timestamp (750 ms, the \
             newer IR frame) does not"
        );
    }

    #[test]
    fn test_live_preview_builds_window_from_shown_event() {
        let targets_px = FOUR_BY_FOUR_CENTRES[..1].to_vec();
        let config = kappa_ray_config(&targets_px, [0.0, 0.0], None);
        let cameras: Vec<CameraInfo> = config.cameras.iter().map(|c| c.to_info()).collect();
        let pipeline =
            Pipeline::from_config(&fake_registry(), &config, synthetic_rig(), &cameras, None)
                .expect("builds without I/O");
        let protocol = TargetProtocol::new(ProtocolConfig::default()).unwrap();
        let timing = protocol.timing();
        let mut live = LiveFeedback::with_pipeline(
            pipeline,
            protocol,
            FitConfig::default(),
            ProfileMeta::default(),
        );

        let shown = TargetShown {
            index: 3,
            output: OutputId::from("eDP-1"),
            px_logical: Point2::new(targets_px[0].0, targets_px[0].1),
            shown_at: Timestamp::from_nanos(5_000_000_000),
            clock: TargetClock::Commit,
        };
        live.on_shown(&shown);

        assert_eq!(live.windows.len(), 1);
        let window = &live.windows[0];
        assert_eq!(window.index, 3);
        assert_eq!(window.start, Timestamp(shown.shown_at.0 + timing.settle));
        assert_eq!(window.end, Timestamp(window.start.0 + timing.window));
    }

    #[test]
    fn test_live_preview_retry_keeps_latest_presentation_only() {
        use eye_core::GazeRay;
        use nalgebra::{Matrix2, Matrix3, Point3, Unit};

        let targets_px = FOUR_BY_FOUR_CENTRES[..1].to_vec();
        let config = kappa_ray_config(&targets_px, [0.0, 0.0], None);
        let cameras: Vec<CameraInfo> = config.cameras.iter().map(|c| c.to_info()).collect();
        let pipeline =
            Pipeline::from_config(&fake_registry(), &config, synthetic_rig(), &cameras, None)
                .expect("builds without I/O");
        let protocol = TargetProtocol::new(ProtocolConfig::default()).unwrap();
        let timing = protocol.timing();
        let mut live = LiveFeedback::with_pipeline(
            pipeline,
            protocol,
            FitConfig::default(),
            ProfileMeta::default(),
        );

        let px = Point2::new(targets_px[0].0, targets_px[0].1);
        let eye = Point3::new(155.0, 85.0, -500.0);
        let direction = Unit::new_normalize(Point3::new(0.0, 0.0, 0.0) - eye);
        let fake_ray = |ts_ns: u64| GazeRay {
            side: None,
            timestamp: Timestamp::from_nanos(ts_ns),
            origin: eye,
            direction,
            angular_cov: Matrix2::identity() * 1e-6,
            origin_cov: Matrix3::zeros(),
            head_rotation: None,
        };

        let onset_a = 0u64;
        live.on_shown(&TargetShown {
            index: 0,
            output: OutputId::from("eDP-1"),
            px_logical: px,
            shown_at: Timestamp::from_nanos(onset_a),
            clock: TargetClock::Commit,
        });
        let mid_a = onset_a + timing.settle.as_nanos() as u64 + timing.window.as_nanos() as u64 / 2;
        live.batches.push(RayBatch {
            timestamp: Timestamp::from_nanos(mid_a),
            rays: vec![fake_ray(mid_a)],
        });

        let onset_b = 10_000_000_000u64;
        live.on_shown(&TargetShown {
            index: 9,
            output: OutputId::from("eDP-1"),
            px_logical: px,
            shown_at: Timestamp::from_nanos(onset_b),
            clock: TargetClock::Commit,
        });
        let mid_b = onset_b + timing.settle.as_nanos() as u64 + timing.window.as_nanos() as u64 / 2;
        live.batches.push(RayBatch {
            timestamp: Timestamp::from_nanos(mid_b),
            rays: vec![fake_ray(mid_b)],
        });

        let samples = live.fit_samples();
        assert_eq!(
            samples.len(),
            1,
            "only the retry's (index 9) presentation must be kept"
        );
        assert_eq!(samples[0].ray.timestamp, Timestamp::from_nanos(mid_b));
    }

    #[test]
    fn test_live_preview_samples_equal_offline_selection_on_same_events() {
        use eye_core::session::TargetRecord;

        let targets_px = FOUR_BY_FOUR_CENTRES[..4].to_vec();
        let config = kappa_ray_config(&targets_px, [0.0, 0.0], None);
        let cameras: Vec<CameraInfo> = config.cameras.iter().map(|c| c.to_info()).collect();
        let pipeline =
            Pipeline::from_config(&fake_registry(), &config, synthetic_rig(), &cameras, None)
                .expect("builds without I/O");
        let protocol = TargetProtocol::new(ProtocolConfig {
            grid: [2, 2],
            ..ProtocolConfig::default()
        })
        .unwrap();
        let (cols, rows) = protocol.grid();
        let base_len = cols * rows;
        let screen = synthetic_rig().screen().clone();
        let mut live = LiveFeedback::with_pipeline(
            pipeline,
            protocol,
            FitConfig::default(),
            ProfileMeta::default(),
        );

        let mut shown_events = Vec::new();
        let dwell_ns = 2_500_000_000u64;
        for (k, &px) in targets_px.iter().enumerate() {
            let onset_ns = k as u64 * dwell_ns;
            let shown = TargetShown {
                index: k,
                output: OutputId::from("eDP-1"),
                px_logical: Point2::new(px.0, px.1),
                shown_at: Timestamp::from_nanos(onset_ns),
                clock: TargetClock::Commit,
            };
            live.on_shown(&shown);
            shown_events.push(shown);
            live.process(&frame_at(k as u64, onset_ns + 1_100_000_000, (k + 1) as u8))
                .expect("process succeeds");
            if k == 0 {
                live.process(&frame_at(100, onset_ns + 1_900_000_000, 1))
                    .expect("process succeeds");
            }
            live.on_hidden(k, Timestamp::from_nanos(onset_ns + dwell_ns));
        }

        // Pushed after target 3's on_shown, but its timestamp falls in target 1's window:
        // `process` must not gate on what the "current" target is at push time.
        let late_ray_ts_ns = dwell_ns + 1_000_000_000;
        live.process(&frame_at(150, late_ray_ts_ns, 2))
            .expect("process succeeds");

        let retry_index = targets_px.len();
        let retry_onset_ns = targets_px.len() as u64 * dwell_ns;
        let shown_a_retry = TargetShown {
            index: retry_index,
            output: OutputId::from("eDP-1"),
            px_logical: shown_events[0].px_logical,
            shown_at: Timestamp::from_nanos(retry_onset_ns),
            clock: TargetClock::Commit,
        };
        live.on_shown(&shown_a_retry);
        shown_events.push(shown_a_retry.clone());
        live.process(&frame_at(200, retry_onset_ns + 1_100_000_000, 1))
            .expect("process succeeds");
        live.on_hidden(
            retry_index,
            Timestamp::from_nanos(retry_onset_ns + dwell_ns),
        );

        let records: Vec<TargetRecord> = shown_events
            .iter()
            .map(|s| target_record(s, None, &screen))
            .collect();
        let independent_windows = protocol.fixation_windows(&records, &screen).unwrap();
        assert_eq!(live.windows, independent_windows);

        let expected = fit_samples(
            &independent_windows,
            live.batches.iter(),
            latest_presentation(&independent_windows, base_len),
            legacy_source(&live.meta.estimator),
        );
        let fingerprint = |samples: &[FitSample]| -> Vec<(u64, u64, u64)> {
            samples
                .iter()
                .map(|s| {
                    (
                        s.ray.timestamp.as_nanos(),
                        s.target_mm.x.to_bits(),
                        s.target_mm.y.to_bits(),
                    )
                })
                .collect()
        };
        assert_eq!(fingerprint(&live.fit_samples()), fingerprint(&expected));

        let target1_window = independent_windows
            .iter()
            .find(|w| w.index == 1)
            .expect("target 1 has a window");
        assert!(
            expected
                .iter()
                .any(|s| s.ray.timestamp.as_nanos() == late_ray_ts_ns
                    && s.target_mm == target1_window.target_mm),
            "the late ray, pushed after target 3's on_shown, must still be selected for \
             target 1's (non-superseded) window by timestamp, not by push order"
        );

        assert_eq!(
            expected.len(),
            5,
            "target 0's base presentation (one in-window sample) is superseded by its retry; \
             targets 1-3 and the retry each contribute one sample, plus the late sample for \
             target 1"
        );
    }

    #[test]
    fn test_refit_after_min_targets_applies_correction() {
        let three = FOUR_BY_FOUR_CENTRES[..3].to_vec();
        let targets_px: Vec<(f64, f64)> = three.iter().chain(three.iter()).copied().collect();
        let config = kappa_ray_config(&targets_px, [3.0, -1.0], None);
        let cameras: Vec<CameraInfo> = config.cameras.iter().map(|c| c.to_info()).collect();
        let pipeline =
            Pipeline::from_config(&fake_registry(), &config, synthetic_rig(), &cameras, None)
                .expect("builds without I/O");
        let protocol = TargetProtocol::new(ProtocolConfig::default()).unwrap();
        let timing = protocol.timing();
        let fit_cfg = FitConfig::default();
        let mut live =
            LiveFeedback::with_pipeline(pipeline, protocol, fit_cfg, ProfileMeta::default());

        let frames_per_target = (fit_cfg.min_samples_per_target * 2) as u64;
        let messages = live_session(&targets_px, timing, frames_per_target);
        let (fb_rx, _append_rx) = run_live_session(&mut live, messages);

        let feedback: Vec<Feedback> = fb_rx.try_iter().collect();
        assert!(!feedback.is_empty());
        assert!(live.is_calibrated());

        let dwell_ns = timing.dwell.as_nanos() as u64;
        let mut before_errors = Vec::new();
        let mut after_errors = Vec::new();
        for fb in &feedback {
            let index = (fb.at.as_nanos() / dwell_ns) as usize;
            let target_px = Point2::new(targets_px[index].0, targets_px[index].1);
            let error = px_error(fb, target_px);
            assert_eq!(
                fb.calibrated,
                index >= fit_cfg.min_targets_offset,
                "feedback for target {index} has calibrated={} but min_targets_offset={}",
                fb.calibrated,
                fit_cfg.min_targets_offset
            );
            if fb.calibrated {
                after_errors.push(error);
            } else {
                before_errors.push(error);
            }
        }
        assert!(!before_errors.is_empty(), "expected uncalibrated feedback");
        assert!(!after_errors.is_empty(), "expected calibrated feedback");
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        assert!(
            mean(&after_errors) < mean(&before_errors),
            "calibrated mean {} not below uncalibrated mean {}",
            mean(&after_errors),
            mean(&before_errors)
        );
    }

    #[test]
    fn test_refit_failure_keeps_previous_correction() {
        let targets_px = FOUR_BY_FOUR_CENTRES[..3].to_vec();
        let config = kappa_ray_config(&targets_px, [3.0, -1.0], None);
        let cameras: Vec<CameraInfo> = config.cameras.iter().map(|c| c.to_info()).collect();
        let pipeline =
            Pipeline::from_config(&fake_registry(), &config, synthetic_rig(), &cameras, None)
                .expect("builds without I/O");
        let protocol = TargetProtocol::new(ProtocolConfig::default()).unwrap();
        let timing = protocol.timing();
        let fit_cfg = FitConfig::default();
        let mut live =
            LiveFeedback::with_pipeline(pipeline, protocol, fit_cfg, ProfileMeta::default());

        let frames_per_target = (fit_cfg.min_samples_per_target * 2) as u64;
        let messages = live_session(&targets_px, timing, frames_per_target);
        let (fb_rx, _append_rx) = run_live_session(&mut live, messages);
        assert!(live.is_calibrated(), "setup: expected a successful refit");
        fb_rx.try_iter().for_each(drop);

        let probe = frame_at(90_000, 90_000_000_000, 1);
        live.process(&probe).expect("process succeeds");
        let before = fb_rx.try_recv().expect("feedback for the probe frame");
        assert!(before.calibrated);

        live.batches.clear();
        live.refit(99, None);
        assert!(
            live.is_calibrated(),
            "a failed refit must not clear an existing correction"
        );

        live.process(&probe).expect("process succeeds");
        let after = fb_rx
            .try_recv()
            .expect("feedback for the repeated probe frame");
        assert!(after.calibrated);
        assert_eq!(
            after.px_logical, before.px_logical,
            "the correction changed even though the refit failed"
        );
        assert_eq!(after.cov_px, before.cov_px);
    }

    struct CountingSink {
        frames: usize,
    }

    impl RecordSink for CountingSink {
        fn frame(&mut self, _frame: &Frame) -> anyhow::Result<()> {
            self.frames += 1;
            Ok(())
        }

        fn target(&mut self, _record: &eye_core::session::TargetRecord) -> anyhow::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn test_slow_rays_skips_sets_but_sink_still_sees_every_frame() {
        let targets_px = FOUR_BY_FOUR_CENTRES[..1].to_vec();
        let config = kappa_ray_config(&targets_px, [3.0, -1.0], None);
        let cameras: Vec<CameraInfo> = config.cameras.iter().map(|c| c.to_info()).collect();
        let pipeline =
            Pipeline::from_config(&fake_registry(), &config, synthetic_rig(), &cameras, None)
                .expect("builds without I/O");
        let protocol = TargetProtocol::new(ProtocolConfig::default()).unwrap();
        let timing = protocol.timing();
        let fit_cfg = FitConfig::default();
        let mut live =
            LiveFeedback::with_pipeline(pipeline, protocol, fit_cfg, ProfileMeta::default());
        live.force_rays_duration(Duration::from_millis(100));

        let frames_per_target = 6u64;
        let messages = live_session(&targets_px, timing, frames_per_target);
        let frame_count = messages
            .iter()
            .filter(|m| matches!(m, LiveMsg::Frame(_)))
            .count();

        let mut sink = CountingSink { frames: 0 };
        let (fb_rx, _append_rx) = run_live_session_with_sink(&mut live, messages, &mut sink);

        assert_eq!(
            sink.frames, frame_count,
            "every frame must still reach the RecordSink regardless of processing skips"
        );
        let feedback: Vec<Feedback> = fb_rx.try_iter().collect();
        assert!(
            feedback.len() < frame_count,
            "a slow observer must skip processing some sets ({} feedback for {} frames)",
            feedback.len(),
            frame_count
        );
    }

    #[test]
    fn test_rejected_target_is_represented_once() {
        let targets_px = FOUR_BY_FOUR_CENTRES.to_vec();
        let bad = targets_px.len() - 2;
        let config = kappa_ray_config(&targets_px, [0.0, 0.0], Some((bad, [10.0, 0.0])));
        let cameras: Vec<CameraInfo> = config.cameras.iter().map(|c| c.to_info()).collect();
        let pipeline =
            Pipeline::from_config(&fake_registry(), &config, synthetic_rig(), &cameras, None)
                .expect("builds without I/O");
        let protocol = TargetProtocol::new(ProtocolConfig::default()).unwrap();
        let timing = protocol.timing();
        let fit_cfg = FitConfig::default();
        let mut live =
            LiveFeedback::with_pipeline(pipeline, protocol, fit_cfg, ProfileMeta::default());

        let frames_per_target = (fit_cfg.min_samples_per_target * 2) as u64;
        let messages = live_session(&targets_px, timing, frames_per_target);
        let (_fb_rx, append_rx) = run_live_session(&mut live, messages);

        let appended = drain_appended(&append_rx);
        assert_eq!(appended.len(), 1, "{appended:?}");
        let expected_px = Point2::new(targets_px[bad].0, targets_px[bad].1);
        assert_eq!(appended[0].px_logical, expected_px);
        assert!(appended[0].retry);
    }

    #[test]
    fn test_no_verdict_before_baseline() {
        let targets_px = FOUR_BY_FOUR_CENTRES[..2].to_vec();
        let config = kappa_ray_config(&targets_px, [0.0, 0.0], Some((0, [10.0, 0.0])));
        let cameras: Vec<CameraInfo> = config.cameras.iter().map(|c| c.to_info()).collect();
        let pipeline =
            Pipeline::from_config(&fake_registry(), &config, synthetic_rig(), &cameras, None)
                .expect("builds without I/O");
        let protocol = TargetProtocol::new(ProtocolConfig::default()).unwrap();
        let timing = protocol.timing();
        let fit_cfg = FitConfig::default();
        assert!(
            2 < fit_cfg.min_targets_offset,
            "test assumes no baseline yet"
        );
        let mut live =
            LiveFeedback::with_pipeline(pipeline, protocol, fit_cfg, ProfileMeta::default());

        let frames_per_target = (fit_cfg.min_samples_per_target * 2) as u64;
        let messages = live_session(&targets_px, timing, frames_per_target);
        let (_fb_rx, append_rx) = run_live_session(&mut live, messages);

        assert!(!live.is_calibrated());
        assert_eq!(drain_appended(&append_rx).len(), 0);
    }

    #[test]
    fn test_retry_with_good_samples_replaces_bad_ones_and_is_not_rejected_again() {
        use eye_core::GazeRay;
        use nalgebra::{Matrix2, Matrix3, Point3, Unit};

        let targets_px = FOUR_BY_FOUR_CENTRES.to_vec();
        let bad = targets_px.len() - 2;
        let config = kappa_ray_config(&targets_px, [0.0, 0.0], Some((bad, [10.0, 0.0])));
        let cameras: Vec<CameraInfo> = config.cameras.iter().map(|c| c.to_info()).collect();
        let pipeline =
            Pipeline::from_config(&fake_registry(), &config, synthetic_rig(), &cameras, None)
                .expect("builds without I/O");
        let protocol = TargetProtocol::new(ProtocolConfig::default()).unwrap();
        let timing = protocol.timing();
        let fit_cfg = FitConfig::default();
        let mut live =
            LiveFeedback::with_pipeline(pipeline, protocol, fit_cfg, ProfileMeta::default());

        let frames_per_target = (fit_cfg.min_samples_per_target * 2) as u64;
        let messages = live_session(&targets_px, timing, frames_per_target);
        let (_fb_rx, append_rx) = run_live_session(&mut live, messages);
        let appended = drain_appended(&append_rx);
        assert_eq!(appended.len(), 1, "setup: expected one retry request");

        let bad_px = Point2::new(targets_px[bad].0, targets_px[bad].1);
        let screen = synthetic_rig().screen().clone();
        let bad_mm = px_logical_to_mm(&screen, &bad_px);

        let onset_ns = 10_000_000_000_000u64;
        let shown = TargetShown {
            index: targets_px.len(),
            output: OutputId::from("eDP-1"),
            px_logical: bad_px,
            shown_at: Timestamp::from_nanos(onset_ns),
            clock: TargetClock::Commit,
        };
        live.on_shown(&shown);
        assert!(
            !live.fit_samples().iter().any(|s| s.target_mm == bad_mm),
            "the rejected presentation's samples must be superseded once its retry is shown"
        );

        let eye = Point3::new(155.0, 85.0, -500.0);
        let target_point = Point3::new(bad_mm.x, bad_mm.y, 0.0);
        let direction = Unit::new_normalize(target_point - eye);
        let mid_ns =
            onset_ns + timing.settle.as_nanos() as u64 + timing.window.as_nanos() as u64 / 2;
        live.batches.push(RayBatch {
            timestamp: Timestamp::from_nanos(mid_ns),
            rays: (0..(fit_cfg.min_samples_per_target * 2))
                .map(|_| GazeRay {
                    side: None,
                    timestamp: Timestamp::from_nanos(mid_ns),
                    origin: eye,
                    direction,
                    angular_cov: Matrix2::identity() * 1e-6,
                    origin_cov: Matrix3::zeros(),
                    head_rotation: None,
                })
                .collect(),
        });
        live.on_hidden(
            targets_px.len(),
            Timestamp::from_nanos(onset_ns + 100_000_000_000),
        );

        assert_eq!(
            drain_appended(&append_rx).len(),
            0,
            "a retry whose samples are now good must not be re-appended"
        );

        let samples = live.fit_samples();
        let outcome = DotSessionFit::fit_with(
            &samples,
            live.pipeline.as_ref().unwrap().rig(),
            &fit_cfg,
            ProfileMeta::default(),
        )
        .unwrap();
        let index = target_index_in_samples(&samples, bad_mm).unwrap();
        let verdict = outcome
            .verdicts()
            .into_iter()
            .find(|v| v.index == index)
            .unwrap();
        assert!(
            !verdict.rejected,
            "verdict for the retried target should no longer be rejected: {verdict:?}"
        );
    }

    #[test]
    fn test_retries_stop_appending_at_max_retries() {
        use eye_core::GazeRay;
        use nalgebra::{Matrix2, Matrix3, Point3, Unit};

        let targets_px = FOUR_BY_FOUR_CENTRES.to_vec();
        let bad = targets_px.len() - 2;
        let config = kappa_ray_config(&targets_px, [0.0, 0.0], Some((bad, [10.0, 0.0])));
        let cameras: Vec<CameraInfo> = config.cameras.iter().map(|c| c.to_info()).collect();
        let pipeline =
            Pipeline::from_config(&fake_registry(), &config, synthetic_rig(), &cameras, None)
                .expect("builds without I/O");
        let protocol = TargetProtocol::new(ProtocolConfig::default()).unwrap();
        let timing = protocol.timing();
        let fit_cfg = FitConfig::default();
        let mut live =
            LiveFeedback::with_pipeline(pipeline, protocol, fit_cfg, ProfileMeta::default());

        let frames_per_target = (fit_cfg.min_samples_per_target * 2) as u64;
        let messages = live_session(&targets_px, timing, frames_per_target);
        let (_fb_rx, append_rx) = run_live_session(&mut live, messages);
        let appended = drain_appended(&append_rx);
        assert_eq!(appended.len(), 1, "setup: expected the first retry request");
        assert!(appended[0].retry);

        let bad_px = Point2::new(targets_px[bad].0, targets_px[bad].1);
        let screen = synthetic_rig().screen().clone();
        let bad_mm = px_logical_to_mm(&screen, &bad_px);
        let eye = Point3::new(155.0, 85.0, -500.0);
        let target_point = Point3::new(bad_mm.x, bad_mm.y, 0.0);
        let direction = Unit::new_normalize(target_point - eye);

        let present_bad_retry = |live: &mut LiveFeedback, retry_index: usize, at_ns: u64| {
            let shown = TargetShown {
                index: retry_index,
                output: OutputId::from("eDP-1"),
                px_logical: bad_px,
                shown_at: Timestamp::from_nanos(at_ns),
                clock: TargetClock::Commit,
            };
            live.on_shown(&shown);
            // Below `min_samples_per_target`: the fitter rejects this target as TooFewSamples
            // on every refit, regardless of the shared `direction`.
            let mid_ns =
                at_ns + timing.settle.as_nanos() as u64 + timing.window.as_nanos() as u64 / 2;
            live.batches.push(RayBatch {
                timestamp: Timestamp::from_nanos(mid_ns),
                rays: (0..2)
                    .map(|_| GazeRay {
                        side: None,
                        timestamp: Timestamp::from_nanos(mid_ns),
                        origin: eye,
                        direction,
                        angular_cov: Matrix2::identity() * 1e-6,
                        origin_cov: Matrix3::zeros(),
                        head_rotation: None,
                    })
                    .collect(),
            });
            live.on_hidden(retry_index, Timestamp::from_nanos(at_ns + 100_000_000_000));
        };

        present_bad_retry(&mut live, targets_px.len(), 10_000_000_000_000);
        let second_retry = drain_appended(&append_rx);
        assert_eq!(
            second_retry.len(),
            1,
            "the second rejection is still under MAX_RETRIES and must append once more"
        );

        present_bad_retry(&mut live, targets_px.len() + 1, 10_200_000_000_000);
        let third_retry = drain_appended(&append_rx);
        assert_eq!(
            third_retry.len(),
            0,
            "a third rejection of the same target exceeds MAX_RETRIES and must not append"
        );
    }
}
