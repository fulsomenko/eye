use std::collections::BTreeMap;
use std::fs::Permissions;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crossbeam_channel::Receiver;
use eye::config::Config;
use eye_calibration::protocol::{ProtocolConfig, TargetProtocol};
use eye_calibration::store::{ProfileStore, rig_to_table};
use eye_capture::session::{
    EmitterState, FORMAT_VERSION, RecordedCamera, SessionId, SessionMeta, SessionWriter,
};
use eye_core::log::{field, span};
use eye_core::session::TargetRecord;
use eye_core::{CameraInfo, Frame, Rig, ScreenModel, Timestamp};
use eye_geometry::screen::px_logical_to_mm;
use eye_overlay::targets::{
    AppendSender, FeedbackSender, FinishPolicy, TargetDisplay, TargetEvent, TargetShown, TargetSpec,
};
use eye_platform::EmitterGuard;

use crate::capture::{Capture, CaptureMsg};
use crate::cli::GIT_REV;
use crate::commands::emitter::emitter_guards;
use crate::commands::probe::{ProbeReport, Probes, collect};
use crate::ctx::Ctx;
use crate::rig;
use crate::shutdown;

#[derive(Debug, clap::Args)]
pub struct Args {
    /// Dot targets: 16 (4x4 cell centres, serves 3x3 and 4x4 evaluation), 9 (3x3), 0 (none)
    #[arg(long, default_value_t = 16, value_parser = parse_target_count)]
    pub targets: u32,
    /// Time each dot is shown (default: the protocol's 1500 ms)
    #[arg(long, value_name = "MS")]
    pub dwell_ms: Option<u64>,
    /// Recording length; required with --targets 0, rejected otherwise
    #[arg(long, value_name = "SECONDS")]
    pub duration_s: Option<u64>,
}

pub fn parse_target_count(s: &str) -> Result<u32, String> {
    match s {
        "16" => Ok(16),
        "9" => Ok(9),
        "0" => Ok(0),
        _ => Err("allowed: 16, 9, 0".to_string()),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RecordOptions {
    pub protocol: Option<ProtocolConfig>,
    pub duration: Option<Duration>,
}

impl RecordOptions {
    pub fn from_args(args: &Args) -> anyhow::Result<Self> {
        if args.targets == 0 {
            let duration_s = args
                .duration_s
                .ok_or_else(|| anyhow::anyhow!("--duration-s is required with --targets 0"))?;
            return Ok(Self {
                protocol: None,
                duration: Some(Duration::from_secs(duration_s)),
            });
        }
        anyhow::ensure!(
            args.duration_s.is_none(),
            "--duration-s is only used with --targets 0"
        );
        let grid = if args.targets == 16 { [4, 4] } else { [3, 3] };
        let mut protocol = ProtocolConfig {
            grid,
            ..ProtocolConfig::default()
        };
        if let Some(dwell_ms) = args.dwell_ms {
            protocol.dwell_ms = dwell_ms;
        }
        Ok(Self {
            protocol: Some(protocol),
            duration: None,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SessionLocation {
    pub root: PathBuf,
    pub id: SessionId,
}

impl SessionLocation {
    /// `--output DIR` = the session directory itself: root = parent (or "."), id = file name.
    /// Default: root `recordings`, id `SessionId::now()`.
    pub fn resolve(output: Option<&Path>) -> anyhow::Result<Self> {
        match output {
            Some(path) => {
                let name = path.file_name().and_then(|n| n.to_str()).ok_or_else(|| {
                    anyhow::anyhow!("--output has no file name: {}", path.display())
                })?;
                let id = SessionId::new(name)?;
                let root = match path.parent() {
                    Some(parent) if !parent.as_os_str().is_empty() => parent.to_path_buf(),
                    _ => PathBuf::from("."),
                };
                Ok(Self { root, id })
            }
            None => Ok(Self {
                root: PathBuf::from("recordings"),
                id: SessionId::now(),
            }),
        }
    }

    pub fn dir(&self) -> PathBuf {
        self.root.join(self.id.as_str())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RecordSummary {
    pub dir: PathBuf,
    pub frames: BTreeMap<String, u64>,
    pub dropped_frames: u64,
    pub targets: usize,
    pub interrupted: bool,
}

pub trait RecordSink {
    fn frame(&mut self, frame: &Frame) -> anyhow::Result<()>;
    fn target(&mut self, record: &TargetRecord) -> anyhow::Result<()>;
}

#[derive(Debug)]
pub struct SessionSink {
    writer: SessionWriter,
    frames: BTreeMap<String, u64>,
    targets: usize,
}

impl SessionSink {
    pub fn new(writer: SessionWriter) -> Self {
        Self {
            writer,
            frames: BTreeMap::new(),
            targets: 0,
        }
    }

    pub fn finish(self) -> anyhow::Result<(PathBuf, BTreeMap<String, u64>, usize)> {
        let dir = self.writer.finish()?;
        Ok((dir, self.frames, self.targets))
    }
}

impl RecordSink for SessionSink {
    fn frame(&mut self, frame: &Frame) -> anyhow::Result<()> {
        self.writer.write_frame(frame)?;
        *self
            .frames
            .entry(frame.header().camera.to_string())
            .or_insert(0) += 1;
        Ok(())
    }

    fn target(&mut self, record: &TargetRecord) -> anyhow::Result<()> {
        self.writer.write_target(record)?;
        self.targets += 1;
        Ok(())
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum PumpEnd {
    Finished,
    Deadline,
    Interrupted,
}

/// Hooks `pump` drives alongside the frame/target recording. All methods default to no-ops.
pub trait PumpObserver {
    /// Whether `record_targets` should hold `Finished` back until the pump has acked every
    /// `Hidden`, giving this observer a chance to append a retry first.
    fn finish_policy(&self) -> FinishPolicy {
        FinishPolicy::Immediate
    }

    /// Called once by `record_targets` before pumping, with the overlay's feedback sender and an
    /// `AppendSender` for the running `TargetDisplay` (e.g. to re-present a target after an
    /// online rejection).
    fn attach(&mut self, _feedback: FeedbackSender, _append: AppendSender) {}

    /// Called once `record_session` has resolved the rig and opened the camera sources, before
    /// any frame is pumped, so an observer that needs them (e.g. to build a `Pipeline`) can set
    /// itself up.
    fn prepare(&mut self, _rig: &Rig, _cameras: &[CameraInfo]) -> anyhow::Result<()> {
        Ok(())
    }

    fn on_frame(&mut self, _frame: &Frame) -> anyhow::Result<()> {
        Ok(())
    }

    fn on_shown(&mut self, _shown: &TargetShown) {}

    fn on_hidden(&mut self, _index: usize, _at: Timestamp) {}
}

#[derive(Debug, Default)]
pub struct NoopObserver;

impl PumpObserver for NoopObserver {}

/// Hands the observer both senders exactly once and returns the pump's settle sender only when
/// the policy waits for acks.
fn wire_observer(
    observer: &mut dyn PumpObserver,
    finish: FinishPolicy,
    feedback: FeedbackSender,
    appender: AppendSender,
) -> Option<AppendSender> {
    let settle = (finish == FinishPolicy::AwaitSettle).then(|| appender.clone());
    observer.attach(feedback, appender);
    settle
}

#[allow(clippy::too_many_arguments)]
pub fn pump(
    frames: &Receiver<CaptureMsg>,
    targets: &Receiver<TargetEvent>,
    settle: Option<&AppendSender>,
    deadline: &Receiver<Instant>,
    shutdown: &Receiver<()>,
    to_record: &dyn Fn(&TargetShown, Option<Timestamp>) -> TargetRecord,
    sink: &mut dyn RecordSink,
    observer: &mut dyn PumpObserver,
) -> anyhow::Result<PumpEnd> {
    let mut pending: Option<TargetShown> = None;
    let end = loop {
        crossbeam_channel::select! {
            recv(frames) -> msg => match msg {
                Ok(CaptureMsg::Frame(frame)) => {
                    sink.frame(&frame)?;
                    observer.on_frame(&frame)?;
                }
                Ok(CaptureMsg::Failed { camera, error }) => anyhow::bail!("camera {camera} failed: {error}"),
                Err(_) => anyhow::bail!("all capture threads stopped"),
            },
            recv(targets) -> event => match event {
                Ok(TargetEvent::Shown(shown)) => {
                    observer.on_shown(&shown);
                    if let Some(prev) = pending.replace(shown) {
                        sink.target(&to_record(&prev, None))?;
                    }
                }
                Ok(TargetEvent::Hidden { index, at, .. }) => {
                    observer.on_hidden(index, at);
                    if let Some(settle) = settle {
                        settle.settle();
                    }
                    if pending.as_ref().is_some_and(|s| s.index == index)
                        && let Some(shown) = pending.take()
                    {
                        sink.target(&to_record(&shown, Some(at)))?;
                    }
                }
                Ok(TargetEvent::Finished) => break PumpEnd::Finished,
                Err(_) => anyhow::bail!("target overlay stopped before the sequence finished"),
            },
            recv(deadline) -> _ => break PumpEnd::Deadline,
            recv(shutdown) -> _ => break PumpEnd::Interrupted,
        }
    };
    if let Some(shown) = pending.take() {
        sink.target(&to_record(&shown, None))?;
    }
    Ok(end)
}

pub fn target_record(
    shown: &TargetShown,
    hidden: Option<Timestamp>,
    screen: &ScreenModel,
) -> TargetRecord {
    let mm = px_logical_to_mm(screen, &shown.px_logical);
    TargetRecord {
        seq: shown.index as u64,
        shown_ns: shown.shown_at.as_nanos(),
        hidden_ns: hidden.map(Timestamp::as_nanos),
        clock: shown.clock,
        output: shown.output.clone(),
        px_logical: [shown.px_logical.x, shown.px_logical.y],
        mm: [mm.x, mm.y],
    }
}

pub fn session_meta(
    id: &SessionId,
    cameras: &[(CameraInfo, Option<PathBuf>)],
    rig: &Rig,
    probe: &ProbeReport,
    emitter_on: bool,
    protocol: Option<ProtocolConfig>,
    now: SystemTime,
) -> anyhow::Result<SessionMeta> {
    let recorded_cameras = cameras
        .iter()
        .map(|(info, device)| RecordedCamera::from_info(info, device.as_deref()))
        .collect::<Result<Vec<_>, _>>()?;
    let created_unix_s = now.duration_since(UNIX_EPOCH)?.as_secs();
    Ok(SessionMeta {
        format_version: FORMAT_VERSION,
        session_id: id.clone(),
        created_unix_s,
        git_rev: Some(GIT_REV.to_string()),
        emitter: emitter_on.then_some(EmitterState::On),
        cameras: recorded_cameras,
        rig: Some(rig_to_table(rig)),
        probe: Some(toml::Table::try_from(probe)?),
        protocol,
    })
}

/// Shows the target sequence, pumping frames and targets into `sink` until it finishes.
fn record_targets(
    protocol: &TargetProtocol,
    active_rig: &Rig,
    capture: &Capture,
    shutdown: &Receiver<()>,
    to_record: &dyn Fn(&TargetShown, Option<Timestamp>) -> TargetRecord,
    sink: &mut dyn RecordSink,
    observer: &mut dyn PumpObserver,
) -> anyhow::Result<PumpEnd> {
    let output = active_rig.screen().output.clone();
    let specs: Vec<TargetSpec> = protocol
        .display_sequence(Timestamp::now().as_nanos(), active_rig.screen())
        .into_iter()
        .map(TargetSpec::from)
        .collect();
    let finish = observer.finish_policy();
    let (display, feedback) = TargetDisplay::spawn(&output, protocol.lead_in(), specs, finish)?;
    let settle = wire_observer(observer, finish, feedback, display.appender());
    let end = pump(
        capture.frames(),
        display.events(),
        settle.as_ref(),
        &crossbeam_channel::never(),
        shutdown,
        to_record,
        sink,
        observer,
    )?;
    if end == PumpEnd::Finished {
        capture.write_for(Duration::from_millis(250), sink)?;
        display.wait()?;
    } else {
        display.close()?;
    }
    Ok(end)
}

/// Records one session. Shared with `eye calibrate`.
pub fn record_session(
    config: &Config,
    location: &SessionLocation,
    opts: &RecordOptions,
    shutdown: &Receiver<()>,
    observer: &mut dyn PumpObserver,
) -> anyhow::Result<RecordSummary> {
    let _session =
        tracing::info_span!(span::SESSION, { field::SESSION_ID } = location.id.as_str()).entered();
    let protocol = opts.protocol.map(TargetProtocol::new).transpose()?;
    let probes = Probes::system();
    let report = collect(&probes);
    let output = rig::target_output(config, &report.outputs)?;
    let (active_rig, _) = rig::resolve_rig(config, output, &ProfileStore::open_default()?)?;
    let guards = emitter_guards(
        &rig::msxu_cameras(config),
        &report.cameras,
        EmitterGuard::enable,
    )?;
    let sources = eye::tracker::open_sources(config)?;
    let cameras: Vec<(CameraInfo, Option<PathBuf>)> = sources
        .iter()
        .zip(&config.cameras)
        .map(|(s, c)| (s.camera().clone(), Some(c.device.clone())))
        .collect();
    let camera_infos: Vec<CameraInfo> = cameras.iter().map(|(info, _)| info.clone()).collect();
    observer.prepare(&active_rig, &camera_infos)?;
    let meta = session_meta(
        &location.id,
        &cameras,
        &active_rig,
        &report,
        !guards.is_empty(),
        opts.protocol,
        SystemTime::now(),
    )?;
    let writer = SessionWriter::create(&location.root, &meta)?;
    std::fs::set_permissions(writer.dir(), Permissions::from_mode(0o700))?;
    tracing::warn!(
        "recording to {}: it contains images of your face (biometric data); keep it local, never upload or attach it",
        writer.dir().display()
    );
    let mut sink = SessionSink::new(writer);
    let capture = Capture::spawn(sources)?;
    capture.preroll(Duration::from_secs(3), &mut sink)?;
    let to_record =
        |s: &TargetShown, hidden: Option<Timestamp>| target_record(s, hidden, active_rig.screen());
    let end = match &protocol {
        Some(p) => record_targets(
            p,
            &active_rig,
            &capture,
            shutdown,
            &to_record,
            &mut sink,
            observer,
        )?,
        None => {
            let deadline = opts
                .duration
                .map_or_else(crossbeam_channel::never, crossbeam_channel::after);
            pump(
                capture.frames(),
                &crossbeam_channel::never(),
                None,
                &deadline,
                shutdown,
                &to_record,
                &mut sink,
                observer,
            )?
        }
    };
    let dropped_frames = capture.stop_and_drain(&mut sink)?;
    let (dir, frames, targets) = sink.finish()?;
    drop(guards);
    Ok(RecordSummary {
        dir,
        frames,
        dropped_frames,
        targets,
        interrupted: end == PumpEnd::Interrupted,
    })
}

fn dir_size(path: &Path) -> anyhow::Result<u64> {
    let mut total = 0u64;
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            total += dir_size(&entry.path())?;
        } else if file_type.is_file() {
            total += entry.metadata()?.len();
        }
    }
    Ok(total)
}

pub fn run(ctx: &Ctx, args: Args) -> anyhow::Result<()> {
    let location = SessionLocation::resolve(ctx.output.as_deref())?;
    let opts = RecordOptions::from_args(&args)?;
    let config = Config::load(ctx.config_path.as_deref())?;
    let shutdown = shutdown::install()?;
    let summary = record_session(
        &config,
        &location,
        &opts,
        shutdown.receiver(),
        &mut NoopObserver,
    )?;

    let frames = summary
        .frames
        .iter()
        .map(|(id, n)| format!("{id} {n} frames"))
        .collect::<Vec<_>>()
        .join(", ");
    let mb = dir_size(&summary.dir)? as f64 / 1_000_000.0;
    print!(
        "recorded {}: {frames}; {} targets; {} dropped; {mb:.1} MB",
        summary.dir.display(),
        summary.targets,
        summary.dropped_frames,
    );
    if summary.interrupted {
        tracing::warn!(
            "recording to {} was interrupted; it is shorter than requested",
            summary.dir.display()
        );
        print!("; interrupted");
    }
    println!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use eye_calibration::store::rig_to_table;
    use eye_core::session::TargetClock;
    use eye_core::{CameraId, CameraModel, FrameHeader, Illumination, OutputId, PixelFormat};
    use eye_overlay::targets::AppendMsg;
    use eye_platform::{Compositor, SessionInfo, SessionType};
    use nalgebra::{Isometry3, Point2, Vector2};

    use super::*;
    use crate::testing::edp1;

    struct FakeSink {
        frames: Vec<(String, u64)>,
        targets: Vec<TargetRecord>,
    }

    impl FakeSink {
        fn new() -> Self {
            Self {
                frames: Vec::new(),
                targets: Vec::new(),
            }
        }
    }

    impl RecordSink for FakeSink {
        fn frame(&mut self, frame: &Frame) -> anyhow::Result<()> {
            self.frames
                .push((frame.header().camera.to_string(), frame.header().seq));
            Ok(())
        }

        fn target(&mut self, record: &TargetRecord) -> anyhow::Result<()> {
            self.targets.push(record.clone());
            Ok(())
        }
    }

    fn frame(camera: &str, seq: u64, t_ns: u64) -> Frame {
        Frame::new(
            FrameHeader {
                camera: CameraId::from(camera),
                seq,
                timestamp: Timestamp::from_nanos(t_ns),
                width: 2,
                height: 2,
                format: PixelFormat::Gray8,
                illumination: Illumination::Unknown,
            },
            Arc::from(vec![0u8; 4]),
        )
        .expect("valid frame")
    }

    fn screen_fixture() -> ScreenModel {
        ScreenModel {
            output: OutputId::from("eDP-1"),
            size_mm: Vector2::new(310.0, 170.0),
            size_px: (3840, 2160),
            scale: 2.0,
        }
    }

    fn shown(index: usize, at_ns: u64) -> TargetShown {
        TargetShown {
            index,
            output: OutputId::from("eDP-1"),
            px_logical: Point2::new(0.0, 0.0),
            shown_at: Timestamp::from_nanos(at_ns),
            clock: TargetClock::Presentation,
        }
    }

    fn to_record(shown: &TargetShown, hidden: Option<Timestamp>) -> TargetRecord {
        target_record(shown, hidden, &screen_fixture())
    }

    #[test]
    fn test_target_count_accepts_16_9_0() {
        assert_eq!(parse_target_count("16"), Ok(16));
        assert_eq!(parse_target_count("9"), Ok(9));
        assert_eq!(parse_target_count("0"), Ok(0));
        assert_eq!(
            parse_target_count("4"),
            Err("allowed: 16, 9, 0".to_string())
        );
    }

    #[test]
    fn test_options_zero_targets_requires_duration() {
        let args = Args {
            targets: 0,
            dwell_ms: None,
            duration_s: None,
        };
        let err = RecordOptions::from_args(&args).unwrap_err();
        assert!(err.to_string().contains("--duration-s is required"));

        let args = Args {
            targets: 0,
            dwell_ms: None,
            duration_s: Some(5),
        };
        let opts = RecordOptions::from_args(&args).unwrap();
        assert_eq!(opts.protocol, None);
        assert_eq!(opts.duration, Some(Duration::from_secs(5)));
    }

    #[test]
    fn test_options_duration_with_targets_is_error() {
        let args = Args {
            targets: 16,
            dwell_ms: None,
            duration_s: Some(5),
        };
        let err = RecordOptions::from_args(&args).unwrap_err();
        assert_eq!(
            err.to_string(),
            "--duration-s is only used with --targets 0"
        );
    }

    #[test]
    fn test_options_default_is_4x4_protocol() {
        let args = Args {
            targets: 16,
            dwell_ms: None,
            duration_s: None,
        };
        let opts = RecordOptions::from_args(&args).unwrap();
        assert_eq!(
            opts.protocol,
            Some(ProtocolConfig {
                grid: [4, 4],
                ..ProtocolConfig::default()
            })
        );

        let args = Args {
            targets: 9,
            dwell_ms: Some(2000),
            duration_s: None,
        };
        let opts = RecordOptions::from_args(&args).unwrap();
        let protocol = opts.protocol.unwrap();
        assert_eq!(protocol.grid, [3, 3]);
        assert_eq!(protocol.dwell_ms, 2000);
    }

    #[test]
    fn test_session_location_default_and_output() {
        let loc = SessionLocation::resolve(None).unwrap();
        assert_eq!(loc.root, PathBuf::from("recordings"));

        let loc = SessionLocation::resolve(Some(Path::new("/tmp/x/s1"))).unwrap();
        assert_eq!(loc.root, PathBuf::from("/tmp/x"));
        assert_eq!(loc.id.as_str(), "s1");

        let err = SessionLocation::resolve(Some(Path::new("/tmp/x/.hidden"))).unwrap_err();
        assert!(err.to_string().contains("invalid session id"));
    }

    #[test]
    fn test_pump_writes_frames_and_targets_until_finished() {
        let (frames_tx, frames_rx) = crossbeam_channel::unbounded();
        frames_tx
            .send(CaptureMsg::Frame(frame("ir", 0, 0)))
            .unwrap();
        frames_tx
            .send(CaptureMsg::Frame(frame("ir", 1, 1)))
            .unwrap();
        frames_tx
            .send(CaptureMsg::Frame(frame("ir", 2, 2)))
            .unwrap();

        let (targets_tx, targets_rx) = crossbeam_channel::unbounded();
        targets_tx.send(TargetEvent::Shown(shown(0, 10))).unwrap();
        targets_tx
            .send(TargetEvent::Hidden {
                index: 0,
                at: Timestamp::from_nanos(20),
                clock: TargetClock::Presentation,
            })
            .unwrap();
        targets_tx.send(TargetEvent::Shown(shown(1, 20))).unwrap();
        targets_tx.send(TargetEvent::Finished).unwrap();

        let mut sink = FakeSink::new();
        let end = pump(
            &frames_rx,
            &targets_rx,
            None,
            &crossbeam_channel::never(),
            &crossbeam_channel::never(),
            &to_record,
            &mut sink,
            &mut NoopObserver,
        )
        .unwrap();

        assert_eq!(end, PumpEnd::Finished);
        assert_eq!(
            sink.targets,
            vec![
                to_record(&shown(0, 10), Some(Timestamp::from_nanos(20))),
                to_record(&shown(1, 20), None),
            ]
        );
        let queued = [
            ("ir".to_string(), 0u64),
            ("ir".to_string(), 1u64),
            ("ir".to_string(), 2u64),
        ];
        assert!(queued.starts_with(&sink.frames));
        drop(frames_tx);
    }

    #[test]
    fn test_pump_pending_target_written_at_end() {
        let (_frames_tx, frames_rx) = crossbeam_channel::unbounded::<CaptureMsg>();
        let (targets_tx, targets_rx) = crossbeam_channel::unbounded();
        targets_tx.send(TargetEvent::Shown(shown(0, 10))).unwrap();
        let deadline = crossbeam_channel::after(Duration::from_millis(200));

        let mut sink = FakeSink::new();
        let end = pump(
            &frames_rx,
            &targets_rx,
            None,
            &deadline,
            &crossbeam_channel::never(),
            &to_record,
            &mut sink,
            &mut NoopObserver,
        )
        .unwrap();

        assert_eq!(end, PumpEnd::Deadline);
        assert_eq!(sink.targets.len(), 1);
        assert_eq!(sink.targets[0].hidden_ns, None);
        drop(targets_tx);
    }

    #[test]
    fn test_pump_stops_on_shutdown() {
        let (_frames_tx, frames_rx) = crossbeam_channel::unbounded::<CaptureMsg>();
        let (_targets_tx, targets_rx) = crossbeam_channel::unbounded::<TargetEvent>();
        let (shutdown_tx, shutdown_rx) = crossbeam_channel::bounded(1);
        shutdown_tx.send(()).unwrap();

        let mut sink = FakeSink::new();
        let end = pump(
            &frames_rx,
            &targets_rx,
            None,
            &crossbeam_channel::never(),
            &shutdown_rx,
            &to_record,
            &mut sink,
            &mut NoopObserver,
        )
        .unwrap();

        assert_eq!(end, PumpEnd::Interrupted);
        assert!(sink.targets.is_empty());
    }

    #[test]
    fn test_pump_overlay_disconnect_is_error() {
        let (_frames_tx, frames_rx) = crossbeam_channel::unbounded::<CaptureMsg>();
        let (targets_tx, targets_rx) = crossbeam_channel::unbounded::<TargetEvent>();
        drop(targets_tx);

        let mut sink = FakeSink::new();
        let err = pump(
            &frames_rx,
            &targets_rx,
            None,
            &crossbeam_channel::never(),
            &crossbeam_channel::never(),
            &to_record,
            &mut sink,
            &mut NoopObserver,
        )
        .unwrap_err();

        assert_eq!(
            err.to_string(),
            "target overlay stopped before the sequence finished"
        );
    }

    #[test]
    fn test_pump_stops_on_deadline() {
        let (_frames_tx, frames_rx) = crossbeam_channel::unbounded::<CaptureMsg>();
        let (_targets_tx, targets_rx) = crossbeam_channel::unbounded::<TargetEvent>();
        let deadline = crossbeam_channel::after(Duration::from_millis(10));

        let mut sink = FakeSink::new();
        let end = pump(
            &frames_rx,
            &targets_rx,
            None,
            &deadline,
            &crossbeam_channel::never(),
            &to_record,
            &mut sink,
            &mut NoopObserver,
        )
        .unwrap();

        assert_eq!(end, PumpEnd::Deadline);
    }

    #[test]
    fn test_pump_errors_when_capture_threads_stop() {
        let (frames_tx, frames_rx) = crossbeam_channel::unbounded();
        frames_tx
            .send(CaptureMsg::Frame(frame("ir", 0, 0)))
            .unwrap();
        frames_tx
            .send(CaptureMsg::Frame(frame("ir", 1, 1)))
            .unwrap();
        frames_tx
            .send(CaptureMsg::Frame(frame("ir", 2, 2)))
            .unwrap();
        drop(frames_tx);
        let targets_rx = crossbeam_channel::never();

        let mut sink = FakeSink::new();
        let err = pump(
            &frames_rx,
            &targets_rx,
            None,
            &crossbeam_channel::never(),
            &crossbeam_channel::never(),
            &to_record,
            &mut sink,
            &mut NoopObserver,
        )
        .unwrap_err();

        assert_eq!(err.to_string(), "all capture threads stopped");
        assert_eq!(sink.frames.len(), 3);
    }

    #[derive(Default)]
    struct CountingObserver {
        seqs: Vec<u64>,
    }

    impl PumpObserver for CountingObserver {
        fn on_frame(&mut self, frame: &Frame) -> anyhow::Result<()> {
            self.seqs.push(frame.header().seq);
            Ok(())
        }
    }

    #[test]
    fn test_pump_calls_on_frame_for_every_frame() {
        let (frames_tx, frames_rx) = crossbeam_channel::unbounded();
        for seq in 0..3 {
            frames_tx
                .send(CaptureMsg::Frame(frame("ir", seq, seq)))
                .unwrap();
        }
        drop(frames_tx);
        let targets_rx = crossbeam_channel::never();

        let mut sink = FakeSink::new();
        let mut observer = CountingObserver::default();
        let err = pump(
            &frames_rx,
            &targets_rx,
            None,
            &crossbeam_channel::never(),
            &crossbeam_channel::never(),
            &to_record,
            &mut sink,
            &mut observer,
        )
        .unwrap_err();

        assert_eq!(err.to_string(), "all capture threads stopped");
        assert_eq!(observer.seqs, vec![0, 1, 2]);
    }

    #[test]
    fn test_pump_capture_failure_is_error() {
        let (frames_tx, frames_rx) = crossbeam_channel::unbounded();
        frames_tx
            .send(CaptureMsg::Failed {
                camera: "ir".to_string(),
                error: "gone".to_string(),
            })
            .unwrap();
        let targets_rx = crossbeam_channel::never();

        let mut sink = FakeSink::new();
        let err = pump(
            &frames_rx,
            &targets_rx,
            None,
            &crossbeam_channel::never(),
            &crossbeam_channel::never(),
            &to_record,
            &mut sink,
            &mut NoopObserver,
        )
        .unwrap_err();

        assert_eq!(err.to_string(), "camera ir failed: gone");
    }

    #[test]
    fn test_pump_acks_every_hidden_even_if_observer_does_not() {
        let (_frames_tx, frames_rx) = crossbeam_channel::unbounded::<CaptureMsg>();
        let (targets_tx, targets_rx) = crossbeam_channel::unbounded();
        targets_tx.send(TargetEvent::Shown(shown(0, 10))).unwrap();
        targets_tx
            .send(TargetEvent::Hidden {
                index: 0,
                at: Timestamp::from_nanos(20),
                clock: TargetClock::Presentation,
            })
            .unwrap();
        targets_tx.send(TargetEvent::Shown(shown(1, 30))).unwrap();
        targets_tx
            .send(TargetEvent::Hidden {
                index: 1,
                at: Timestamp::from_nanos(40),
                clock: TargetClock::Presentation,
            })
            .unwrap();
        targets_tx.send(TargetEvent::Finished).unwrap();

        let (append_tx, append_rx) = crossbeam_channel::unbounded();
        let settle = AppendSender::new(append_tx);

        let mut sink = FakeSink::new();
        let end = pump(
            &frames_rx,
            &targets_rx,
            Some(&settle),
            &crossbeam_channel::never(),
            &crossbeam_channel::never(),
            &to_record,
            &mut sink,
            &mut NoopObserver,
        )
        .unwrap();

        assert_eq!(end, PumpEnd::Finished);
        let acks: Vec<_> = append_rx.try_iter().collect();
        assert_eq!(acks.iter().filter(|m| **m == AppendMsg::Settled).count(), 2);
        assert!(!acks.iter().any(|m| matches!(m, AppendMsg::Append(_))));
    }

    #[test]
    fn test_pump_without_settle_sender_sends_nothing() {
        let (_frames_tx, frames_rx) = crossbeam_channel::unbounded::<CaptureMsg>();
        let (targets_tx, targets_rx) = crossbeam_channel::unbounded();
        targets_tx.send(TargetEvent::Shown(shown(0, 10))).unwrap();
        targets_tx
            .send(TargetEvent::Hidden {
                index: 0,
                at: Timestamp::from_nanos(20),
                clock: TargetClock::Presentation,
            })
            .unwrap();
        targets_tx.send(TargetEvent::Finished).unwrap();

        let (_append_tx, append_rx) = crossbeam_channel::unbounded::<AppendMsg>();

        let mut sink = FakeSink::new();
        let end = pump(
            &frames_rx,
            &targets_rx,
            None,
            &crossbeam_channel::never(),
            &crossbeam_channel::never(),
            &to_record,
            &mut sink,
            &mut NoopObserver,
        )
        .unwrap();

        assert_eq!(end, PumpEnd::Finished);
        assert_eq!(append_rx.try_iter().count(), 0);
    }

    struct AppendingObserver {
        sender: AppendSender,
    }

    impl PumpObserver for AppendingObserver {
        fn on_hidden(&mut self, _index: usize, _at: Timestamp) {
            self.sender
                .append(TargetSpec {
                    px_logical: Point2::new(1.0, 1.0),
                    timing: eye_core::session::TargetTiming {
                        settle: Duration::from_millis(100),
                        window: Duration::from_millis(150),
                        dwell: Duration::from_millis(250),
                    },
                    retry: true,
                })
                .unwrap();
        }
    }

    #[test]
    fn test_pump_settles_after_observer_append() {
        let (_frames_tx, frames_rx) = crossbeam_channel::unbounded::<CaptureMsg>();
        let (targets_tx, targets_rx) = crossbeam_channel::unbounded();
        targets_tx.send(TargetEvent::Shown(shown(0, 10))).unwrap();
        targets_tx
            .send(TargetEvent::Hidden {
                index: 0,
                at: Timestamp::from_nanos(20),
                clock: TargetClock::Presentation,
            })
            .unwrap();
        targets_tx.send(TargetEvent::Finished).unwrap();

        let (append_tx, append_rx) = crossbeam_channel::unbounded();
        let settle = AppendSender::new(append_tx.clone());
        let mut observer = AppendingObserver {
            sender: AppendSender::new(append_tx),
        };

        let mut sink = FakeSink::new();
        let end = pump(
            &frames_rx,
            &targets_rx,
            Some(&settle),
            &crossbeam_channel::never(),
            &crossbeam_channel::never(),
            &to_record,
            &mut sink,
            &mut observer,
        )
        .unwrap();

        assert_eq!(end, PumpEnd::Finished);
        let acks: Vec<_> = append_rx.try_iter().collect();
        assert_eq!(acks.len(), 2);
        assert!(matches!(acks[0], AppendMsg::Append(_)));
        assert_eq!(acks[1], AppendMsg::Settled);
    }

    #[derive(Default)]
    struct AttachCountingObserver {
        finish: FinishPolicy,
        attach_calls: u32,
    }

    impl PumpObserver for AttachCountingObserver {
        fn finish_policy(&self) -> FinishPolicy {
            self.finish
        }

        fn attach(&mut self, _feedback: FeedbackSender, _append: AppendSender) {
            self.attach_calls += 1;
        }
    }

    #[test]
    fn test_wire_observer_attaches_once_and_returns_settle_only_for_await_settle() {
        let (feedback_tx, _feedback_rx) = crossbeam_channel::unbounded();
        let feedback = FeedbackSender::new(feedback_tx);
        let (append_tx, append_rx) = crossbeam_channel::unbounded();
        let appender = AppendSender::new(append_tx);

        let mut immediate = AttachCountingObserver {
            finish: FinishPolicy::Immediate,
            attach_calls: 0,
        };
        let settle = wire_observer(
            &mut immediate,
            FinishPolicy::Immediate,
            feedback.clone(),
            appender.clone(),
        );
        assert!(settle.is_none());
        assert_eq!(immediate.attach_calls, 1);

        let mut await_settle = AttachCountingObserver {
            finish: FinishPolicy::AwaitSettle,
            attach_calls: 0,
        };
        let settle = wire_observer(
            &mut await_settle,
            FinishPolicy::AwaitSettle,
            feedback,
            appender,
        );
        assert_eq!(await_settle.attach_calls, 1);
        let settle = settle.expect("await-settle policy returns a settle sender");
        settle.settle();
        assert_eq!(append_rx.try_recv().unwrap(), AppendMsg::Settled);
    }

    #[test]
    fn test_target_record_maps_px_to_mm() {
        let shown = TargetShown {
            index: 3,
            output: OutputId::from("eDP-1"),
            px_logical: Point2::new(960.0, 540.0),
            shown_at: Timestamp::from_nanos(5),
            clock: TargetClock::Commit,
        };
        let record = target_record(&shown, Some(Timestamp::from_nanos(7)), &screen_fixture());

        assert_eq!(record.seq, 3);
        assert_eq!(record.shown_ns, 5);
        assert_eq!(record.hidden_ns, Some(7));
        assert_eq!(record.clock, TargetClock::Commit);
        assert_eq!(record.output, OutputId::from("eDP-1"));
        assert_eq!(record.px_logical, [960.0, 540.0]);
        assert!((record.mm[0] - 155.0).abs() < 1e-9);
        assert!((record.mm[1] - 85.0).abs() < 1e-9);
    }

    fn probe_report() -> ProbeReport {
        ProbeReport {
            session: SessionInfo {
                session_type: SessionType::Wayland,
                wayland_display: None,
                compositor: Compositor::Unknown,
                runtime_dir: None,
            },
            display_backend: Some("wayland"),
            outputs: vec![edp1()],
            display_error: None,
            cameras: vec![],
            camera_error: None,
            emitters: vec![],
            hardware_profile: None,
        }
    }

    fn rig_fixture() -> Rig {
        Rig::new(
            vec![CameraModel {
                id: CameraId::from("ir"),
                width: 640,
                height: 360,
                fx: 500.0,
                fy: 500.0,
                cx: 320.0,
                cy: 180.0,
                distortion: [0.0; 5],
                screen_from_camera: Isometry3::identity(),
            }],
            screen_fixture(),
        )
        .expect("valid rig")
    }

    fn cameras_fixture() -> Vec<(CameraInfo, Option<PathBuf>)> {
        vec![
            (
                CameraInfo {
                    id: CameraId::from("rgb"),
                    width: 1280,
                    height: 720,
                    format: PixelFormat::Mjpeg,
                    frame_interval: Duration::from_nanos(33_333_333),
                },
                Some(PathBuf::from("/dev/video0")),
            ),
            (
                CameraInfo {
                    id: CameraId::from("ir"),
                    width: 640,
                    height: 360,
                    format: PixelFormat::Gray8,
                    frame_interval: Duration::from_nanos(33_333_333),
                },
                Some(PathBuf::from("/dev/video2")),
            ),
        ]
    }

    #[test]
    fn test_session_meta_contains_git_rev_probe_and_rig() {
        let id = SessionId::new("20261007T221500Z").unwrap();
        let report = probe_report();
        let active_rig = rig_fixture();
        let now = UNIX_EPOCH + Duration::from_secs(1_791_409_623);

        let meta = session_meta(
            &id,
            &cameras_fixture(),
            &active_rig,
            &report,
            true,
            None,
            now,
        )
        .unwrap();
        assert_eq!(meta.git_rev, Some(GIT_REV.to_string()));
        assert_eq!(meta.created_unix_s, 1_791_409_623);
        assert_eq!(meta.format_version, 1);
        let probe_table = meta.probe.as_ref().unwrap();
        assert_eq!(probe_table["outputs"][0]["name"].as_str(), Some("eDP-1"));
        assert_eq!(meta.rig, Some(rig_to_table(&active_rig)));
        assert_eq!(meta.emitter, Some(EmitterState::On));

        let meta_off = session_meta(
            &id,
            &cameras_fixture(),
            &active_rig,
            &report,
            false,
            None,
            now,
        )
        .unwrap();
        assert_eq!(meta_off.emitter, None);
    }

    #[test]
    fn test_session_meta_records_protocol() {
        let id = SessionId::new("20261007T221500Z").unwrap();
        let report = probe_report();
        let active_rig = rig_fixture();
        let now = UNIX_EPOCH + Duration::from_secs(1_791_409_623);
        let protocol = ProtocolConfig {
            grid: [4, 4],
            ..ProtocolConfig::default()
        };

        let meta = session_meta(
            &id,
            &cameras_fixture(),
            &active_rig,
            &report,
            true,
            Some(protocol),
            now,
        )
        .unwrap();
        assert_eq!(meta.protocol, Some(protocol));

        let meta_without = session_meta(
            &id,
            &cameras_fixture(),
            &active_rig,
            &report,
            true,
            None,
            now,
        )
        .unwrap();
        assert_eq!(meta_without.protocol, None);
    }
}
