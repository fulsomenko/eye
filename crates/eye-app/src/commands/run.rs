use std::time::{Duration, Instant};

use crossbeam_channel::Receiver;
use eye::config::{OutputConfig, OutputMode};
use eye::tracker::{Tracker, TrackerStats};
use eye_calibration::correction::{UserProfile, rig_fingerprint};
use eye_calibration::store::ProfileStore;
use eye_core::stage::GazeCorrection;
use eye_core::{GazePoint, GazeSink, Rig};
use eye_overlay::sink::{LayerShellOverlay, OverlayMode, OverlayOptions};
use eye_platform::EmitterGuard;

use crate::commands::emitter::emitter_guards;
use crate::commands::probe::{Probes, collect};
use crate::ctx::Ctx;
use crate::rig;

#[derive(Debug, clap::Args)]
pub struct Args {
    /// Overrides [output].mode
    #[arg(long, value_enum)]
    pub mode: Option<Mode>,
    /// Region grid, e.g. 4x4; overrides [output].grid (region mode only)
    #[arg(long, value_parser = parse_grid, value_name = "COLSxROWS")]
    pub grid: Option<(u32, u32)>,
    /// User profile from `eye calibrate`
    #[arg(long, default_value = "default", conflicts_with = "no_profile")]
    pub profile: String,
    /// Run uncalibrated
    #[arg(long)]
    pub no_profile: bool,
    /// Log live latency (frame timestamp to emit) every 5 s
    #[arg(long)]
    pub stats: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Mode {
    Point,
    Region,
}

pub fn parse_grid(s: &str) -> Result<(u32, u32), String> {
    let (cols, rows) = s
        .split_once('x')
        .ok_or_else(|| format!("expected COLSxROWS, got \"{s}\""))?;
    let cols: u32 = cols
        .parse()
        .map_err(|_| format!("expected COLSxROWS, got \"{s}\""))?;
    let rows: u32 = rows
        .parse()
        .map_err(|_| format!("expected COLSxROWS, got \"{s}\""))?;
    if !(1..=16).contains(&cols) || !(1..=16).contains(&rows) {
        return Err(format!("cols and rows must be 1 to 16, got \"{s}\""));
    }
    Ok((cols, rows))
}

pub fn apply_output_overrides(output: &mut OutputConfig, args: &Args) -> anyhow::Result<()> {
    if let Some(mode) = args.mode {
        output.mode = match mode {
            Mode::Point => OutputMode::Point,
            Mode::Region => OutputMode::Region,
        };
    }
    if let Some((cols, rows)) = args.grid {
        anyhow::ensure!(
            output.mode == OutputMode::Region,
            "--grid only applies to region mode"
        );
        output.grid = [cols, rows];
    }
    Ok(())
}

pub fn overlay_mode(output: &OutputConfig) -> OverlayMode {
    match output.mode {
        OutputMode::Point => OverlayMode::Point,
        OutputMode::Region => OverlayMode::Region {
            cols: output.grid[0],
            rows: output.grid[1],
        },
    }
}

/// `--no-profile`: skips the store entirely. Otherwise the stored profile (or `None`) plus
/// warnings for a missing profile, a different rig fingerprint and a different estimator.
pub fn select_profile(
    store: &ProfileStore,
    args: &Args,
    rig: &Rig,
    estimator: &str,
) -> anyhow::Result<(Option<UserProfile>, Vec<String>)> {
    if args.no_profile {
        return Ok((None, Vec::new()));
    }
    let mut warnings = Vec::new();
    let Some(profile) = store.load_profile(&args.profile)? else {
        warnings.push(format!(
            "no profile \"{}\": uncalibrated gaze is typically 4 to 5 deg off; run eye calibrate",
            args.profile
        ));
        return Ok((None, warnings));
    };
    if profile.rig_fingerprint != rig_fingerprint(rig) {
        warnings.push(format!(
            "profile \"{}\" was fitted on another rig; run eye calibrate again",
            args.profile
        ));
    }
    if profile.estimator != estimator {
        warnings.push(format!(
            "profile \"{}\" was fitted with estimator \"{}\", running \"{estimator}\"; run eye calibrate again",
            args.profile, profile.estimator
        ));
    }
    Ok((Some(profile), warnings))
}

pub fn stats_line(stats: &TrackerStats, window: Duration, points_before: u64) -> String {
    let points = stats.points_emitted - points_before;
    let frames_dropped: u64 = stats.frames_dropped.values().sum();
    let secs = window.as_secs_f64();
    let rate = if secs > 0.0 {
        points as f64 / secs
    } else {
        0.0
    };
    format!(
        "{points} points in {secs:.1} s ({rate:.1}/s), latency p50 {p50:.1} ms, p95 {p95:.1} ms, max {max:.1} ms, {frames_dropped} frames dropped",
        p50 = stats.capture_to_emit.p50.as_secs_f64() * 1000.0,
        p95 = stats.capture_to_emit.p95.as_secs_f64() * 1000.0,
        max = stats.capture_to_emit.max.as_secs_f64() * 1000.0,
    )
}

#[derive(Debug)]
pub struct StatsTicker {
    log: bool,
    interval: Duration,
    last: Instant,
    last_points: u64,
}

impl StatsTicker {
    pub fn new(log: bool, interval: Duration, now: Instant) -> Self {
        Self {
            log,
            interval,
            last: now,
            last_points: 0,
        }
    }

    pub fn due(&self, now: Instant) -> bool {
        now.duration_since(self.last) >= self.interval
    }

    /// Restarts the window; returns `stats_line(..)` if `log`.
    pub fn report(&mut self, stats: &TrackerStats, now: Instant) -> Option<String> {
        let window = now.duration_since(self.last);
        let points_before = self.last_points;
        self.last = now;
        self.last_points = stats.points_emitted;
        self.log.then(|| stats_line(stats, window, points_before))
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum LoopEnd {
    Shutdown,
    TrackerStopped,
    OverlayClosed(String),
}

pub fn run_loop(
    points: &Receiver<GazePoint>,
    sink: &mut dyn GazeSink,
    shutdown: &Receiver<()>,
    ticker: &mut StatsTicker,
    stats: &dyn Fn() -> TrackerStats,
    now: &dyn Fn() -> Instant,
) -> LoopEnd {
    loop {
        crossbeam_channel::select! {
            recv(shutdown) -> _ => return LoopEnd::Shutdown,
            recv(points) -> msg => match msg {
                Ok(point) => {
                    if let Err(err) = sink.push(&point) {
                        return LoopEnd::OverlayClosed(err.to_string());
                    }
                }
                Err(_) => return LoopEnd::TrackerStopped,
            },
            default(Duration::from_millis(100)) => {}
        }
        let t = now();
        if ticker.due(t)
            && let Some(line) = ticker.report(&stats(), t)
        {
            tracing::info!("{line}");
        }
    }
}

pub fn run(ctx: &Ctx, args: Args) -> anyhow::Result<()> {
    ctx.reject_output("run")?;
    let mut config = eye::config::Config::load(ctx.config_path.as_deref())?;
    apply_output_overrides(&mut config.output, &args)?;
    config.validate()?;
    let probes = Probes::system();
    let report = collect(&probes);
    let output = rig::target_output(&config, &report.outputs)?;
    let store = ProfileStore::open_default()?;
    let (rig, _) = rig::resolve_rig(&config, output, &store)?;
    let (profile, warnings) = select_profile(&store, &args, &rig, &config.estimate.kind)?;
    for warning in &warnings {
        tracing::warn!("{warning}");
    }
    let correction = profile.map(|p| Box::new(p) as Box<dyn GazeCorrection>);
    let shutdown = crate::shutdown::install()?;
    let guards = emitter_guards(
        &rig::msxu_cameras(&config),
        &report.cameras,
        EmitterGuard::enable,
    )?;
    let mut overlay = LayerShellOverlay::spawn(OverlayOptions::new(
        rig.screen().output.clone(),
        overlay_mode(&config.output),
        rig.screen(),
    ))?;
    let tracker = Tracker::from_config(&config, rig, correction)?;
    let points = tracker.subscribe();
    let mut ticker = StatsTicker::new(args.stats, Duration::from_secs(5), Instant::now());
    let end = run_loop(
        &points,
        &mut overlay,
        &shutdown,
        &mut ticker,
        &|| tracker.stats(),
        &Instant::now,
    );
    drop(points);
    let tracker_result = tracker.shutdown();
    let overlay_result = overlay.shutdown();
    drop(guards);
    match end {
        LoopEnd::Shutdown => {
            tracker_result?;
            overlay_result?;
            Ok(())
        }
        LoopEnd::TrackerStopped => match tracker_result {
            Err(err) => Err(anyhow::Error::new(err).context("tracker stopped")),
            Ok(()) => anyhow::bail!("tracker stopped without an error"),
        },
        LoopEnd::OverlayClosed(reason) => anyhow::bail!("overlay closed: {reason}"),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use eye_bench::testing::synthetic_rig;
    use eye_core::{OutputId, SinkError, Timestamp};
    use nalgebra::{Matrix2, Point2};

    use super::*;

    struct FakeSink {
        points: Vec<GazePoint>,
        closed: bool,
    }

    impl FakeSink {
        fn new(closed: bool) -> Self {
            Self {
                points: Vec::new(),
                closed,
            }
        }
    }

    impl GazeSink for FakeSink {
        fn name(&self) -> &'static str {
            "fake"
        }

        fn push(&mut self, point: &GazePoint) -> Result<(), SinkError> {
            if self.closed {
                return Err(SinkError::Closed);
            }
            self.points.push(point.clone());
            Ok(())
        }
    }

    fn point(ts_ms: u64) -> GazePoint {
        GazePoint {
            timestamp: Timestamp::from_nanos(ts_ms * 1_000_000),
            output: OutputId::from("eDP-1"),
            mm: Point2::new(0.0, 0.0),
            px_physical: Point2::new(0.0, 0.0),
            px_logical: Point2::new(0.0, 0.0),
            cov_mm: Matrix2::identity(),
            confidence: 1.0,
        }
    }

    #[test]
    fn test_parse_grid_accepts_cols_x_rows() {
        assert_eq!(parse_grid("4x4"), Ok((4, 4)));
        assert_eq!(parse_grid("3x2"), Ok((3, 2)));
    }

    #[test]
    fn test_parse_grid_rejects_bad_input() {
        assert!(parse_grid("4").is_err());
        assert!(parse_grid("0x3").is_err());
        assert!(parse_grid("17x2").is_err());
        assert!(parse_grid("axb").is_err());
    }

    fn args_with(mode: Option<Mode>, grid: Option<(u32, u32)>) -> Args {
        Args {
            mode,
            grid,
            profile: "default".to_string(),
            no_profile: false,
            stats: false,
        }
    }

    #[test]
    fn test_output_overrides_apply_mode_and_grid() {
        let mut output = OutputConfig {
            target: Some("eDP-1".to_string()),
            ..Default::default()
        };
        let args = args_with(Some(Mode::Region), Some((3, 3)));
        apply_output_overrides(&mut output, &args).expect("overrides apply");
        assert_eq!(output.mode, OutputMode::Region);
        assert_eq!(output.grid, [3, 3]);
        assert_eq!(output.target, Some("eDP-1".to_string()));
    }

    #[test]
    fn test_output_overrides_grid_with_point_mode_is_error() {
        let mut output = OutputConfig::default();
        let args = args_with(None, Some((3, 3)));
        let err =
            apply_output_overrides(&mut output, &args).expect_err("grid without region errors");
        assert_eq!(err.to_string(), "--grid only applies to region mode");
    }

    #[test]
    fn test_overlay_mode_from_output_config() {
        let output = OutputConfig {
            mode: OutputMode::Region,
            grid: [4, 3],
            ..Default::default()
        };
        assert_eq!(
            overlay_mode(&output),
            OverlayMode::Region { cols: 4, rows: 3 }
        );

        let output = OutputConfig {
            mode: OutputMode::Point,
            ..Default::default()
        };
        assert_eq!(overlay_mode(&output), OverlayMode::Point);
    }

    #[test]
    fn test_select_profile_missing_warns_and_is_none() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = ProfileStore::at(dir.path());
        let args = args_with(None, None);
        let (profile, warnings) =
            select_profile(&store, &args, &synthetic_rig(), "fused").expect("resolves");
        assert_eq!(profile, None);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("no profile \"default\""));
    }

    #[test]
    fn test_select_profile_no_profile_skips_store() {
        let store = ProfileStore::at("/nonexistent/eye-store");
        let mut args = args_with(None, None);
        args.no_profile = true;
        let result = select_profile(&store, &args, &synthetic_rig(), "fused").expect("resolves");
        assert_eq!(result, (None, Vec::new()));
    }

    #[test]
    fn test_profile_mismatch_warns() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = ProfileStore::at(dir.path());
        let profile = UserProfile {
            version: 1,
            name: "default".to_string(),
            created_unix_s: 0,
            rig_fingerprint: "0000".to_string(),
            estimator: "ir-pupil".to_string(),
            eyes: BTreeMap::new(),
        };
        store
            .save_profile("default", &profile)
            .expect("save profile");
        let args = args_with(None, None);

        let (found, warnings) =
            select_profile(&store, &args, &synthetic_rig(), "fused").expect("resolves");
        assert_eq!(found, Some(profile.clone()));
        assert_eq!(warnings.len(), 2);
        assert!(warnings[0].contains("another rig"));
        assert!(warnings[1].contains("estimator \"ir-pupil\", running \"fused\""));

        let matching = UserProfile {
            rig_fingerprint: eye_calibration::correction::rig_fingerprint(&synthetic_rig()),
            estimator: "fused".to_string(),
            ..profile
        };
        store
            .save_profile("default", &matching)
            .expect("save profile");
        let (found, warnings) =
            select_profile(&store, &args, &synthetic_rig(), "fused").expect("resolves");
        assert_eq!(found, Some(matching));
        assert_eq!(warnings, Vec::<String>::new());
    }

    #[test]
    fn test_run_loop_forwards_points_until_tracker_stops() {
        let (points_tx, points_rx) = crossbeam_channel::unbounded();
        let (_shutdown_tx, shutdown_rx) = crossbeam_channel::bounded(1);
        points_tx.send(point(1)).unwrap();
        points_tx.send(point(2)).unwrap();
        points_tx.send(point(3)).unwrap();
        drop(points_tx);

        let mut sink = FakeSink::new(false);
        let mut ticker = StatsTicker::new(false, Duration::from_secs(5), Instant::now());
        let end = run_loop(
            &points_rx,
            &mut sink,
            &shutdown_rx,
            &mut ticker,
            &TrackerStats::default,
            &Instant::now,
        );

        assert_eq!(end, LoopEnd::TrackerStopped);
        assert_eq!(
            sink.points.iter().map(|p| p.timestamp).collect::<Vec<_>>(),
            vec![
                Timestamp::from_nanos(1_000_000),
                Timestamp::from_nanos(2_000_000),
                Timestamp::from_nanos(3_000_000),
            ]
        );
    }

    #[test]
    fn test_run_loop_stops_on_shutdown() {
        let (_points_tx, points_rx) = crossbeam_channel::unbounded();
        let (shutdown_tx, shutdown_rx) = crossbeam_channel::bounded(1);
        shutdown_tx.send(()).unwrap();

        let mut sink = FakeSink::new(false);
        let mut ticker = StatsTicker::new(false, Duration::from_secs(5), Instant::now());
        let end = run_loop(
            &points_rx,
            &mut sink,
            &shutdown_rx,
            &mut ticker,
            &TrackerStats::default,
            &Instant::now,
        );

        assert_eq!(end, LoopEnd::Shutdown);
        assert!(sink.points.is_empty());
    }

    #[test]
    fn test_run_loop_overlay_closed_ends_loop() {
        let (points_tx, points_rx) = crossbeam_channel::unbounded();
        let (_shutdown_tx, shutdown_rx) = crossbeam_channel::bounded(1);
        points_tx.send(point(1)).unwrap();

        let mut sink = FakeSink::new(true);
        let mut ticker = StatsTicker::new(false, Duration::from_secs(5), Instant::now());
        let end = run_loop(
            &points_rx,
            &mut sink,
            &shutdown_rx,
            &mut ticker,
            &TrackerStats::default,
            &Instant::now,
        );

        assert_eq!(end, LoopEnd::OverlayClosed(SinkError::Closed.to_string()));
    }

    #[test]
    fn test_stats_ticker_logs_only_with_flag() {
        let t0 = Instant::now();
        let stats = TrackerStats::default();

        let mut ticker = StatsTicker::new(false, Duration::from_secs(5), t0);
        assert_eq!(ticker.report(&stats, t0 + Duration::from_secs(6)), None);

        let mut ticker = StatsTicker::new(true, Duration::from_secs(5), t0);
        assert!(ticker.report(&stats, t0 + Duration::from_secs(6)).is_some());

        let ticker = StatsTicker::new(true, Duration::from_secs(5), t0);
        assert!(!ticker.due(t0 + Duration::from_secs(4)));
        assert!(ticker.due(t0 + Duration::from_secs(5)));
    }

    #[test]
    fn test_stats_line_format() {
        let stats = TrackerStats {
            points_emitted: 242,
            capture_to_emit: eye::tracker::LatencySummary {
                count: 142,
                p50: Duration::from_millis(31),
                p95: Duration::from_millis(44),
                max: Duration::from_millis(61),
            },
            ..Default::default()
        };
        let line = stats_line(&stats, Duration::from_secs(5), 100);
        assert_eq!(
            line,
            "142 points in 5.0 s (28.4/s), latency p50 31.0 ms, p95 44.0 ms, max 61.0 ms, 0 frames dropped"
        );

        let mut stats = stats;
        stats.frames_dropped = BTreeMap::from([
            (eye_core::CameraId::from("ir"), 3),
            (eye_core::CameraId::from("rgb"), 2),
        ]);
        let line = stats_line(&stats, Duration::from_secs(5), 100);
        assert!(line.ends_with("5 frames dropped"));
    }
}
