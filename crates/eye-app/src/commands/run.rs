use std::time::{Duration, Instant};

use crossbeam_channel::Receiver;
use eye::config::{OutputConfig, OutputMode};
use eye::registry::Registry;
use eye::tracker::{Tracker, TrackerStats};
#[cfg(test)]
use eye_calibration::correction::Provenance;
use eye_calibration::correction::{UserProfile, rig_fingerprint, text_fingerprint};
use eye_calibration::store::ProfileStore;
use eye_core::stage::GazeCorrection;
use eye_core::{GazePoint, Rig};
use eye_overlay::sink::{LayerShellOverlay, OverlayMode, OverlayOptions};
use eye_overlay::stats::PresentSummary;
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
    /// Overrides [output].easing_ms; 0 disables point-mode easing
    #[arg(long)]
    pub easing_ms: Option<u64>,
    /// Overrides [output].hide_margin_px (point mode only)
    #[arg(long)]
    pub hide_margin_px: Option<f64>,
    /// Overrides [output].hide_below_confidence (point mode only)
    #[arg(long)]
    pub hide_below_confidence: Option<f64>,
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
    if let Some(easing_ms) = args.easing_ms {
        output.easing_ms = easing_ms;
    }
    if let Some(hide_margin_px) = args.hide_margin_px {
        output.hide_margin_px = hide_margin_px;
    }
    if let Some(hide_below_confidence) = args.hide_below_confidence {
        output.hide_below_confidence = hide_below_confidence;
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

pub fn overlay_options(output: &OutputConfig, screen: &eye_core::ScreenModel) -> OverlayOptions {
    let mut options = OverlayOptions::new(screen.output.clone(), overlay_mode(output), screen);
    options.interpolation = eye_overlay::point::Interpolation::from_millis(output.easing_ms);
    options.hide = eye_overlay::point::HideRules {
        margin_px: output.hide_margin_px,
        min_confidence: output.hide_below_confidence,
    };
    options.style = eye_overlay::point::PointStyle::default();
    options
}

/// `--no-profile`: skips the store entirely. Otherwise the stored profile (or `None`) plus
/// warnings for a missing profile, a different rig fingerprint and a different estimator.
pub fn select_profile(
    store: &ProfileStore,
    args: &Args,
    rig: &Rig,
    estimator: &str,
    estimator_fingerprint: &str,
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
    if let Some(fp) = &profile.provenance.estimator_fingerprint
        && fp != estimator_fingerprint
    {
        warnings.push(format!(
            "profile \"{}\" was fitted with different estimator options; run eye calibrate again",
            args.profile
        ));
    }
    Ok((Some(profile), warnings))
}

/// `[estimate]` table as written (kind first) hashed for `Provenance::estimator_fingerprint`.
pub fn estimator_fingerprint(section: &eye::config::StageSection) -> String {
    let mut table = toml::Table::new();
    table.insert("kind".into(), toml::Value::String(section.kind.clone()));
    table.extend(section.options.clone());
    text_fingerprint(&toml::to_string(&table).expect("a toml table serializes"))
}

pub fn stats_line(
    stats: &TrackerStats,
    present: &PresentSummary,
    window: Duration,
    points_before: u64,
) -> String {
    let points = stats.points_emitted - points_before;
    let frames_dropped: u64 = stats.frames_dropped.values().sum();
    let sink_drops: u64 = stats.sink_drops.values().sum();
    let secs = window.as_secs_f64();
    let rate = if secs > 0.0 {
        points as f64 / secs
    } else {
        0.0
    };
    format!(
        "{points} points in {secs:.1} s ({rate:.1}/s), emit p50 {emit_p50:.1} ms, p95 {emit_p95:.1} ms (n = {emit_window}), present p50 {present_p50:.1} ms, p95 {present_p95:.1} ms, {frames_dropped} frames dropped, {sink_drops} sink drops",
        emit_p50 = stats.capture_to_emit.p50.as_secs_f64() * 1000.0,
        emit_p95 = stats.capture_to_emit.p95.as_secs_f64() * 1000.0,
        emit_window = stats.capture_to_emit.window,
        present_p50 = present.p50.as_secs_f64() * 1000.0,
        present_p95 = present.p95.as_secs_f64() * 1000.0,
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
    pub fn report(
        &mut self,
        stats: &TrackerStats,
        present: &PresentSummary,
        now: Instant,
    ) -> Option<String> {
        let window = now.duration_since(self.last);
        let points_before = self.last_points;
        self.last = now;
        self.last_points = stats.points_emitted;
        self.log
            .then(|| stats_line(stats, present, window, points_before))
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
    shutdown: &Receiver<()>,
    ticker: &mut StatsTicker,
    stats: &dyn Fn() -> TrackerStats,
    present: &dyn Fn() -> PresentSummary,
    now: &dyn Fn() -> Instant,
) -> LoopEnd {
    loop {
        crossbeam_channel::select! {
            recv(shutdown) -> _ => return LoopEnd::Shutdown,
            recv(points) -> msg => if msg.is_err() { return LoopEnd::TrackerStopped },
            default(Duration::from_millis(100)) => {}
        }
        let s = stats();
        if s.sinks == 0 {
            return LoopEnd::OverlayClosed("overlay sink removed".into());
        }
        let t = now();
        if ticker.due(t)
            && let Some(line) = ticker.report(&s, &present(), t)
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
    let (profile, warnings) = select_profile(
        &store,
        &args,
        &rig,
        &config.estimate.kind,
        &estimator_fingerprint(&config.estimate),
    )?;
    for warning in &warnings {
        tracing::warn!("{warning}");
    }
    if let Some(profile) = &profile {
        tracing::info!(
            name = %profile.name,
            rig_fingerprint = %profile.rig_fingerprint,
            estimator = %profile.estimator,
            session_id = ?profile.provenance.session_id,
            expected_loto_mean_deg = ?profile.provenance.expected_loto_mean_deg,
            "profile applied"
        );
    }
    let correction = profile.map(|p| Box::new(p) as Box<dyn GazeCorrection>);
    let shutdown = crate::shutdown::install()?;
    let guards = emitter_guards(
        &rig::msxu_cameras(&config),
        &report.cameras,
        EmitterGuard::enable,
    )?;
    let overlay = LayerShellOverlay::spawn(overlay_options(&config.output, rig.screen()))?;
    let present = overlay.present_stats();
    let tracker = Tracker::from_config_with(
        &Registry::with_defaults(),
        &config,
        rig,
        correction,
        vec![Box::new(overlay)],
    )?;
    let points = tracker.subscribe();
    let mut ticker = StatsTicker::new(args.stats, Duration::from_secs(5), Instant::now());
    let end = run_loop(
        &points,
        &shutdown,
        &mut ticker,
        &|| tracker.stats(),
        &|| present.summary(),
        &Instant::now,
    );
    drop(points);
    let tracker_result = tracker.shutdown();
    drop(guards);
    match end {
        LoopEnd::Shutdown => {
            tracker_result?;
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

    use super::*;

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
            easing_ms: None,
            hide_margin_px: None,
            hide_below_confidence: None,
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
    fn test_output_overrides_apply_easing_ms() {
        let mut output = OutputConfig::default();
        let mut args = args_with(None, None);
        args.easing_ms = Some(0);
        apply_output_overrides(&mut output, &args).expect("overrides apply");
        assert_eq!(output.easing_ms, 0);
    }

    #[test]
    fn test_output_overrides_apply_hide_margin_and_confidence() {
        let mut output = OutputConfig::default();
        let mut args = args_with(None, None);
        args.hide_margin_px = Some(10.0);
        args.hide_below_confidence = Some(0.5);
        apply_output_overrides(&mut output, &args).expect("overrides apply");
        assert_eq!(output.hide_margin_px, 10.0);
        assert_eq!(output.hide_below_confidence, 0.5);
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
    fn test_overlay_options_maps_every_output_knob() {
        let screen = eye_core::ScreenModel {
            output: eye_core::OutputId::from("eDP-1"),
            size_mm: nalgebra::Vector2::new(310.0, 170.0),
            size_px: (3840, 2160),
            scale: 2.0,
        };
        let output = OutputConfig {
            mode: OutputMode::Region,
            grid: [4, 2],
            easing_ms: 120,
            hide_margin_px: 10.0,
            hide_below_confidence: 0.5,
            ..Default::default()
        };
        let options = overlay_options(&output, &screen);
        assert_eq!(options.mode, OverlayMode::Region { cols: 4, rows: 2 });
        assert_eq!(options.interpolation.max_lag, Duration::from_millis(120));
        assert_eq!(
            options.hide,
            eye_overlay::point::HideRules {
                margin_px: 10.0,
                min_confidence: 0.5,
            }
        );
        assert_eq!(options.output, screen.output);
        assert_eq!(options.style, eye_overlay::point::PointStyle::default());
    }

    #[test]
    fn test_select_profile_missing_warns_and_is_none() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = ProfileStore::at(dir.path());
        let args = args_with(None, None);
        let (profile, warnings) =
            select_profile(&store, &args, &synthetic_rig(), "fused", "").expect("resolves");
        assert_eq!(profile, None);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("no profile \"default\""));
    }

    #[test]
    fn test_select_profile_no_profile_skips_store() {
        let store = ProfileStore::at("/nonexistent/eye-store");
        let mut args = args_with(None, None);
        args.no_profile = true;
        let result =
            select_profile(&store, &args, &synthetic_rig(), "fused", "").expect("resolves");
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
            calibration_pose: None,
            provenance: Provenance::default(),
        };
        store
            .save_profile("default", &profile)
            .expect("save profile");
        let args = args_with(None, None);

        let (found, warnings) =
            select_profile(&store, &args, &synthetic_rig(), "fused", "").expect("resolves");
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
            select_profile(&store, &args, &synthetic_rig(), "fused", "").expect("resolves");
        assert_eq!(found, Some(matching));
        assert_eq!(warnings, Vec::<String>::new());
    }

    #[test]
    fn test_select_profile_warns_on_estimator_fingerprint_mismatch() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = ProfileStore::at(dir.path());
        let profile = UserProfile {
            version: 1,
            name: "default".to_string(),
            created_unix_s: 0,
            rig_fingerprint: eye_calibration::correction::rig_fingerprint(&synthetic_rig()),
            estimator: "fused".to_string(),
            eyes: BTreeMap::new(),
            calibration_pose: None,
            provenance: Provenance {
                estimator_fingerprint: Some("x".to_string()),
                ..Provenance::default()
            },
        };
        store
            .save_profile("default", &profile)
            .expect("save profile");
        let args = args_with(None, None);

        let (_, warnings) =
            select_profile(&store, &args, &synthetic_rig(), "fused", "y").expect("resolves");
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("different estimator options"));

        let (_, warnings) =
            select_profile(&store, &args, &synthetic_rig(), "fused", "x").expect("resolves");
        assert_eq!(warnings, Vec::<String>::new());
    }

    #[test]
    fn test_estimator_fingerprint_depends_on_options() {
        let bare = eye::config::StageSection {
            kind: "fused".to_string(),
            options: toml::Table::new(),
        };
        let mut options = toml::Table::new();
        options.insert("ir".to_string(), toml::Value::String("pccr".to_string()));
        let with_options = eye::config::StageSection {
            kind: "fused".to_string(),
            options,
        };

        assert_eq!(estimator_fingerprint(&bare), estimator_fingerprint(&bare));
        assert_ne!(
            estimator_fingerprint(&bare),
            estimator_fingerprint(&with_options)
        );
    }

    #[test]
    fn test_run_loop_does_not_end_before_first_point_with_one_sink() {
        let (_points_tx, points_rx) = crossbeam_channel::unbounded();
        let (shutdown_tx, shutdown_rx) = crossbeam_channel::bounded(1);
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(250));
            shutdown_tx.send(()).unwrap();
        });

        let mut ticker = StatsTicker::new(false, Duration::from_secs(5), Instant::now());
        let end = run_loop(
            &points_rx,
            &shutdown_rx,
            &mut ticker,
            &|| TrackerStats {
                sinks: 1,
                ..Default::default()
            },
            &PresentSummary::default,
            &Instant::now,
        );

        assert_eq!(end, LoopEnd::Shutdown);
    }

    #[test]
    fn test_run_loop_ends_when_sinks_reach_zero() {
        let (_points_tx, points_rx) = crossbeam_channel::unbounded();
        let (_shutdown_tx, shutdown_rx) = crossbeam_channel::bounded(1);

        let mut ticker = StatsTicker::new(false, Duration::from_secs(5), Instant::now());
        let end = run_loop(
            &points_rx,
            &shutdown_rx,
            &mut ticker,
            &TrackerStats::default,
            &PresentSummary::default,
            &Instant::now,
        );

        assert_eq!(
            end,
            LoopEnd::OverlayClosed("overlay sink removed".to_string())
        );
    }

    #[test]
    fn test_run_loop_tracker_stopped_when_signal_closes() {
        let (points_tx, points_rx) = crossbeam_channel::unbounded::<GazePoint>();
        drop(points_tx);
        let (_shutdown_tx, shutdown_rx) = crossbeam_channel::bounded(1);

        let mut ticker = StatsTicker::new(false, Duration::from_secs(5), Instant::now());
        let end = run_loop(
            &points_rx,
            &shutdown_rx,
            &mut ticker,
            &|| TrackerStats {
                sinks: 1,
                ..Default::default()
            },
            &PresentSummary::default,
            &Instant::now,
        );

        assert_eq!(end, LoopEnd::TrackerStopped);
    }

    #[test]
    fn test_run_loop_stops_on_shutdown() {
        let (_points_tx, points_rx) = crossbeam_channel::unbounded();
        let (shutdown_tx, shutdown_rx) = crossbeam_channel::bounded(1);
        shutdown_tx.send(()).unwrap();

        let mut ticker = StatsTicker::new(false, Duration::from_secs(5), Instant::now());
        let end = run_loop(
            &points_rx,
            &shutdown_rx,
            &mut ticker,
            &TrackerStats::default,
            &PresentSummary::default,
            &Instant::now,
        );

        assert_eq!(end, LoopEnd::Shutdown);
    }

    #[test]
    fn test_stats_ticker_logs_only_with_flag() {
        let t0 = Instant::now();
        let stats = TrackerStats::default();
        let present = PresentSummary::default();

        let mut ticker = StatsTicker::new(false, Duration::from_secs(5), t0);
        assert_eq!(
            ticker.report(&stats, &present, t0 + Duration::from_secs(6)),
            None
        );

        let mut ticker = StatsTicker::new(true, Duration::from_secs(5), t0);
        assert!(
            ticker
                .report(&stats, &present, t0 + Duration::from_secs(6))
                .is_some()
        );

        let ticker = StatsTicker::new(true, Duration::from_secs(5), t0);
        assert!(!ticker.due(t0 + Duration::from_secs(4)));
        assert!(ticker.due(t0 + Duration::from_secs(5)));
    }

    #[test]
    fn test_stats_line_includes_present_latency_and_sink_drops() {
        let stats = TrackerStats {
            points_emitted: 242,
            capture_to_emit: eye::tracker::LatencySummary {
                count: 142,
                window: 100,
                p50: Duration::from_millis(31),
                p95: Duration::from_millis(44),
                max: Duration::from_millis(61),
            },
            sink_drops: BTreeMap::from([("layer-shell", 3)]),
            ..Default::default()
        };
        let present = PresentSummary {
            count: 142,
            p50: Duration::from_millis(48),
            p95: Duration::from_millis(61),
            max: Duration::from_millis(80),
        };
        let line = stats_line(&stats, &present, Duration::from_secs(5), 100);
        assert_eq!(
            line,
            "142 points in 5.0 s (28.4/s), emit p50 31.0 ms, p95 44.0 ms (n = 100), present p50 48.0 ms, p95 61.0 ms, 0 frames dropped, 3 sink drops"
        );

        let mut stats = stats;
        stats.frames_dropped = BTreeMap::from([
            (eye_core::CameraId::from("ir"), 3),
            (eye_core::CameraId::from("rgb"), 2),
        ]);
        let line = stats_line(&stats, &present, Duration::from_secs(5), 100);
        assert!(line.ends_with("5 frames dropped, 3 sink drops"));
    }
}
