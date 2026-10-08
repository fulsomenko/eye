use std::collections::BTreeSet;
use std::fs::Permissions;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use eye::config::Config;
use eye::registry::Registry;
use eye_bench::calibration::{dot_session_fitter, fit_samples, loto, position_key};
use eye_bench::metrics::{MetricParams, SessionMetrics, compute};
use eye_bench::runner::replay_session;
use eye_calibration::correction::UserProfile;
use eye_calibration::protocol::ProtocolConfig;
use eye_calibration::store::ProfileStore;
use eye_calibration::user_fit::{DotSessionFit, FitConfig, ProfileMeta};

use crate::commands::record::{RecordOptions, RecordSummary, SessionLocation, record_session};
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
}

pub fn fit_recording(
    dir: &Path,
    config: &Config,
    registry: &Registry,
    meta: ProfileMeta,
) -> anyhow::Result<FitResult> {
    let mut replayed = replay_session(dir, config, registry, &ProtocolConfig::default())?;
    let run = &replayed.run;
    let samples = fit_samples(
        &run.windows,
        run.steps.iter().filter_map(|s| s.batch.as_ref()),
        |_| true,
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
    let profile = DotSessionFit::fit_with(&samples, &run.rig, &FitConfig::default(), meta)?.profile;
    let expected = match loto(&mut replayed, &dot_session_fitter) {
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
    Ok(FitResult {
        profile,
        samples: samples.len(),
        targets,
        expected,
    })
}

fn protocol_for(targets: u32, dwell_ms: Option<u64>) -> ProtocolConfig {
    let default = ProtocolConfig::default();
    ProtocolConfig {
        grid: if targets == 16 { [4, 4] } else { [3, 3] },
        dwell_ms: dwell_ms.unwrap_or(default.dwell_ms),
        ..default
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
    format!(
        "{} targets, {} samples; expected accuracy (leave-one-target-out): {accuracy}",
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
    if let Some(location) = recording.location() {
        let shutdown_rx = shutdown::install()?;
        let opts = RecordOptions {
            protocol: Some(protocol_for(args.targets, args.dwell_ms)),
            duration: None,
        };
        let summary: RecordSummary = record_session(&config, location, &opts, &shutdown_rx)?;
        anyhow::ensure!(
            !summary.interrupted,
            "calibration interrupted; nothing saved"
        );
    }
    tracing::info!("fitting...");
    let meta = ProfileMeta {
        name: args.profile.clone(),
        created_unix_s: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        estimator: config.estimate.kind.clone(),
    };
    let result = fit_recording(
        &recording.session_dir(),
        &config,
        &Registry::with_defaults(),
        meta,
    )?;
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
    use clap::{CommandFactory, Parser, error::ErrorKind};
    use eye_bench::testing::{
        FOUR_BY_FOUR_CENTRES, SyntheticSession, fake_registry, kappa_ray_config,
        write_synthetic_session,
    };

    use super::*;
    use crate::cli::Cli;

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
    fn test_from_conflicts_with_keep_and_targets() {
        Cli::command().debug_assert();

        let err = Cli::try_parse_from(["eye", "calibrate", "--from", "d", "--keep"]).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::ArgumentConflict);

        let err = Cli::try_parse_from(["eye", "calibrate", "--from", "d", "--targets", "16"])
            .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::ArgumentConflict);
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
        let result = fit_recording(&session_dir, &config, &fake_registry(), meta).unwrap();

        assert_eq!(result.targets, 16);
        assert_eq!(result.samples, 384);
        let mean = result.expected.unwrap().angular_error_deg.unwrap().mean;
        assert!(mean < 0.1, "mean angular error {mean} not < 0.1");
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
        };
        let result = fit_recording(&session_dir, &config, &fake_registry(), meta).unwrap();

        assert_eq!(result.profile.name, "alice");
        assert_eq!(result.profile.created_unix_s, 1_791_409_623);
        assert_eq!(result.profile.estimator, "test-kappa-ray");
        assert!(!result.profile.rig_fingerprint.is_empty());
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
        }
    }

    fn fixture_profile() -> UserProfile {
        UserProfile {
            version: 1,
            name: "default".into(),
            created_unix_s: 0,
            rig_fingerprint: String::new(),
            estimator: String::new(),
            eyes: std::collections::BTreeMap::new(),
        }
    }

    #[test]
    fn test_summary_line_format() {
        let result = fit_result(9, 2140, Some(1.84), 3.90, Some(1.0));
        assert_eq!(
            summary_line(&result),
            "9 targets, 2140 samples; expected accuracy (leave-one-target-out): mean 1.84 deg, p95 3.90 deg, 3x3 hit 100.0 %"
        );

        let no_expected = FitResult {
            profile: fixture_profile(),
            samples: 2140,
            targets: 9,
            expected: None,
        };
        assert_eq!(
            summary_line(&no_expected),
            "9 targets, 2140 samples; expected accuracy (leave-one-target-out): n/a"
        );
    }
}
