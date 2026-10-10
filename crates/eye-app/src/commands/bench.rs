use std::path::PathBuf;
use std::time::SystemTime;

use eye::config::{Config, ConfigSource, EnvOverrides};
use eye_bench::matrix::{BenchMatrix, RigSource};
use eye_bench::report::BenchReport;
use eye_bench::row::{CalibrationMode, RowKind, RowOutcome};
use eye_bench::runner::run_matrix_with_store;

use crate::cli::GIT_REV;
use crate::ctx::Ctx;
use crate::paths::utc_stamp;

#[derive(Debug, clap::Args)]
pub struct Args {
    /// Bench matrix (bench.toml). Without it, the pipeline from --config runs over RECORDING...
    #[arg(long, value_name = "FILE")]
    pub matrix: Option<PathBuf>,
    /// Calibration modes to evaluate, comma-separated or repeated; overrides the matrix (default: none,loto)
    #[arg(long, value_enum, value_delimiter = ',')]
    pub calibration: Vec<CalibrationArg>,
    /// Profile file for calibration mode `profile`; overrides the matrix's evaluation.profile
    #[arg(long, value_name = "FILE")]
    pub profile: Option<PathBuf>,
    /// Rig to score every row with: `session`, `stored` or `file:PATH`; overrides evaluation.rig
    #[arg(long, value_parser = RigSource::parse, value_name = "SOURCE")]
    pub rig: Option<RigSource>,
    /// Recording directories; override the matrix's list
    #[arg(value_name = "RECORDING")]
    pub recordings: Vec<PathBuf>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum CalibrationArg {
    None,
    Loto,
    Profile,
    Cross,
}

impl From<CalibrationArg> for CalibrationMode {
    fn from(value: CalibrationArg) -> Self {
        match value {
            CalibrationArg::None => CalibrationMode::None,
            CalibrationArg::Loto => CalibrationMode::Loto,
            CalibrationArg::Profile => CalibrationMode::Profile,
            CalibrationArg::Cross => CalibrationMode::Cross,
        }
    }
}

pub fn build_matrix(config_path: Option<PathBuf>, args: &Args) -> anyhow::Result<BenchMatrix> {
    let mut matrix = match &args.matrix {
        Some(path) => {
            if config_path.is_some() {
                tracing::debug!("--config is not used with --matrix");
            }
            let mut m = BenchMatrix::from_path(path)?;
            if !args.recordings.is_empty() {
                m.recordings = args.recordings.clone();
            }
            m
        }
        None => {
            anyhow::ensure!(
                !args.recordings.is_empty(),
                "give RECORDING... or --matrix FILE"
            );
            BenchMatrix::single(config_path, args.recordings.clone())
        }
    };
    if !args.calibration.is_empty() {
        matrix.evaluation.calibration = args.calibration.iter().copied().map(Into::into).collect();
    }
    if args.profile.is_some() {
        matrix.evaluation.profile = args.profile.clone();
    }
    if let Some(rig) = &args.rig {
        matrix.evaluation.rig = rig.clone();
    }
    matrix.validate()?;
    Ok(matrix)
}

pub fn report_dir(output: Option<PathBuf>, now: SystemTime) -> PathBuf {
    output.unwrap_or_else(|| PathBuf::from("bench-reports").join(utc_stamp(now)))
}

/// Ok if at least one SESSION row is `RowOutcome::Ok`; aggregate rows are ignored for the decision.
pub fn check_outcome(report: &BenchReport) -> anyhow::Result<()> {
    let any_ok = report
        .rows
        .iter()
        .any(|row| row.kind == RowKind::Session && matches!(row.outcome, RowOutcome::Ok { .. }));
    anyhow::ensure!(any_ok, "no pipeline produced a result");
    Ok(())
}

pub fn run(ctx: &Ctx, args: Args) -> anyhow::Result<()> {
    let config_path = match &ctx.config_path {
        Some(p) => Some(p.clone()),
        None => match Config::load_with(None, &EnvOverrides::default())?.source {
            ConfigSource::File(p) => Some(p),
            ConfigSource::BuiltinDefault => None,
        },
    };
    let matrix = build_matrix(config_path, &args)?;
    let store = eye_calibration::store::ProfileStore::open_default()?;
    let report = run_matrix_with_store(
        &matrix,
        &eye::registry::Registry::with_defaults(),
        &|output| store.load_rig(output),
    )
    .with_provenance(
        Some(GIT_REV.to_string()),
        args.matrix.as_ref().map(|p| p.display().to_string()),
    );
    let dir = report_dir(ctx.output.clone(), SystemTime::now());
    report.write_to(&dir)?;
    print!("{}", report.to_markdown());
    tracing::info!(dir = %dir.display(), "report written");
    check_outcome(&report)
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::time::{Duration, UNIX_EPOCH};

    use eye_bench::metrics::MetricParams;
    use eye_bench::row::BenchRow;

    use super::*;

    fn args(
        matrix: Option<PathBuf>,
        calibration: Vec<CalibrationArg>,
        recordings: Vec<&str>,
    ) -> Args {
        Args {
            matrix,
            calibration,
            profile: None,
            rig: None,
            recordings: recordings.into_iter().map(PathBuf::from).collect(),
        }
    }

    #[test]
    fn test_build_matrix_single_from_config_and_recordings() {
        let matrix = build_matrix(
            Some(PathBuf::from("cfg/ir.toml")),
            &args(None, vec![], vec!["a", "b"]),
        )
        .unwrap();
        assert_eq!(matrix.pipelines.len(), 1);
        assert_eq!(matrix.pipelines[0].name, "ir");
        assert_eq!(
            matrix.pipelines[0].config,
            Some(PathBuf::from("cfg/ir.toml"))
        );
        assert_eq!(
            matrix.recordings,
            vec![PathBuf::from("a"), PathBuf::from("b")]
        );
    }

    #[test]
    fn test_build_matrix_without_inputs_is_error() {
        let err = build_matrix(None, &args(None, vec![], vec![])).unwrap_err();
        assert!(
            err.to_string()
                .contains("give RECORDING... or --matrix FILE")
        );
    }

    #[test]
    fn test_build_matrix_positional_recordings_override_matrix() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bench.toml");
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(b"recordings = [\"x\"]\n\n[[pipeline]]\nname = \"a\"\n")
            .unwrap();

        let matrix = build_matrix(None, &args(Some(path.clone()), vec![], vec!["y"])).unwrap();
        assert_eq!(matrix.recordings, vec![PathBuf::from("y")]);
    }

    #[test]
    fn test_build_matrix_calibration_override() {
        let matrix =
            build_matrix(None, &args(None, vec![CalibrationArg::None], vec!["r"])).unwrap();
        assert_eq!(matrix.evaluation.calibration, vec![CalibrationMode::None]);

        let matrix = build_matrix(
            None,
            &args(
                None,
                vec![CalibrationArg::None, CalibrationArg::Loto],
                vec!["r"],
            ),
        )
        .unwrap();
        assert_eq!(
            matrix.evaluation.calibration,
            vec![CalibrationMode::None, CalibrationMode::Loto]
        );
    }

    #[test]
    fn test_bench_calibration_then_recording_parses() {
        use clap::Parser;

        use crate::cli::{Cli, Command};

        let cli = Cli::try_parse_from(["eye", "bench", "--calibration", "none", "rec1"]).unwrap();
        let Command::Bench(a) = cli.command else {
            panic!("expected bench command");
        };
        assert_eq!(a.calibration, vec![CalibrationArg::None]);
        assert_eq!(a.recordings, vec![PathBuf::from("rec1")]);

        let cli = Cli::try_parse_from([
            "eye",
            "bench",
            "--calibration",
            "none",
            "--calibration",
            "loto",
            "r",
        ])
        .unwrap();
        let Command::Bench(a) = cli.command else {
            panic!("expected bench command");
        };
        assert_eq!(
            a.calibration,
            vec![CalibrationArg::None, CalibrationArg::Loto]
        );
    }

    #[test]
    fn test_bench_profile_and_cross_args_parse() {
        use clap::Parser;

        use crate::cli::{Cli, Command};

        let cli = Cli::try_parse_from([
            "eye",
            "bench",
            "--calibration",
            "profile,cross",
            "--profile",
            "p.toml",
            "r",
        ])
        .unwrap();
        let Command::Bench(a) = cli.command else {
            panic!("expected bench command");
        };
        assert_eq!(
            a.calibration,
            vec![CalibrationArg::Profile, CalibrationArg::Cross]
        );
        assert_eq!(a.profile, Some(PathBuf::from("p.toml")));

        let matrix = build_matrix(None, &a).unwrap();
        assert_eq!(
            matrix.evaluation.calibration,
            vec![CalibrationMode::Profile, CalibrationMode::Cross]
        );
        assert_eq!(matrix.evaluation.profile, Some(PathBuf::from("p.toml")));
    }

    #[test]
    fn test_report_dir_default_and_override() {
        let now = UNIX_EPOCH + Duration::from_secs(1_791_409_623);
        assert_eq!(
            report_dir(None, now),
            PathBuf::from("bench-reports").join("20261007T214703Z")
        );
        assert_eq!(
            report_dir(Some(PathBuf::from("p")), now),
            PathBuf::from("p")
        );
    }

    fn row(kind: RowKind, outcome: RowOutcome) -> BenchRow {
        BenchRow {
            pipeline: "p".to_owned(),
            calibration: CalibrationMode::None,
            kind,
            session: "s".to_owned(),
            rig_source: "session".to_owned(),
            rig_fingerprint: String::new(),
            step_errors: 0,
            warnings: vec![],
            outcome,
        }
    }

    fn ok_outcome() -> RowOutcome {
        RowOutcome::Ok {
            metrics: eye_bench::metrics::SessionMetrics {
                windows: 0,
                samples: 0,
                angular_error_deg: None,
                px_error_logical: None,
                accuracy_deg: None,
                precision_rms_s2s_deg: None,
                precision_pooled_rms_s2s_deg: None,
                nees: None,
                regions: vec![],
                processing_ms: None,
                dropout_rate: None,
                output_rate_hz: None,
            },
        }
    }

    fn error_outcome() -> RowOutcome {
        RowOutcome::Error {
            message: "boom".to_owned(),
        }
    }

    #[test]
    fn test_check_outcome_fails_only_when_all_rows_error() {
        let report = BenchReport::new(
            MetricParams::default(),
            vec![
                row(RowKind::Session, ok_outcome()),
                row(RowKind::Session, error_outcome()),
            ],
        );
        assert!(check_outcome(&report).is_ok());

        let report = BenchReport::new(
            MetricParams::default(),
            vec![
                row(RowKind::Session, error_outcome()),
                row(RowKind::Session, error_outcome()),
                row(RowKind::Aggregate, ok_outcome()),
            ],
        );
        let err = check_outcome(&report).unwrap_err();
        assert!(err.to_string().contains("no pipeline produced a result"));
    }
}
