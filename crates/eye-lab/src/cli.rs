use std::{
    io::Write,
    path::PathBuf,
    process::ExitCode,
    sync::{Arc, atomic::AtomicBool},
    time::{Duration, SystemTime},
};

use clap::Parser;

use eye::config::EnvOverrides;
use eye_core::log::{field, span};

use crate::{
    case::{RunOptions, TestRegistry},
    mode::ModeHost,
    modes::{CameraSelection, LiveHost},
    report::{self, Report},
    runner,
    sequence::{LoadError, MAX_TIMEOUT_S, Sequence},
    signals, suites,
};

#[derive(Debug, clap::Parser)]
#[command(
    name = "eye-lab",
    version,
    about = "Headless camera-mode and test-sequence runner"
)]
pub struct Cli {
    #[command(flatten)]
    pub log: eye_log::cli::LogArgs,
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, clap::Subcommand)]
pub enum Command {
    /// Run a sequence of tests
    Run(RunArgs),
    /// List the registered test cases
    ListTests,
    /// List the built-in suites
    Suites,
    /// Print the camera mode matrix, the emitter byte and the metadata node
    Modes(ModesArgs),
}

impl Command {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Run(_) => "run",
            Self::ListTests => "list-tests",
            Self::Suites => "suites",
            Self::Modes(_) => "modes",
        }
    }
}

#[derive(Debug, clap::Args)]
pub struct CameraArgs {
    /// RGB video node (default: $EYE_CAMERA, else the first RGB camera)
    #[arg(long, value_name = "PATH")]
    pub rgb: Option<PathBuf>,
    /// IR video node (default: $EYE_IR_CAMERA, else the first IR camera)
    #[arg(long, value_name = "PATH")]
    pub ir: Option<PathBuf>,
}

#[derive(Debug, clap::Args)]
pub struct ModesArgs {
    #[command(flatten)]
    pub cameras: CameraArgs,
    /// Machine-readable output (the matrix rows only)
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, clap::Args)]
pub struct RunArgs {
    #[command(flatten)]
    pub source: SequenceSource,
    #[command(flatten)]
    pub cameras: CameraArgs,
    /// Skip every test that needs a person in front of the camera
    #[arg(long)]
    pub no_subject: bool,
    /// Report directory (default: lab-reports/<unix seconds>-<sequence>)
    #[arg(long, value_name = "DIR")]
    pub out: Option<PathBuf>,
    /// Seconds a timed-out or interrupted test may take to stop before it is abandoned
    #[arg(long, value_name = "SECONDS", default_value_t = 5.0, value_parser = parse_positive_seconds)]
    pub grace_s: f64,
    /// eye.toml for pipeline tests (default: the embedded IR-only config)
    #[arg(long, value_name = "PATH")]
    pub eye_config: Option<PathBuf>,
}

#[derive(Debug, clap::Args)]
#[group(required = true, multiple = false)]
pub struct SequenceSource {
    /// A built-in suite (see `eye-lab suites`)
    #[arg(long)]
    pub suite: Option<String>,
    /// A lab.toml file
    #[arg(long, value_name = "PATH")]
    pub file: Option<PathBuf>,
}

fn parse_positive_seconds(s: &str) -> Result<f64, String> {
    let v: f64 = s.parse().map_err(|_| format!("{s:?} is not a number"))?;
    if v.is_finite() && v > 0.0 && v <= MAX_TIMEOUT_S {
        Ok(v)
    } else {
        Err(format!("must be > 0 and <= {MAX_TIMEOUT_S}, got {v}"))
    }
}

pub fn main() -> ExitCode {
    ExitCode::from(run_main())
}

fn load_sequence(
    source: &SequenceSource,
) -> Result<(String, Vec<crate::sequence::ResolvedStep>, Vec<String>), LoadError> {
    if let Some(name) = &source.suite {
        let def = suites::find(name).ok_or_else(|| LoadError::UnknownSuite {
            name: name.clone(),
            available: suites::names(),
        })?;
        Ok((def.name.to_owned(), def.load()?, def.sources()))
    } else {
        let path = source
            .file
            .as_ref()
            .expect("clap enforces exactly one of --suite/--file");
        let sequence = Sequence::load(path)?;
        let origin = path.display().to_string();
        let steps = sequence.resolve(&origin)?;
        Ok((sequence.name.clone(), steps, vec![origin]))
    }
}

fn run_main() -> u8 {
    let cli = Cli::parse();
    let command = cli.command.name();
    let _log = match eye_log::cli::init(&cli.log, command) {
        Ok(guard) => guard,
        Err(e) => {
            eprintln!("eye-lab: {e}");
            return 2;
        }
    };

    match cli.command {
        Command::ListTests => {
            let registry = TestRegistry::builtin();
            for info in registry.infos() {
                let case = (info.factory)(&toml::Table::new())
                    .expect("every builtin case builds with empty params");
                println!(
                    "{:<24} {:<22} {:>5}  {}",
                    info.name,
                    case.needs().to_string(),
                    format!("{}s", case.default_timeout().as_secs()),
                    info.summary
                );
            }
            0
        }
        Command::Suites => {
            for def in suites::BUILTIN {
                let steps = match def.load() {
                    Ok(steps) => steps.len().to_string(),
                    Err(_) => "ERR".to_owned(),
                };
                println!("{:<16} {:>3} steps  {}", def.name, steps, def.summary);
            }
            0
        }
        Command::Run(args) => run_command(args),
        Command::Modes(args) => modes_command(args),
    }
}

fn build_host(cameras: &CameraArgs) -> Result<LiveHost, u8> {
    let selection = CameraSelection::resolve(
        cameras.rgb.clone(),
        cameras.ir.clone(),
        &EnvOverrides::from_process_env(),
    );
    LiveHost::probe(&selection).map_err(|e| {
        eprintln!("eye-lab: {e}");
        2
    })
}

fn modes_command(args: ModesArgs) -> u8 {
    let host = match build_host(&args.cameras) {
        Ok(host) => host,
        Err(code) => return code,
    };

    if args.json {
        match serde_json::to_string_pretty(&host.matrix()) {
            Ok(s) => println!("{s}"),
            Err(e) => {
                eprintln!("eye-lab: {e}");
                return 2;
            }
        }
    } else {
        println!("STREAMS EMITTER RGB                IR                 RUNNABLE");
        for row in host.matrix() {
            let rgb = row.rgb.as_deref().unwrap_or("-");
            let ir = row.ir.as_deref().unwrap_or("-");
            let runnable = match (row.runnable, &row.note) {
                (true, _) => "yes".to_owned(),
                (false, Some(note)) => format!("no: {note}"),
                (false, None) => "no".to_owned(),
            };
            println!(
                "{:<7} {:<7} {:<18} {:<18} {}",
                row.streams, row.emitter, rgb, ir, runnable
            );
        }
        println!();
        println!("{}", host.emitter_status());
        println!("{}", host.metadata_status());
    }
    0
}

fn run_command(args: RunArgs) -> u8 {
    let (name, steps, sources) = match load_sequence(&args.source) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("eye-lab: {e}");
            return 2;
        }
    };
    let planned_len = steps.len();
    let registry = TestRegistry::builtin();
    let planned = match runner::plan(steps, &registry) {
        Ok(planned) => planned,
        Err(e) => {
            eprintln!("eye-lab: {e}");
            return 2;
        }
    };

    let cancel = Arc::new(AtomicBool::new(false));
    if let Err(e) = signals::install(&cancel) {
        eprintln!("eye-lab: installing signal handlers: {e}");
        return 2;
    }

    let mut host: Box<dyn ModeHost> = match build_host(&args.cameras) {
        Ok(host) => Box::new(host),
        Err(code) => return code,
    };

    println!("running {name} ({planned_len} steps)");

    let options = RunOptions {
        subject: !args.no_subject,
        grace: Duration::from_secs_f64(args.grace_s),
        eye_config: args.eye_config.clone(),
    };

    let started = SystemTime::now();
    let started_unix_s = started
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let start_instant = std::time::Instant::now();

    let session_id = format!("{started_unix_s}-{name}");
    let _session =
        tracing::info_span!(span::SESSION, { field::SESSION_ID } = session_id.as_str()).entered();
    let outcome = runner::run(
        &planned,
        host.as_mut(),
        &options,
        &cancel,
        &mut std::io::stdout(),
    );

    let duration_ms = u64::try_from(start_instant.elapsed().as_millis()).unwrap_or(u64::MAX);

    let tally = report::tally(&outcome.results);
    let aborted = outcome.aborted.map(|a| a.to_string());
    let exit_code = report::exit_code(&tally, aborted.is_some());

    let mut environment = report::base_environment();
    environment.extend(host.environment());

    let report = Report {
        schema: report::REPORT_SCHEMA,
        tool: format!("eye-lab {}", env!("CARGO_PKG_VERSION")),
        sequence: name.clone(),
        sources,
        started_unix_s,
        duration_ms,
        subject: options.subject,
        environment,
        steps: outcome.results,
        tally,
        aborted,
        exit_code,
    };

    let out_dir = args
        .out
        .unwrap_or_else(|| PathBuf::from(format!("lab-reports/{started_unix_s}-{name}")));

    match report::write_files(&report, &out_dir) {
        Ok((json_path, md_path)) => {
            println!("report: {}", json_path.display());
            println!("report: {}", md_path.display());
        }
        Err(e) => {
            eprintln!("eye-lab: writing report: {e}");
            return 2;
        }
    }

    let _ = std::io::stdout().flush();
    println!(
        "{}",
        report::result_sentence(&report.tally, report.exit_code)
    );

    exit_code
}
