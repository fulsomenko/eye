use std::{
    io::Write,
    path::PathBuf,
    process::ExitCode,
    sync::{Arc, atomic::AtomicBool},
    time::{Duration, SystemTime},
};

use clap::Parser;
use tracing_subscriber::EnvFilter;

use crate::{
    case::{RunOptions, TestRegistry},
    mode::{ModeHost, NullHost},
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
}

#[derive(Debug, clap::Args)]
pub struct RunArgs {
    #[command(flatten)]
    pub source: SequenceSource,
    /// Skip every test that needs a person in front of the camera
    #[arg(long)]
    pub no_subject: bool,
    /// Report directory (default: lab-reports/<unix seconds>-<sequence>)
    #[arg(long, value_name = "DIR")]
    pub out: Option<PathBuf>,
    /// Seconds a timed-out or interrupted test may take to stop before it is abandoned
    #[arg(long, value_name = "SECONDS", default_value_t = 5.0, value_parser = parse_positive_seconds)]
    pub grace_s: f64,
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
    let _ = tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .try_init();

    let cli = Cli::parse();

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
    }
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

    let mut host: Box<dyn ModeHost> = Box::new(NullHost);

    println!("running {name} ({planned_len} steps)");

    let options = RunOptions {
        subject: !args.no_subject,
        grace: Duration::from_secs_f64(args.grace_s),
        eye_config: None,
    };

    let started = SystemTime::now();
    let started_unix_s = started
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let start_instant = std::time::Instant::now();

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
