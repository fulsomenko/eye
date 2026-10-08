use std::path::PathBuf;

use clap::{Parser, Subcommand};

use crate::commands;
use crate::ctx::Ctx;

pub const GIT_REV: &str = env!("EYE_GIT_REV");
pub const VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), " (", env!("EYE_GIT_REV"), ")");

#[derive(Debug, Parser)]
#[command(name = "eye", version = VERSION, about = "Webcam gaze tracking")]
pub struct Cli {
    /// Pipeline config (eye.toml). Default: $XDG_CONFIG_HOME/eye/eye.toml, then ./eye.toml, then built-in defaults.
    #[arg(long, global = true, env = "EYE_CONFIG", value_name = "PATH")]
    pub config: Option<PathBuf>,
    /// Where the subcommand writes its primary artifact (see `eye <command> --help`).
    #[arg(long, global = true, value_name = "PATH")]
    pub output: Option<PathBuf>,
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Probe displays, cameras and the IR emitter
    Probe(commands::probe::Args),
    /// Control the IR emitter
    Emitter(commands::emitter::Args),
    /// Record the cameras while showing dot targets
    Record(commands::record::Args),
    /// Fit a user profile from a dot session
    Calibrate(commands::calibrate::Args),
    /// Track gaze live and draw it on the desktop
    Run(commands::run::Args),
    /// Score pipeline configs against recordings
    Bench(commands::bench::Args),
}

pub fn init_tracing() {
    use tracing_subscriber::filter::LevelFilter;

    let filter = tracing_subscriber::EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init();
}

pub fn dispatch(cli: Cli) -> anyhow::Result<()> {
    let ctx = Ctx::new(cli.config, cli.output)?;
    match cli.command {
        Command::Probe(a) => commands::probe::run(&ctx, a),
        Command::Emitter(a) => commands::emitter::run(&ctx, a),
        Command::Record(a) => commands::record::run(&ctx, a),
        Command::Calibrate(a) => commands::calibrate::run(&ctx, a),
        Command::Run(a) => commands::run::run(&ctx, a),
        Command::Bench(a) => commands::bench::run(&ctx, a),
    }
}

#[cfg(test)]
mod tests {
    use clap::CommandFactory;

    use super::*;

    #[test]
    fn test_cli_definition_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn test_global_flags_parse_after_subcommand() {
        let cli = Cli::try_parse_from(["eye", "probe", "--config", "a.toml", "--output", "o"])
            .expect("parses");
        assert_eq!(cli.config, Some(PathBuf::from("a.toml")));
        assert_eq!(cli.output, Some(PathBuf::from("o")));
    }
}
