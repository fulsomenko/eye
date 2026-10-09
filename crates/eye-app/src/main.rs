#![forbid(unsafe_code)]

use std::process::ExitCode;

use clap::Parser;

fn main() -> ExitCode {
    let cli = eye_app::cli::Cli::parse();
    let command = cli.command.name();
    let _log = match eye_log::cli::init(&cli.log, command) {
        Ok(guard) => guard,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::from(2);
        }
    };
    match eye_app::cli::dispatch(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err:#}");
            ExitCode::FAILURE
        }
    }
}
