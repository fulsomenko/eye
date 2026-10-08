#![forbid(unsafe_code)]

use std::process::ExitCode;

use clap::Parser;

fn main() -> ExitCode {
    let cli = eye_app::cli::Cli::parse();
    eye_app::cli::init_tracing();
    match eye_app::cli::dispatch(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err:#}");
            ExitCode::FAILURE
        }
    }
}
