use std::io;
use std::path::PathBuf;
use std::time::SystemTime;

use eye_core::log::{field, span};
use tracing_subscriber::filter::EnvFilter;
use tracing_subscriber::fmt;
use tracing_subscriber::layer::{Layer, SubscriberExt};
use tracing_subscriber::registry;

use crate::layer::{SinkHandle, SinkLayer};
use crate::sink::JsonLinesSink;

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum LogFormat {
    Text,
    Json,
}

pub const DEFAULT_SINK_FILTER: &str = "trace,ort=info,tract=info";

#[derive(clap::Args, Debug, Clone)]
pub struct LogArgs {
    /// Terminal filter (EnvFilter syntax). Falls back to EYE_LOG, then RUST_LOG, then `info`.
    #[arg(long, global = true, env = "EYE_LOG", value_name = "FILTER")]
    pub log_level: Option<String>,
    /// JSON Lines sink path, or `auto` for $XDG_STATE_HOME/eye/logs/<run.id>.jsonl
    #[arg(long, global = true, env = "EYE_LOG_FILE", value_name = "PATH")]
    pub log_file: Option<String>,
    #[arg(
        long,
        global = true,
        env = "EYE_LOG_FILE_LEVEL",
        default_value = DEFAULT_SINK_FILTER,
        value_name = "FILTER"
    )]
    pub log_file_level: String,
    #[arg(long, global = true, value_enum, default_value_t = LogFormat::Text)]
    pub log_format: LogFormat,
}

#[derive(Debug, thiserror::Error)]
pub enum LogError {
    #[error("invalid --log-level {spec:?}: {source}")]
    Filter {
        spec: String,
        source: tracing_subscriber::filter::ParseError,
    },
    #[error("invalid --log-file-level {spec:?}: {source}")]
    FileFilter {
        spec: String,
        source: tracing_subscriber::filter::ParseError,
    },
    #[error("opening log file {}: {source}", path.display())]
    Open { path: PathBuf, source: io::Error },
    #[error("no home directory for --log-file auto")]
    NoHome,
}

pub struct LogGuard {
    span: Option<tracing::span::EnteredSpan>,
    sink: Option<SinkHandle>,
}

impl std::fmt::Debug for LogGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LogGuard").finish_non_exhaustive()
    }
}

impl Drop for LogGuard {
    fn drop(&mut self) {
        self.span.take();
        if let Some(handle) = self.sink.take() {
            let report = handle.shutdown();
            tracing::info!(
                written = report.written,
                dropped = report.dropped,
                "log sink shut down"
            );
        }
    }
}

pub fn run_id() -> String {
    format!(
        "{}-{}",
        crate::stamp::utc_stamp(SystemTime::now()),
        std::process::id()
    )
}

pub fn auto_path(run_id: &str) -> Result<PathBuf, LogError> {
    use etcetera::BaseStrategy as _;
    let strategy = etcetera::choose_base_strategy().map_err(|_| LogError::NoHome)?;
    let state_dir = strategy.state_dir().ok_or(LogError::NoHome)?;
    Ok(state_dir.join("eye/logs").join(format!("{run_id}.jsonl")))
}

fn default_terminal_filter() -> String {
    let runtime_pins = DEFAULT_SINK_FILTER
        .split_once(',')
        .map(|(_, rest)| rest)
        .unwrap_or("");
    format!("info,{runtime_pins}")
}

fn resolve_level(flag: Option<&str>, env_rust_log: Option<String>) -> String {
    flag.map(str::to_string)
        .filter(|s| !s.trim().is_empty())
        .or_else(|| env_rust_log.filter(|s| !s.trim().is_empty()))
        .unwrap_or_else(default_terminal_filter)
}

fn terminal_filter(spec: &str) -> Result<EnvFilter, LogError> {
    EnvFilter::try_new(spec).map_err(|source| LogError::Filter {
        spec: spec.to_string(),
        source,
    })
}

fn file_filter(spec: &str) -> Result<EnvFilter, LogError> {
    EnvFilter::try_new(spec).map_err(|source| LogError::FileFilter {
        spec: spec.to_string(),
        source,
    })
}

pub fn init(args: &LogArgs, command: &str) -> Result<LogGuard, LogError> {
    let spec = resolve_level(args.log_level.as_deref(), std::env::var("RUST_LOG").ok());
    let filter = terminal_filter(&spec)?;

    let run_id = run_id();

    let terminal = match args.log_format {
        LogFormat::Text => fmt::layer()
            .with_writer(io::stderr)
            .with_filter(filter)
            .boxed(),
        LogFormat::Json => fmt::layer()
            .json()
            .with_writer(io::stderr)
            .with_filter(filter)
            .boxed(),
    };

    let mut sink_handle = None;
    let mut file_path = None;
    let file_layer = match &args.log_file {
        Some(log_file) => {
            let path = if log_file == "auto" {
                auto_path(&run_id)?
            } else {
                PathBuf::from(log_file)
            };
            let sink = JsonLinesSink::create(&path).map_err(|source| LogError::Open {
                path: path.clone(),
                source,
            })?;
            let filter = file_filter(&args.log_file_level)?;
            let (layer, handle) = SinkLayer::spawn(Box::new(sink), 4096);
            sink_handle = Some(handle);
            file_path = Some(path);
            Some(layer.with_filter(filter))
        }
        None => None,
    };

    let subscriber = registry().with(terminal);
    match file_layer {
        Some(layer) => {
            tracing::subscriber::set_global_default(subscriber.with(layer))
                .expect("eye_log::cli::init must be called at most once per process");
        }
        None => {
            tracing::subscriber::set_global_default(subscriber)
                .expect("eye_log::cli::init must be called at most once per process");
        }
    }

    let span = tracing::info_span!(
        span::RUN,
        { field::RUN_ID } = run_id.as_str(),
        { field::COMMAND } = command,
    )
    .entered();

    tracing::info!({ field::RUN_ID } = %run_id, "run started");
    if let Some(path) = &file_path {
        tracing::info!(path = %path.display(), "logging to file");
    }

    Ok(LogGuard {
        span: Some(span),
        sink: sink_handle,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, clap::Parser)]
    struct TestCli {
        #[command(subcommand)]
        command: TestCommand,
        #[command(flatten)]
        log: LogArgs,
    }

    #[derive(Debug, clap::Subcommand)]
    enum TestCommand {
        Sub,
    }

    struct EnvGuard {
        key: &'static str,
        previous: Option<std::ffi::OsString>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: impl AsRef<std::ffi::OsStr>) -> Self {
            let previous = std::env::var_os(key);
            unsafe { std::env::set_var(key, value) };
            Self { key, previous }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match self.previous.take() {
                Some(v) => unsafe { std::env::set_var(self.key, v) },
                None => unsafe { std::env::remove_var(self.key) },
            }
        }
    }

    #[test]
    fn test_log_args_parse_env_and_flags() {
        use clap::Parser;

        let cli = TestCli::try_parse_from(["t", "sub", "--log-file", "x"]).unwrap();
        assert_eq!(cli.log.log_file, Some("x".to_string()));

        {
            let _guard = EnvGuard::set("EYE_LOG", "debug");
            let cli = TestCli::try_parse_from(["t", "sub"]).unwrap();
            assert_eq!(cli.log.log_level, Some("debug".to_string()));
        }

        let had_eye_log = std::env::var_os("EYE_LOG").is_some();
        assert!(!had_eye_log, "EYE_LOG leaked out of the guard");
        let cli = TestCli::try_parse_from(["t", "sub"]).unwrap();
        assert_eq!(cli.log.log_level, None);
    }

    #[test]
    fn test_level_resolution_prefers_flag_then_env_then_info() {
        assert_eq!(
            resolve_level(Some("warn"), Some("debug".to_string())),
            "warn"
        );
        assert_eq!(resolve_level(None, Some("debug".to_string())), "debug");
        assert_eq!(resolve_level(None, None), "info,ort=info,tract=info");
        assert_eq!(resolve_level(Some(""), None), "info,ort=info,tract=info");
    }

    #[test]
    fn test_invalid_filter_is_an_error() {
        let err = terminal_filter("eye[").unwrap_err();
        match err {
            LogError::Filter { spec, source } => {
                assert_eq!(spec, "eye[");
                assert!(source.to_string().contains("invalid filter directive"));
            }
            other => panic!("expected LogError::Filter, got {other:?}"),
        }
    }

    #[test]
    fn test_auto_path_uses_xdg_state_home() {
        let dir = tempfile::tempdir().unwrap();
        let _guard = EnvGuard::set("XDG_STATE_HOME", dir.path());
        let path = auto_path("r").unwrap();
        assert_eq!(path, dir.path().join("eye/logs/r.jsonl"));
    }

    #[test]
    fn test_run_id_is_stamp_dash_pid() {
        let id = run_id();
        let re_ok = id
            .split_once('-')
            .map(|(stamp, pid)| {
                stamp.len() == 16
                    && stamp.starts_with(|c: char| c.is_ascii_digit())
                    && pid == std::process::id().to_string()
            })
            .unwrap_or(false);
        assert!(re_ok, "run id {id:?} does not match <stamp>-<pid>");
        assert!(id.contains('T') && id.contains('Z'));
    }

    fn capture_with_sink_filter(spec: &str, f: impl FnOnce()) -> Vec<crate::Record> {
        let buf = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = crate::testing::VecSink(std::sync::Arc::clone(&buf));
        let (layer, handle) = SinkLayer::spawn(Box::new(sink), 1024);
        let filter = file_filter(spec).expect("valid filter spec");
        let subscriber = registry().with(layer.with_filter(filter));
        tracing::subscriber::with_default(subscriber, f);
        handle.shutdown();
        buf.lock().expect("VecSink mutex poisoned").clone()
    }

    #[test]
    fn test_default_file_filter_excludes_runtime_trace() {
        let records = capture_with_sink_filter(DEFAULT_SINK_FILTER, || {
            tracing::event!(target: "ort::lifetime", tracing::Level::TRACE, "ort trace");
            tracing::event!(
                target: "eye_detect::mediapipe::pipeline",
                tracing::Level::TRACE,
                "eye trace"
            );
        });

        assert_eq!(
            records.len(),
            1,
            "expected only the eye_detect record, got {records:?}"
        );
        assert_eq!(records[0].target, "eye_detect::mediapipe::pipeline");
        assert_eq!(records[0].level, crate::Level::Trace);
    }

    #[test]
    fn test_file_level_override_restores_runtime_trace() {
        let records = capture_with_sink_filter("trace", || {
            tracing::event!(target: "ort::lifetime", tracing::Level::TRACE, "ort trace");
        });

        assert_eq!(records.len(), 1, "expected the ort record, got {records:?}");
        assert_eq!(records[0].target, "ort::lifetime");
        assert_eq!(records[0].level, crate::Level::Trace);
    }

    #[test]
    fn test_log_args_help_names_default_filter() {
        use clap::CommandFactory;

        let cmd = TestCli::command();
        let arg = cmd
            .get_arguments()
            .find(|a| a.get_id() == "log_file_level")
            .expect("log_file_level arg exists");
        let defaults: Vec<String> = arg
            .get_default_values()
            .iter()
            .map(|v| v.to_string_lossy().into_owned())
            .collect();
        assert_eq!(defaults, vec![DEFAULT_SINK_FILTER.to_string()]);
    }
}
