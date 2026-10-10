use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum BenchError {
    #[error("i/o error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("toml: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("pipeline {pipeline:?}: {source}")]
    Config {
        pipeline: String,
        #[source]
        source: eye::error::ConfigError,
    },
    #[error("recording {}: {source}", session.display())]
    Capture {
        session: PathBuf,
        #[source]
        source: eye_capture::CaptureError,
    },
    #[error("calibration: {0}")]
    Calibration(#[source] Box<dyn std::error::Error + Send + Sync + 'static>),
    #[error("pipeline: {0}")]
    Pipeline(#[source] Box<dyn std::error::Error + Send + Sync + 'static>),
    #[error("config camera {camera:?} not in recording {session}")]
    MissingCamera { camera: String, session: String },
    #[error("recording {session} has no [rig] snapshot")]
    NoRig { session: String },
    #[error("rig mismatch for session {session}: {reason}")]
    RigMismatch { session: String, reason: String },
    #[error("rig source: {0}")]
    RigSource(String),
    #[error("bench matrix: {0}")]
    Matrix(String),
}

#[cfg(test)]
mod tests {
    use std::error::Error as _;

    use super::*;

    #[test]
    fn test_boxed_calibration_error_keeps_its_source() {
        let inner = std::io::Error::other("fit diverged");
        let e = BenchError::Calibration(Box::new(inner));
        assert_eq!(e.to_string(), "calibration: fit diverged");
        assert_eq!(
            e.source().map(ToString::to_string),
            Some("fit diverged".to_string())
        );
    }
}
