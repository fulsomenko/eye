use std::path::PathBuf;

use eye_core::stage::StageError;

#[derive(Debug, thiserror::Error)]
pub enum DetectError {
    #[error("invalid options for detector {detector}: {source}")]
    Config {
        detector: &'static str,
        #[source]
        source: toml::de::Error,
    },
    #[error(transparent)]
    Core(#[from] eye_core::CoreError),
    #[error("MJPG decode failed: {0}")]
    Decode(String),
    #[error("pixel format {0} is not supported here")]
    UnsupportedFormat(String),
    #[error("model {path}: {reason}")]
    Model { path: PathBuf, reason: String },
    #[error("inference failed: {0}")]
    Inference(String),
}

impl From<DetectError> for StageError {
    fn from(e: DetectError) -> Self {
        match e {
            DetectError::Config { .. } | DetectError::Model { .. } => Self::Config(e.to_string()),
            DetectError::UnsupportedFormat(_) => Self::Unsupported(e.to_string()),
            _ => Self::Failed(e.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toml_error() -> toml::de::Error {
        toml::from_str::<toml::Table>("x =").expect_err("`x =` is not valid TOML")
    }

    #[test]
    fn test_detect_error_maps_to_stage_error_kinds() {
        let config = DetectError::Config {
            detector: "ir-classic",
            source: toml_error(),
        };
        assert!(
            matches!(StageError::from(config), StageError::Config(m) if m.starts_with("invalid options for detector ir-classic"))
        );
        let model = DetectError::Model {
            path: PathBuf::from("/m/face.onnx"),
            reason: "missing".into(),
        };
        assert_eq!(
            StageError::from(model),
            StageError::Config("model /m/face.onnx: missing".into())
        );
        assert_eq!(
            StageError::from(DetectError::UnsupportedFormat("Gray8".into())),
            StageError::Unsupported("pixel format Gray8 is not supported here".into())
        );
        assert_eq!(
            StageError::from(DetectError::Decode("truncated".into())),
            StageError::Failed("MJPG decode failed: truncated".into())
        );
        assert_eq!(
            StageError::from(DetectError::Inference("shape".into())),
            StageError::Failed("inference failed: shape".into())
        );
    }
}
