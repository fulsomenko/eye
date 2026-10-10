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
        match &e {
            DetectError::Config { .. } | DetectError::Model { .. } => Self::Config(Box::new(e)),
            DetectError::UnsupportedFormat(_) => Self::Unsupported(Box::new(e)),
            _ => Self::Failed(Box::new(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error as _;

    use eye_core::stage::StageErrorKind;

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
        let e = StageError::from(config);
        assert_eq!(e.kind(), StageErrorKind::Config);
        assert!(
            e.to_string()
                .starts_with("invalid configuration: invalid options for detector ir-classic")
        );
        assert!(e.source().unwrap().downcast_ref::<DetectError>().is_some());

        let model = DetectError::Model {
            path: PathBuf::from("/m/face.onnx"),
            reason: "missing".into(),
        };
        let e = StageError::from(model);
        assert_eq!(e.kind(), StageErrorKind::Config);
        assert_eq!(
            e.to_string(),
            "invalid configuration: model /m/face.onnx: missing"
        );
        assert!(e.source().unwrap().downcast_ref::<DetectError>().is_some());

        let e = StageError::from(DetectError::UnsupportedFormat("Gray8".into()));
        assert_eq!(e.kind(), StageErrorKind::Unsupported);
        assert_eq!(
            e.to_string(),
            "unsupported: pixel format Gray8 is not supported here"
        );
        assert!(e.source().unwrap().downcast_ref::<DetectError>().is_some());

        let e = StageError::from(DetectError::Decode("truncated".into()));
        assert_eq!(e.kind(), StageErrorKind::Failed);
        assert_eq!(e.to_string(), "stage failed: MJPG decode failed: truncated");
        assert!(e.source().unwrap().downcast_ref::<DetectError>().is_some());

        let e = StageError::from(DetectError::Inference("shape".into()));
        assert_eq!(e.kind(), StageErrorKind::Failed);
        assert_eq!(e.to_string(), "stage failed: inference failed: shape");
        assert!(e.source().unwrap().downcast_ref::<DetectError>().is_some());
    }
}
