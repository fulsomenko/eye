use std::{fmt, path::PathBuf};

use eye_core::stage::StageError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StageKind {
    Detector,
    Estimator,
    Filter,
}

impl fmt::Display for StageKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Detector => "detector",
            Self::Estimator => "estimator",
            Self::Filter => "filter",
        })
    }
}

impl StageKind {
    pub fn span_kind(self) -> &'static str {
        match self {
            Self::Detector => "detect",
            Self::Estimator => "estimate",
            Self::Filter => "filter",
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("reading {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("parsing {origin}: {source}")]
    Parse {
        origin: String,
        #[source]
        source: toml::de::Error,
    },
    #[error("no [[camera]] entries")]
    NoCameras,
    #[error("duplicate camera id {0:?}")]
    DuplicateCamera(String),
    #[error("camera {camera:?}: {reason}")]
    InvalidCamera {
        camera: String,
        reason: &'static str,
    },
    #[error("[detect] names camera {camera:?}, but the cameras are {available:?}")]
    UnknownDetectCamera {
        camera: String,
        available: Vec<String>,
    },
    #[error("[detect] is empty: at least one camera needs a detector")]
    NoDetectors,
    #[error("[capture] channel_capacity must be at least 1")]
    ZeroChannelCapacity,
    #[error("[output] {0}")]
    InvalidOutput(&'static str),
    #[error("unknown {kind} {name:?}; available: {}", available.join(", "))]
    UnknownStage {
        kind: StageKind,
        name: String,
        available: Vec<&'static str>,
    },
    #[error("{kind} {name:?} is not compiled in; rebuild with `--features {feature}`")]
    NotCompiled {
        kind: StageKind,
        name: String,
        feature: &'static str,
    },
    #[error("building {kind} {name:?}: {source}")]
    Stage {
        kind: StageKind,
        name: String,
        #[source]
        source: StageError,
    },
    #[error("{kind} {name:?} registered twice")]
    DuplicateRegistration { kind: StageKind, name: &'static str },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_kind_display_is_lowercase_name() {
        assert_eq!(StageKind::Detector.to_string(), "detector");
        assert_eq!(StageKind::Estimator.to_string(), "estimator");
        assert_eq!(StageKind::Filter.to_string(), "filter");
    }

    #[test]
    fn test_stage_kind_span_kind_matches_vocabulary() {
        assert_eq!(StageKind::Detector.span_kind(), "detect");
        assert_eq!(StageKind::Estimator.span_kind(), "estimate");
        assert_eq!(StageKind::Filter.span_kind(), "filter");
    }
}
