use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum CalibrationError {
    #[error("invalid calibration config: {0}")]
    Config(#[from] toml::de::Error),
    #[error("invalid parameter `{name}`: {reason}")]
    Param { name: &'static str, reason: String },
    #[error("unknown camera `{0}`")]
    UnknownCamera(String),
    #[error("need at least {need} {what}, got {got}")]
    InsufficientData {
        what: &'static str,
        need: usize,
        got: usize,
    },
    #[error(transparent)]
    Geometry(#[from] eye_geometry::GeometryError),
    #[error(transparent)]
    Core(#[from] eye_core::CoreError),
    #[error(transparent)]
    Store(#[from] StoreError),
}

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("cannot locate the home directory")]
    NoHome,
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("{path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },
    #[error("serialize: {0}")]
    Serialize(#[from] toml::ser::Error),
    #[error("{what}: unsupported format version {found} (expected 1)")]
    UnsupportedVersion { what: &'static str, found: u32 },
    #[error("invalid name `{0}` (allowed: [A-Za-z0-9._-]{{1,64}}, not starting with '.')")]
    InvalidName(String),
}
