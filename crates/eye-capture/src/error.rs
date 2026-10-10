use std::{path::PathBuf, time::Duration};

use eye_core::CameraId;

#[derive(Debug, thiserror::Error)]
pub enum CaptureError {
    #[error("open {path}: {source}")]
    Open {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("{path}: requested {requested}, driver chose {actual}")]
    FormatRejected {
        path: PathBuf,
        requested: String,
        actual: String,
    },
    #[error("{path}: pixel format {format} cannot be captured")]
    UnsupportedFormat { path: PathBuf, format: String },
    #[error("camera {camera}: buffer timestamps are not CLOCK_MONOTONIC (flags {flags:#010x})")]
    NotMonotonic { camera: String, flags: u32 },
    #[error("camera {camera}: no frame within {timeout:?}")]
    Timeout { camera: String, timeout: Duration },
    #[error("camera {camera}: device disconnected")]
    Disconnected { camera: String },
    #[error("camera {camera}: {source}")]
    Io {
        camera: String,
        #[source]
        source: std::io::Error,
    },
    #[error("end of stream")]
    EndOfStream,
    #[error("camera {camera}: invalid options: {reason}")]
    Config { camera: String, reason: String },
    #[error("invalid session id {id:?}")]
    InvalidSessionId { id: String },
    #[error("cameras [{}]: {reason}", cameras.iter().map(|c| c.as_str()).collect::<Vec<_>>().join(", "))]
    Pairing {
        cameras: Vec<CameraId>,
        reason: &'static str,
    },
    #[error("recording {path}: {source}")]
    RecordingIo {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("recording {path}: {reason}")]
    RecordingFormat { path: PathBuf, reason: String },
    #[error(transparent)]
    Core(#[from] eye_core::CoreError),
}
