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
    NotMonotonic { camera: CameraId, flags: u32 },
    #[error("camera {camera}: no frame within {timeout:?}")]
    Timeout { camera: CameraId, timeout: Duration },
    #[error("camera {camera}: device disconnected")]
    Disconnected { camera: CameraId },
    #[error("camera {camera}: {source}")]
    Io {
        camera: CameraId,
        #[source]
        source: std::io::Error,
    },
    #[error("end of stream")]
    EndOfStream,
    #[error("camera {camera}: invalid options: {reason}")]
    Config { camera: CameraId, reason: String },
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capture_error_display_keeps_camera_prefix_with_typed_id() {
        assert_eq!(
            CaptureError::Timeout {
                camera: CameraId::from("ir"),
                timeout: Duration::from_millis(50),
            }
            .to_string(),
            "camera ir: no frame within 50ms"
        );
        assert_eq!(
            CaptureError::Config {
                camera: CameraId::from("ir"),
                reason: "x".into(),
            }
            .to_string(),
            "camera ir: invalid options: x"
        );
    }
}
