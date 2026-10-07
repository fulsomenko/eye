use crate::{CameraId, PixelFormat};

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum CoreError {
    #[error("frame dimensions must be non-zero, got {width}x{height}")]
    ZeroDimension { width: u32, height: u32 },
    #[error("{format:?} frame of {width}x{height} needs {expected} bytes, got {actual}")]
    FrameSize {
        format: PixelFormat,
        width: u32,
        height: u32,
        expected: usize,
        actual: usize,
    },
    #[error("{format:?} frame has no data")]
    EmptyFrame { format: PixelFormat },
    #[error("frame set is empty")]
    EmptyFrameSet,
    #[error("camera {0} appears more than once")]
    DuplicateCamera(CameraId),
    #[error("sigma must be finite and non-negative, got {0}")]
    InvalidSigma(f64),
    #[error(
        "ellipse needs a finite centre and angle and positive finite semi-axes, got a={semi_a} b={semi_b} angle={angle}"
    )]
    InvalidEllipse {
        semi_a: f64,
        semi_b: f64,
        angle: f64,
    },
}
