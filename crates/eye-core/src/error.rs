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
}
