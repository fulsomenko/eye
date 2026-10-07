use nalgebra::{Matrix2, Matrix3};

use crate::{CameraId, OutputId, PixelFormat};

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
    #[error("covariance must be finite, symmetric and positive semi-definite, got {0:?}")]
    InvalidCovariance(Matrix2<f64>),
    #[error("gaze value is not finite or out of range: {0}")]
    InvalidGaze(&'static str),
    #[error("camera model {camera}: {reason}")]
    InvalidCameraModel {
        camera: CameraId,
        reason: &'static str,
    },
    #[error("screen model {output}: {reason}")]
    InvalidScreen {
        output: OutputId,
        reason: &'static str,
    },
    #[error("rig has no camera")]
    EmptyRig,
    #[error("3x3 covariance must be finite, symmetric and positive semi-definite, got {0:?}")]
    InvalidCovariance3(Matrix3<f64>),
}
