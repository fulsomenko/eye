//! Shared types of the eye gaze-tracking pipeline.
#![forbid(unsafe_code)]

mod camera;
mod error;
mod frame;
mod gaze;
mod id;
mod rig;
mod sink;
mod time;

pub mod grid;
pub mod image;
pub mod log;
pub mod observation;
pub mod session;
pub mod stage;

pub use camera::{CameraId, CameraInfo, Illumination, PixelFormat};
pub use error::CoreError;
pub use frame::{Frame, FrameHeader, FrameSet};
pub use gaze::{GazePoint, GazeRay, OutputId, validate_covariance2, validate_covariance3};
pub use observation::{
    Ellipse2, EyeCorners, EyeObservation, FaceObservation, Measured, Observations, Side,
};
pub use rig::{CameraModel, Rig, ScreenModel};
pub use sink::{GazeSink, SinkError};
pub use time::Timestamp;
