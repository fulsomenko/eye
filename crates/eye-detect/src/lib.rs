//! Frame to Observations detectors (IR bright pupil, MediaPipe via ort or tract).
#![forbid(unsafe_code)]

mod error;

pub mod ellipse;
pub mod image;
pub mod ir;
pub mod mediapipe;
pub mod mjpeg;
pub mod models;
pub mod options;

pub use error::DetectError;
