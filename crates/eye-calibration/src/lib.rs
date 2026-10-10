//! Camera, rig and per-user gaze calibration.
#![forbid(unsafe_code)]

pub mod checkerboard;
pub mod corners;
pub mod correction;
pub mod error;
pub mod intrinsics;
pub mod nominal;
pub mod profiles;
pub mod protocol;
pub mod stereo;
pub mod store;
#[cfg(test)]
mod testutil;
pub mod user_fit;
mod views;
