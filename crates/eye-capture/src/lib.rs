//! Camera frame sources, frame pairing, recording and replay.
#![forbid(unsafe_code)]

pub mod error;
pub mod illumination;
pub mod pairing;
pub mod session;
pub mod source;
#[cfg(test)]
pub(crate) mod testing;
pub mod uvc_meta;
pub mod v4l2;

pub use error::CaptureError;
