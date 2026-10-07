//! Observations to gaze rays (IR pupil, RGB landmarks, fused).
#![forbid(unsafe_code)]

mod error;

pub mod fused;
pub mod ir;
pub mod ir_pupil;
pub mod landmark;
pub mod options;
pub mod pccr;
#[cfg(test)]
mod testutil;

pub use error::EstimateError;
