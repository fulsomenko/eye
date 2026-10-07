//! Temporal gaze filters and fixation labelling.
#![forbid(unsafe_code)]

mod error;
mod point;

pub mod fixation;
pub mod kalman;
pub mod one_euro;

pub use error::FilterError;
