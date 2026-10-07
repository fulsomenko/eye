//! Scores gaze pipelines against recorded ground truth.
#![forbid(unsafe_code)]

pub mod calibration;
pub mod error;
pub mod matrix;
pub mod metrics;
pub mod report;
pub mod row;
pub mod runner;
#[cfg(any(test, feature = "testing"))]
pub mod testing;

pub use error::BenchError;
