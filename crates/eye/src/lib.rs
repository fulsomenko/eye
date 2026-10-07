//! Gaze tracker facade: builds the pipeline from config.
#![forbid(unsafe_code)]

pub mod config;
pub mod error;
pub mod pipeline;
pub mod registry;
#[cfg(test)]
mod testkit;
pub mod tracker;
