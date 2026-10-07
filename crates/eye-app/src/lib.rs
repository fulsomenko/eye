//! The eye command-line application.
#![forbid(unsafe_code)]

pub mod capture;
pub mod cli;
pub mod commands;
pub mod ctx;
pub mod paths;
pub mod rig;
pub mod shutdown;
#[cfg(test)]
mod testing;
