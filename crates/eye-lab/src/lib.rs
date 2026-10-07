//! Headless hardware and pipeline test runner: camera modes and test sequences.
#![forbid(unsafe_code)]

pub mod case;
pub mod cli;
pub mod hw;
pub mod mode;
pub mod modes;
pub mod pipeline;
pub mod regress;
pub mod report;
pub mod runner;
pub mod selftest;
pub mod sequence;
pub mod signals;
pub mod stats;
pub mod suites;
pub mod testkit;
