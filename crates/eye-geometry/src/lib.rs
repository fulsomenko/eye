//! Camera, head-pose, eyeball and screen geometry for gaze estimation.
#![forbid(unsafe_code)]

mod error;

pub mod angles;
pub mod camera;
pub mod eyeball;
pub mod face_template;
pub mod lsq;
pub mod pnp;
pub mod screen;
#[cfg(any(test, feature = "synth"))]
pub mod synth;
pub mod triangulation;
pub mod uncertainty;

pub use error::GeometryError;
