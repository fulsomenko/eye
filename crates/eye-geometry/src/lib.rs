//! Camera, head-pose, eyeball and screen geometry for gaze estimation.
#![forbid(unsafe_code)]

mod error;

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
pub use eye_core::angles;

#[cfg(test)]
mod tests {
    #[test]
    fn test_angles_reexport_resolves() {
        type F = fn(&nalgebra::Unit<nalgebra::Vector3<f64>>) -> nalgebra::Vector2<f64>;
        let a: F = crate::angles::yaw_pitch_from_direction;
        let b: F = eye_core::angles::yaw_pitch_from_direction;
        assert!(std::ptr::fn_addr_eq(a, b));
    }
}
