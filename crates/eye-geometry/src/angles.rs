//! Gaze-angle convention, normative for the workspace.
//!
//! Angles are defined relative to the screen frame (R1): origin at the top-left corner of the
//! active area, x right, y down, z into the panel, mm, right-handed. The user's eyes are at
//! `z < 0`; a gaze ray that hits the screen has `d.z > 0`.
//!
//! ```text
//! yaw   = atan2(d.x, d.z)                 // > 0: towards screen +x (the user's right)
//! pitch = atan2(-d.y, hypot(d.x, d.z))     // > 0: up (screen -y)
//! d(yaw, pitch) = (cos(pitch) sin(yaw), -sin(pitch), cos(pitch) cos(yaw))
//! ```
//!
//! `yaw = pitch = 0` is perpendicular into the screen. This is the precise meaning of
//! `GazeRay::angular_cov`: the 2x2 covariance of `[yaw, pitch]` in rad².

use nalgebra::{Unit, Vector2, Vector3};

/// Unit direction for a given `[yaw, pitch]` (rad), in the screen frame (R1).
pub fn direction_from_yaw_pitch(angles: &Vector2<f64>) -> Unit<Vector3<f64>> {
    let (y, p) = (angles.x, angles.y);
    Unit::new_normalize(Vector3::new(p.cos() * y.sin(), -p.sin(), p.cos() * y.cos()))
}

/// `[yaw, pitch]` (rad) for a given unit direction, in the screen frame (R1).
pub fn yaw_pitch_from_direction(d: &Unit<Vector3<f64>>) -> Vector2<f64> {
    Vector2::new(d.x.atan2(d.z), (-d.y).atan2(d.x.hypot(d.z)))
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use nalgebra::{Vector2, Vector3};
    use proptest::prelude::*;

    use super::{direction_from_yaw_pitch, yaw_pitch_from_direction};

    #[test]
    fn test_yaw_pitch_zero_is_into_screen() {
        let d = direction_from_yaw_pitch(&Vector2::new(0.0, 0.0));
        assert_abs_diff_eq!(d.into_inner(), Vector3::new(0.0, 0.0, 1.0), epsilon = 1e-15);
    }

    #[test]
    fn test_yaw_positive_points_right_pitch_positive_points_up() {
        let d_yaw = direction_from_yaw_pitch(&Vector2::new(0.1, 0.0));
        assert!(d_yaw.x > 0.0);

        let d_pitch = direction_from_yaw_pitch(&Vector2::new(0.0, 0.1));
        assert!(d_pitch.y < 0.0);
    }

    proptest! {
        #[test]
        fn prop_yaw_pitch_round_trip(yaw in -1.4f64..1.4, pitch in -1.4f64..1.4) {
            let input = Vector2::new(yaw, pitch);
            let d = direction_from_yaw_pitch(&input);
            let back = yaw_pitch_from_direction(&d);
            assert_abs_diff_eq!(back, input, epsilon = 1e-12);
        }
    }
}
