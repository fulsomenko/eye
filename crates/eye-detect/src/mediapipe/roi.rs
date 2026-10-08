use std::f64::consts::PI;

use nalgebra::{Point2, Vector2};

use crate::mediapipe::blazeface::{FaceDetection, Letterbox};
use crate::mediapipe::landmarks::index;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RotatedRect {
    pub center: Point2<f64>,
    pub size: f64,
    pub rotation: f64,
}

impl RotatedRect {
    /// Continuous crop coordinates (0..n) to image coordinates.
    pub fn crop_to_image(&self, crop: &Point2<f64>, n: usize) -> Point2<f64> {
        let n = n as f64;
        let q = Vector2::new(
            (crop.x / n - 0.5) * self.size,
            (crop.y / n - 0.5) * self.size,
        );
        let (sin, cos) = self.rotation.sin_cos();
        let rotated = Vector2::new(cos * q.x - sin * q.y, sin * q.x + cos * q.y);
        self.center + rotated
    }
}

fn normalize_radians(angle: f64) -> f64 {
    let wrapped = (angle + PI).rem_euclid(2.0 * PI) - PI;
    if wrapped <= -PI {
        wrapped + 2.0 * PI
    } else {
        wrapped
    }
}

fn rotation_from_eye_line(e0: Point2<f64>, e1: Point2<f64>) -> f64 {
    normalize_radians(-(-(e1.y - e0.y)).atan2(e1.x - e0.x))
}

pub fn roi_from_detection(det: &FaceDetection, letterbox: &Letterbox, n: usize) -> RotatedRect {
    let e0 = letterbox.to_image(&det.keypoints[0], n);
    let e1 = letterbox.to_image(&det.keypoints[1], n);
    let rotation = rotation_from_eye_line(e0, e1);
    let center = letterbox.to_image(&det.center, n);
    let w = det.size.x * n as f64 / letterbox.scale;
    let h = det.size.y * n as f64 / letterbox.scale;
    RotatedRect {
        center,
        size: 1.5 * w.max(h),
        rotation,
    }
}

pub fn roi_from_landmarks(landmarks: &[Point2<f64>]) -> RotatedRect {
    let rotation = rotation_from_eye_line(
        landmarks[index::RIGHT_EYE_LATERAL],
        landmarks[index::LEFT_EYE_LATERAL],
    );

    let (mut min_x, mut min_y) = (f64::INFINITY, f64::INFINITY);
    let (mut max_x, mut max_y) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for p in landmarks {
        min_x = min_x.min(p.x);
        max_x = max_x.max(p.x);
        min_y = min_y.min(p.y);
        max_y = max_y.max(p.y);
    }

    RotatedRect {
        center: Point2::new((min_x + max_x) / 2.0, (min_y + max_y) / 2.0),
        size: 1.5 * (max_x - min_x).max(max_y - min_y),
        rotation,
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;

    use super::*;
    use crate::mediapipe::blazeface::NUM_KEYPOINTS;

    fn detection(
        center: (f64, f64),
        size: (f64, f64),
        kp0: (f64, f64),
        kp1: (f64, f64),
    ) -> FaceDetection {
        let mut keypoints = [Point2::new(0.0, 0.0); NUM_KEYPOINTS];
        keypoints[0] = Point2::new(kp0.0, kp0.1);
        keypoints[1] = Point2::new(kp1.0, kp1.1);
        FaceDetection {
            score: 1.0,
            center: Point2::new(center.0, center.1),
            size: Vector2::new(size.0, size.1),
            keypoints,
        }
    }

    #[test]
    fn test_roi_from_level_eyes_has_zero_rotation() {
        let letterbox = Letterbox::new(1280, 720, 128);
        let det = detection((0.5, 0.5), (0.2, 0.2), (0.4, 0.5), (0.6, 0.5));
        let roi = roi_from_detection(&det, &letterbox, 128);
        assert_abs_diff_eq!(roi.rotation, 0.0, epsilon = 1e-9);
    }

    #[test]
    fn test_roi_rotation_follows_eye_line() {
        let e0 = Point2::new(100.0, 200.0);
        let e1 = Point2::new(160.0, 210.0);
        let rotation = rotation_from_eye_line(e0, e1);
        assert_abs_diff_eq!(rotation, 10.0f64.atan2(60.0), epsilon = 1e-9);
    }

    #[test]
    fn test_roi_side_is_1_5x_long_edge() {
        let letterbox = Letterbox::new(1280, 720, 128);
        let det = detection((0.5, 0.5), (0.2, 0.3), (0.4, 0.5), (0.6, 0.5));
        let roi = roi_from_detection(&det, &letterbox, 128);
        assert_abs_diff_eq!(roi.size, 1.5 * 0.3 * 128.0 / 0.1, epsilon = 1e-9);
    }

    proptest! {
        #[test]
        fn prop_crop_to_image_roundtrip(
            cx in 0.0f64..1280.0,
            cy in 0.0f64..720.0,
            size in 50.0f64..600.0,
            rotation in -PI..=PI,
            cu in 0.0f64..256.0,
            cv in 0.0f64..256.0,
        ) {
            let roi = RotatedRect { center: Point2::new(cx, cy), size, rotation };
            let crop = Point2::new(cu, cv);
            let image = roi.crop_to_image(&crop, 256);

            let delta = image - roi.center;
            let (sin, cos) = (-rotation).sin_cos();
            let q = Vector2::new(cos * delta.x - sin * delta.y, sin * delta.x + cos * delta.y);
            let recovered = Point2::new(
                (q.x / roi.size + 0.5) * 256.0,
                (q.y / roi.size + 0.5) * 256.0,
            );
            prop_assert!((recovered.x - crop.x).abs() < 1e-9);
            prop_assert!((recovered.y - crop.y).abs() < 1e-9);
        }
    }
}
