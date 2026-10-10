use eye_core::{Ellipse2, EyeCorners, EyeObservation, Measured, Side};
use nalgebra::Point2;

use crate::DetectError;
use crate::mediapipe::roi::RotatedRect;
use crate::mediapipe::{LANDMARK_INPUT, MediaPipeOptions, NUM_LANDMARKS};

pub use eye_core::observation::mediapipe478 as index;

pub fn unproject_landmarks(
    raw: &[f32],
    roi: &RotatedRect,
    n: usize,
) -> Result<Vec<Point2<f64>>, DetectError> {
    if raw.len() != NUM_LANDMARKS * 3 {
        return Err(DetectError::Inference(format!(
            "expected {} landmark values, got {}",
            NUM_LANDMARKS * 3,
            raw.len()
        )));
    }
    Ok((0..NUM_LANDMARKS)
        .map(|i| {
            let crop = Point2::new(f64::from(raw[i * 3]), f64::from(raw[i * 3 + 1]));
            roi.crop_to_image(&crop, n)
        })
        .collect())
}

fn iris_points(landmarks: &[Point2<f64>], idx: [usize; 5]) -> [Point2<f64>; 5] {
    idx.map(|i| landmarks[i])
}

fn iris_ellipse(points: [Point2<f64>; 5], sigma: f64) -> Result<Measured<Ellipse2>, DetectError> {
    let [c, p1, p2, p3, p4] = points;
    let d1 = p1 - p3;
    let d2 = p2 - p4;
    let ellipse = Ellipse2::new(c, d1.norm() / 2.0, d2.norm() / 2.0, d1.y.atan2(d1.x))?;
    Ok(Measured::new(ellipse, sigma)?)
}

/// `[right, left]`.
pub fn eyes_from_landmarks(
    landmarks: &[Point2<f64>],
    roi: &RotatedRect,
    options: &MediaPipeOptions,
) -> Result<[EyeObservation; 2], DetectError> {
    let crop_to_image_px = roi.size / LANDMARK_INPUT as f64;
    let sigma_corner = options.corner_sigma_crop_px * crop_to_image_px;
    let sigma_iris = options.iris_sigma_crop_px * crop_to_image_px;

    let mut right = EyeObservation::new(Side::Right);
    right.corners = Some(EyeCorners {
        lateral: Measured::new(landmarks[index::RIGHT_EYE_LATERAL], sigma_corner)?,
        medial: Measured::new(landmarks[index::RIGHT_EYE_MEDIAL], sigma_corner)?,
    });
    right.iris = Some(iris_ellipse(
        iris_points(landmarks, index::RIGHT_IRIS),
        sigma_iris,
    )?);

    let mut left = EyeObservation::new(Side::Left);
    left.corners = Some(EyeCorners {
        lateral: Measured::new(landmarks[index::LEFT_EYE_LATERAL], sigma_corner)?,
        medial: Measured::new(landmarks[index::LEFT_EYE_MEDIAL], sigma_corner)?,
    });
    left.iris = Some(iris_ellipse(
        iris_points(landmarks, index::LEFT_IRIS),
        sigma_iris,
    )?);

    Ok([right, left])
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;
    use crate::mediapipe::roi::RotatedRect;

    fn roi(size: f64) -> RotatedRect {
        RotatedRect {
            center: Point2::new(640.0, 360.0),
            size,
            rotation: 0.0,
        }
    }

    #[test]
    fn test_unproject_wrong_length_is_inference_error() {
        let raw = vec![0.0f32; NUM_LANDMARKS * 3 - 1];
        let err = unproject_landmarks(&raw, &roi(256.0), LANDMARK_INPUT).unwrap_err();
        assert!(matches!(err, DetectError::Inference(_)));
    }

    fn landmarks_fixture() -> Vec<Point2<f64>> {
        let mut lm = vec![Point2::new(640.0, 360.0); NUM_LANDMARKS];
        lm[index::RIGHT_EYE_LATERAL] = Point2::new(500.0, 350.0);
        lm[index::RIGHT_EYE_MEDIAL] = Point2::new(560.0, 352.0);
        lm[index::LEFT_EYE_LATERAL] = Point2::new(780.0, 350.0);
        lm[index::LEFT_EYE_MEDIAL] = Point2::new(720.0, 352.0);
        for (k, &i) in index::RIGHT_IRIS.iter().enumerate() {
            lm[i] = Point2::new(530.0 + k as f64, 351.0);
        }
        for (k, &i) in index::LEFT_IRIS.iter().enumerate() {
            lm[i] = Point2::new(750.0 + k as f64, 351.0);
        }
        lm
    }

    #[test]
    fn test_eyes_assign_subject_sides_and_corners() {
        let lm = landmarks_fixture();
        let eyes = eyes_from_landmarks(&lm, &roi(256.0), &MediaPipeOptions::default()).unwrap();

        assert_eq!(eyes[0].side, Side::Right);
        assert_eq!(eyes[1].side, Side::Left);
        assert_eq!(
            *eyes[0].corners.as_ref().unwrap().lateral.value(),
            lm[index::RIGHT_EYE_LATERAL]
        );
        assert_eq!(
            *eyes[0].corners.as_ref().unwrap().medial.value(),
            lm[index::RIGHT_EYE_MEDIAL]
        );
        assert_eq!(
            *eyes[1].corners.as_ref().unwrap().lateral.value(),
            lm[index::LEFT_EYE_LATERAL]
        );
        assert_eq!(
            eyes[0].iris.as_ref().unwrap().value().center(),
            lm[index::RIGHT_IRIS[0]]
        );
    }

    #[test]
    fn test_iris_ellipse_from_circle_points() {
        let c = Point2::new(10.0, 10.0);
        let points = [
            c,
            Point2::new(15.0, 10.0),
            Point2::new(10.0, 15.0),
            Point2::new(5.0, 10.0),
            Point2::new(10.0, 5.0),
        ];
        let measured = iris_ellipse(points, 0.1).unwrap();
        assert_abs_diff_eq!(measured.value().semi_major(), 5.0, epsilon = 1e-12);
        assert_abs_diff_eq!(measured.value().semi_minor(), 5.0, epsilon = 1e-12);
    }

    #[test]
    fn test_iris_ellipse_from_stretched_points() {
        let c = Point2::new(0.0, 0.0);
        let points = [
            c,
            Point2::new(6.0, 0.0),
            Point2::new(0.0, 4.0),
            Point2::new(-6.0, 0.0),
            Point2::new(0.0, -4.0),
        ];
        let measured = iris_ellipse(points, 0.1).unwrap();
        assert_abs_diff_eq!(measured.value().semi_major(), 6.0, epsilon = 1e-12);
        assert_abs_diff_eq!(measured.value().semi_minor(), 4.0, epsilon = 1e-12);
        assert_abs_diff_eq!(measured.value().angle(), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_sigma_scales_with_roi_size() {
        let lm = landmarks_fixture();
        let options = MediaPipeOptions::default();
        let eyes = eyes_from_landmarks(&lm, &roi(512.0), &options).unwrap();
        assert_abs_diff_eq!(
            eyes[0].iris.as_ref().unwrap().sigma(),
            2.0 * options.iris_sigma_crop_px,
            epsilon = 1e-12
        );
    }
}
