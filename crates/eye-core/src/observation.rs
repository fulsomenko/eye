use std::f64::consts::{FRAC_PI_2, PI};

use nalgebra::Point2;

use crate::{CameraId, CoreError, Timestamp};

/// Landmark layout of a [`FaceObservation`]. An open set: a detector that emits a new layout
/// adds a constant here; consumers compare by value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LandmarkScheme(&'static str);

impl LandmarkScheme {
    /// `landmarks` holds the two pupil centres, the subject's right eye first; there is no face
    /// mesh.
    pub const IR_PUPIL_PAIR: Self = Self("ir-pupil-pair");
    /// MediaPipe Face Landmarker: 478 points, iris 468..=477 (see [`mediapipe478`]).
    pub const MEDIAPIPE_478: Self = Self("mediapipe-478");

    pub const fn new(name: &'static str) -> Self {
        Self(name)
    }

    pub const fn as_str(self) -> &'static str {
        self.0
    }
}

impl std::fmt::Display for LandmarkScheme {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

/// Layout of the MediaPipe Face Landmarker output named by [`LandmarkScheme::MEDIAPIPE_478`].
pub mod mediapipe478 {
    pub const COUNT: usize = 478;
    pub const RIGHT_EYE_LATERAL: usize = 33;
    pub const RIGHT_EYE_MEDIAL: usize = 133;
    pub const LEFT_EYE_LATERAL: usize = 263;
    pub const LEFT_EYE_MEDIAL: usize = 362;
    /// Centre first.
    pub const RIGHT_IRIS: [usize; 5] = [468, 469, 470, 471, 472];
    pub const LEFT_IRIS: [usize; 5] = [473, 474, 475, 476, 477];
}

/// A value with its isotropic 1-sigma uncertainty in the value's unit (px for image features;
/// for an [`Ellipse2`] the sigma of its centre).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Measured<T> {
    value: T,
    sigma: f64,
}

impl<T> Measured<T> {
    pub fn new(value: T, sigma: f64) -> Result<Self, CoreError> {
        if sigma.is_finite() && sigma >= 0.0 {
            Ok(Self { value, sigma })
        } else {
            Err(CoreError::InvalidSigma(sigma))
        }
    }

    pub fn value(&self) -> &T {
        &self.value
    }

    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    pub fn into_value(self) -> T {
        self.value
    }

    /// Keeps `sigma`, so `f` must not change the value's scale (a translation such as an ROI
    /// offset is fine, a resize is not).
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Measured<U> {
        Measured {
            value: f(self.value),
            sigma: self.sigma,
        }
    }
}

/// Image-frame ellipse in canonical form: `semi_major >= semi_minor > 0` and the major-axis
/// angle in `[-pi/2, pi/2)` radians, measured from image +x towards +y (0 for a circle).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ellipse2 {
    center: Point2<f64>,
    semi_major: f64,
    semi_minor: f64,
    angle: f64,
}

impl Ellipse2 {
    /// `angle` is the direction of the `semi_a` axis; the axes may come in either order.
    pub fn new(
        center: Point2<f64>,
        semi_a: f64,
        semi_b: f64,
        angle: f64,
    ) -> Result<Self, CoreError> {
        let positive = |v: f64| v.is_finite() && v > 0.0;
        if !(center.x.is_finite()
            && center.y.is_finite()
            && angle.is_finite()
            && positive(semi_a)
            && positive(semi_b))
        {
            return Err(CoreError::InvalidEllipse {
                semi_a,
                semi_b,
                angle,
            });
        }
        let (semi_major, semi_minor, major_angle) = if semi_a >= semi_b {
            (semi_a, semi_b, angle)
        } else {
            (semi_b, semi_a, angle + FRAC_PI_2)
        };
        let angle = if semi_major - semi_minor <= 1e-12 * semi_major {
            0.0
        } else {
            wrap_half_turn(major_angle)
        };
        Ok(Self {
            center,
            semi_major,
            semi_minor,
            angle,
        })
    }

    pub fn circle(center: Point2<f64>, radius: f64) -> Result<Self, CoreError> {
        Self::new(center, radius, radius, 0.0)
    }

    pub fn center(&self) -> Point2<f64> {
        self.center
    }

    pub fn semi_major(&self) -> f64 {
        self.semi_major
    }

    pub fn semi_minor(&self) -> f64 {
        self.semi_minor
    }

    pub fn angle(&self) -> f64 {
        self.angle
    }

    pub fn area(&self) -> f64 {
        PI * self.semi_major * self.semi_minor
    }
}

fn wrap_half_turn(angle: f64) -> f64 {
    let wrapped = (angle + FRAC_PI_2).rem_euclid(PI) - FRAC_PI_2;
    // rem_euclid can round up to exactly PI.
    if wrapped >= FRAC_PI_2 {
        wrapped - PI
    } else {
        wrapped
    }
}

/// The subject's anatomical side. In an unmirrored camera image the subject's right eye
/// appears on the image left.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Side {
    Left,
    Right,
}

impl Side {
    pub const fn opposite(self) -> Self {
        match self {
            Self::Left => Self::Right,
            Self::Right => Self::Left,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EyeCorners {
    /// Outer canthus, towards the ear.
    pub lateral: Measured<Point2<f64>>,
    /// Inner canthus, towards the nose.
    pub medial: Measured<Point2<f64>>,
}

impl EyeCorners {
    pub fn midpoint(&self) -> Point2<f64> {
        nalgebra::center(self.lateral.value(), self.medial.value())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EyeObservation {
    pub side: Side,
    pub corners: Option<EyeCorners>,
    pub iris: Option<Measured<Ellipse2>>,
    pub pupil: Option<Measured<Ellipse2>>,
    pub glints: Vec<Measured<Point2<f64>>>,
}

impl EyeObservation {
    pub fn new(side: Side) -> Self {
        Self {
            side,
            corners: None,
            iris: None,
            pupil: None,
            glints: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FaceObservation {
    pub scheme: LandmarkScheme,
    pub landmarks: Vec<Point2<f64>>,
    /// At most one per side.
    pub eyes: Vec<EyeObservation>,
}

impl FaceObservation {
    pub fn eye(&self, side: Side) -> Option<&EyeObservation> {
        self.eyes.iter().find(|eye| eye.side == side)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Observations {
    pub camera: CameraId,
    pub timestamp: Timestamp,
    pub face: Option<FaceObservation>,
}

impl Observations {
    pub fn empty(camera: CameraId, timestamp: Timestamp) -> Self {
        Self {
            camera,
            timestamp,
            face: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::{assert_abs_diff_eq, assert_relative_eq};
    use nalgebra::{Matrix2, Rotation2};
    use proptest::prelude::*;

    use super::*;

    fn center_fixture() -> Point2<f64> {
        Point2::new(320.5, 180.25)
    }

    #[test]
    fn test_landmark_scheme_count_matches_iris_range() {
        assert_eq!(mediapipe478::LEFT_IRIS[4] + 1, mediapipe478::COUNT);
        let first_iris = mediapipe478::RIGHT_IRIS[0];
        for idx in [
            mediapipe478::RIGHT_EYE_LATERAL,
            mediapipe478::RIGHT_EYE_MEDIAL,
            mediapipe478::LEFT_EYE_LATERAL,
            mediapipe478::LEFT_EYE_MEDIAL,
        ] {
            assert!(idx < first_iris, "corner index {idx} overlaps iris range");
        }
        let right: std::collections::HashSet<_> = mediapipe478::RIGHT_IRIS.iter().collect();
        let left: std::collections::HashSet<_> = mediapipe478::LEFT_IRIS.iter().collect();
        assert!(right.is_disjoint(&left));
        assert_eq!(
            mediapipe478::RIGHT_IRIS[4] + 1,
            mediapipe478::LEFT_IRIS[0],
            "iris ranges are contiguous"
        );
    }

    #[test]
    fn test_measured_accepts_zero_sigma() {
        let m = Measured::new(1.5, 0.0).unwrap();
        assert_eq!(*m.value(), 1.5);
        assert_eq!(m.sigma(), 0.0);
    }

    #[test]
    fn test_measured_rejects_negative_sigma() {
        assert_eq!(Measured::new(1.5, -0.1), Err(CoreError::InvalidSigma(-0.1)));
    }

    #[test]
    fn test_measured_rejects_non_finite_sigma() {
        let err = Measured::new(1.5, f64::NAN).unwrap_err();
        assert!(matches!(err, CoreError::InvalidSigma(s) if s.is_nan()));
        assert_eq!(
            Measured::new(1.5, f64::INFINITY),
            Err(CoreError::InvalidSigma(f64::INFINITY))
        );
    }

    #[test]
    fn test_measured_map_keeps_sigma() {
        let m = Measured::new((10.5, 20.25), 0.3).unwrap();
        let mapped = m.map(|(x, y)| (x + 100.0, y + 50.0));
        assert_eq!(*mapped.value(), (110.5, 70.25));
        assert_eq!(mapped.sigma(), 0.3);
    }

    #[test]
    fn test_ellipse_swaps_axes_and_rotates_angle() {
        let c = center_fixture();
        let e1 = Ellipse2::new(c, 2.0, 3.0, 0.0).unwrap();
        assert_eq!(e1.semi_major(), 3.0);
        assert_eq!(e1.semi_minor(), 2.0);
        assert_abs_diff_eq!(e1.angle(), -FRAC_PI_2, epsilon = 1e-12);

        let e2 = Ellipse2::new(c, 3.0, 2.0, FRAC_PI_2).unwrap();
        assert_abs_diff_eq!(e2.angle(), e1.angle(), epsilon = 1e-12);
    }

    #[test]
    fn test_ellipse_angle_wraps_into_half_open_range() {
        let c = center_fixture();
        let e1 = Ellipse2::new(c, 3.0, 2.0, 3.0 * PI / 4.0).unwrap();
        assert_abs_diff_eq!(e1.angle(), -PI / 4.0, epsilon = 1e-12);

        let e2 = Ellipse2::new(c, 3.0, 2.0, -FRAC_PI_2).unwrap();
        assert_abs_diff_eq!(e2.angle(), -FRAC_PI_2, epsilon = 1e-12);
    }

    #[test]
    fn test_circle_has_zero_angle() {
        let c = center_fixture();
        assert_eq!(Ellipse2::new(c, 2.0, 2.0, 1.0).unwrap().angle(), 0.0);
        assert_eq!(Ellipse2::circle(c, 3.0).unwrap().semi_minor(), 3.0);
    }

    #[test]
    fn test_ellipse_rejects_non_positive_axis() {
        let c = center_fixture();
        assert!(Ellipse2::new(c, 0.0, 2.0, 0.0).is_err());
        assert!(Ellipse2::new(c, -1.0, 2.0, 0.0).is_err());
    }

    #[test]
    fn test_ellipse_rejects_nan_centre() {
        let c = Point2::new(f64::NAN, 1.0);
        assert!(Ellipse2::new(c, 1.0, 2.0, 0.0).is_err());
    }

    #[test]
    fn test_ellipse_area() {
        let c = center_fixture();
        let e = Ellipse2::new(c, 3.0, 2.0, 0.0).unwrap();
        assert_relative_eq!(e.area(), 6.0 * PI, epsilon = 1e-12);
    }

    #[test]
    fn test_eye_corners_midpoint() {
        let corners = EyeCorners {
            lateral: Measured::new(Point2::new(300.0, 180.0), 0.0).unwrap(),
            medial: Measured::new(Point2::new(330.0, 182.0), 0.0).unwrap(),
        };
        let mid = corners.midpoint();
        assert_eq!(mid, Point2::new(315.0, 181.0));
    }

    #[test]
    fn test_face_observation_eye_finds_side() {
        let face = FaceObservation {
            scheme: LandmarkScheme::new("test"),
            landmarks: Vec::new(),
            eyes: vec![EyeObservation::new(Side::Right)],
        };
        assert!(face.eye(Side::Right).is_some());
        assert!(face.eye(Side::Left).is_none());
        assert_eq!(Side::Left.opposite(), Side::Right);
    }

    #[test]
    fn test_landmark_scheme_values_are_distinct_and_display_their_name() {
        assert_ne!(LandmarkScheme::IR_PUPIL_PAIR, LandmarkScheme::MEDIAPIPE_478);
        assert_eq!(LandmarkScheme::MEDIAPIPE_478.as_str(), "mediapipe-478");
        assert_eq!(
            format!("{}", LandmarkScheme::IR_PUPIL_PAIR),
            "ir-pupil-pair"
        );
        assert_eq!(
            LandmarkScheme::new("mediapipe-478"),
            LandmarkScheme::MEDIAPIPE_478
        );
        let face = FaceObservation {
            scheme: LandmarkScheme::MEDIAPIPE_478,
            landmarks: Vec::new(),
            eyes: Vec::new(),
        };
        assert_ne!(face.scheme, LandmarkScheme::IR_PUPIL_PAIR);
    }

    #[test]
    fn test_observations_empty_has_no_face() {
        let obs = Observations::empty(CameraId::new("cam0"), Timestamp::from_nanos(0));
        assert!(obs.face.is_none());
    }

    fn conic(a: f64, b: f64, angle: f64) -> Matrix2<f64> {
        let r = Rotation2::new(angle).into_inner();
        r * Matrix2::new(1.0 / (a * a), 0.0, 0.0, 1.0 / (b * b)) * r.transpose()
    }

    proptest! {
        #[test]
        fn test_ellipse_normalisation_preserves_conic(a in 0.1f64..100.0, b in 0.1f64..100.0, angle in -10.0f64..10.0) {
            let c = center_fixture();
            let e = Ellipse2::new(c, a, b, angle).unwrap();
            prop_assert!(e.semi_major() >= e.semi_minor());
            prop_assert!(e.angle() >= -FRAC_PI_2 && e.angle() < FRAC_PI_2);

            let input_conic = conic(a, b, angle);
            let canonical_conic = conic(e.semi_major(), e.semi_minor(), e.angle());
            let max_abs = input_conic.iter().chain(canonical_conic.iter()).fold(0.0f64, |m, v| m.max(v.abs()));
            let tol = 1e-9 * max_abs;
            for (x, y) in input_conic.iter().zip(canonical_conic.iter()) {
                prop_assert!((x - y).abs() <= tol);
            }
        }
    }
}
