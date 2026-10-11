use nalgebra::{Matrix2, Matrix3, Point2, Point3, Unit, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};

use crate::{CoreError, Side, Timestamp, id::string_id};

string_id!(
    /// Compositor output name, e.g. `eDP-1`.
    OutputId
);

/// Gaze ray in the reference (screen) frame, mm.
#[derive(Debug, Clone, PartialEq)]
pub struct GazeRay {
    /// `None` for a cyclopean or fused ray.
    pub side: Option<Side>,
    /// The instant this ray describes: the observed frame's time, or the alignment time for a
    /// ray built from observations of more than one frame.
    pub timestamp: Timestamp,
    /// Eye centre.
    pub origin: Point3<f64>,
    pub direction: Unit<Vector3<f64>>,
    /// Covariance of `(yaw, pitch)` as defined by [`crate::angles`], rad².
    pub angular_cov: Matrix2<f64>,
    /// Covariance of `origin`, mm².
    pub origin_cov: Matrix3<f64>,
    /// Screen-frame head orientation from a measured PnP pose (or one lent inside the same
    /// fused batch). `None` when no pose was measured for this ray.
    pub head_rotation: Option<UnitQuaternion<f64>>,
}

impl GazeRay {
    pub fn validate(&self) -> Result<(), CoreError> {
        if !self.origin.iter().all(|v| v.is_finite()) {
            return Err(CoreError::InvalidGaze("ray origin"));
        }
        if !self.direction.iter().all(|v| v.is_finite()) {
            return Err(CoreError::InvalidGaze("ray direction"));
        }
        validate_covariance2(&self.angular_cov)?;
        validate_covariance3(&self.origin_cov)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GazePoint {
    pub timestamp: Timestamp,
    pub output: OutputId,
    /// Screen frame of `output`: mm from the top-left corner of the active area.
    pub mm: Point2<f64>,
    /// Derived from `mm` by the eye-geometry screen mapping; a stage that changes `mm` must
    /// re-derive both pixel fields.
    pub px_physical: Point2<f64>,
    pub px_logical: Point2<f64>,
    pub cov_mm: Matrix2<f64>,
    /// In `[0, 1]`.
    pub confidence: f64,
}

impl GazePoint {
    pub fn validate(&self) -> Result<(), CoreError> {
        let points = [self.mm, self.px_physical, self.px_logical];
        if !points.iter().all(|p| p.x.is_finite() && p.y.is_finite()) {
            return Err(CoreError::InvalidGaze("point coordinates"));
        }
        if !(0.0..=1.0).contains(&self.confidence) {
            return Err(CoreError::InvalidGaze("confidence"));
        }
        validate_covariance2(&self.cov_mm)
    }
}

/// The ray model that produced a gaze ray. Keys per-source corrections; for the per-chain
/// variants the serde names equal `eye_estimate::fused::FusedSource::as_str`. `Fused` tags
/// the fused estimator's selected output and has no `FusedSource` counterpart. `RgbOnly`
/// sorts first: it is the primary source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RaySource {
    RgbOnly,
    IrOnly,
    IrOnRgbEyeball,
    IrGlintOnRgbEyeball,
    Stereo,
    Fused,
}

impl RaySource {
    pub const ALL: [RaySource; 6] = [
        Self::RgbOnly,
        Self::IrOnly,
        Self::IrOnRgbEyeball,
        Self::IrGlintOnRgbEyeball,
        Self::Stereo,
        Self::Fused,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::RgbOnly => "rgb-only",
            Self::IrOnly => "ir-only",
            Self::IrOnRgbEyeball => "ir-on-rgb-eyeball",
            Self::IrGlintOnRgbEyeball => "ir-glint-on-rgb-eyeball",
            Self::Stereo => "stereo",
            Self::Fused => "fused",
        }
    }
}

/// A candidate ray tagged with the model that produced it.
#[derive(Debug, Clone, PartialEq)]
pub struct SourcedRay {
    pub source: RaySource,
    pub ray: GazeRay,
}

impl SourcedRay {
    pub fn tag(source: RaySource, rays: Vec<GazeRay>) -> Vec<SourcedRay> {
        rays.into_iter()
            .map(|ray| SourcedRay { source, ray })
            .collect()
    }
}

/// Finite, symmetric (relative tolerance 1e-9) and positive semi-definite.
pub fn validate_covariance2(cov: &Matrix2<f64>) -> Result<(), CoreError> {
    let (a, b, c, d) = (cov[(0, 0)], cov[(0, 1)], cov[(1, 0)], cov[(1, 1)]);
    let scale = a.abs().max(d.abs()).max(f64::MIN_POSITIVE);
    let valid = cov.iter().all(|v| v.is_finite())
        && (b - c).abs() <= 1e-9 * scale
        && a >= 0.0
        && d >= 0.0
        && a * d - b * c >= -1e-12 * scale * scale;
    if valid {
        Ok(())
    } else {
        Err(CoreError::InvalidCovariance(*cov))
    }
}

/// Finite, symmetric (relative tolerance 1e-9) and positive semi-definite (Cholesky of
/// `cov + 1e-12 * scale * I`).
pub fn validate_covariance3(cov: &Matrix3<f64>) -> Result<(), CoreError> {
    let scale = cov.diagonal().amax().max(f64::MIN_POSITIVE);
    let symmetric = (cov - cov.transpose()).amax() <= 1e-9 * scale;
    let psd = (cov + Matrix3::identity() * (1e-12 * scale))
        .cholesky()
        .is_some();
    if cov.iter().all(|v| v.is_finite()) && symmetric && psd {
        Ok(())
    } else {
        Err(CoreError::InvalidCovariance3(*cov))
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{Matrix3, Point3, Vector3};

    use super::*;

    fn eye_fixture() -> Point3<f64> {
        Point3::new(155.0, 85.0, -500.0)
    }

    fn valid_ray() -> GazeRay {
        GazeRay {
            side: None,
            timestamp: Timestamp::from_nanos(0),
            origin: eye_fixture(),
            direction: Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0)),
            angular_cov: Matrix2::identity() * 1e-4,
            origin_cov: Matrix3::zeros(),
            head_rotation: None,
        }
    }

    fn valid_point() -> GazePoint {
        GazePoint {
            timestamp: Timestamp::from_nanos(1_000_000_000),
            output: OutputId::from("eDP-1"),
            mm: Point2::new(155.0, 85.0),
            px_physical: Point2::new(1920.0, 1080.0),
            px_logical: Point2::new(960.0, 540.0),
            cov_mm: Matrix2::new(16.0, 2.0, 2.0, 9.0),
            confidence: 0.8,
        }
    }

    #[test]
    fn test_gaze_ray_validate_rejects_asymmetric_cov() {
        let mut ray = valid_ray();
        ray.angular_cov = Matrix2::new(1e-4, 1e-5, 0.0, 1e-4);
        assert!(matches!(
            ray.validate(),
            Err(CoreError::InvalidCovariance(_))
        ));
    }

    #[test]
    fn test_gaze_ray_validate_rejects_nan_origin() {
        let mut ray = valid_ray();
        ray.origin = Point3::new(f64::NAN, 85.0, -500.0);
        assert_eq!(ray.validate(), Err(CoreError::InvalidGaze("ray origin")));
    }

    #[test]
    fn test_covariance_rejects_indefinite() {
        assert!(validate_covariance2(&Matrix2::new(1.0, 2.0, 2.0, 1.0)).is_err());
        assert!(validate_covariance2(&Matrix2::new(-1.0, 0.0, 0.0, 1.0)).is_err());
        assert!(validate_covariance2(&Matrix2::zeros()).is_ok());
    }

    #[test]
    fn test_gaze_ray_validate_rejects_indefinite_origin_cov() {
        let mut ray = valid_ray();
        ray.origin_cov = Matrix3::new(1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0);
        assert!(matches!(
            ray.validate(),
            Err(CoreError::InvalidCovariance3(_))
        ));

        ray.origin_cov = Matrix3::zeros();
        assert!(ray.validate().is_ok());

        ray.origin_cov = Matrix3::new(4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 25.0);
        assert!(ray.validate().is_ok());
    }

    #[test]
    fn test_covariance3_rejects_asymmetric() {
        let cov = Matrix3::new(1.0, 1e-3, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        assert!(matches!(
            validate_covariance3(&cov),
            Err(CoreError::InvalidCovariance3(_))
        ));
    }

    #[test]
    fn test_gaze_point_validate_rejects_confidence_above_one() {
        let mut point = valid_point();
        point.confidence = 1.5;
        assert_eq!(point.validate(), Err(CoreError::InvalidGaze("confidence")));

        let point = valid_point();
        assert!(point.validate().is_ok());
    }

    #[test]
    fn test_gaze_point_json_roundtrip_equal() {
        let point = valid_point();
        let json = serde_json::to_string(&point).unwrap();
        assert!(json.contains("\"output\":\"eDP-1\""));
        assert!(json.contains("\"timestamp\":1000000000"));
        let parsed: GazePoint = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, point);
    }

    #[test]
    fn test_ray_source_serde_names_match_as_str() {
        for source in RaySource::ALL {
            let json = serde_json::to_string(&source).unwrap();
            assert_eq!(json, format!("\"{}\"", source.as_str()));
            let back: RaySource = serde_json::from_str(&json).unwrap();
            assert_eq!(back, source);
        }
    }

    #[test]
    fn test_ray_source_orders_rgb_only_first() {
        assert_eq!(RaySource::ALL.iter().min(), Some(&RaySource::RgbOnly));
    }

    #[test]
    fn test_ray_source_fused_round_trips_serde() {
        let json = serde_json::to_string(&RaySource::Fused).unwrap();
        assert_eq!(json, "\"fused\"");
        let back: RaySource = serde_json::from_str(&json).unwrap();
        assert_eq!(back, RaySource::Fused);
    }
}
