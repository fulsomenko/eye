use crate::{
    FrameSet, GazePoint, GazeRay, Illumination, Observations, PixelFormat, RaySource, Rig,
    SourcedRay,
};

pub use crate::sink::{GazeSink, SinkError};

pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

/// Error shared by every pipeline stage trait. Implementation crates convert
/// their own error enums into it at the trait boundary; the original error
/// stays reachable through `source()`.
#[derive(Debug, thiserror::Error)]
pub enum StageError {
    #[error("invalid configuration: {0}")]
    Config(#[source] BoxError),
    #[error("unsupported: {0}")]
    Unsupported(#[source] BoxError),
    #[error("stage failed: {0}")]
    Failed(#[source] BoxError),
    #[error("stage closed")]
    Closed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StageErrorKind {
    Config,
    Unsupported,
    Failed,
    Closed,
}

impl StageError {
    pub fn kind(&self) -> StageErrorKind {
        match self {
            Self::Config(_) => StageErrorKind::Config,
            Self::Unsupported(_) => StageErrorKind::Unsupported,
            Self::Failed(_) => StageErrorKind::Failed,
            Self::Closed => StageErrorKind::Closed,
        }
    }

    pub fn config(source: impl Into<BoxError>) -> Self {
        Self::Config(source.into())
    }

    pub fn unsupported(source: impl Into<BoxError>) -> Self {
        Self::Unsupported(source.into())
    }

    pub fn failed(source: impl Into<BoxError>) -> Self {
        Self::Failed(source.into())
    }
}

/// Frames to observations. One call per paired `FrameSet`; returns one `Observations` per frame
/// it used.
pub trait Detector: Send {
    fn name(&self) -> &'static str;
    fn accepts(&self, format: PixelFormat, illumination: Illumination) -> bool;
    fn detect(&mut self, frames: &FrameSet) -> Result<Vec<Observations>, StageError>;
}

/// Observations to gaze rays in the screen frame (R1).
pub trait GazeEstimator: Send {
    fn name(&self) -> &'static str;
    /// Uncorrected candidate rays, each tagged with the model that produced it. Independent of
    /// any user correction, so the pipeline can cache a batch and select from it again.
    fn estimate(&mut self, obs: &[Observations], rig: &Rig) -> Result<Vec<SourcedRay>, StageError>;
    /// The rays to intersect with the screen, chosen from one batch's candidates after applying
    /// `correction`. Default: every candidate, corrected where the correction has an entry and
    /// unchanged otherwise (an estimator with one source emits at most one candidate per side).
    fn select(
        &mut self,
        candidates: &[SourcedRay],
        correction: Option<&dyn GazeCorrection>,
    ) -> Vec<GazeRay> {
        candidates
            .iter()
            .map(|c| {
                correction
                    .and_then(|k| k.correct(c.source, &c.ray))
                    .unwrap_or_else(|| c.ray.clone())
            })
            .collect()
    }
    /// Drops selection state that depends on the correction; `Pipeline::set_correction` calls it.
    fn reset_selection(&mut self) {}
}

/// Per-user correction of a ray from one source.
pub trait GazeCorrection: Send {
    /// `None` when there is no correction for `(ray.side, source)` or the ray cannot be corrected.
    fn correct(&self, source: RaySource, ray: &GazeRay) -> Option<GazeRay>;
}

/// Temporal filter over screen points.
pub trait GazeFilter: Send {
    fn name(&self) -> &'static str;
    fn apply(&mut self, point: GazePoint) -> GazePoint;
    fn reset(&mut self);
}

#[cfg(test)]
mod tests {
    use nalgebra::{Matrix2, Matrix3, Point3, Unit, Vector3};

    use super::*;
    use crate::{CameraId, CameraModel, Frame, FrameHeader, OutputId, ScreenModel, Timestamp};

    #[derive(Debug)]
    struct FakeDetector {
        calls: usize,
    }

    impl Detector for FakeDetector {
        fn name(&self) -> &'static str {
            "fake"
        }

        fn accepts(&self, format: PixelFormat, illumination: Illumination) -> bool {
            matches!(
                (format, illumination),
                (PixelFormat::Gray8, Illumination::IrLit)
            )
        }

        fn detect(&mut self, frames: &FrameSet) -> Result<Vec<Observations>, StageError> {
            self.calls += 1;
            if self.calls == 3 {
                return Err(StageError::Closed);
            }
            Ok(frames
                .frames()
                .iter()
                .map(|f| Observations::empty(f.header().camera.clone(), f.header().timestamp))
                .collect())
        }
    }

    #[derive(Debug)]
    struct FixedEstimator;

    impl GazeEstimator for FixedEstimator {
        fn name(&self) -> &'static str {
            "fixed"
        }

        fn estimate(
            &mut self,
            obs: &[Observations],
            _rig: &Rig,
        ) -> Result<Vec<SourcedRay>, StageError> {
            if obs.is_empty() {
                return Ok(vec![]);
            }
            Ok(SourcedRay::tag(
                RaySource::RgbOnly,
                vec![GazeRay {
                    side: None,
                    timestamp: obs[0].timestamp,
                    origin: Point3::new(155.0, 85.0, -500.0),
                    direction: Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0)),
                    angular_cov: Matrix2::zeros(),
                    origin_cov: Matrix3::zeros(),
                    head_rotation: None,
                }],
            ))
        }
    }

    #[derive(Debug)]
    struct ShiftCorrection(f64);

    impl GazeCorrection for ShiftCorrection {
        fn correct(&self, _source: RaySource, ray: &GazeRay) -> Option<GazeRay> {
            let mut shifted = ray.clone();
            shifted.origin.x += self.0;
            Some(shifted)
        }
    }

    #[derive(Debug)]
    struct RgbOnlyShift(f64);

    impl GazeCorrection for RgbOnlyShift {
        fn correct(&self, source: RaySource, ray: &GazeRay) -> Option<GazeRay> {
            if source != RaySource::RgbOnly {
                return None;
            }
            let mut shifted = ray.clone();
            shifted.origin.x += self.0;
            Some(shifted)
        }
    }

    #[derive(Debug)]
    struct CountingFilter {
        applied: u32,
    }

    impl GazeFilter for CountingFilter {
        fn name(&self) -> &'static str {
            "counting"
        }

        fn apply(&mut self, point: GazePoint) -> GazePoint {
            self.applied += 1;
            point
        }

        fn reset(&mut self) {
            self.applied = 0;
        }
    }

    const fn is_send<T: Send + ?Sized>() {}

    #[test]
    fn test_stage_traits_are_object_safe_and_send() {
        is_send::<Box<dyn Detector>>();
        is_send::<Box<dyn GazeEstimator>>();
        is_send::<Box<dyn GazeCorrection>>();
        is_send::<Box<dyn GazeFilter>>();
        is_send::<Box<dyn GazeSink>>();
    }

    fn ir_frame() -> Frame {
        let data: std::sync::Arc<[u8]> = vec![0u8; 230_400].into();
        Frame::new(
            FrameHeader {
                camera: CameraId::from("ir"),
                seq: 0,
                timestamp: Timestamp::from_nanos(0),
                width: 640,
                height: 360,
                format: PixelFormat::Gray8,
                illumination: Illumination::IrLit,
            },
            data,
        )
        .unwrap()
    }

    fn rgb_frame() -> Frame {
        let data: std::sync::Arc<[u8]> = vec![0xFF, 0xD8, 0xFF, 0xD9].into();
        Frame::new(
            FrameHeader {
                camera: CameraId::from("rgb"),
                seq: 0,
                timestamp: Timestamp::from_nanos(0),
                width: 1280,
                height: 720,
                format: PixelFormat::Mjpeg,
                illumination: Illumination::Ambient,
            },
            data,
        )
        .unwrap()
    }

    #[test]
    fn test_boxed_detector_returns_one_observation_per_frame() {
        let mut detector: Box<dyn Detector> = Box::new(FakeDetector { calls: 0 });
        let frames = FrameSet::new(vec![ir_frame(), rgb_frame()]).unwrap();
        let obs = detector.detect(&frames).unwrap();
        assert_eq!(obs.len(), 2);
        assert_eq!(obs[0].camera, CameraId::from("ir"));
        assert_eq!(obs[1].camera, CameraId::from("rgb"));
        assert!(detector.accepts(PixelFormat::Gray8, Illumination::IrLit));
        assert!(!detector.accepts(PixelFormat::Mjpeg, Illumination::Ambient));
    }

    #[test]
    fn test_detector_error_is_stage_error() {
        let mut detector: Box<dyn Detector> = Box::new(FakeDetector { calls: 0 });
        let frames = FrameSet::new(vec![ir_frame()]).unwrap();
        detector.detect(&frames).unwrap();
        detector.detect(&frames).unwrap();
        let err = detector.detect(&frames).unwrap_err();
        assert_eq!(err.kind(), StageErrorKind::Closed);
        assert_eq!(err.to_string(), "stage closed");
    }

    fn nominal_rig() -> Rig {
        let camera = CameraModel {
            id: CameraId::from("ir"),
            width: 640,
            height: 360,
            fx: 500.0,
            fy: 500.0,
            cx: 320.0,
            cy: 180.0,
            distortion: [0.0; 5],
            screen_from_camera: nalgebra::Isometry3::identity(),
        };
        let screen = ScreenModel {
            output: OutputId::from("eDP-1"),
            size_mm: nalgebra::Vector2::new(310.0, 170.0),
            size_px: (3840, 2160),
            scale: 2.0,
        };
        Rig::new(vec![camera], screen).unwrap()
    }

    #[test]
    fn test_estimator_correction_filter_chain() {
        let mut estimator = FixedEstimator;
        let rig = nominal_rig();
        let obs = vec![Observations::empty(
            CameraId::from("ir"),
            Timestamp::from_nanos(0),
        )];
        let rays = estimator.estimate(&obs, &rig).unwrap();
        assert_eq!(rays.len(), 1);

        let correction = ShiftCorrection(2.0);
        let shifted = correction
            .correct(RaySource::RgbOnly, &rays[0].ray)
            .unwrap();
        approx::assert_abs_diff_eq!(shifted.origin.x, 157.0, epsilon = 1e-12);

        let mut filter = CountingFilter { applied: 0 };
        let point = GazePoint {
            timestamp: Timestamp::from_nanos(0),
            output: OutputId::from("eDP-1"),
            mm: nalgebra::Point2::new(0.0, 0.0),
            px_physical: nalgebra::Point2::new(0.0, 0.0),
            px_logical: nalgebra::Point2::new(0.0, 0.0),
            cov_mm: Matrix2::zeros(),
            confidence: 1.0,
        };
        filter.apply(point.clone());
        filter.apply(point);
        assert_eq!(filter.applied, 2);
        filter.reset();
        assert_eq!(filter.applied, 0);
    }

    #[test]
    fn test_stage_error_display() {
        assert_eq!(
            StageError::config("x").to_string(),
            "invalid configuration: x"
        );
        assert_eq!(StageError::unsupported("x").to_string(), "unsupported: x");
        assert_eq!(StageError::failed("boom").to_string(), "stage failed: boom");
    }

    #[test]
    fn test_stage_error_exposes_source_chain() {
        use std::error::Error as _;

        let e = StageError::config(std::io::Error::other("disk"));
        assert_eq!(e.kind(), StageErrorKind::Config);
        assert_eq!(e.to_string(), "invalid configuration: disk");
        assert!(
            e.source()
                .unwrap()
                .downcast_ref::<std::io::Error>()
                .is_some()
        );
        assert!(StageError::Closed.source().is_none());
    }

    fn fixed_ray(x: f64) -> GazeRay {
        GazeRay {
            side: None,
            timestamp: Timestamp::from_nanos(0),
            origin: Point3::new(x, 85.0, -500.0),
            direction: Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0)),
            angular_cov: Matrix2::zeros(),
            origin_cov: Matrix3::zeros(),
            head_rotation: None,
        }
    }

    #[test]
    fn test_sourced_ray_tag_keeps_order() {
        let tagged = SourcedRay::tag(RaySource::IrOnly, vec![fixed_ray(1.0), fixed_ray(2.0)]);
        assert_eq!(tagged.len(), 2);
        assert_eq!(tagged[0].source, RaySource::IrOnly);
        assert_eq!(tagged[0].ray.origin.x, 1.0);
        assert_eq!(tagged[1].source, RaySource::IrOnly);
        assert_eq!(tagged[1].ray.origin.x, 2.0);
    }

    #[test]
    fn test_default_select_corrects_each_candidate_by_source() {
        let mut estimator = FixedEstimator;
        let candidates = vec![
            SourcedRay {
                source: RaySource::RgbOnly,
                ray: fixed_ray(155.0),
            },
            SourcedRay {
                source: RaySource::IrOnly,
                ray: fixed_ray(200.0),
            },
        ];

        let correction = RgbOnlyShift(2.0);
        let selected = estimator.select(&candidates, Some(&correction));
        assert_eq!(selected.len(), 2);
        approx::assert_abs_diff_eq!(selected[0].origin.x, 157.0, epsilon = 1e-12);
        approx::assert_abs_diff_eq!(selected[1].origin.x, 200.0, epsilon = 1e-12);

        let unselected = estimator.select(&candidates, None);
        approx::assert_abs_diff_eq!(unselected[0].origin.x, 155.0, epsilon = 1e-12);
        approx::assert_abs_diff_eq!(unselected[1].origin.x, 200.0, epsilon = 1e-12);
    }
}
