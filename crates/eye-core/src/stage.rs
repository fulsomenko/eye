use crate::{FrameSet, GazePoint, GazeRay, Illumination, Observations, PixelFormat, Rig};

pub use crate::sink::{GazeSink, SinkError};

/// Error shared by every pipeline stage trait. Implementation crates convert
/// their own error enums into it at the trait boundary.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum StageError {
    #[error("invalid configuration: {0}")]
    Config(String),
    #[error("unsupported: {0}")]
    Unsupported(String),
    #[error("stage failed: {0}")]
    Failed(String),
    #[error("stage closed")]
    Closed,
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
    fn estimate(&mut self, obs: &[Observations], rig: &Rig) -> Result<Vec<GazeRay>, StageError>;
}

/// Per-user correction of a ray. Infallible: an implementation that cannot correct a ray
/// returns it unchanged.
pub trait GazeCorrection: Send {
    fn correct(&self, ray: &GazeRay) -> GazeRay;
}

/// Temporal filter over screen points.
pub trait GazeFilter: Send {
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
        ) -> Result<Vec<GazeRay>, StageError> {
            if obs.is_empty() {
                return Ok(vec![]);
            }
            Ok(vec![GazeRay {
                side: None,
                origin: Point3::new(155.0, 85.0, -500.0),
                direction: Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0)),
                angular_cov: Matrix2::zeros(),
                origin_cov: Matrix3::zeros(),
            }])
        }
    }

    #[derive(Debug)]
    struct ShiftCorrection(f64);

    impl GazeCorrection for ShiftCorrection {
        fn correct(&self, ray: &GazeRay) -> GazeRay {
            let mut shifted = ray.clone();
            shifted.origin.x += self.0;
            shifted
        }
    }

    #[derive(Debug)]
    struct CountingFilter {
        applied: u32,
    }

    impl GazeFilter for CountingFilter {
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
        assert_eq!(err, StageError::Closed);
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
        let shifted = correction.correct(&rays[0]);
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
            StageError::Config("x".into()).to_string(),
            "invalid configuration: x"
        );
        assert_eq!(
            StageError::Unsupported("x".into()).to_string(),
            "unsupported: x"
        );
        assert_eq!(
            StageError::Failed("boom".into()).to_string(),
            "stage failed: boom"
        );
    }
}
