//! Synchronous pipeline core: frames in, `GazePoint`s out, no threads, no I/O.

use std::collections::VecDeque;
use std::fmt;

use eye_capture::CaptureError;
use eye_capture::pairing::{Pairer, PairingConfig};
use eye_core::{
    CameraId, CameraInfo, Frame, FrameSet, GazePoint, GazeRay, PixelFormat, Rig, ScreenModel,
    Timestamp,
    stage::{Detector, GazeCorrection, GazeEstimator, GazeFilter, StageError},
};
use eye_geometry::screen::{confidence_from_cov, gaze_point, mm_to_px_logical, mm_to_px_physical};
use nalgebra::{Matrix2, Point2, Vector2};

use crate::config::Config;
use crate::error::{ConfigError, StageKind};
use crate::registry::Registry;

/// Uncorrected rays of one frame set; `timestamp` is the newest frame's.
#[derive(Debug, Clone, PartialEq)]
pub struct RayBatch {
    pub timestamp: Timestamp,
    pub rays: Vec<GazeRay>,
}

#[derive(Debug)]
pub enum RayStep {
    /// No detector accepted a frame, or no observation / ray resulted.
    NoGaze,
    /// A detector or the estimator failed; the set is skipped (counted towards the threshold).
    Skipped(StageError),
    Rays(RayBatch),
}

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error(transparent)]
    Config(#[from] ConfigError),
    #[error("[detect] names camera {camera:?}, but no CameraInfo was given for it")]
    MissingCamera { camera: String },
    #[error("pairing: {0}")]
    Pairing(#[source] CaptureError),
    #[error("{kind} {name:?} failed {count} times in a row: {source}")]
    Stage {
        kind: StageKind,
        name: &'static str,
        count: u32,
        #[source]
        source: StageError,
    },
}

pub struct Pipeline {
    rig: Rig,
    pairer: Pairer,
    pending: VecDeque<FrameSet>,
    detectors: Vec<(CameraId, Box<dyn Detector>)>,
    estimator: Box<dyn GazeEstimator>,
    filter: Box<dyn GazeFilter>,
    correction: Option<Box<dyn GazeCorrection>>,
    max_consecutive_errors: u32,
    consecutive_errors: u32,
}

impl Pipeline {
    /// Builds every stage through `registry` (no I/O). `cameras` are the streams' infos (the
    /// recording's in the bench, `CameraConfig::to_info` live). A Gray8 camera becomes the
    /// Pairer's primary; `config.capture.rgb_offset_ns` becomes `offset_secondary_ns`.
    pub fn from_config(
        registry: &Registry,
        config: &Config,
        rig: Rig,
        cameras: &[CameraInfo],
        correction: Option<Box<dyn GazeCorrection>>,
    ) -> Result<Pipeline, PipelineError> {
        let mut detectors = Vec::with_capacity(config.detect.len());
        for (camera, section) in &config.detect {
            if !cameras.iter().any(|c| c.id.as_str() == camera) {
                return Err(PipelineError::MissingCamera {
                    camera: camera.clone(),
                });
            }
            detectors.push((
                CameraId::from(camera.as_str()),
                registry.detector(section, &rig)?,
            ));
        }
        let estimator = registry.estimator(&config.estimate, &rig)?;
        let filter = registry.filter(&config.filter, &rig)?;

        let mut ordered = cameras.to_vec();
        ordered.sort_by_key(|c| c.format != PixelFormat::Gray8);
        let pairer = match ordered.as_slice() {
            [_, _] => Pairer::with_config(
                &ordered,
                PairingConfig {
                    offset_secondary_ns: config.capture.rgb_offset_ns,
                    ..PairingConfig::default()
                },
            ),
            _ => Pairer::new(&ordered),
        }
        .map_err(PipelineError::Pairing)?;

        Ok(Self::new(
            rig,
            pairer,
            detectors,
            estimator,
            filter,
            correction,
            config.tracker.max_consecutive_stage_errors,
        ))
    }

    pub fn new(
        rig: Rig,
        pairer: Pairer,
        detectors: Vec<(CameraId, Box<dyn Detector>)>,
        estimator: Box<dyn GazeEstimator>,
        filter: Box<dyn GazeFilter>,
        correction: Option<Box<dyn GazeCorrection>>,
        max_consecutive_errors: u32,
    ) -> Pipeline {
        Self {
            rig,
            pairer,
            pending: VecDeque::new(),
            detectors,
            estimator,
            filter,
            correction,
            max_consecutive_errors,
            consecutive_errors: 0,
        }
    }

    /// Cameras that have a detector, in `[detect]` key order.
    pub fn cameras(&self) -> Vec<CameraId> {
        self.detectors
            .iter()
            .map(|(camera, _)| camera.clone())
            .collect()
    }

    pub fn rig(&self) -> &Rig {
        &self.rig
    }

    /// Pushes one frame into the Pairer (arrival order); never drops a frame. The Pairer may
    /// complete more than one set per push (e.g. a stale held frame released by a new arrival);
    /// extras are queued and returned by the next `pair`/`flush` call before any new frame.
    pub fn pair(&mut self, frame: Frame) -> Option<FrameSet> {
        self.pending.extend(self.pairer.push(frame));
        self.pending.pop_front()
    }

    /// Releases a held frame alone (end of stream, or a stalled camera).
    pub fn flush(&mut self) -> Option<FrameSet> {
        self.pending.pop_front().or_else(|| self.pairer.flush())
    }

    /// Detect + estimate. A detector/estimator error returns `Ok(Skipped(e))` (warn) until more
    /// than `max_consecutive_errors` in a row, then `Err(Stage)`. Any non-error result resets the
    /// count.
    pub fn rays(&mut self, set: &FrameSet) -> Result<RayStep, PipelineError> {
        let timestamp = set
            .frames()
            .iter()
            .map(|f| f.header().timestamp)
            .max()
            .expect("a FrameSet is non-empty");

        let mut observations = Vec::new();
        let mut failure = None;
        for (camera, detector) in &mut self.detectors {
            let frames: Vec<Frame> = set
                .frames()
                .iter()
                .filter(|f| {
                    let h = f.header();
                    h.camera == *camera && detector.accepts(h.format, h.illumination)
                })
                .cloned()
                .collect();
            if frames.is_empty() {
                continue;
            }
            let subset = FrameSet::new(frames).expect("a subset of a valid FrameSet is valid");
            match detector.detect(&subset) {
                Ok(obs) => observations.extend(obs),
                Err(e) => {
                    failure = Some((StageKind::Detector, detector.name(), e));
                    break;
                }
            }
        }
        if let Some((kind, name, e)) = failure {
            return self.stage_failed(kind, name, e);
        }
        if observations.is_empty() {
            self.consecutive_errors = 0;
            return Ok(RayStep::NoGaze);
        }
        match self.estimator.estimate(&observations, &self.rig) {
            Ok(rays) => {
                self.consecutive_errors = 0;
                Ok(if rays.is_empty() {
                    RayStep::NoGaze
                } else {
                    RayStep::Rays(RayBatch { timestamp, rays })
                })
            }
            Err(e) => {
                let name = self.estimator.name();
                self.stage_failed(StageKind::Estimator, name, e)
            }
        }
    }

    fn stage_failed(
        &mut self,
        kind: StageKind,
        name: &'static str,
        e: StageError,
    ) -> Result<RayStep, PipelineError> {
        self.consecutive_errors += 1;
        if self.consecutive_errors > self.max_consecutive_errors {
            return Err(PipelineError::Stage {
                kind,
                name,
                count: self.consecutive_errors,
                source: e,
            });
        }
        tracing::warn!(
            %kind,
            name,
            error = %e,
            consecutive = self.consecutive_errors,
            "stage error, frame set skipped"
        );
        Ok(RayStep::Skipped(e))
    }

    /// Correct each ray, intersect, fuse, filter. Under `cfg!(debug_assertions)` an invalid point
    /// is dropped with `warn!`.
    pub fn finish(&mut self, batch: &RayBatch) -> Option<GazePoint> {
        let screen = self.rig.screen();
        let points: Vec<GazePoint> = batch
            .rays
            .iter()
            .map(|ray| {
                self.correction
                    .as_ref()
                    .map_or_else(|| ray.clone(), |c| c.correct(ray))
            })
            .filter_map(|ray| gaze_point(&ray, screen, batch.timestamp))
            .collect();
        let fused = fuse_points(&points, screen)?;
        let point = self.filter.apply(fused);
        if cfg!(debug_assertions)
            && let Err(e) = point.validate()
        {
            tracing::warn!(error = %e, "invalid gaze point dropped");
            return None;
        }
        Some(point)
    }

    /// Replaces the correction and calls `filter.reset()` once.
    pub fn set_correction(&mut self, correction: Option<Box<dyn GazeCorrection>>) {
        self.correction = correction;
        self.filter.reset();
    }
}

impl fmt::Debug for Pipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Pipeline")
            .field("cameras", &self.cameras())
            .field(
                "detectors",
                &self
                    .detectors
                    .iter()
                    .map(|(_, d)| d.name())
                    .collect::<Vec<_>>(),
            )
            .field("estimator", &self.estimator.name())
            .field("has_correction", &self.correction.is_some())
            .field("max_consecutive_errors", &self.max_consecutive_errors)
            .finish_non_exhaustive()
    }
}

/// Information-form fusion in mm: `W_i = C_i^-1`, `C = (sum W_i)^-1`, `mu = C sum W_i mu_i`.
/// Points with a non-finite or singular `cov_mm` are dropped; none left gives `None`; one point
/// is returned unchanged.
pub fn fuse_points(points: &[GazePoint], screen: &ScreenModel) -> Option<GazePoint> {
    let usable: Vec<(&GazePoint, Matrix2<f64>)> = points
        .iter()
        .filter(|p| p.cov_mm.iter().all(|v| v.is_finite()))
        .filter_map(|p| p.cov_mm.try_inverse().map(|w| (p, w)))
        .collect();
    match usable.as_slice() {
        [] => None,
        [(p, _)] => Some((*p).clone()),
        many => {
            let info: Matrix2<f64> = many.iter().map(|(_, w)| w).sum();
            let cov = info.try_inverse()?;
            let mm = Point2::from(
                cov * many
                    .iter()
                    .map(|(p, w)| w * p.mm.coords)
                    .sum::<Vector2<f64>>(),
            );
            Some(GazePoint {
                timestamp: many
                    .iter()
                    .map(|(p, _)| p.timestamp)
                    .max()
                    .expect("non-empty"),
                output: screen.output.clone(),
                px_physical: mm_to_px_physical(screen, &mm),
                px_logical: mm_to_px_logical(screen, &mm),
                confidence: confidence_from_cov(&cov),
                mm,
                cov_mm: cov,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    use approx::assert_relative_eq;
    use eye_core::Illumination;

    use super::*;
    use crate::registry::PassThroughFilter;
    use crate::testkit::{
        self, CountingFilter, FailingEstimator, FakeDetector, FakeEstimator, TWO_CAMERA_TOML,
        YawOffset, fake_registry,
    };

    fn one_camera_pipeline(max_consecutive_errors: u32) -> Pipeline {
        let info = testkit::info("ir", PixelFormat::Gray8, 66);
        Pipeline::new(
            testkit::rig(),
            Pairer::new(&[info]).expect("single-camera pairer"),
            vec![(CameraId::from("ir"), Box::new(FakeDetector::new("fake")))],
            Box::new(FakeEstimator),
            Box::new(PassThroughFilter),
            None,
            max_consecutive_errors,
        )
    }

    #[test]
    fn test_finish_hits_screen_centre() {
        let mut pipeline = one_camera_pipeline(3);
        let frame = testkit::frame("ir", 0, 100, Illumination::IrLit);
        let set = FrameSet::single(frame);

        let RayStep::Rays(batch) = pipeline.rays(&set).expect("rays succeeds") else {
            panic!("expected a ray batch")
        };
        assert_eq!(batch.timestamp, Timestamp::from_nanos(100_000_000));

        let point = pipeline.finish(&batch).expect("ray hits the panel");
        assert_eq!(point.timestamp, batch.timestamp);
        assert_relative_eq!(point.mm.x, 155.0, epsilon = 1e-6);
        assert_relative_eq!(point.mm.y, 85.0, epsilon = 1e-6);
        assert_relative_eq!(point.px_logical.x, 960.0, epsilon = 1e-3);
        assert_relative_eq!(point.px_logical.y, 540.0, epsilon = 1e-3);
    }

    #[test]
    fn test_finish_misses_panel_behind_eye_returns_none() {
        let mut pipeline = one_camera_pipeline(3);
        let batch = RayBatch {
            timestamp: Timestamp::from_nanos(0),
            rays: vec![GazeRay {
                side: None,
                origin: nalgebra::Point3::new(155.0, 85.0, -500.0),
                direction: -nalgebra::Vector3::z_axis(),
                angular_cov: Matrix2::identity() * 1e-6,
                origin_cov: nalgebra::Matrix3::zeros(),
            }],
        };
        assert!(pipeline.finish(&batch).is_none());
    }

    #[test]
    fn test_finish_ray_pointing_away_from_panel_returns_none() {
        let mut pipeline = one_camera_pipeline(3);
        let batch = RayBatch {
            timestamp: Timestamp::from_nanos(0),
            rays: vec![GazeRay {
                side: None,
                origin: nalgebra::Point3::new(155.0, 85.0, 500.0),
                direction: nalgebra::Vector3::z_axis(),
                angular_cov: Matrix2::identity() * 1e-6,
                origin_cov: nalgebra::Matrix3::zeros(),
            }],
        };
        assert!(pipeline.finish(&batch).is_none());
    }

    #[test]
    fn test_finish_applies_correction() {
        let mut pipeline = one_camera_pipeline(3);
        pipeline.set_correction(Some(Box::new(YawOffset(0.01))));

        let set = FrameSet::single(testkit::frame("ir", 0, 100, Illumination::IrLit));
        let RayStep::Rays(batch) = pipeline.rays(&set).expect("rays succeeds") else {
            panic!("expected a ray batch")
        };
        let point = pipeline.finish(&batch).expect("ray hits the panel");
        assert_relative_eq!(point.mm.x, 155.0 + 500.0 * 0.01_f64.tan(), epsilon = 1e-9);
        assert_relative_eq!(point.mm.y, 85.0, epsilon = 1e-9);
    }

    #[test]
    fn test_rays_routes_frames_to_bound_detector_only() {
        let a = FakeDetector::new("a");
        let b = FakeDetector::new("b");
        let a_seen = a.seen.clone();
        let b_seen = b.seen.clone();
        let mut pipeline = Pipeline::new(
            testkit::rig(),
            Pairer::new(&[testkit::info("ir", PixelFormat::Gray8, 66)])
                .expect("single-camera pairer"),
            vec![
                (CameraId::from("rgb"), Box::new(a)),
                (CameraId::from("ir"), Box::new(b)),
            ],
            Box::new(FakeEstimator),
            Box::new(PassThroughFilter),
            None,
            3,
        );

        let set = FrameSet::new(vec![
            testkit::frame("ir", 0, 105, Illumination::IrLit),
            testkit::frame_fmt("rgb", 0, 100, PixelFormat::Mjpeg, Illumination::Ambient),
        ])
        .expect("distinct cameras");

        let RayStep::Rays(batch) = pipeline.rays(&set).expect("rays succeeds") else {
            panic!("expected a ray batch")
        };
        assert_eq!(batch.timestamp, Timestamp::from_nanos(105_000_000));
        assert_eq!(*a_seen.lock().unwrap(), vec![CameraId::from("rgb")]);
        assert_eq!(*b_seen.lock().unwrap(), vec![CameraId::from("ir")]);
    }

    #[test]
    fn test_rays_skips_unaccepted_frames_no_gaze() {
        let detector = FakeDetector::new("fake").accepting(Illumination::IrLit);
        let mut pipeline = Pipeline::new(
            testkit::rig(),
            Pairer::new(&[testkit::info("ir", PixelFormat::Gray8, 66)])
                .expect("single-camera pairer"),
            vec![(CameraId::from("ir"), Box::new(detector))],
            Box::new(FakeEstimator),
            Box::new(PassThroughFilter),
            None,
            3,
        );

        let set = FrameSet::single(testkit::frame("ir", 0, 100, Illumination::IrDark));
        assert!(matches!(pipeline.rays(&set), Ok(RayStep::NoGaze)));
    }

    #[test]
    fn test_rays_errors_skip_then_fail_after_threshold() {
        let detector = FakeDetector::new("fake").failing_always();
        let mut pipeline = Pipeline::new(
            testkit::rig(),
            Pairer::new(&[testkit::info("ir", PixelFormat::Gray8, 66)])
                .expect("single-camera pairer"),
            vec![(CameraId::from("ir"), Box::new(detector))],
            Box::new(FakeEstimator),
            Box::new(PassThroughFilter),
            None,
            3,
        );
        let set = FrameSet::single(testkit::frame("ir", 0, 100, Illumination::IrLit));

        for _ in 0..3 {
            assert!(matches!(pipeline.rays(&set), Ok(RayStep::Skipped(_))));
        }
        let Err(PipelineError::Stage {
            kind, name, count, ..
        }) = pipeline.rays(&set)
        else {
            panic!("expected a fatal stage error")
        };
        assert_eq!(kind, StageKind::Detector);
        assert_eq!(name, "fake");
        assert_eq!(count, 4);
    }

    #[test]
    fn test_rays_success_resets_error_count() {
        let detector = FakeDetector::new("fake").failing(&[true, true, false, true, true, true]);
        let mut pipeline = Pipeline::new(
            testkit::rig(),
            Pairer::new(&[testkit::info("ir", PixelFormat::Gray8, 66)])
                .expect("single-camera pairer"),
            vec![(CameraId::from("ir"), Box::new(detector))],
            Box::new(FakeEstimator),
            Box::new(PassThroughFilter),
            None,
            3,
        );
        let set = FrameSet::single(testkit::frame("ir", 0, 100, Illumination::IrLit));

        for _ in 0..6 {
            assert!(pipeline.rays(&set).is_ok());
        }
    }

    #[test]
    fn test_rays_estimator_error_is_skipped() {
        let mut pipeline = Pipeline::new(
            testkit::rig(),
            Pairer::new(&[testkit::info("ir", PixelFormat::Gray8, 66)])
                .expect("single-camera pairer"),
            vec![(CameraId::from("ir"), Box::new(FakeDetector::new("fake")))],
            Box::new(FailingEstimator),
            Box::new(PassThroughFilter),
            None,
            3,
        );
        let set = FrameSet::single(testkit::frame("ir", 0, 100, Illumination::IrLit));
        assert!(matches!(pipeline.rays(&set), Ok(RayStep::Skipped(_))));
    }

    #[test]
    fn test_set_correction_resets_filter_once() {
        let resets = Arc::new(AtomicU32::new(0));
        let mut pipeline = Pipeline::new(
            testkit::rig(),
            Pairer::new(&[testkit::info("ir", PixelFormat::Gray8, 66)])
                .expect("single-camera pairer"),
            vec![(CameraId::from("ir"), Box::new(FakeDetector::new("fake")))],
            Box::new(FakeEstimator),
            Box::new(CountingFilter {
                resets: resets.clone(),
            }),
            None,
            3,
        );
        pipeline.set_correction(Some(Box::new(YawOffset(0.0))));
        assert_eq!(resets.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_pair_single_camera_emits_each_frame() {
        let infos = [testkit::info("ir", PixelFormat::Gray8, 66)];
        let mut pipeline = Pipeline::new(
            testkit::rig(),
            Pairer::new(&infos).expect("single-camera pairer"),
            vec![(CameraId::from("ir"), Box::new(FakeDetector::new("fake")))],
            Box::new(FakeEstimator),
            Box::new(PassThroughFilter),
            None,
            3,
        );
        for seq in 0..3 {
            let set = pipeline
                .pair(testkit::frame("ir", seq, seq * 66, Illumination::IrLit))
                .expect("single camera emits every frame");
            assert_eq!(set.frames().len(), 1);
        }
    }

    #[test]
    fn test_pair_gray_camera_is_primary() {
        let config = Config::from_toml_str(TWO_CAMERA_TOML).expect("parses");
        let mut pipeline = Pipeline::from_config(
            &fake_registry(),
            &config,
            testkit::rig(),
            &[
                testkit::info("rgb", PixelFormat::Mjpeg, 33),
                testkit::info("ir", PixelFormat::Gray8, 66),
            ],
            None,
        )
        .expect("builds without I/O");

        assert!(
            pipeline
                .pair(testkit::frame_fmt(
                    "rgb",
                    0,
                    100,
                    PixelFormat::Mjpeg,
                    Illumination::Ambient
                ))
                .is_none()
        );
        let set = pipeline
            .pair(testkit::frame("ir", 1, 108, Illumination::IrLit))
            .expect("ir release pairs with the held rgb frame");
        let ids: Vec<&str> = set
            .frames()
            .iter()
            .map(|f| f.header().camera.as_str())
            .collect();
        assert_eq!(ids, vec!["ir", "rgb"]);
    }

    #[test]
    fn test_flush_releases_held_frame() {
        let config = Config::from_toml_str(TWO_CAMERA_TOML).expect("parses");
        let cameras: Vec<CameraInfo> = config.cameras.iter().map(|c| c.to_info()).collect();
        let mut pipeline =
            Pipeline::from_config(&fake_registry(), &config, testkit::rig(), &cameras, None)
                .expect("builds without I/O");

        let rgb = config.camera("rgb").expect("rgb exists").to_info();
        assert!(
            pipeline
                .pair(testkit::frame_fmt(
                    "rgb",
                    0,
                    100,
                    rgb.format,
                    Illumination::Ambient
                ))
                .is_none()
        );
        let held = pipeline.flush().expect("held rgb frame is released");
        assert_eq!(held.frames().len(), 1);
        assert_eq!(held.frames()[0].header().camera.as_str(), "rgb");
        assert!(pipeline.flush().is_none());
        assert_eq!(pipeline.cameras(), vec![CameraId::from("ir")]);
    }

    #[test]
    fn test_from_config_with_fake_registry_builds_without_io() {
        let config = Config::from_toml_str(TWO_CAMERA_TOML).expect("parses");
        let cameras: Vec<CameraInfo> = config.cameras.iter().map(|c| c.to_info()).collect();
        assert!(
            Pipeline::from_config(&fake_registry(), &config, testkit::rig(), &cameras, None)
                .is_ok()
        );
    }

    #[test]
    fn test_from_config_detect_camera_without_info_errors() {
        let config = Config::from_toml_str(TWO_CAMERA_TOML).expect("parses");
        let cameras = [testkit::info("rgb", PixelFormat::Mjpeg, 33)];
        let Err(PipelineError::MissingCamera { camera }) =
            Pipeline::from_config(&fake_registry(), &config, testkit::rig(), &cameras, None)
        else {
            panic!("expected MissingCamera")
        };
        assert_eq!(camera, "ir");
    }

    #[test]
    fn test_from_config_unknown_estimator_is_config_error() {
        let toml_str = TWO_CAMERA_TOML.replace("estimate = \"fake\"", "estimate = \"nope\"");
        let config = Config::from_toml_str(&toml_str).expect("parses");
        let cameras: Vec<CameraInfo> = config.cameras.iter().map(|c| c.to_info()).collect();
        let Err(PipelineError::Config(ConfigError::UnknownStage { .. })) =
            Pipeline::from_config(&fake_registry(), &config, testkit::rig(), &cameras, None)
        else {
            panic!("expected a Config(UnknownStage) error")
        };
    }

    #[test]
    fn test_fuse_single_point_is_identity() {
        let p = testkit::point(100.0, 50.0, Matrix2::identity());
        assert_eq!(
            fuse_points(std::slice::from_ref(&p), &testkit::screen()),
            Some(p)
        );
    }

    #[test]
    fn test_fuse_two_points_inverse_covariance_weighted() {
        let a = testkit::point(100.0, 50.0, Matrix2::identity());
        let b = testkit::point(110.0, 50.0, Matrix2::identity() * 4.0);
        let fused = fuse_points(&[a, b], &testkit::screen()).expect("two valid points fuse");
        assert_relative_eq!(fused.mm.x, 102.0, epsilon = 1e-12);
        assert_relative_eq!(fused.cov_mm, Matrix2::identity() * 0.8, epsilon = 1e-12);
        assert_eq!(
            fused.px_logical.x,
            mm_to_px_logical(&testkit::screen(), &Point2::new(102.0, 50.0)).x
        );
        assert_eq!(fused.output.as_str(), "eDP-1");
    }

    #[test]
    fn test_fuse_drops_singular_and_nonfinite_covariances() {
        let zero = testkit::point(0.0, 0.0, Matrix2::zeros());
        let nan = testkit::point(0.0, 0.0, Matrix2::new(f64::NAN, 0.0, 0.0, f64::NAN));
        assert_eq!(
            fuse_points(&[zero.clone(), nan.clone()], &testkit::screen()),
            None
        );

        let valid = testkit::point(10.0, 20.0, Matrix2::identity());
        assert_eq!(
            fuse_points(&[zero, nan, valid.clone()], &testkit::screen()),
            Some(valid)
        );
    }

    #[test]
    fn test_fuse_timestamp_is_newest() {
        let mut a = testkit::point(0.0, 0.0, Matrix2::identity());
        let mut b = testkit::point(0.0, 0.0, Matrix2::identity());
        a.timestamp = Timestamp::from_nanos(10_000_000);
        b.timestamp = Timestamp::from_nanos(12_000_000);
        let fused = fuse_points(&[a, b], &testkit::screen()).expect("two valid points fuse");
        assert_eq!(fused.timestamp, Timestamp::from_nanos(12_000_000));
    }
}
