//! Test fakes shared by `pipeline` tests.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use eye_core::{
    CameraId, CameraInfo, CameraModel, Frame, FrameHeader, FrameSet, GazePoint, GazeRay,
    Illumination, Observations, OutputId, PixelFormat, Rig, ScreenModel, Timestamp,
    stage::{Detector, GazeCorrection, GazeEstimator, GazeFilter, StageError},
};
use nalgebra::{Matrix2, Matrix3, Point2, Rotation3, Vector2, Vector3};

use crate::registry::Registry;

pub(crate) const TWO_CAMERA_TOML: &str = r#"
estimate = "fake"
[[camera]]
id = "rgb"
device = "/dev/video-does-not-exist-0"
format = "mjpeg"
size = [1280, 720]
[[camera]]
id = "ir"
device = "/dev/video-does-not-exist-2"
format = "gray"
size = [640, 360]
[detect]
ir = "fake"
"#;

pub(crate) fn screen() -> ScreenModel {
    ScreenModel {
        output: OutputId::from("eDP-1"),
        size_mm: Vector2::new(310.0, 170.0),
        size_px: (3840, 2160),
        scale: 2.0,
    }
}

pub(crate) fn rig() -> Rig {
    let camera = CameraModel {
        id: CameraId::from("ir"),
        width: 640,
        height: 360,
        fx: 430.0,
        fy: 430.0,
        cx: 320.0,
        cy: 180.0,
        distortion: [0.0; 5],
        screen_from_camera: nalgebra::Isometry3::identity(),
    };
    Rig::new(vec![camera], screen()).expect("testkit rig is valid")
}

pub(crate) fn info(camera: &str, format: PixelFormat, interval_ms: u64) -> CameraInfo {
    CameraInfo {
        id: CameraId::from(camera),
        format,
        width: 4,
        height: 2,
        frame_interval: std::time::Duration::from_millis(interval_ms),
    }
}

pub(crate) fn frame_fmt(
    camera: &str,
    seq: u64,
    t_ms: u64,
    format: PixelFormat,
    illumination: Illumination,
) -> Frame {
    let (width, height) = (4u32, 2u32);
    let data: Arc<[u8]> = match format.bytes_per_pixel() {
        Some(bpp) => vec![0u8; width as usize * height as usize * bpp].into(),
        None => vec![0xFF, 0xD8, 0xFF, 0xD9].into(),
    };
    Frame::new(
        FrameHeader {
            camera: CameraId::from(camera),
            seq,
            timestamp: Timestamp::from_nanos(t_ms * 1_000_000),
            width,
            height,
            format,
            illumination,
        },
        data,
    )
    .expect("testkit frame is valid")
}

pub(crate) fn frame(camera: &str, seq: u64, t_ms: u64, illumination: Illumination) -> Frame {
    frame_fmt(camera, seq, t_ms, PixelFormat::Gray8, illumination)
}

pub(crate) fn point(x: f64, y: f64, cov_mm: Matrix2<f64>) -> GazePoint {
    GazePoint {
        timestamp: Timestamp::from_nanos(0),
        output: OutputId::from("eDP-1"),
        mm: Point2::new(x, y),
        px_physical: Point2::new(0.0, 0.0),
        px_logical: Point2::new(0.0, 0.0),
        cov_mm,
        confidence: 0.5,
    }
}

/// Records the camera ids it saw; `.accepting(Illumination)` restricts `accepts`; `.failing(&[bool])`
/// fails the calls marked `true` then succeeds; `.failing_always()`. Returns one
/// `Observations::empty(camera, timestamp)` per frame.
#[derive(Debug)]
pub(crate) struct FakeDetector {
    name: &'static str,
    only: Option<Illumination>,
    script: VecDeque<bool>,
    fail_rest: bool,
    pub(crate) seen: Arc<Mutex<Vec<CameraId>>>,
}

impl FakeDetector {
    pub(crate) fn new(name: &'static str) -> Self {
        Self {
            name,
            only: None,
            script: VecDeque::new(),
            fail_rest: false,
            seen: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub(crate) fn accepting(mut self, illumination: Illumination) -> Self {
        self.only = Some(illumination);
        self
    }

    pub(crate) fn failing(mut self, script: &[bool]) -> Self {
        self.script = script.iter().copied().collect();
        self
    }

    pub(crate) fn failing_always(mut self) -> Self {
        self.fail_rest = true;
        self
    }
}

impl Detector for FakeDetector {
    fn name(&self) -> &'static str {
        self.name
    }

    fn accepts(&self, _format: PixelFormat, illumination: Illumination) -> bool {
        self.only.is_none_or(|only| only == illumination)
    }

    fn detect(&mut self, frames: &FrameSet) -> Result<Vec<Observations>, StageError> {
        {
            let mut seen = self.seen.lock().expect("testkit mutex is never poisoned");
            for f in frames.frames() {
                seen.push(f.header().camera.clone());
            }
        }
        let fail = self.script.pop_front().unwrap_or(self.fail_rest);
        if fail {
            return Err(StageError::Failed("fake detector failure".into()));
        }
        Ok(frames
            .frames()
            .iter()
            .map(|f| Observations::empty(f.header().camera.clone(), f.header().timestamp))
            .collect())
    }
}

/// One ray per call: origin (155, 85, -500) mm, direction +z, angular_cov 1e-6 I, origin_cov 0.
#[derive(Debug)]
pub(crate) struct FakeEstimator;

impl GazeEstimator for FakeEstimator {
    fn name(&self) -> &'static str {
        "fake"
    }

    fn estimate(&mut self, _obs: &[Observations], _rig: &Rig) -> Result<Vec<GazeRay>, StageError> {
        Ok(vec![GazeRay {
            side: None,
            origin: nalgebra::Point3::new(155.0, 85.0, -500.0),
            direction: Vector3::z_axis(),
            angular_cov: Matrix2::identity() * 1e-6,
            origin_cov: Matrix3::zeros(),
        }])
    }
}

/// name "failing"; `estimate` always returns `Err(StageError::Failed("fake estimator failure".into()))`.
#[derive(Debug)]
pub(crate) struct FailingEstimator;

impl GazeEstimator for FailingEstimator {
    fn name(&self) -> &'static str {
        "failing"
    }

    fn estimate(&mut self, _obs: &[Observations], _rig: &Rig) -> Result<Vec<GazeRay>, StageError> {
        Err(StageError::Failed("fake estimator failure".into()))
    }
}

/// Rotates the ray about screen +y by `.0` rad: positive yaw moves the hit towards +x (R1).
#[derive(Debug)]
pub(crate) struct YawOffset(pub f64);

impl GazeCorrection for YawOffset {
    fn correct(&self, ray: &GazeRay) -> GazeRay {
        GazeRay {
            direction: Rotation3::from_axis_angle(&Vector3::y_axis(), self.0) * ray.direction,
            ..ray.clone()
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct CountingFilter {
    pub(crate) resets: Arc<AtomicU32>,
}

impl GazeFilter for CountingFilter {
    fn apply(&mut self, point: GazePoint) -> GazePoint {
        point
    }

    fn reset(&mut self) {
        self.resets.fetch_add(1, Ordering::SeqCst);
    }
}

/// `Registry::with_defaults()` plus detector "fake" (`FakeDetector::new("fake")`) and estimator
/// "fake" (`FakeEstimator`).
pub(crate) fn fake_registry() -> Registry {
    let mut registry = Registry::with_defaults();
    registry
        .register_detector("fake", |_, _| {
            Ok(Box::new(FakeDetector::new("fake")) as Box<dyn Detector>)
        })
        .expect("\"fake\" detector name is unique");
    registry
        .register_estimator("fake", |_, _| {
            Ok(Box::new(FakeEstimator) as Box<dyn GazeEstimator>)
        })
        .expect("\"fake\" estimator name is unique");
    registry
}
