//! Synthetic sessions and fake pipeline stages for `eye-bench` tests.
//! No face images, no biometric data: every pixel is a flat gray value.

use std::f64::consts::PI;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use eye::config::Config;
use eye::error::ConfigError;
use eye::registry::Registry;
use eye_calibration::protocol::ProtocolConfig;
use eye_calibration::store::rig_to_table;
use eye_capture::session::{FORMAT_VERSION, RecordedCamera, SessionId, SessionMeta, SessionWriter};
use eye_core::session::{TargetClock, TargetRecord};
use eye_core::stage::{Detector, GazeEstimator, StageError};
use eye_core::{
    CameraId, CameraInfo, CameraModel, FaceObservation, Frame, FrameHeader, FrameSet, GazeRay,
    Illumination, Observations, OutputId, PixelFormat, Rig, ScreenModel, Timestamp,
};
use eye_geometry::angles::{direction_from_yaw_pitch, yaw_pitch_from_direction};
use eye_geometry::screen::px_logical_to_mm;
use nalgebra::{
    Isometry3, Matrix2, Matrix3, Point2, Point3, Translation3, Unit, UnitQuaternion, Vector2,
    Vector3,
};
use serde::Deserialize;

use crate::error::BenchError;

pub const FOUR_BY_FOUR_CENTRES: [(f64, f64); 16] = [
    (240.0, 135.0),
    (240.0, 405.0),
    (240.0, 675.0),
    (240.0, 945.0),
    (720.0, 135.0),
    (720.0, 405.0),
    (720.0, 675.0),
    (720.0, 945.0),
    (1200.0, 135.0),
    (1200.0, 405.0),
    (1200.0, 675.0),
    (1200.0, 945.0),
    (1680.0, 135.0),
    (1680.0, 405.0),
    (1680.0, 675.0),
    (1680.0, 945.0),
];

#[derive(Debug, Clone, PartialEq)]
pub struct SyntheticSession {
    /// Logical px, shown in this order (target k = k-th entry).
    pub targets: Vec<(f64, f64)>,
    pub size: (u32, u32),
    /// Pixel value k + 1 while target k is shown (0 otherwise); `false`: every pixel 0.
    pub code_frames: bool,
    /// Camera ids, each recorded with identical Gray8 frames.
    pub cameras: Vec<&'static str>,
    pub with_rig: bool,
    pub protocol: Option<ProtocolConfig>,
}

impl Default for SyntheticSession {
    fn default() -> Self {
        Self {
            targets: Vec::new(),
            size: (8, 8),
            code_frames: false,
            cameras: vec!["ir"],
            with_rig: true,
            protocol: None,
        }
    }
}

/// eDP-1, 310 x 170 mm, 3840x2160, scale 2 (1920x1080 logical).
pub fn synthetic_screen() -> ScreenModel {
    ScreenModel {
        output: OutputId::from("eDP-1"),
        size_mm: Vector2::new(310.0, 170.0),
        size_px: (3840, 2160),
        scale: 2.0,
    }
}

pub fn synthetic_rig() -> Rig {
    let camera = CameraModel {
        id: CameraId::from("ir"),
        width: 640,
        height: 360,
        fx: 472.0,
        fy: 472.0,
        cx: 320.0,
        cy: 180.0,
        distortion: [0.0; 5],
        screen_from_camera: Isometry3::from_parts(
            Translation3::new(155.0, -7.0, 0.0),
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), PI),
        ),
    };
    Rig::new(vec![camera], synthetic_screen()).expect("synthetic rig is valid")
}

const LEAD_IN_NS: u64 = 1_000_000_000;
const DWELL_NS: u64 = 1_500_000_000;
const FRAME_INTERVAL_NS: u64 = 33_333_333;

fn target_onset(k: u64) -> u64 {
    LEAD_IN_NS + LEAD_IN_NS + k * DWELL_NS
}

/// Writes `<root>/<id>/` through the real `SessionWriter`; returns the session directory.
#[allow(clippy::result_large_err)]
pub fn write_synthetic_session(
    root: &Path,
    id: &str,
    spec: &SyntheticSession,
) -> Result<PathBuf, BenchError> {
    let session_dir = root.join(id);
    let capture = |source| BenchError::Capture {
        session: session_dir.clone(),
        source,
    };

    let cameras = spec
        .cameras
        .iter()
        .map(|&camera_id| {
            let info = CameraInfo {
                id: CameraId::from(camera_id),
                format: PixelFormat::Gray8,
                width: spec.size.0,
                height: spec.size.1,
                frame_interval: Duration::from_nanos(FRAME_INTERVAL_NS),
            };
            RecordedCamera::from_info(&info, None).map_err(capture)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let meta = SessionMeta {
        format_version: FORMAT_VERSION,
        session_id: SessionId::new(id).map_err(capture)?,
        created_unix_s: 1_791_409_623,
        git_rev: None,
        emitter: None,
        cameras,
        rig: spec.with_rig.then(|| rig_to_table(&synthetic_rig())),
        probe: None,
        protocol: spec.protocol,
    };

    let mut writer = SessionWriter::create(root, &meta).map_err(capture)?;

    let n = spec.targets.len() as u64;
    let max_t = LEAD_IN_NS + LEAD_IN_NS + n * DWELL_NS + LEAD_IN_NS;
    let mut j = 0u64;
    loop {
        let t = LEAD_IN_NS + j * FRAME_INTERVAL_NS;
        if t > max_t {
            break;
        }
        let pixel = if spec.code_frames {
            (0..n)
                .find_map(|k| {
                    let onset = target_onset(k);
                    (onset <= t && t < onset + DWELL_NS).then_some((k + 1) as u8)
                })
                .unwrap_or(0)
        } else {
            0
        };
        let data: Arc<[u8]> = vec![pixel; (spec.size.0 * spec.size.1) as usize].into();
        for &camera_id in &spec.cameras {
            let frame = Frame::new(
                FrameHeader {
                    camera: CameraId::from(camera_id),
                    seq: j,
                    timestamp: Timestamp::from_nanos(t),
                    width: spec.size.0,
                    height: spec.size.1,
                    format: PixelFormat::Gray8,
                    illumination: Illumination::Unknown,
                },
                Arc::clone(&data),
            )
            .expect("synthetic frame is valid");
            writer.write_frame(&frame).map_err(capture)?;
        }
        j += 1;
    }

    for (k, &(x, y)) in spec.targets.iter().enumerate() {
        let shown_ns = target_onset(k as u64);
        let mm = px_logical_to_mm(&synthetic_screen(), &Point2::new(x, y));
        let target = TargetRecord {
            seq: k as u64,
            shown_ns,
            hidden_ns: Some(shown_ns + DWELL_NS),
            clock: TargetClock::Commit,
            output: OutputId::from("eDP-1"),
            px_logical: [x, y],
            mm: [mm.x, mm.y],
        };
        writer.write_target(&target).map_err(capture)?;
    }

    writer.finish().map_err(capture)
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NoOptions {}

#[derive(Debug)]
struct NullDetector;

impl NullDetector {
    fn from_config(options: &toml::Table) -> Result<Self, StageError> {
        let NoOptions {} = options
            .clone()
            .try_into()
            .map_err(|e: toml::de::Error| StageError::Config(Box::new(e)))?;
        Ok(Self)
    }
}

impl Detector for NullDetector {
    fn name(&self) -> &'static str {
        "test-null"
    }

    fn accepts(&self, _format: PixelFormat, _illumination: Illumination) -> bool {
        true
    }

    fn detect(&mut self, frames: &FrameSet) -> Result<Vec<Observations>, StageError> {
        Ok(frames
            .frames()
            .iter()
            .map(|f| Observations {
                camera: f.header().camera.clone(),
                timestamp: f.header().timestamp,
                face: Some(eye_core::FaceObservation {
                    scheme: "test",
                    landmarks: Vec::new(),
                    eyes: Vec::new(),
                }),
            })
            .collect())
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FixedRayOptions {
    origin: [f64; 3],
    toward: [f64; 2],
}

#[derive(Debug)]
struct FixedRayEstimator {
    origin: Point3<f64>,
    toward: Point2<f64>,
}

impl FixedRayEstimator {
    fn from_config(options: &toml::Table) -> Result<Self, StageError> {
        let FixedRayOptions { origin, toward } = options
            .clone()
            .try_into()
            .map_err(|e: toml::de::Error| StageError::Config(Box::new(e)))?;
        if origin[2] >= 0.0 {
            return Err(StageError::Config(
                "origin must be in front of the screen (z < 0)".into(),
            ));
        }
        Ok(Self {
            origin: Point3::new(origin[0], origin[1], origin[2]),
            toward: Point2::new(toward[0], toward[1]),
        })
    }
}

impl GazeEstimator for FixedRayEstimator {
    fn name(&self) -> &'static str {
        "test-fixed-ray"
    }

    fn estimate(&mut self, obs: &[Observations], _rig: &Rig) -> Result<Vec<GazeRay>, StageError> {
        let direction =
            Unit::new_normalize(Point3::new(self.toward.x, self.toward.y, 0.0) - self.origin);
        Ok(vec![GazeRay {
            side: None,
            timestamp: obs
                .first()
                .map_or(Timestamp::from_nanos(0), |o| o.timestamp),
            origin: self.origin,
            direction,
            origin_cov: Matrix3::zeros(),
            angular_cov: Matrix2::identity() * 1e-6,
            head_rotation: None,
        }])
    }
}

#[derive(Debug)]
struct TargetCodeDetector;

impl TargetCodeDetector {
    fn from_config(options: &toml::Table) -> Result<Self, StageError> {
        let NoOptions {} = options
            .clone()
            .try_into()
            .map_err(|e: toml::de::Error| StageError::Config(Box::new(e)))?;
        Ok(Self)
    }
}

impl Detector for TargetCodeDetector {
    fn name(&self) -> &'static str {
        "test-target-code"
    }

    fn accepts(&self, _format: PixelFormat, _illumination: Illumination) -> bool {
        true
    }

    fn detect(&mut self, frames: &FrameSet) -> Result<Vec<Observations>, StageError> {
        Ok(frames
            .frames()
            .iter()
            .map(|f| {
                let code = f.data().first().copied().unwrap_or(0);
                Observations {
                    camera: f.header().camera.clone(),
                    timestamp: f.header().timestamp,
                    face: Some(FaceObservation {
                        scheme: "test",
                        landmarks: vec![Point2::new(f64::from(code), 0.0)],
                        eyes: Vec::new(),
                    }),
                }
            })
            .collect())
    }
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct KappaOutlier {
    index: usize,
    offset_deg: [f64; 2],
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct KappaRayOptions {
    eye: [f64; 3],
    targets_mm: Vec<[f64; 2]>,
    offset_deg: [f64; 2],
    #[serde(default)]
    outlier: Option<KappaOutlier>,
}

#[derive(Debug)]
struct KappaRayEstimator {
    eye: Point3<f64>,
    targets_mm: Vec<[f64; 2]>,
    offset_deg: [f64; 2],
    outlier: Option<KappaOutlier>,
}

impl KappaRayEstimator {
    fn from_config(options: &toml::Table) -> Result<Self, StageError> {
        let KappaRayOptions {
            eye,
            targets_mm,
            offset_deg,
            outlier,
        } = options
            .clone()
            .try_into()
            .map_err(|e: toml::de::Error| StageError::Config(Box::new(e)))?;
        if eye[2] >= 0.0 {
            return Err(StageError::Config(
                "eye must be in front of the screen (z < 0)".into(),
            ));
        }
        Ok(Self {
            eye: Point3::new(eye[0], eye[1], eye[2]),
            targets_mm,
            offset_deg,
            outlier,
        })
    }
}

impl GazeEstimator for KappaRayEstimator {
    fn name(&self) -> &'static str {
        "test-kappa-ray"
    }

    fn estimate(&mut self, obs: &[Observations], _rig: &Rig) -> Result<Vec<GazeRay>, StageError> {
        let Some(code) = obs
            .iter()
            .find_map(|o| o.face.as_ref())
            .and_then(|face| face.landmarks.first())
            .map(|p| p.x.round() as i64)
        else {
            return Ok(Vec::new());
        };
        if code <= 0 {
            return Ok(Vec::new());
        }
        let idx = (code - 1) as usize;
        let t = *self
            .targets_mm
            .get(idx)
            .ok_or_else(|| StageError::Failed(format!("target index {idx} out of range").into()))?;
        let off = match &self.outlier {
            Some(o) if o.index == idx => o.offset_deg,
            _ => self.offset_deg,
        };
        let base = yaw_pitch_from_direction(&Unit::new_normalize(
            Point3::new(t[0], t[1], 0.0) - self.eye,
        ));
        let direction = direction_from_yaw_pitch(
            &(base + Vector2::new(off[0].to_radians(), off[1].to_radians())),
        );
        Ok(vec![GazeRay {
            side: None,
            timestamp: obs
                .first()
                .map_or(Timestamp::from_nanos(0), |o| o.timestamp),
            origin: self.eye,
            direction,
            origin_cov: Matrix3::zeros(),
            angular_cov: Matrix2::identity() * 1e-6,
            head_rotation: None,
        }])
    }
}

pub fn register_fakes(registry: &mut Registry) -> Result<(), ConfigError> {
    registry.register_detector("test-null", |o, _rig| {
        Ok(Box::new(NullDetector::from_config(o)?))
    })?;
    registry.register_estimator("test-fixed-ray", |o, _rig| {
        Ok(Box::new(FixedRayEstimator::from_config(o)?))
    })?;
    registry.register_detector("test-target-code", |o, _rig| {
        Ok(Box::new(TargetCodeDetector::from_config(o)?))
    })?;
    registry.register_estimator("test-kappa-ray", |o, _rig| {
        Ok(Box::new(KappaRayEstimator::from_config(o)?))
    })?;
    Ok(())
}

/// `eye.toml` text: one camera "ir" (device "/dev/null", gray, 8x8), detector test-target-code,
/// estimator test-kappa-ray (eye fixed at `[155.0, 85.0, -500.0]`), filter none.
pub fn kappa_ray_toml(
    targets_px: &[(f64, f64)],
    offset_deg: [f64; 2],
    outlier: Option<(usize, [f64; 2])>,
) -> String {
    let screen = synthetic_screen();
    let targets_mm_str = targets_px
        .iter()
        .map(|&(x, y)| {
            let mm = px_logical_to_mm(&screen, &Point2::new(x, y));
            format!("[{:?}, {:?}]", mm.x, mm.y)
        })
        .collect::<Vec<_>>()
        .join(", ");
    let outlier_line = match outlier {
        Some((index, offset)) => format!(
            "outlier = {{ index = {index}, offset_deg = [{:?}, {:?}] }}\n",
            offset[0], offset[1]
        ),
        None => String::new(),
    };
    format!(
        "[[camera]]\n\
         id = \"ir\"\n\
         device = \"/dev/null\"\n\
         format = \"gray\"\n\
         size = [8, 8]\n\
         \n\
         [detect]\n\
         ir = \"test-target-code\"\n\
         \n\
         [estimate]\n\
         kind = \"test-kappa-ray\"\n\
         eye = [155.0, 85.0, -500.0]\n\
         targets_mm = [{targets_mm_str}]\n\
         offset_deg = [{:?}, {:?}]\n\
         {outlier_line}\
         \n\
         [filter]\n\
         kind = \"none\"\n",
        offset_deg[0], offset_deg[1]
    )
}

pub fn kappa_ray_config(
    targets_px: &[(f64, f64)],
    offset_deg: [f64; 2],
    outlier: Option<(usize, [f64; 2])>,
) -> Config {
    Config::from_toml_str(&kappa_ray_toml(targets_px, offset_deg, outlier))
        .expect("valid synthetic config")
}

/// `Registry::with_defaults()` (for the built-in "none" filter) plus `register_fakes`.
pub fn fake_registry() -> Registry {
    let mut registry = Registry::with_defaults();
    register_fakes(&mut registry).expect("fake stage names are unique");
    registry
}

/// `eye.toml` text: one camera "ir" (device "/dev/null", gray, 8x8), detector test-null, estimator
/// test-fixed-ray, filter none.
pub fn fixed_ray_toml(origin: [f64; 3], toward_mm: [f64; 2]) -> String {
    format!(
        "[[camera]]\n\
         id = \"ir\"\n\
         device = \"/dev/null\"\n\
         format = \"gray\"\n\
         size = [8, 8]\n\
         \n\
         [detect]\n\
         ir = \"test-null\"\n\
         \n\
         [estimate]\n\
         kind = \"test-fixed-ray\"\n\
         origin = [{:?}, {:?}, {:?}]\n\
         toward = [{:?}, {:?}]\n\
         \n\
         [filter]\n\
         kind = \"none\"\n",
        origin[0], origin[1], origin[2], toward_mm[0], toward_mm[1]
    )
}

pub fn fixed_ray_config(origin: [f64; 3], toward_mm: [f64; 2]) -> Config {
    Config::from_toml_str(&fixed_ray_toml(origin, toward_mm)).expect("valid synthetic config")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_ray_toml_is_exact() {
        let text = fixed_ray_toml([155.0, 85.0, -500.0], [161.458333, 94.444444]);
        assert_eq!(
            text,
            "[[camera]]\n\
             id = \"ir\"\n\
             device = \"/dev/null\"\n\
             format = \"gray\"\n\
             size = [8, 8]\n\
             \n\
             [detect]\n\
             ir = \"test-null\"\n\
             \n\
             [estimate]\n\
             kind = \"test-fixed-ray\"\n\
             origin = [155.0, 85.0, -500.0]\n\
             toward = [161.458333, 94.444444]\n\
             \n\
             [filter]\n\
             kind = \"none\"\n"
        );
    }

    #[test]
    fn test_write_synthetic_session_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let spec = SyntheticSession::default();
        let session_dir = write_synthetic_session(dir.path(), "s1", &spec).unwrap();
        let recording = eye_capture::session::Recording::open(&session_dir).unwrap();
        assert_eq!(recording.meta().session_id.as_str(), "s1");
        assert!(recording.meta().rig.is_some());
        assert_eq!(recording.merged_index().len(), 61);
    }
}
