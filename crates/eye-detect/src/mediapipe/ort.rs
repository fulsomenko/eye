use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use ::ort::session::Session;
use ::ort::session::builder::GraphOptimizationLevel;
use ::ort::value::{Outlet, TensorRef};
use eye_core::stage::StageError;

use crate::DetectError;
use crate::mediapipe::blazeface::{DETECTOR_INPUT, FaceDetectorOutput};
use crate::mediapipe::pipeline::MediaPipeDetector;
use crate::mediapipe::{
    DetectorRoles, LANDMARK_INPUT, LandmarkOutput, LandmarkRoles, MediaPipeOptions,
    MediaPipeRuntime, ModelIo, resolve_detector_roles, resolve_landmark_roles,
};
use crate::models::{ModelDir, ModelFile};
use crate::options::parse_options;

pub const NAME: &str = "mediapipe-ort";
pub const ORT_DYLIB_PATH: &str = "ORT_DYLIB_PATH";

#[derive(Debug)]
pub struct OrtRuntime {
    detector: Session,
    landmarks: Session,
    detector_roles: DetectorRoles,
    landmark_roles: LandmarkRoles,
}

static ORT_INIT: OnceLock<Result<(), String>> = OnceLock::new();

fn dylib_path(env_value: Option<OsString>) -> Result<PathBuf, DetectError> {
    match env_value {
        Some(v) if !v.is_empty() => Ok(PathBuf::from(v)),
        _ => Err(DetectError::Inference(format!(
            "{ORT_DYLIB_PATH} is not set; run inside nix develop or point it at libonnxruntime.so"
        ))),
    }
}

fn init_ort() -> Result<(), DetectError> {
    ORT_INIT
        .get_or_init(|| {
            let path = dylib_path(std::env::var_os(ORT_DYLIB_PATH)).map_err(|e| e.to_string())?;
            ::ort::init_from(&path)
                .map_err(|e| e.to_string())?
                .with_name("eye")
                .commit();
            Ok(())
        })
        .clone()
        .map_err(DetectError::Inference)
}

fn io(outlets: &[Outlet]) -> ModelIo {
    outlets
        .iter()
        .map(|o| {
            (
                o.name().to_owned(),
                o.dtype()
                    .tensor_shape()
                    .map(|s| s.to_vec())
                    .unwrap_or_default(),
            )
        })
        .collect()
}

fn open(path: &Path, threads: usize) -> Result<Session, DetectError> {
    let build = || -> ::ort::Result<Session> {
        Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_file(path)
    };
    build().map_err(|e| DetectError::Model {
        path: path.to_owned(),
        reason: e.to_string(),
    })
}

fn inference(e: impl std::fmt::Display) -> DetectError {
    DetectError::Inference(e.to_string())
}

impl OrtRuntime {
    /// Resolves both model paths BEFORE touching ONNX Runtime, so a missing model never loads the library.
    pub fn load(dir: &ModelDir, threads: usize) -> Result<Self, DetectError> {
        let detector_path = dir.path(ModelFile::FaceDetector)?;
        let landmarks_path = dir.path(ModelFile::FaceLandmarks)?;
        init_ort()?;
        let detector = open(&detector_path, threads)?;
        let landmarks = open(&landmarks_path, threads)?;
        let detector_roles =
            resolve_detector_roles(&io(detector.inputs()), &io(detector.outputs())).map_err(
                |reason| DetectError::Model {
                    path: detector_path.clone(),
                    reason,
                },
            )?;
        let landmark_roles =
            resolve_landmark_roles(&io(landmarks.inputs()), &io(landmarks.outputs())).map_err(
                |reason| DetectError::Model {
                    path: landmarks_path.clone(),
                    reason,
                },
            )?;
        tracing::info!(
            runtime = "mediapipe-ort",
            detector = %detector_path.display(),
            landmarks = %landmarks_path.display(),
            threads = threads as u64,
            "mediapipe models loaded"
        );
        Ok(Self {
            detector,
            landmarks,
            detector_roles,
            landmark_roles,
        })
    }
}

impl MediaPipeRuntime for OrtRuntime {
    fn run_face_detector(&mut self, input: &[f32]) -> Result<FaceDetectorOutput, DetectError> {
        let tensor =
            TensorRef::from_array_view(([1usize, DETECTOR_INPUT, DETECTOR_INPUT, 3], input))
                .map_err(inference)?;
        let outputs = self
            .detector
            .run(::ort::inputs![tensor])
            .map_err(inference)?;
        let (_, regressors) = outputs[self.detector_roles.regressors]
            .try_extract_tensor::<f32>()
            .map_err(inference)?;
        let (_, logits) = outputs[self.detector_roles.logits]
            .try_extract_tensor::<f32>()
            .map_err(inference)?;
        Ok(FaceDetectorOutput {
            regressors: regressors.to_vec(),
            logits: logits.to_vec(),
        })
    }

    fn run_landmarks(&mut self, input: &[f32]) -> Result<LandmarkOutput, DetectError> {
        let tensor =
            TensorRef::from_array_view(([1usize, LANDMARK_INPUT, LANDMARK_INPUT, 3], input))
                .map_err(inference)?;
        let outputs = self
            .landmarks
            .run(::ort::inputs![tensor])
            .map_err(inference)?;
        let (_, landmarks) = outputs[self.landmark_roles.landmarks]
            .try_extract_tensor::<f32>()
            .map_err(inference)?;
        let (_, presence) = outputs[self.landmark_roles.presence]
            .try_extract_tensor::<f32>()
            .map_err(inference)?;
        let presence_logit = presence
            .first()
            .copied()
            .ok_or_else(|| inference("empty presence output"))?;
        Ok(LandmarkOutput {
            landmarks: landmarks.to_vec(),
            presence_logit,
        })
    }
}

pub fn from_config(
    table: &toml::Table,
    _rig: &eye_core::Rig,
) -> Result<MediaPipeDetector<OrtRuntime>, StageError> {
    let options: MediaPipeOptions = parse_options(NAME, table)?;
    let dir = ModelDir::resolve(options.model_dir.as_deref())?;
    let runtime = OrtRuntime::load(&dir, options.threads)?;
    Ok(MediaPipeDetector::new(NAME, runtime, options))
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;

    use nalgebra::{Isometry3, Vector2};

    use super::*;

    fn nominal_rig() -> eye_core::Rig {
        let camera = eye_core::CameraModel {
            id: eye_core::CameraId::from("rgb"),
            width: 1280,
            height: 720,
            fx: 500.0,
            fy: 500.0,
            cx: 640.0,
            cy: 360.0,
            distortion: [0.0; 5],
            screen_from_camera: Isometry3::identity(),
        };
        let screen = eye_core::ScreenModel {
            output: eye_core::OutputId::from("eDP-1"),
            size_mm: Vector2::new(310.0, 170.0),
            size_px: (3840, 2160),
            scale: 2.0,
        };
        eye_core::Rig::new(vec![camera], screen).unwrap()
    }

    #[test]
    fn test_dylib_path_unset_is_error_naming_env_var() {
        let err = dylib_path(None).unwrap_err();
        assert!(err.to_string().contains("ORT_DYLIB_PATH"));
    }

    #[test]
    fn test_dylib_path_empty_is_error() {
        let err = dylib_path(Some(OsString::new())).unwrap_err();
        assert!(err.to_string().contains("ORT_DYLIB_PATH"));
    }

    #[test]
    fn test_from_config_missing_model_file_is_model_error_without_loading_ort() {
        let dir = tempfile::tempdir().unwrap();
        let mut table = toml::Table::new();
        table.insert("model_dir".into(), dir.path().to_str().unwrap().into());
        let rig = nominal_rig();

        let err = from_config(&table, &rig).unwrap_err();
        match err {
            StageError::Config(source) => {
                let msg = source.to_string();
                assert!(
                    msg.contains("mediapipe/face_detector.onnx"),
                    "message was: {msg}"
                );
            }
            other => panic!("expected StageError::Config, got {other:?}"),
        }
    }

    #[test]
    fn test_from_config_rejects_unknown_option() {
        let mut table = toml::Table::new();
        table.insert("bogus".into(), 1.into());
        let rig = nominal_rig();

        let err = from_config(&table, &rig).unwrap_err();
        assert!(matches!(err, StageError::Config(_)));
    }

    #[test]
    #[ignore = "needs EYE_MODEL_DIR"]
    fn test_model_load_resolves_roles() {
        let dir = ModelDir::resolve(None).unwrap();
        OrtRuntime::load(&dir, 2).unwrap();
    }

    #[test]
    #[ignore = "needs EYE_MODEL_DIR"]
    fn test_logs_models_loaded_at_info() {
        use eye_log::testing::capture_logs;
        use eye_log::{Level as LogLevel, Value};

        let dir = ModelDir::resolve(None).unwrap();
        let (_, logs) = capture_logs(tracing::Level::INFO, || OrtRuntime::load(&dir, 2).unwrap());

        let rec = logs
            .iter()
            .find(|r| r.message == "mediapipe models loaded")
            .expect("event emitted");
        assert_eq!(rec.level, LogLevel::Info);
        assert_eq!(rec.fields["runtime"], Value::Str("mediapipe-ort".into()));
        assert!(matches!(rec.fields["detector"], Value::Str(_)));
    }

    #[test]
    #[ignore = "needs EYE_MODEL_DIR"]
    fn test_model_blank_landmark_input_has_low_presence() {
        let dir = ModelDir::resolve(None).unwrap();
        let mut runtime = OrtRuntime::load(&dir, 2).unwrap();
        let input = vec![0.0f32; LANDMARK_INPUT * LANDMARK_INPUT * 3];
        let output = runtime.run_landmarks(&input).unwrap();
        let presence = 1.0 / (1.0 + (-output.presence_logit).exp());
        assert!(presence < 0.5, "presence was {presence}");
    }

    #[test]
    #[ignore = "needs EYE_MODEL_DIR"]
    fn test_model_blank_image_has_no_face() {
        let dir = ModelDir::resolve(None).unwrap();
        let runtime = OrtRuntime::load(&dir, 2).unwrap();
        let options = MediaPipeOptions::default();
        let mut detector = MediaPipeDetector::new(NAME, runtime, options);
        let image = crate::image::RgbImage::owned(1280, 720, vec![0u8; 1280 * 720 * 3]);
        let camera = eye_core::CameraId::from("cam0");
        let result = detector.detect_image(&camera, &image).unwrap();
        assert!(result.is_none());
    }

    #[test]
    #[ignore = "benchmark: needs EYE_MODEL_DIR, run in release"]
    fn test_model_landmark_latency_benchmark() {
        let dir = ModelDir::resolve(None).unwrap();
        let mut runtime = OrtRuntime::load(&dir, 2).unwrap();
        let input = vec![0.0f32; LANDMARK_INPUT * LANDMARK_INPUT * 3];
        for _ in 0..10 {
            runtime.run_landmarks(&input).unwrap();
        }
        let mut samples = Vec::with_capacity(100);
        for _ in 0..100 {
            let start = std::time::Instant::now();
            runtime.run_landmarks(&input).unwrap();
            samples.push(start.elapsed());
        }
        samples.sort();
        let p50 = samples[49];
        let p95 = samples[94];
        println!("landmark latency p50={p50:?} p95={p95:?}");
    }

    #[test]
    #[ignore = "needs hardware"]
    fn test_real_frame_face_and_sides() {
        use crate::mediapipe::landmarks::index;

        let path = std::env::var("EYE_MJPG_FRAME").expect("EYE_MJPG_FRAME not set");
        let bytes = std::fs::read(path).expect("failed to read EYE_MJPG_FRAME");
        let image = crate::mjpeg::decode_mjpeg_rgb(&bytes).unwrap();

        let dir = ModelDir::resolve(None).unwrap();
        let runtime = OrtRuntime::load(&dir, 2).unwrap();
        let mut detector = MediaPipeDetector::new(NAME, runtime, MediaPipeOptions::default());
        let camera = eye_core::CameraId::from("cam0");
        let face = detector
            .detect_image(&camera, &image)
            .unwrap()
            .expect("expected a face in the real frame");

        assert!(
            face.landmarks[index::LEFT_EYE_LATERAL].x > face.landmarks[index::RIGHT_EYE_LATERAL].x,
            "left landmark x {} was not right of right landmark x {}",
            face.landmarks[index::LEFT_EYE_LATERAL].x,
            face.landmarks[index::RIGHT_EYE_LATERAL].x
        );

        for eye in &face.eyes {
            let corners = eye.corners.as_ref().expect("eye corners present");
            let iris = eye.iris.as_ref().expect("iris present");
            let center_x = iris.value().center().x;
            let (lo, hi) = (
                corners.lateral.value().x.min(corners.medial.value().x),
                corners.lateral.value().x.max(corners.medial.value().x),
            );
            assert!(
                (lo..=hi).contains(&center_x),
                "{:?} iris center x {center_x} not between corners {lo}..{hi}",
                eye.side
            );
        }
    }
}
