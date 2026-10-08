use std::path::Path;
use std::sync::Arc;

use eye_core::stage::StageError;
use tract_onnx::prelude::*;

use crate::DetectError;
use crate::mediapipe::blazeface::{DETECTOR_INPUT, FaceDetectorOutput};
use crate::mediapipe::pipeline::MediaPipeDetector;
use crate::mediapipe::{
    DetectorRoles, LANDMARK_INPUT, LandmarkOutput, LandmarkRoles, MediaPipeOptions,
    MediaPipeRuntime, ModelIo, resolve_detector_roles, resolve_landmark_roles,
};
use crate::models::{ModelDir, ModelFile};
use crate::options::parse_options;

pub const NAME: &str = "mediapipe-tract";

#[derive(Debug)]
pub struct TractRuntime {
    detector: Arc<TypedRunnableModel>,
    landmarks: Arc<TypedRunnableModel>,
    detector_roles: DetectorRoles,
    landmark_roles: LandmarkRoles,
}

fn describe_io(model: &TypedModel) -> TractResult<(ModelIo, ModelIo)> {
    let f = |o: &OutletId| -> TractResult<(String, Vec<i64>)> {
        let name = model.outlet_label(*o).unwrap_or_default().to_owned();
        let shape = model
            .outlet_fact(*o)?
            .shape
            .as_concrete()
            .map(|s| s.iter().map(|d| *d as i64).collect())
            .unwrap_or_default();
        Ok((name, shape))
    };
    let inputs = model
        .input_outlets()?
        .iter()
        .map(f)
        .collect::<TractResult<Vec<_>>>()?;
    let outputs = model
        .output_outlets()?
        .iter()
        .map(f)
        .collect::<TractResult<Vec<_>>>()?;
    Ok((inputs, outputs))
}

fn load_model(
    path: &Path,
    n: usize,
) -> Result<(Arc<TypedRunnableModel>, ModelIo, ModelIo), DetectError> {
    let load = || -> TractResult<_> {
        let model = tract_onnx::onnx()
            .model_for_path(path)?
            .with_input_fact(0, f32::fact([1, n, n, 3]).into())?
            .into_optimized()?;
        let (inputs, outputs) = describe_io(&model)?;
        Ok((model.into_runnable()?, inputs, outputs))
    };
    load().map_err(|e| DetectError::Model {
        path: path.to_owned(),
        reason: format!("{e:#}"),
    })
}

fn inference(e: impl std::fmt::Display) -> DetectError {
    DetectError::Inference(e.to_string())
}

fn run(
    model: &Arc<TypedRunnableModel>,
    n: usize,
    input: &[f32],
) -> Result<TVec<TValue>, DetectError> {
    let tensor = Tensor::from_shape(&[1, n, n, 3], input).map_err(inference)?;
    model.run(tvec!(tensor.into_tvalue())).map_err(inference)
}

fn to_vec(outputs: &TVec<TValue>, i: usize) -> Result<Vec<f32>, DetectError> {
    Ok(outputs[i]
        .try_as_plain_ram()
        .map_err(inference)?
        .as_slice::<f32>()
        .map_err(inference)?
        .to_vec())
}

impl TractRuntime {
    /// Resolves both model paths BEFORE touching tract, so a missing model never loads a model.
    pub fn load(dir: &ModelDir) -> Result<Self, DetectError> {
        let detector_path = dir.path(ModelFile::FaceDetector)?;
        let landmarks_path = dir.path(ModelFile::FaceLandmarks)?;
        let (detector, di, dout) = load_model(&detector_path, DETECTOR_INPUT)?;
        let (landmarks, li, lout) = load_model(&landmarks_path, LANDMARK_INPUT)?;
        let detector_roles =
            resolve_detector_roles(&di, &dout).map_err(|reason| DetectError::Model {
                path: detector_path,
                reason,
            })?;
        let landmark_roles =
            resolve_landmark_roles(&li, &lout).map_err(|reason| DetectError::Model {
                path: landmarks_path,
                reason,
            })?;
        Ok(Self {
            detector,
            landmarks,
            detector_roles,
            landmark_roles,
        })
    }
}

impl MediaPipeRuntime for TractRuntime {
    fn run_face_detector(&mut self, input: &[f32]) -> Result<FaceDetectorOutput, DetectError> {
        let outputs = run(&self.detector, DETECTOR_INPUT, input)?;
        Ok(FaceDetectorOutput {
            regressors: to_vec(&outputs, self.detector_roles.regressors)?,
            logits: to_vec(&outputs, self.detector_roles.logits)?,
        })
    }

    fn run_landmarks(&mut self, input: &[f32]) -> Result<LandmarkOutput, DetectError> {
        let outputs = run(&self.landmarks, LANDMARK_INPUT, input)?;
        let presence = to_vec(&outputs, self.landmark_roles.presence)?;
        let presence_logit = presence
            .first()
            .copied()
            .ok_or_else(|| inference("empty presence output"))?;
        Ok(LandmarkOutput {
            landmarks: to_vec(&outputs, self.landmark_roles.landmarks)?,
            presence_logit,
        })
    }
}

pub fn from_config(
    table: &toml::Table,
    _rig: &eye_core::Rig,
) -> Result<MediaPipeDetector<TractRuntime>, StageError> {
    let options: MediaPipeOptions = parse_options(NAME, table)?;
    let dir = ModelDir::resolve(options.model_dir.as_deref())?;
    Ok(MediaPipeDetector::new(
        NAME,
        TractRuntime::load(&dir)?,
        options,
    ))
}

#[cfg(test)]
mod tests {
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
    fn test_from_config_missing_model_file_is_model_error() {
        let dir = tempfile::tempdir().unwrap();
        let mut table = toml::Table::new();
        table.insert("model_dir".into(), dir.path().to_str().unwrap().into());
        let rig = nominal_rig();

        let err = from_config(&table, &rig).unwrap_err();
        match err {
            StageError::Config(msg) => {
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
    fn test_model_tract_loads_both_models() {
        let dir = ModelDir::resolve(None).unwrap();
        TractRuntime::load(&dir).unwrap();
    }

    #[test]
    #[ignore = "needs EYE_MODEL_DIR"]
    fn test_model_tract_blank_image_has_no_face() {
        let dir = ModelDir::resolve(None).unwrap();
        let runtime = TractRuntime::load(&dir).unwrap();
        let options = MediaPipeOptions::default();
        let mut detector = MediaPipeDetector::new(NAME, runtime, options);
        let image = crate::image::RgbImage {
            width: 1280,
            height: 720,
            data: vec![0u8; 1280 * 720 * 3],
        };
        let camera = eye_core::CameraId::from("cam0");
        let result = detector.detect_image(&camera, &image).unwrap();
        assert!(result.is_none());
    }

    #[test]
    #[ignore = "needs EYE_MODEL_DIR"]
    #[cfg(all(feature = "mediapipe-ort", feature = "mediapipe-tract"))]
    fn test_model_tract_matches_ort_on_same_input() {
        use crate::mediapipe::ort::OrtRuntime;

        fn gradient_input(n: usize) -> Vec<f32> {
            let mut data = vec![0.0f32; n * n * 3];
            for y in 0..n {
                for x in 0..n {
                    let base = ((x + y) % 256) as f32 / 255.0;
                    let disc = |cx: usize, cy: usize| {
                        let dx = x as isize - cx as isize;
                        let dy = y as isize - cy as isize;
                        (dx * dx + dy * dy) as f64 <= (n as f64 / 8.0).powi(2)
                    };
                    let value = if disc(n / 3, n / 3) || disc(2 * n / 3, 2 * n / 3) {
                        0.0
                    } else {
                        base
                    };
                    let idx = (y * n + x) * 3;
                    data[idx] = value;
                    data[idx + 1] = value;
                    data[idx + 2] = value;
                }
            }
            data
        }

        let dir = ModelDir::resolve(None).unwrap();
        let mut tract = TractRuntime::load(&dir).unwrap();
        let mut ort = OrtRuntime::load(&dir, 2).unwrap();

        let detector_input = gradient_input(DETECTOR_INPUT);
        let tract_det = tract.run_face_detector(&detector_input).unwrap();
        let ort_det = ort.run_face_detector(&detector_input).unwrap();
        let regressor_diff = tract_det
            .regressors
            .iter()
            .zip(&ort_det.regressors)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let logit_diff = tract_det
            .logits
            .iter()
            .zip(&ort_det.logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("detector max abs diff: regressors={regressor_diff} logits={logit_diff}");
        assert!(regressor_diff < 1e-3, "regressor diff was {regressor_diff}");
        assert!(logit_diff < 1e-3, "logit diff was {logit_diff}");

        let landmark_input = gradient_input(LANDMARK_INPUT);
        let tract_lm = tract.run_landmarks(&landmark_input).unwrap();
        let ort_lm = ort.run_landmarks(&landmark_input).unwrap();
        let landmark_diff = tract_lm
            .landmarks
            .iter()
            .zip(&ort_lm.landmarks)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let presence_diff = (tract_lm.presence_logit - ort_lm.presence_logit).abs();
        println!("landmark max abs diff: landmarks={landmark_diff} presence_logit={presence_diff}");
        assert!(landmark_diff < 0.05, "landmark diff was {landmark_diff}");
        assert!(presence_diff < 1e-3, "presence diff was {presence_diff}");
    }

    #[test]
    #[ignore = "benchmark: needs EYE_MODEL_DIR, run in release"]
    fn test_model_tract_landmark_latency_benchmark() {
        let dir = ModelDir::resolve(None).unwrap();
        let mut runtime = TractRuntime::load(&dir).unwrap();
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
}
