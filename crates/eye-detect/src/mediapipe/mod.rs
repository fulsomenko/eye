pub mod anchors;
pub mod blazeface;
pub mod landmarks;
#[cfg(feature = "mediapipe-ort")]
pub mod ort;
pub mod pipeline;
pub mod roi;
#[cfg(feature = "mediapipe-tract")]
pub mod tract;
pub mod warp;

use std::path::PathBuf;

use crate::DetectError;
use crate::mediapipe::blazeface::FaceDetectorOutput;

pub const LANDMARK_INPUT: usize = 256;
pub const NUM_LANDMARKS: usize = eye_core::observation::mediapipe478::COUNT;

pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[derive(Debug, Clone, PartialEq)]
pub struct LandmarkOutput {
    /// `478 * 3` crop-pixel `(x, y, z)` triples; `z` is dropped downstream.
    pub landmarks: Vec<f32>,
    pub presence_logit: f32,
}

pub trait MediaPipeRuntime: Send + std::fmt::Debug {
    /// `input`: NHWC [1, 128, 128, 3], RGB, values in [-1, 1].
    fn run_face_detector(&mut self, input: &[f32]) -> Result<FaceDetectorOutput, DetectError>;
    /// `input`: NHWC [1, 256, 256, 3], RGB, values in [0, 1].
    fn run_landmarks(&mut self, input: &[f32]) -> Result<LandmarkOutput, DetectError>;
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct MediaPipeOptions {
    /// `None`: resolved from `EYE_MODEL_DIR` via [`crate::models::ModelDir`].
    pub model_dir: Option<PathBuf>,
    /// Intra-op threads; ignored by `mediapipe-tract`.
    pub threads: usize,
    pub min_detection_score: f32,
    pub nms_iou: f32,
    pub min_presence: f32,
    /// When true, reuse the previous frame's landmark ROI and skip the detector.
    pub track: bool,
    pub iris_sigma_crop_px: f64,
    pub corner_sigma_crop_px: f64,
}

impl Default for MediaPipeOptions {
    fn default() -> Self {
        Self {
            model_dir: None,
            threads: 2,
            min_detection_score: 0.5,
            nms_iou: 0.3,
            min_presence: 0.5,
            track: true,
            iris_sigma_crop_px: 0.15,
            corner_sigma_crop_px: 0.3,
        }
    }
}

/// `(name, shape)` per input or output; dynamic dims are `-1`.
pub type ModelIo = Vec<(String, Vec<i64>)>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DetectorRoles {
    pub regressors: usize,
    pub logits: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LandmarkRoles {
    pub landmarks: usize,
    pub presence: usize,
}

fn elements(shape: &[i64]) -> i64 {
    shape.iter().map(|d| d.max(&1)).product()
}

fn input_ok(inputs: &ModelIo, n: usize) -> bool {
    let n = n as i64;
    matches!(inputs.as_slice(), [(_, s)] if s.len() == 4 && (s[0] == 1 || s[0] == -1) && s[1..] == [n, n, 3])
}

/// Roles are matched by output shape, not name.
///
/// Err(reason listing every input/output name and shape); the adapter wraps it in
/// `DetectError::Model { path, reason }`.
pub fn resolve_detector_roles(
    inputs: &ModelIo,
    outputs: &ModelIo,
) -> Result<DetectorRoles, String> {
    use crate::mediapipe::blazeface::{DETECTOR_INPUT, NUM_ANCHORS, NUM_COORDS};

    let find = |count: usize| {
        outputs
            .iter()
            .position(|(_, s)| elements(s) == count as i64)
    };
    match (
        input_ok(inputs, DETECTOR_INPUT),
        find(NUM_ANCHORS * NUM_COORDS),
        find(NUM_ANCHORS),
    ) {
        (true, Some(regressors), Some(logits)) => Ok(DetectorRoles { regressors, logits }),
        _ => Err(format!(
            "unexpected model I/O: inputs {inputs:?}, outputs {outputs:?}"
        )),
    }
}

/// Two 1-element outputs exist in the real model (`Identity_1 [1,1,1,1]` the presence logit,
/// `Identity_2 [1,1]` unused); presence is picked as the rank-4 one.
///
/// Err(reason listing every input/output name and shape); the adapter wraps it in
/// `DetectError::Model { path, reason }`.
pub fn resolve_landmark_roles(
    inputs: &ModelIo,
    outputs: &ModelIo,
) -> Result<LandmarkRoles, String> {
    let landmarks = outputs
        .iter()
        .position(|(_, s)| elements(s) == (NUM_LANDMARKS * 3) as i64);
    let presence = outputs
        .iter()
        .position(|(_, s)| s.len() == 4 && elements(s) == 1);
    match (input_ok(inputs, LANDMARK_INPUT), landmarks, presence) {
        (true, Some(landmarks), Some(presence)) => Ok(LandmarkRoles {
            landmarks,
            presence,
        }),
        _ => Err(format!(
            "unexpected model I/O: inputs {inputs:?}, outputs {outputs:?}"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid_midpoint_and_symmetry() {
        assert_eq!(sigmoid(0.0), 0.5);
        let x = 3.0f32;
        assert!((sigmoid(x) + sigmoid(-x) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_resolve_detector_roles_by_element_count() {
        let inputs: ModelIo = vec![("".into(), vec![1, 128, 128, 3])];
        let outputs: ModelIo = vec![
            ("a".into(), vec![1, 896, 1]),
            ("b".into(), vec![1, 896, 16]),
        ];
        let roles = resolve_detector_roles(&inputs, &outputs).unwrap();
        assert_eq!(
            roles,
            DetectorRoles {
                regressors: 1,
                logits: 0
            }
        );
    }

    #[test]
    fn test_resolve_detector_roles_wrong_input_shape_is_error() {
        let inputs: ModelIo = vec![("".into(), vec![1, 3, 128, 128])];
        let outputs: ModelIo = vec![
            ("a".into(), vec![1, 896, 1]),
            ("b".into(), vec![1, 896, 16]),
        ];
        let err = resolve_detector_roles(&inputs, &outputs).unwrap_err();
        assert!(err.contains("[1, 3, 128, 128]"));
    }

    #[test]
    fn test_resolve_landmark_roles_picks_rank4_scalar_for_presence() {
        let inputs: ModelIo = vec![("input_12".into(), vec![-1, 256, 256, 3])];
        let outputs: ModelIo = vec![
            ("x".into(), vec![1, 1]),
            ("y".into(), vec![1, 1, 1, 1434]),
            ("z".into(), vec![1, 1, 1, 1]),
        ];
        let roles = resolve_landmark_roles(&inputs, &outputs).unwrap();
        assert_eq!(
            roles,
            LandmarkRoles {
                landmarks: 1,
                presence: 2
            }
        );
    }

    #[test]
    fn test_options_reject_unknown_field() {
        let mut table = toml::Table::new();
        table.insert("unknown_option".into(), true.into());
        let err =
            crate::options::parse_options::<MediaPipeOptions>("mediapipe", &table).unwrap_err();
        assert!(matches!(
            err,
            DetectError::Config {
                detector: "mediapipe",
                ..
            }
        ));
    }
}
