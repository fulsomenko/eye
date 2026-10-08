//! Resolves the on-disk location of the MediaPipe ONNX models the detectors load.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use crate::error::DetectError;

pub const EYE_MODEL_DIR: &str = "EYE_MODEL_DIR";
const FIX_HINT: &str = "set model_dir in eye.toml or EYE_MODEL_DIR (nix develop sets it; or: nix build .#mediapipe-models and export EYE_MODEL_DIR=$(readlink -f result))";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFile {
    FaceDetector,
    FaceLandmarks,
}

impl ModelFile {
    pub fn relative_path(self) -> &'static str {
        match self {
            ModelFile::FaceDetector => "mediapipe/face_detector.onnx",
            ModelFile::FaceLandmarks => "mediapipe/face_landmarks_detector.onnx",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelDir(PathBuf);

impl ModelDir {
    /// Precedence: `configured` (the detector's `model_dir` option) > `$EYE_MODEL_DIR`. No implicit fallback.
    pub fn resolve(configured: Option<&Path>) -> Result<Self, DetectError> {
        Self::resolve_with(configured, std::env::var_os(EYE_MODEL_DIR))
    }

    pub fn resolve_with(
        configured: Option<&Path>,
        env_value: Option<OsString>,
    ) -> Result<Self, DetectError> {
        let dir = match (configured, env_value) {
            (Some(dir), _) => dir.to_path_buf(),
            (None, Some(v)) if !v.is_empty() => PathBuf::from(v),
            _ => {
                return Err(DetectError::Model {
                    path: PathBuf::new(),
                    reason: format!("model directory not set: {FIX_HINT}"),
                });
            }
        };
        if !dir.is_dir() {
            return Err(DetectError::Model {
                path: dir,
                reason: format!("not a directory: {FIX_HINT}"),
            });
        }
        Ok(Self(dir))
    }

    pub fn path(&self, file: ModelFile) -> Result<PathBuf, DetectError> {
        let path = self.0.join(file.relative_path());
        if path.is_file() {
            Ok(path)
        } else {
            Err(DetectError::Model {
                path,
                reason: format!("model file missing: {FIX_HINT}"),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_prefers_configured_over_env() {
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        let resolved =
            ModelDir::resolve_with(Some(a.path()), Some(OsString::from(b.path()))).unwrap();
        assert_eq!(resolved, ModelDir(a.path().to_path_buf()));
    }

    #[test]
    fn test_resolve_falls_back_to_env() {
        let dir = tempfile::tempdir().unwrap();
        let resolved = ModelDir::resolve_with(None, Some(OsString::from(dir.path()))).unwrap();
        assert_eq!(resolved, ModelDir(dir.path().to_path_buf()));
    }

    #[test]
    fn test_resolve_without_any_source_errors_with_fix_hint() {
        let err = ModelDir::resolve_with(None, None).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("EYE_MODEL_DIR"), "message was: {msg}");
        assert!(
            msg.contains("nix build .#mediapipe-models"),
            "message was: {msg}"
        );
    }

    #[test]
    fn test_resolve_empty_env_value_is_treated_as_unset() {
        let err = ModelDir::resolve_with(None, Some(OsString::new())).unwrap_err();
        let expected = ModelDir::resolve_with(None, None).unwrap_err();
        assert_eq!(err.to_string(), expected.to_string());
    }

    #[test]
    fn test_resolve_nonexistent_configured_dir_is_model_error() {
        let missing = PathBuf::from("/nonexistent/eye-detect-models-missing");
        let err = ModelDir::resolve_with(Some(&missing), None).unwrap_err();
        match err {
            DetectError::Model { path, .. } => assert_eq!(path, missing),
            other => panic!("expected DetectError::Model, got {other:?}"),
        }
    }

    #[test]
    fn test_path_missing_file_names_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let resolved = ModelDir::resolve_with(Some(dir.path()), None).unwrap();
        let err = resolved.path(ModelFile::FaceLandmarks).unwrap_err();
        match err {
            DetectError::Model { path, .. } => {
                assert!(
                    path.ends_with("mediapipe/face_landmarks_detector.onnx"),
                    "path was: {path:?}"
                );
            }
            other => panic!("expected DetectError::Model, got {other:?}"),
        }
    }

    #[test]
    fn test_path_existing_file_is_returned() {
        let dir = tempfile::tempdir().unwrap();
        let mediapipe_dir = dir.path().join("mediapipe");
        std::fs::create_dir_all(&mediapipe_dir).unwrap();
        let file_path = mediapipe_dir.join("face_detector.onnx");
        std::fs::write(&file_path, b"stub").unwrap();

        let resolved = ModelDir::resolve_with(Some(dir.path()), None).unwrap();
        let path = resolved.path(ModelFile::FaceDetector).unwrap();
        assert_eq!(path, file_path);
    }

    #[test]
    #[ignore = "needs EYE_MODEL_DIR"]
    fn test_nix_model_dir_contains_all_files() {
        let resolved = ModelDir::resolve(None).unwrap();
        for file in [ModelFile::FaceDetector, ModelFile::FaceLandmarks] {
            resolved.path(file).unwrap();
        }
        let dir = &resolved.0;
        assert!(dir.join("mediapipe/tflite/face_detector.tflite").is_file());
        assert!(
            dir.join("mediapipe/tflite/face_landmarks_detector.tflite")
                .is_file()
        );
    }
}
