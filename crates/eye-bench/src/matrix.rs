//! `bench.toml`: a matrix of recordings x pipeline configs to evaluate.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use eye_calibration::protocol::{ProtocolConfig, TargetProtocol};
use eye_calibration::user_fit::FitConfig;
use serde::Deserialize;

use crate::error::BenchError;
use crate::metrics::MetricParams;
use crate::row::CalibrationMode;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchMatrix {
    pub recordings: Vec<PathBuf>,
    #[serde(rename = "pipeline")]
    pub pipelines: Vec<PipelineSpec>,
    #[serde(default)]
    pub evaluation: Evaluation,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PipelineSpec {
    pub name: String,
    /// `None` runs `Config::builtin_default()`.
    pub config: Option<PathBuf>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Evaluation {
    pub calibration: Vec<CalibrationMode>,
    pub protocol: ProtocolConfig,
    pub metrics: MetricParams,
    pub fit: FitConfig,
}

impl Default for Evaluation {
    fn default() -> Self {
        Self {
            calibration: vec![CalibrationMode::None, CalibrationMode::Loto],
            protocol: ProtocolConfig::default(),
            metrics: MetricParams::default(),
            fit: FitConfig::default(),
        }
    }
}

fn resolve(dir: &Path, path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        dir.join(path)
    }
}

fn valid_pipeline_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '+' | '-'))
}

impl BenchMatrix {
    /// Reads, parses, resolves relative paths against `path`'s directory, then `validate`s.
    #[allow(clippy::result_large_err)]
    pub fn from_path(path: &Path) -> Result<Self, BenchError> {
        let text = std::fs::read_to_string(path).map_err(|source| BenchError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let mut matrix: BenchMatrix = toml::from_str(&text)?;
        let dir = path.parent().unwrap_or_else(|| Path::new("."));
        matrix.recordings = matrix
            .recordings
            .into_iter()
            .map(|p| resolve(dir, p))
            .collect();
        for pipeline in &mut matrix.pipelines {
            pipeline.config = pipeline.config.take().map(|p| resolve(dir, p));
        }
        matrix.validate()?;
        Ok(matrix)
    }

    /// One pipeline, named after the config file stem (or "default" for `None`), over `recordings`.
    pub fn single(config: Option<PathBuf>, recordings: Vec<PathBuf>) -> Self {
        let name = config
            .as_ref()
            .and_then(|p| p.file_stem())
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "default".to_string());
        Self {
            recordings,
            pipelines: vec![PipelineSpec { name, config }],
            evaluation: Evaluation::default(),
        }
    }

    /// `Matrix(..)` for: no recordings, no pipelines, a duplicate or invalid pipeline name, an empty or
    /// duplicated `calibration` list, `metrics.validate()` failing (message passed through);
    /// `Calibration(..)` for `TargetProtocol::new(protocol)` failing.
    #[allow(clippy::result_large_err)]
    pub fn validate(&self) -> Result<(), BenchError> {
        if self.recordings.is_empty() {
            return Err(BenchError::Matrix("no recordings".to_string()));
        }
        if self.pipelines.is_empty() {
            return Err(BenchError::Matrix("no pipelines".to_string()));
        }

        let mut seen_names = HashSet::new();
        for pipeline in &self.pipelines {
            if !valid_pipeline_name(&pipeline.name) {
                return Err(BenchError::Matrix(format!(
                    "invalid pipeline name {:?}",
                    pipeline.name
                )));
            }
            if !seen_names.insert(pipeline.name.as_str()) {
                return Err(BenchError::Matrix(format!(
                    "duplicate pipeline name {:?}",
                    pipeline.name
                )));
            }
        }

        if self.evaluation.calibration.is_empty() {
            return Err(BenchError::Matrix(
                "evaluation.calibration must not be empty".to_string(),
            ));
        }
        let mut seen_modes = HashSet::new();
        for mode in &self.evaluation.calibration {
            if !seen_modes.insert(*mode) {
                return Err(BenchError::Matrix(format!(
                    "duplicate calibration mode {mode:?}"
                )));
            }
        }

        self.evaluation
            .metrics
            .validate()
            .map_err(BenchError::Matrix)?;
        TargetProtocol::new(self.evaluation.protocol)
            .map_err(|e| BenchError::Calibration(Box::new(e)))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    fn write_bench_toml(dir: &Path, text: &str) -> PathBuf {
        let path = dir.join("bench.toml");
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(text.as_bytes()).unwrap();
        path
    }

    #[test]
    fn test_matrix_parses_and_resolves_relative_paths() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_bench_toml(
            dir.path(),
            r#"
recordings = ["rec"]

[[pipeline]]
name = "a"
config = "a.toml"
"#,
        );
        let matrix = BenchMatrix::from_path(&path).unwrap();
        assert_eq!(matrix.recordings, vec![dir.path().join("rec")]);
        assert_eq!(matrix.pipelines[0].config, Some(dir.path().join("a.toml")));
    }

    #[test]
    fn test_matrix_rejects_duplicate_pipeline_names() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_bench_toml(
            dir.path(),
            r#"
recordings = ["rec"]

[[pipeline]]
name = "a"

[[pipeline]]
name = "a"
"#,
        );
        assert!(matches!(
            BenchMatrix::from_path(&path),
            Err(BenchError::Matrix(_))
        ));
    }

    #[test]
    fn test_matrix_rejects_bad_pipeline_name() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_bench_toml(
            dir.path(),
            r#"
recordings = ["rec"]

[[pipeline]]
name = "a b"
"#,
        );
        assert!(matches!(
            BenchMatrix::from_path(&path),
            Err(BenchError::Matrix(_))
        ));
    }

    #[test]
    fn test_matrix_rejects_unknown_key() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_bench_toml(
            dir.path(),
            r#"
recordings = ["rec"]

[[pipeline]]
name = "a"

[evaluation]
foo = 1
"#,
        );
        assert!(matches!(
            BenchMatrix::from_path(&path),
            Err(BenchError::Toml(_))
        ));
    }

    #[test]
    fn test_matrix_single_names_pipeline_after_config_stem() {
        let matrix = BenchMatrix::single(
            Some(PathBuf::from("x/ir-classic.toml")),
            vec![PathBuf::from("rec")],
        );
        assert_eq!(matrix.pipelines[0].name, "ir-classic");

        let matrix = BenchMatrix::single(None, vec![PathBuf::from("rec")]);
        assert_eq!(matrix.pipelines[0].name, "default");
    }
}
