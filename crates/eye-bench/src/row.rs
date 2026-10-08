use serde::{Deserialize, Serialize};

use crate::metrics::SessionMetrics;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CalibrationMode {
    None,
}

impl CalibrationMode {
    /// The serialized name ("none").
    pub fn as_str(self) -> &'static str {
        match self {
            CalibrationMode::None => "none",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum RowKind {
    Session,
    Aggregate,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "lowercase")]
pub enum RowOutcome {
    Ok { metrics: SessionMetrics },
    Error { message: String },
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BenchRow {
    pub pipeline: String,
    pub calibration: CalibrationMode,
    pub kind: RowKind,
    /// Session id, or `"all"` for an aggregate row.
    pub session: String,
    /// FrameSets whose pipeline step was skipped after a stage error (they count as dropout).
    pub step_errors: usize,
    /// Non-fatal problems (e.g. a failed calibration fold); listed under `## Warnings`.
    pub warnings: Vec<String>,
    #[serde(flatten)]
    pub outcome: RowOutcome,
}
