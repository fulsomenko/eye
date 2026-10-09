use eye_core::stage::StageError;
use eye_core::{GazePoint, ScreenModel, Timestamp};
use eye_geometry::screen::{confidence_from_cov, mm_to_px_physical, px_physical_to_logical};
use nalgebra::{Matrix2, Point2};

use crate::FilterError;

impl From<FilterError> for StageError {
    fn from(e: FilterError) -> Self {
        StageError::Config(e.to_string())
    }
}

/// Filters work in mm and rebuild px (and confidence) from mm, so px fields can never drift from mm.
pub(crate) fn with_position(
    point: GazePoint,
    mm: Point2<f64>,
    cov_mm: Matrix2<f64>,
    screen: &ScreenModel,
) -> GazePoint {
    let px_physical = mm_to_px_physical(screen, &mm);
    GazePoint {
        mm,
        px_physical,
        px_logical: px_physical_to_logical(screen, &px_physical),
        cov_mm,
        confidence: confidence_from_cov(&cov_mm),
        ..point
    }
}

/// Seconds from `prev` to `now`; `None` unless strictly increasing.
pub(crate) fn dt_seconds(prev: Timestamp, now: Timestamp) -> Option<f64> {
    now.0
        .checked_sub(prev.0)
        .map(|d| d.as_secs_f64())
        .filter(|&s| s > 0.0)
}

/// `(xx, xy, yy)` entries of a symmetric 2x2 covariance, for logging.
pub(crate) fn cov_fields(m: &Matrix2<f64>) -> (f64, f64, f64) {
    (m[(0, 0)], m[(0, 1)], m[(1, 1)])
}
