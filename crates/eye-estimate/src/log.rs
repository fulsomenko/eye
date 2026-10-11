use eye_core::{GazeRay, Side};
use eye_geometry::angles::yaw_pitch_from_direction;

/// One TRACE per output ray; `source` is the estimator or fused candidate name.
pub(crate) fn trace_ray(source: &'static str, ray: &GazeRay) {
    let a = yaw_pitch_from_direction(&ray.direction);
    tracing::trace!(
        source,
        side = ray.side.map(Side::as_str),
        yaw_rad = a.x,
        pitch_rad = a.y,
        origin_x_mm = ray.origin.x,
        origin_y_mm = ray.origin.y,
        origin_z_mm = ray.origin.z,
        yaw_sigma_rad = ray.angular_cov[(0, 0)].sqrt(),
        pitch_sigma_rad = ray.angular_cov[(1, 1)].sqrt(),
        angular_cov_det = ray.angular_cov.determinant(),
        origin_sigma_mm = (ray.origin_cov.trace() / 3.0).sqrt(),
        "gaze ray"
    );
}
