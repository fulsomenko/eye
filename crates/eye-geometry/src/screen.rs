//! Screen intersection: turns a `GazeRay` (reference/screen frame, R1) into a `GazePoint` on
//! the panel, and owns the mm <-> physical px <-> logical px conversions for `ScreenModel`.

use eye_core::{GazePoint, GazeRay, ScreenModel, Timestamp};
use nalgebra::{Point2, Point3, Unit, Vector2, Vector3, Vector5};

use crate::angles::{direction_from_yaw_pitch, yaw_pitch_from_direction};
use crate::uncertainty::{Cov2, block_diag, propagate_fn};

/// Scale for `confidence_from_cov`: `exp(-sigma_major / CONFIDENCE_SCALE_MM)`.
pub const CONFIDENCE_SCALE_MM: f64 = 20.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenHit {
    pub mm: Point2<f64>,
    pub cov_mm: Cov2,
}

/// Intersection with the plane z = 0. `None` when the ray is parallel to the panel or points
/// away from it.
pub fn intersect_plane(o: &Point3<f64>, d: &Unit<Vector3<f64>>) -> Option<Point2<f64>> {
    if d.z <= f64::EPSILON {
        return None;
    }
    let t = -o.z / d.z;
    (t > 0.0).then(|| Point2::new(o.x + t * d.x, o.y + t * d.y))
}

pub fn intersect_gaze(ray: &GazeRay) -> Option<ScreenHit> {
    let angles = yaw_pitch_from_direction(&ray.direction);
    let x = Vector5::new(ray.origin.x, ray.origin.y, ray.origin.z, angles.x, angles.y);
    let cov_in = block_diag::<3, 2, 5>(&ray.origin_cov, &ray.angular_cov);
    let f = |v: &Vector5<f64>| {
        let d = direction_from_yaw_pitch(&Vector2::new(v[3], v[4]));
        intersect_plane(&Point3::new(v[0], v[1], v[2]), &d).map(|p| p.coords)
    };
    let (mm, cov_mm) = propagate_fn(f, &x, &cov_in)?;
    Some(ScreenHit {
        mm: mm.into(),
        cov_mm,
    })
}

pub fn gaze_point(ray: &GazeRay, screen: &ScreenModel, timestamp: Timestamp) -> Option<GazePoint> {
    let hit = intersect_gaze(ray)?;
    let px_physical = mm_to_px_physical(screen, &hit.mm);
    Some(GazePoint {
        timestamp,
        output: screen.output.clone(),
        mm: hit.mm,
        px_physical,
        px_logical: px_physical_to_logical(screen, &px_physical),
        cov_mm: hit.cov_mm,
        confidence: confidence_from_cov(&hit.cov_mm),
    })
}

pub fn mm_to_px_physical(screen: &ScreenModel, mm: &Point2<f64>) -> Point2<f64> {
    Point2::new(
        mm.x * screen.size_px.0 as f64 / screen.size_mm.x,
        mm.y * screen.size_px.1 as f64 / screen.size_mm.y,
    )
}

pub fn px_physical_to_mm(screen: &ScreenModel, px: &Point2<f64>) -> Point2<f64> {
    Point2::new(
        px.x * screen.size_mm.x / screen.size_px.0 as f64,
        px.y * screen.size_mm.y / screen.size_px.1 as f64,
    )
}

pub fn px_physical_to_logical(screen: &ScreenModel, px: &Point2<f64>) -> Point2<f64> {
    Point2::new(px.x / screen.scale, px.y / screen.scale)
}

pub fn px_logical_to_physical(screen: &ScreenModel, px: &Point2<f64>) -> Point2<f64> {
    Point2::new(px.x * screen.scale, px.y * screen.scale)
}

pub fn mm_to_px_logical(screen: &ScreenModel, mm: &Point2<f64>) -> Point2<f64> {
    px_physical_to_logical(screen, &mm_to_px_physical(screen, mm))
}

pub fn px_logical_to_mm(screen: &ScreenModel, px: &Point2<f64>) -> Point2<f64> {
    px_physical_to_mm(screen, &px_logical_to_physical(screen, px))
}

/// Monotone display/gating hint in (0, 1], not a probability: `exp(-sigma_major / CONFIDENCE_SCALE_MM)`.
pub fn confidence_from_cov(cov_mm: &Cov2) -> f64 {
    let lambda_max = cov_mm.symmetric_eigen().eigenvalues.max().max(0.0);
    (-lambda_max.sqrt() / CONFIDENCE_SCALE_MM).exp()
}

#[cfg(test)]
mod tests {
    use approx::{assert_abs_diff_eq, assert_relative_eq};
    use eye_core::{OutputId, Side};
    use nalgebra::{Matrix2, Matrix3, Vector2};
    use proptest::prelude::*;

    use super::*;
    use crate::synth::SplitMix64;

    fn edp1() -> ScreenModel {
        ScreenModel {
            output: OutputId::new("eDP-1"),
            size_mm: Vector2::new(310.0, 170.0),
            size_px: (3840, 2160),
            scale: 2.0,
        }
    }

    fn ray(
        yaw_deg: f64,
        pitch_deg: f64,
        origin_cov: Matrix3<f64>,
        angular_cov: Matrix2<f64>,
    ) -> GazeRay {
        GazeRay {
            side: None,
            origin: Point3::new(155.0, 85.0, -500.0),
            direction: direction_from_yaw_pitch(&Vector2::new(
                yaw_deg.to_radians(),
                pitch_deg.to_radians(),
            )),
            angular_cov,
            origin_cov,
            head_rotation: nalgebra::UnitQuaternion::identity(),
        }
    }

    #[test]
    fn test_perpendicular_ray_hits_foot_point() {
        let hit = intersect_gaze(&ray(0.0, 0.0, Matrix3::zeros(), Matrix2::zeros())).unwrap();
        assert_abs_diff_eq!(hit.mm, Point2::new(155.0, 85.0), epsilon = 1e-12);
    }

    #[test]
    fn test_ray_parallel_or_away_returns_none() {
        assert!(
            intersect_plane(
                &Point3::new(0.0, 0.0, 0.0),
                &Unit::new_normalize(Vector3::new(1.0, 0.0, 0.0))
            )
            .is_none()
        );
        assert!(
            intersect_plane(
                &Point3::new(0.0, 0.0, 0.0),
                &Unit::new_normalize(Vector3::new(0.0, 0.0, -1.0))
            )
            .is_none()
        );
        assert!(
            intersect_plane(
                &Point3::new(0.0, 0.0, 10.0),
                &Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0))
            )
            .is_none()
        );
    }

    #[test]
    fn test_yaw_10deg_at_500mm_offsets_by_tan() {
        let hit = intersect_gaze(&ray(10.0, 0.0, Matrix3::zeros(), Matrix2::zeros())).unwrap();
        let expected_x = 155.0 + 500.0 * 10f64.to_radians().tan();
        assert_abs_diff_eq!(hit.mm.x, expected_x, epsilon = 1e-9);
    }

    #[test]
    fn test_angular_cov_maps_to_mm_cov() {
        let sigma = 1f64.to_radians();
        let hit = intersect_gaze(&ray(
            0.0,
            0.0,
            Matrix3::zeros(),
            Matrix2::identity() * sigma * sigma,
        ))
        .unwrap();
        let expected = (500.0 * sigma).powi(2);
        assert_relative_eq!(hit.cov_mm[(0, 0)], expected, max_relative = 1e-6);
        assert_relative_eq!(hit.cov_mm[(1, 1)], expected, max_relative = 1e-6);
        assert_abs_diff_eq!(hit.cov_mm[(0, 1)], 0.0, epsilon = 1e-9);
    }

    #[test]
    fn test_depth_uncertainty_leaks_into_x_for_oblique_ray() {
        let origin_cov = Matrix3::from_diagonal(&Vector3::new(0.0, 0.0, 100.0));
        let hit = intersect_gaze(&ray(20.0, 0.0, origin_cov, Matrix2::zeros())).unwrap();
        let expected = 100.0 * 20f64.to_radians().tan().powi(2);
        assert_relative_eq!(hit.cov_mm[(0, 0)], expected, max_relative = 1e-5);
    }

    #[test]
    fn test_cov_matches_monte_carlo() {
        let origin_cov = Matrix3::from_diagonal(&Vector3::new(4.0, 4.0, 225.0));
        let angular_cov = Matrix2::from_diagonal(&Vector2::new(
            1.5f64.to_radians().powi(2),
            1f64.to_radians().powi(2),
        ));
        let r = ray(15.0, -5.0, origin_cov, angular_cov);
        let predicted = intersect_gaze(&r).unwrap().cov_mm;

        let n = 20_000;
        let mut rng = SplitMix64::new(3);
        let origin_chol = origin_cov.cholesky().unwrap();
        let angular_chol = angular_cov.cholesky().unwrap();
        let base_angles = yaw_pitch_from_direction(&r.direction);
        let mut sum = Vector2::zeros();
        let mut samples = Vec::with_capacity(n);
        for _ in 0..n {
            let origin_noise = Vector3::new(rng.gaussian(), rng.gaussian(), rng.gaussian());
            let angle_noise = Vector2::new(rng.gaussian(), rng.gaussian());
            let o = r.origin + origin_chol.l() * origin_noise;
            let a = base_angles + angular_chol.l() * angle_noise;
            let d = direction_from_yaw_pitch(&a);
            let hit = intersect_plane(&o, &d).unwrap();
            sum += hit.coords;
            samples.push(hit.coords);
        }
        let mean = sum / n as f64;
        let mut empirical = Matrix2::zeros();
        for s in &samples {
            let diff = s - mean;
            empirical += diff * diff.transpose();
        }
        empirical /= (n - 1) as f64;

        for i in 0..2 {
            assert!(
                (empirical[(i, i)] - predicted[(i, i)]).abs() <= 0.05 * predicted[(i, i)].abs(),
                "diag {i}: empirical {} predicted {}",
                empirical[(i, i)],
                predicted[(i, i)]
            );
        }
        for i in 0..2 {
            for j in 0..2 {
                if i == j {
                    continue;
                }
                let tol = 0.05 * (predicted[(i, i)] * predicted[(j, j)]).sqrt();
                assert!(
                    (empirical[(i, j)] - predicted[(i, j)]).abs() <= tol,
                    "offdiag ({i},{j}): empirical {} predicted {}",
                    empirical[(i, j)],
                    predicted[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_edp1_centre_mm_to_px() {
        let screen = edp1();
        let mm = Point2::new(155.0, 85.0);
        assert_abs_diff_eq!(
            mm_to_px_physical(&screen, &mm),
            Point2::new(1920.0, 1080.0),
            epsilon = 1e-9
        );
        assert_abs_diff_eq!(
            mm_to_px_logical(&screen, &mm),
            Point2::new(960.0, 540.0),
            epsilon = 1e-9
        );
    }

    proptest! {
        #[test]
        fn prop_mm_px_round_trip(x in -50.0..360.0, y in -50.0..220.0) {
            let screen = edp1();
            let mm = Point2::new(x, y);

            let logical = mm_to_px_logical(&screen, &mm);
            let back_logical = px_logical_to_mm(&screen, &logical);
            prop_assert!((back_logical.x - mm.x).abs() < 1e-9);
            prop_assert!((back_logical.y - mm.y).abs() < 1e-9);

            let physical = mm_to_px_physical(&screen, &mm);
            let back_physical = px_physical_to_mm(&screen, &physical);
            prop_assert!((back_physical.x - mm.x).abs() < 1e-9);
            prop_assert!((back_physical.y - mm.y).abs() < 1e-9);
        }
    }

    #[test]
    fn test_off_panel_hit_is_not_clamped() {
        let screen = edp1();
        let px = mm_to_px_physical(&screen, &Point2::new(-20.0, 85.0));
        assert_abs_diff_eq!(px.x, -20.0 * 3840.0 / 310.0, epsilon = 1e-9);
    }

    #[test]
    fn test_confidence_reference_values() {
        let cov = Matrix2::from_diagonal(&Vector2::new(400.0, 100.0));
        assert_abs_diff_eq!(confidence_from_cov(&cov), (-1.0f64).exp(), epsilon = 1e-12);
        assert_abs_diff_eq!(confidence_from_cov(&Matrix2::zeros()), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_gaze_point_fills_every_field() {
        let origin_cov = Matrix3::from_diagonal(&Vector3::new(4.0, 4.0, 100.0));
        let angular_cov = Matrix2::from_diagonal(&Vector2::new(1e-4, 1e-4));
        let r = ray(5.0, -3.0, origin_cov, angular_cov);
        let screen = edp1();
        let timestamp = Timestamp::from_nanos(42);

        let point = gaze_point(&r, &screen, timestamp).unwrap();

        assert_eq!(point.timestamp, timestamp);
        assert_eq!(point.output, screen.output);
        assert_abs_diff_eq!(
            point.px_logical,
            Point2::new(point.px_physical.x / 2.0, point.px_physical.y / 2.0),
            epsilon = 1e-9
        );
        assert!(point.validate().is_ok());
    }

    #[test]
    fn test_side_field_unused_by_intersection() {
        let mut with_side = ray(0.0, 0.0, Matrix3::zeros(), Matrix2::zeros());
        with_side.side = Some(Side::Left);
        let hit = intersect_gaze(&with_side).unwrap();
        assert_abs_diff_eq!(hit.mm, Point2::new(155.0, 85.0), epsilon = 1e-12);
    }
}
