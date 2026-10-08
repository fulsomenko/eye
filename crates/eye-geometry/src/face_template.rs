//! 3D face template keyed by MediaPipe landmark index, for PnP head-pose solves.

use eye_core::Measured;
use eye_core::observation::SCHEME_MEDIAPIPE_478;
use nalgebra::{Point2, Point3};

use crate::GeometryError;

/// Template points (head frame, mm) and the matching image landmarks, same order.
pub type Correspondences = (Vec<Point3<f64>>, Vec<Measured<Point2<f64>>>);

#[derive(Debug, Clone, Copy)]
pub struct FaceTemplate {
    pub scheme: &'static str,
    pub points: &'static [(usize, [f64; 3])], // (landmark index, head-frame mm)
}

// Generated from canonical_face_model.obj at google-ai-edge/mediapipe commit
// f6988c4769278bde600efd488dfc8645432dc92b (sha256 8bac80443397e113f41a8b565ea72c59390bc031d9defab289dba7bc0c54e618),
// Apache-2.0. Provenance script (already run; do not re-run, it needs the network):
//
// curl -L -o /tmp/cfm.obj \
//   "https://raw.githubusercontent.com/google-ai-edge/mediapipe/f6988c4769278bde600efd488dfc8645432dc92b/mediapipe/modules/face_geometry/data/canonical_face_model.obj"
// sha256sum /tmp/cfm.obj   # 8bac80443397e113f41a8b565ea72c59390bc031d9defab289dba7bc0c54e618
// python3 - <<'EOF'
// ids = [1, 4, 5, 6, 33, 98, 133, 168, 195, 197, 263, 327, 362]
// v = [l.split()[1:4] for l in open("/tmp/cfm.obj") if l.startswith("v ")]
// assert len(v) == 468
// for i in ids:
//     x, y, z = map(float, v[i])
//     print(f"        ({i}, [{10*x:.3f}, {-10*y:.3f}, {-10*z:.3f}]),")
// EOF
/// Rigid subset of the MediaPipe 468/478 mesh: eye corners and nose (no mouth, jaw or brows,
/// which move with expression). Generated from canonical_face_model.obj at google-ai-edge/mediapipe
/// commit f6988c4769278bde600efd488dfc8645432dc92b (sha256 8bac8044…e618), Apache-2.0,
/// converted with head = 10 · (x, −y, −z) mm. See crates/eye-geometry/NOTICE.
pub const MEDIAPIPE_RIGID: FaceTemplate = FaceTemplate {
    scheme: SCHEME_MEDIAPIPE_478,
    points: &[
        (1, [0.000, 11.269, -74.756]),
        (4, [0.000, 4.632, -75.866]),
        (5, [0.000, -3.657, -72.429]),
        (6, [0.000, -24.733, -57.886]),
        (33, [-44.459, -26.640, -31.734]),
        (98, [-14.056, 17.142, -52.411]),
        (133, [-18.564, -25.852, -37.579]),
        (168, [0.000, -32.710, -52.360]),
        (195, [0.000, -10.594, -67.746]),
        (197, [0.000, -17.284, -63.167]),
        (263, [44.459, -26.640, -31.734]),
        (327, [14.056, 17.142, -52.411]),
        (362, [18.564, -25.852, -37.579]),
    ],
};

impl FaceTemplate {
    pub fn point(&self, landmark: usize) -> Option<Point3<f64>> {
        self.points
            .iter()
            .find(|(i, _)| *i == landmark)
            .map(|(_, p)| Point3::new(p[0], p[1], p[2]))
    }

    /// Pairs template points with detected landmarks (σ applied to every landmark).
    pub fn correspondences(
        &self,
        landmarks: &[Point2<f64>],
        sigma_px: f64,
    ) -> Result<Correspondences, GeometryError> {
        let need = self.points.iter().map(|(i, _)| i + 1).max().unwrap_or(0);
        if landmarks.len() < need {
            return Err(GeometryError::TooFewObservations {
                need,
                got: landmarks.len(),
            });
        }
        let mut object = Vec::with_capacity(self.points.len());
        let mut image = Vec::with_capacity(self.points.len());
        for &(i, p) in self.points {
            object.push(Point3::new(p[0], p[1], p[2]));
            image.push(
                Measured::new(landmarks[i], sigma_px)
                    .map_err(|_| GeometryError::Degenerate("sigma_px"))?,
            );
        }
        Ok((object, image))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_template_axes_match_head_frame() {
        let p33 = MEDIAPIPE_RIGID.point(33).unwrap();
        let p263 = MEDIAPIPE_RIGID.point(263).unwrap();
        assert!(p33.x < 0.0);
        assert!(p263.x > 0.0);

        let min_z = MEDIAPIPE_RIGID
            .points
            .iter()
            .map(|(_, p)| p[2])
            .fold(f64::INFINITY, f64::min);
        let p4 = MEDIAPIPE_RIGID.point(4).unwrap();
        assert_eq!(p4.z, min_z);
        let p1 = MEDIAPIPE_RIGID.point(1).unwrap();
        assert!((p1.z - p4.z).abs() <= 1.5);

        let p168 = MEDIAPIPE_RIGID.point(168).unwrap();
        assert!(p168.y < p1.y);
    }

    #[test]
    fn test_template_scale_is_millimetres() {
        let p33 = MEDIAPIPE_RIGID.point(33).unwrap();
        let p263 = MEDIAPIPE_RIGID.point(263).unwrap();
        let outer = (p33 - p263).norm();
        assert!((85.0..=105.0).contains(&outer), "outer canthi = {outer}");

        let p133 = MEDIAPIPE_RIGID.point(133).unwrap();
        let p362 = MEDIAPIPE_RIGID.point(362).unwrap();
        let inner = (p133 - p362).norm();
        assert!((25.0..=40.0).contains(&inner), "inner canthi = {inner}");
    }

    #[test]
    fn test_correspondences_rejects_short_landmark_vector() {
        let landmarks = vec![Point2::new(0.0, 0.0); 100];
        let err = MEDIAPIPE_RIGID
            .correspondences(&landmarks, 1.0)
            .unwrap_err();
        assert_eq!(
            err,
            GeometryError::TooFewObservations {
                need: 363,
                got: 100
            }
        );
    }
}
