use eye_core::{CameraId, CameraModel, Rig, ScreenModel};
use eye_geometry::camera::Intrinsics;
use nalgebra::{Isometry3, Point3, Translation3, UnitQuaternion, Vector2, Vector3};

use crate::error::CalibrationError;

/// Dell Latitude 7420 published diagonal FOV of its two webcam modules (2.7 mm module, 6 mm module).
pub const DELL_DIAG_FOV_DEG: [f64; 2] = [75.8, 87.0];
/// Lens centre above the top edge of the active area (thin-bezel lid).
pub const DEFAULT_LENS_ABOVE_ACTIVE_AREA_MM: f64 = 7.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FocalPrior {
    pub f_px: f64,
    pub sigma_px: f64,
}

/// Midpoint of the two Dell modules' focal lengths at this stream size; sigma = half their difference.
pub fn focal_prior(width: u32, height: u32) -> FocalPrior {
    let [a, b] = DELL_DIAG_FOV_DEG
        .map(|deg| Intrinsics::from_diagonal_fov(width, height, deg.to_radians()).fx);
    FocalPrior {
        f_px: (a + b) / 2.0,
        sigma_px: (a - b).abs() / 2.0,
    }
}

#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NominalRigConfig {
    #[serde(default)]
    pub screen_size_mm: Option<[f64; 2]>,
    #[serde(default, rename = "camera")]
    pub cameras: Vec<NominalCamera>,
}

impl NominalRigConfig {
    /// `None` (no `[rig]` section in eye.toml) gives `Default`; otherwise `table.clone().try_into()?`.
    pub fn from_table(table: Option<&toml::Table>) -> Result<Self, CalibrationError> {
        match table {
            None => Ok(Self::default()),
            Some(table) => Ok(table.clone().try_into()?),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NominalCamera {
    pub id: String,
    #[serde(default)]
    pub position_mm: Option<[f64; 3]>,
    #[serde(default)]
    pub focal_px: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CameraStream {
    pub id: CameraId,
    pub width: u32,
    pub height: u32,
}

pub fn nominal_rig(
    mut screen: ScreenModel,
    streams: &[CameraStream],
    cfg: &NominalRigConfig,
) -> Result<Rig, CalibrationError> {
    if let Some([w, h]) = cfg.screen_size_mm
        && !(w.is_finite() && w > 0.0 && h.is_finite() && h > 0.0)
    {
        return Err(CalibrationError::Param {
            name: "screen_size_mm",
            reason: format!("must be positive and finite, got [{w}, {h}]"),
        });
    }
    screen.size_mm = corrected_screen_size_mm(&screen, cfg.screen_size_mm);

    for config_camera in &cfg.cameras {
        if !streams.iter().any(|s| s.id.as_str() == config_camera.id) {
            return Err(CalibrationError::UnknownCamera(config_camera.id.clone()));
        }
    }

    let default_position = Point3::new(
        screen.size_mm.x / 2.0,
        -DEFAULT_LENS_ABOVE_ACTIVE_AREA_MM,
        0.0,
    );
    let mut cameras = Vec::with_capacity(streams.len());
    let mut positions = Vec::with_capacity(streams.len());

    for stream in streams {
        let config_camera = cfg
            .cameras
            .iter()
            .find(|c| c.id.as_str() == stream.id.as_str());

        let f = match config_camera.and_then(|c| c.focal_px) {
            Some(f) => {
                if !(f.is_finite() && f > 0.0) {
                    return Err(CalibrationError::Param {
                        name: "focal_px",
                        reason: format!("must be positive and finite, got {f}"),
                    });
                }
                f
            }
            None => focal_prior(stream.width, stream.height).f_px,
        };

        let position = match config_camera.and_then(|c| c.position_mm) {
            Some([x, y, z]) => {
                if !(x.is_finite() && y.is_finite() && z.is_finite()) {
                    return Err(CalibrationError::Param {
                        name: "position_mm",
                        reason: format!("must be finite, got [{x}, {y}, {z}]"),
                    });
                }
                Point3::new(x, y, z)
            }
            None => default_position,
        };

        positions.push((stream.id.clone(), position));

        cameras.push(CameraModel {
            id: stream.id.clone(),
            width: stream.width,
            height: stream.height,
            fx: f,
            fy: f,
            cx: f64::from(stream.width) / 2.0,
            cy: f64::from(stream.height) / 2.0,
            distortion: [0.0; 5],
            screen_from_camera: nominal_screen_from_camera(&position),
        });
    }

    for i in 0..positions.len() {
        for j in (i + 1)..positions.len() {
            let (a, pa) = &positions[i];
            let (b, pb) = &positions[j];
            if pa == pb {
                tracing::info!(
                    "cameras {a} and {b} share a nominal position; stereo depth is unavailable until calibration-stereo runs"
                );
            }
        }
    }

    Ok(Rig::new(cameras, screen)?)
}

pub fn corrected_screen_size_mm(
    screen: &ScreenModel,
    override_mm: Option<[f64; 2]>,
) -> Vector2<f64> {
    if let Some([w, h]) = override_mm {
        return Vector2::new(w, h);
    }
    let (pw, ph) = (f64::from(screen.size_px.0), f64::from(screen.size_px.1));
    let (w, h) = (screen.size_mm.x, screen.size_mm.y);
    let px_aspect = ph / pw;
    if ((h / w) / px_aspect - 1.0).abs() <= 0.01 {
        return screen.size_mm;
    }
    if w >= h {
        Vector2::new(w, w * px_aspect)
    } else {
        Vector2::new(h / px_aspect, h)
    }
}

pub fn nominal_screen_from_camera(position_mm: &Point3<f64>) -> Isometry3<f64> {
    Isometry3::from_parts(
        Translation3::from(position_mm.coords),
        UnitQuaternion::from_axis_angle(&Vector3::y_axis(), std::f64::consts::PI),
    )
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use eye_core::OutputId;

    use super::*;

    fn edp1() -> ScreenModel {
        ScreenModel {
            output: OutputId::from("eDP-1"),
            size_mm: Vector2::new(310.0, 170.0),
            size_px: (3840, 2160),
            scale: 2.0,
        }
    }

    fn streams() -> Vec<CameraStream> {
        vec![
            CameraStream {
                id: CameraId::from("rgb"),
                width: 1280,
                height: 720,
            },
            CameraStream {
                id: CameraId::from("ir"),
                width: 640,
                height: 360,
            },
        ]
    }

    #[test]
    fn test_focal_from_diag_fov_matches_dell_numbers() {
        let a = Intrinsics::from_diagonal_fov(1280, 720, 75.8f64.to_radians());
        let b = Intrinsics::from_diagonal_fov(1280, 720, 87f64.to_radians());
        assert_relative_eq!(a.fx, 943.253, epsilon = 1e-3);
        assert_relative_eq!(b.fx, 773.793, epsilon = 1e-3);

        let c = Intrinsics::from_diagonal_fov(640, 360, 75.8f64.to_radians());
        let d = Intrinsics::from_diagonal_fov(640, 360, 87f64.to_radians());
        assert_relative_eq!(c.fx, 471.627, epsilon = 1e-3);
        assert_relative_eq!(d.fx, 386.897, epsilon = 1e-3);
    }

    #[test]
    fn test_focal_prior_covers_both_dell_modules() {
        let rgb = focal_prior(1280, 720);
        assert_relative_eq!(rgb.f_px, 858.523, epsilon = 1e-3);
        assert_relative_eq!(rgb.sigma_px, 84.730, epsilon = 1e-3);

        let ir = focal_prior(640, 360);
        assert_relative_eq!(ir.f_px, 429.262, epsilon = 1e-3);
        assert_relative_eq!(ir.sigma_px, 42.365, epsilon = 1e-3);

        for (w, h, prior) in [(1280, 720, rgb), (640, 360, ir)] {
            for deg in DELL_DIAG_FOV_DEG {
                let f = Intrinsics::from_diagonal_fov(w, h, deg.to_radians()).fx;
                assert!(
                    (f - prior.f_px).abs() <= prior.sigma_px + 1e-9,
                    "module focal {f} outside prior {prior:?} for {w}x{h}"
                );
            }
        }
    }

    #[test]
    fn test_ir_intrinsics_from_prior() {
        let rig = nominal_rig(edp1(), &streams(), &NominalRigConfig::default()).unwrap();
        let ir = rig.camera("ir").unwrap();
        assert_relative_eq!(ir.fx, 429.262, epsilon = 1e-3);
        assert_relative_eq!(ir.fy, 429.262, epsilon = 1e-3);
        assert_relative_eq!(ir.cx, 320.0, epsilon = 1e-9);
        assert_relative_eq!(ir.cy, 180.0, epsilon = 1e-9);
        assert_eq!(ir.distortion, [0.0; 5]);
        assert_relative_eq!(
            ir.screen_from_camera.translation.vector,
            Vector3::new(155.0, -7.0, 0.0),
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_focal_override_wins() {
        let cfg = NominalRigConfig {
            screen_size_mm: None,
            cameras: vec![NominalCamera {
                id: "ir".into(),
                position_mm: None,
                focal_px: Some(455.0),
            }],
        };
        let rig = nominal_rig(edp1(), &streams(), &cfg).unwrap();
        let ir = rig.camera("ir").unwrap();
        assert_relative_eq!(ir.fx, 455.0, epsilon = 1e-12);
        assert_relative_eq!(ir.fy, 455.0, epsilon = 1e-12);
    }

    #[test]
    fn test_invalid_focal_errors() {
        let cfg = NominalRigConfig {
            screen_size_mm: None,
            cameras: vec![NominalCamera {
                id: "ir".into(),
                position_mm: None,
                focal_px: Some(-1.0),
            }],
        };
        let err = nominal_rig(edp1(), &streams(), &cfg).unwrap_err();
        assert!(matches!(
            err,
            CalibrationError::Param {
                name: "focal_px",
                ..
            }
        ));
    }

    #[test]
    fn test_edid_height_corrected_by_square_pixels() {
        let size = corrected_screen_size_mm(&edp1(), None);
        assert_relative_eq!(size.x, 310.0, epsilon = 1e-9);
        assert_relative_eq!(size.y, 174.375, epsilon = 1e-9);

        let rig = nominal_rig(edp1(), &streams(), &NominalRigConfig::default()).unwrap();
        assert_relative_eq!(rig.screen().size_mm.y, 174.375, epsilon = 1e-9);
    }

    #[test]
    fn test_consistent_edid_kept() {
        let mut screen = edp1();
        screen.size_mm = Vector2::new(309.9, 174.3);
        let size = corrected_screen_size_mm(&screen, None);
        assert_relative_eq!(size.x, 309.9, epsilon = 1e-9);
        assert_relative_eq!(size.y, 174.3, epsilon = 1e-9);
    }

    #[test]
    fn test_config_override_wins() {
        let size = corrected_screen_size_mm(&edp1(), Some([300.0, 170.0]));
        assert_relative_eq!(size.x, 300.0, epsilon = 1e-9);
        assert_relative_eq!(size.y, 170.0, epsilon = 1e-9);

        let cfg = NominalRigConfig {
            screen_size_mm: Some([0.0, 170.0]),
            cameras: Vec::new(),
        };
        let err = nominal_rig(edp1(), &streams(), &cfg).unwrap_err();
        assert!(matches!(
            err,
            CalibrationError::Param {
                name: "screen_size_mm",
                ..
            }
        ));
    }

    #[test]
    fn test_nominal_camera_looks_at_user() {
        let pose = nominal_screen_from_camera(&Point3::new(155.0, -7.0, 0.0));
        assert_relative_eq!(
            pose.transform_vector(&Vector3::z()),
            Vector3::new(0.0, 0.0, -1.0),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            pose.transform_vector(&Vector3::x()),
            Vector3::new(-1.0, 0.0, 0.0),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            pose.transform_vector(&Vector3::y()),
            Vector3::new(0.0, 1.0, 0.0),
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_default_position_is_top_centre() {
        let rig = nominal_rig(edp1(), &streams(), &NominalRigConfig::default()).unwrap();
        for id in ["rgb", "ir"] {
            let cam = rig.camera(id).unwrap();
            assert_relative_eq!(
                cam.screen_from_camera.translation.vector,
                Vector3::new(155.0, -7.0, 0.0),
                epsilon = 1e-12
            );
        }
    }

    #[test]
    fn test_unknown_camera_in_config_errors() {
        let cfg = NominalRigConfig {
            screen_size_mm: None,
            cameras: vec![NominalCamera {
                id: "depth".into(),
                position_mm: None,
                focal_px: None,
            }],
        };
        let err = nominal_rig(edp1(), &streams(), &cfg).unwrap_err();
        assert!(matches!(err, CalibrationError::UnknownCamera(id) if id == "depth"));
    }

    #[test]
    fn test_parse_rig_section_from_toml() {
        let toml_str = r#"
            screen_size_mm = [310.0, 174.4]
            [[camera]]
            id = "ir"
            position_mm = [155.0, -7.0, 0.0]
            focal_px = 429.26
        "#;
        let table: toml::Table = toml_str.parse().unwrap();
        let cfg = NominalRigConfig::from_table(Some(&table)).unwrap();
        assert_eq!(cfg.screen_size_mm, Some([310.0, 174.4]));
        assert_eq!(cfg.cameras.len(), 1);
        assert_eq!(cfg.cameras[0].id, "ir");
        assert_eq!(cfg.cameras[0].position_mm, Some([155.0, -7.0, 0.0]));
        assert_eq!(cfg.cameras[0].focal_px, Some(429.26));

        let mut bad_table = table.clone();
        bad_table.insert("tilt".into(), 3.into());
        assert!(matches!(
            NominalRigConfig::from_table(Some(&bad_table)),
            Err(CalibrationError::Config(_))
        ));
    }

    #[test]
    fn test_absent_rig_section_is_default() {
        assert_eq!(
            NominalRigConfig::from_table(None).unwrap(),
            NominalRigConfig::default()
        );
    }

    #[test]
    fn test_point_in_front_of_screen_projects_into_nominal_camera() {
        let rig = nominal_rig(edp1(), &streams(), &NominalRigConfig::default()).unwrap();
        let ir = rig.camera("ir").unwrap();
        let face_point = Point3::new(155.0, 85.0, -500.0);
        let p_cam = ir.screen_from_camera.inverse_transform_point(&face_point);
        assert!(p_cam.z > 0.0);

        let intrinsics = Intrinsics::from_camera_model(ir);
        let px = intrinsics.project(&p_cam).unwrap();
        assert!(px.x >= 0.0 && px.x <= 640.0);
        assert!(px.y >= 0.0 && px.y <= 360.0);
        assert_relative_eq!(px.x, 320.0, epsilon = 1.0);
        assert_relative_eq!(px.y, 259.0, epsilon = 1.0);
    }

    #[test]
    fn test_core_error_converts() {
        let err = nominal_rig(edp1(), &[], &NominalRigConfig::default()).unwrap_err();
        assert!(matches!(
            err,
            CalibrationError::Core(eye_core::CoreError::EmptyRig)
        ));
    }
}
