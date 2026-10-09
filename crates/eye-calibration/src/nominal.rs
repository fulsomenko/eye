use eye_core::log::field;
use eye_core::{CameraId, CameraModel, Rig, ScreenModel};
use eye_geometry::camera::Intrinsics;
use nalgebra::{Isometry3, Point3, Translation3, UnitQuaternion, Vector2, Vector3};

use crate::error::CalibrationError;

/// Lens centre above the top edge of the active area (thin-bezel lid).
pub const DEFAULT_LENS_ABOVE_ACTIVE_AREA_MM: f64 = 7.0;
/// Generic diagonal FOV band for a laptop webcam with no matched hardware profile.
pub const GENERIC_DIAG_FOV_DEG: [f64; 2] = [60.0, 90.0];

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FocalPrior {
    pub f_px: f64,
    pub sigma_px: f64,
}

/// Two or more candidates: midpoint of the min/max focal lengths at this stream size, sigma
/// = half their difference. One candidate `v`: same formula over the band `v - 2 ..= v + 2`
/// (published FOVs are rounded).
pub fn focal_prior(
    width: u32,
    height: u32,
    diag_fov_deg: &[f64],
) -> Result<FocalPrior, CalibrationError> {
    if diag_fov_deg.is_empty() || diag_fov_deg.iter().any(|d| !d.is_finite()) {
        return Err(CalibrationError::Param {
            name: "diag_fov_deg",
            reason: format!("must be non-empty and finite, got {diag_fov_deg:?}"),
        });
    }
    let degs: Vec<f64> = if diag_fov_deg.len() == 1 {
        vec![diag_fov_deg[0] - 2.0, diag_fov_deg[0] + 2.0]
    } else {
        diag_fov_deg.to_vec()
    };
    let (min, max) = degs
        .iter()
        .map(|deg| Intrinsics::from_diagonal_fov(width, height, deg.to_radians()).fx)
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), f| {
            (min.min(f), max.max(f))
        });
    Ok(FocalPrior {
        f_px: (min + max) / 2.0,
        sigma_px: (max - min) / 2.0,
    })
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NominalRigConfig {
    #[serde(default)]
    pub screen_size_mm: Option<[f64; 2]>,
    #[serde(default, rename = "camera")]
    pub cameras: Vec<NominalCamera>,
    #[serde(default = "NominalRigConfig::default_diag_fov_deg")]
    pub default_diag_fov_deg: Vec<f64>,
}

impl Default for NominalRigConfig {
    fn default() -> Self {
        Self {
            screen_size_mm: None,
            cameras: Vec::new(),
            default_diag_fov_deg: Self::default_diag_fov_deg(),
        }
    }
}

impl NominalRigConfig {
    fn default_diag_fov_deg() -> Vec<f64> {
        GENERIC_DIAG_FOV_DEG.to_vec()
    }
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
    /// Overrides `NominalRigConfig::default_diag_fov_deg` for this camera.
    #[serde(default)]
    pub diag_fov_deg: Option<Vec<f64>>,
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
    let edid_size_mm = screen.size_mm;
    screen.size_mm = corrected_screen_size_mm(&screen, cfg.screen_size_mm);
    if screen.size_mm != edid_size_mm {
        tracing::debug!(
            { field::REASON } = if cfg.screen_size_mm.is_some() {
                "config_override"
            } else {
                "aspect_mismatch"
            },
            edid_w_mm = edid_size_mm.x,
            edid_h_mm = edid_size_mm.y,
            w_mm = screen.size_mm.x,
            h_mm = screen.size_mm.y,
            "screen size corrected"
        );
    }

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

        let mut fov_bounds: Option<(f64, f64)> = None;
        let (f, focal_source) = match config_camera.and_then(|c| c.focal_px) {
            Some(f) => {
                if !(f.is_finite() && f > 0.0) {
                    return Err(CalibrationError::Param {
                        name: "focal_px",
                        reason: format!("must be positive and finite, got {f}"),
                    });
                }
                (f, "config.focal_px")
            }
            None => {
                let camera_fov = config_camera.and_then(|c| c.diag_fov_deg.as_deref());
                let diag_fov_deg = camera_fov.unwrap_or(&cfg.default_diag_fov_deg);
                fov_bounds = Some((
                    diag_fov_deg.iter().copied().fold(f64::INFINITY, f64::min),
                    diag_fov_deg
                        .iter()
                        .copied()
                        .fold(f64::NEG_INFINITY, f64::max),
                ));
                let source = if camera_fov.is_some() {
                    "camera.diag_fov_deg"
                } else {
                    "default_diag_fov_deg"
                };
                (
                    focal_prior(stream.width, stream.height, diag_fov_deg)?.f_px,
                    source,
                )
            }
        };

        let (position, position_source) = match config_camera.and_then(|c| c.position_mm) {
            Some([x, y, z]) => {
                if !(x.is_finite() && y.is_finite() && z.is_finite()) {
                    return Err(CalibrationError::Param {
                        name: "position_mm",
                        reason: format!("must be finite, got [{x}, {y}, {z}]"),
                    });
                }
                (Point3::new(x, y, z), "config")
            }
            None => (default_position, "default"),
        };

        positions.push((stream.id.clone(), position));

        tracing::info!(
            { field::CAMERA } = stream.id.as_str(),
            focal_source,
            f_px = f,
            fov_min_deg = fov_bounds.map(|(lo, _)| lo),
            fov_max_deg = fov_bounds.map(|(_, hi)| hi),
            position_source,
            x_mm = position.x,
            y_mm = position.y,
            z_mm = position.z,
            "nominal camera"
        );

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
                    camera_a = a.as_str(),
                    camera_b = b.as_str(),
                    "cameras share a nominal position; stereo depth is unavailable until calibration-stereo runs"
                );
            }
        }
    }

    tracing::info!(
        output = screen.output.as_str(),
        cameras = cameras.len() as u64,
        screen_w_mm = screen.size_mm.x,
        screen_h_mm = screen.size_mm.y,
        "nominal rig built"
    );

    Ok(Rig::new(cameras, screen)?)
}

/// An inconsistent EDID keeps its diagonal.
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
    let d = w.hypot(h);
    let px_diag = pw.hypot(ph);
    Vector2::new(d * pw / px_diag, d * ph / px_diag)
}

pub fn nominal_screen_from_camera(position_mm: &Point3<f64>) -> Isometry3<f64> {
    Isometry3::from_parts(
        Translation3::from(position_mm.coords),
        UnitQuaternion::from_axis_angle(&Vector3::y_axis(), std::f64::consts::PI),
    )
}

#[cfg(test)]
mod tests {
    use approx::{assert_abs_diff_eq, assert_relative_eq};
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

    const DELL_DIAG_FOV_DEG: [f64; 2] = [75.8, 87.0];

    fn dell_cfg() -> NominalRigConfig {
        NominalRigConfig {
            default_diag_fov_deg: DELL_DIAG_FOV_DEG.to_vec(),
            ..NominalRigConfig::default()
        }
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
    fn test_focal_prior_two_candidates_keeps_dell_numbers() {
        let rgb = focal_prior(1280, 720, &DELL_DIAG_FOV_DEG).unwrap();
        assert_relative_eq!(rgb.f_px, 858.5231, epsilon = 1e-4);
        assert_relative_eq!(rgb.sigma_px, 84.7298, epsilon = 1e-4);

        let ir = focal_prior(640, 360, &DELL_DIAG_FOV_DEG).unwrap();
        assert_relative_eq!(ir.f_px, 429.2616, epsilon = 1e-4);
        assert_relative_eq!(ir.sigma_px, 42.3649, epsilon = 1e-4);

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
    fn test_focal_prior_single_candidate_uses_band() {
        let a = Intrinsics::from_diagonal_fov(1280, 720, 86f64.to_radians()).fx;
        let b = Intrinsics::from_diagonal_fov(1280, 720, 90f64.to_radians()).fx;
        let prior = focal_prior(1280, 720, &[88.0]).unwrap();
        assert_relative_eq!(prior.f_px, (a + b) / 2.0, epsilon = 1e-9);
        assert_relative_eq!(prior.sigma_px, (a - b).abs() / 2.0, epsilon = 1e-9);
    }

    #[test]
    fn test_default_config_uses_generic_prior() {
        assert_eq!(
            NominalRigConfig::from_table(None)
                .unwrap()
                .default_diag_fov_deg,
            vec![60.0, 90.0]
        );

        let empty_table: toml::Table = "".parse().unwrap();
        let cfg = NominalRigConfig::from_table(Some(&empty_table)).unwrap();
        assert_eq!(cfg.default_diag_fov_deg, vec![60.0, 90.0]);

        let rig = nominal_rig(edp1(), &streams(), &NominalRigConfig::default()).unwrap();
        let ir = rig.camera("ir").unwrap();
        let expected = focal_prior(640, 360, &[60.0, 90.0]).unwrap();
        assert_relative_eq!(ir.fx, expected.f_px, epsilon = 1e-9);
    }

    #[test]
    fn test_focal_prior_rejects_empty() {
        let err = focal_prior(1280, 720, &[]).unwrap_err();
        assert!(matches!(
            err,
            CalibrationError::Param {
                name: "diag_fov_deg",
                ..
            }
        ));
    }

    #[test]
    fn test_ir_intrinsics_from_prior() {
        let rig = nominal_rig(edp1(), &streams(), &dell_cfg()).unwrap();
        let ir = rig.camera("ir").unwrap();
        assert_relative_eq!(ir.fx, 429.262, epsilon = 1e-3);
        assert_relative_eq!(ir.fy, 429.262, epsilon = 1e-3);
        assert_relative_eq!(ir.cx, 320.0, epsilon = 1e-9);
        assert_relative_eq!(ir.cy, 180.0, epsilon = 1e-9);
        assert_eq!(ir.distortion, [0.0; 5]);
        assert_relative_eq!(
            ir.screen_from_camera.translation.vector,
            Vector3::new(154.07424315426908, -7.0, 0.0),
            epsilon = 1e-9
        );
    }

    #[test]
    fn test_focal_override_wins() {
        let cfg = NominalRigConfig {
            cameras: vec![NominalCamera {
                id: "ir".into(),
                position_mm: None,
                focal_px: Some(455.0),
                diag_fov_deg: None,
            }],
            ..NominalRigConfig::default()
        };
        let rig = nominal_rig(edp1(), &streams(), &cfg).unwrap();
        let ir = rig.camera("ir").unwrap();
        assert_relative_eq!(ir.fx, 455.0, epsilon = 1e-12);
        assert_relative_eq!(ir.fy, 455.0, epsilon = 1e-12);
    }

    #[test]
    fn test_invalid_focal_errors() {
        let cfg = NominalRigConfig {
            cameras: vec![NominalCamera {
                id: "ir".into(),
                position_mm: None,
                focal_px: Some(-1.0),
                diag_fov_deg: None,
            }],
            ..NominalRigConfig::default()
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
        assert_relative_eq!(size.x, 308.15, epsilon = 0.01);
        assert_relative_eq!(size.y, 173.33, epsilon = 0.01);

        let rig = nominal_rig(edp1(), &streams(), &NominalRigConfig::default()).unwrap();
        assert_relative_eq!(rig.screen().size_mm.y, 173.33, epsilon = 0.01);
    }

    #[test]
    fn test_diagonal_is_preserved_by_correction() {
        let edid = edp1();
        let size = corrected_screen_size_mm(&edid, None);
        assert_relative_eq!(
            size.x.hypot(size.y),
            edid.size_mm.x.hypot(edid.size_mm.y),
            epsilon = 1e-9
        );
        let (pw, ph) = (f64::from(edid.size_px.0), f64::from(edid.size_px.1));
        assert_relative_eq!(size.x / size.y, pw / ph, epsilon = 1e-12);
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
    fn test_logs_screen_size_corrected_at_debug() {
        let (_, records) = eye_log::testing::capture_logs(tracing::Level::DEBUG, || {
            nominal_rig(edp1(), &streams(), &NominalRigConfig::default()).unwrap();
        });
        let rec = records
            .iter()
            .find(|r| r.message == "screen size corrected")
            .expect("no 'screen size corrected' record");
        assert_eq!(rec.level, eye_log::Level::Debug);
        assert_eq!(
            rec.fields.get("reason"),
            Some(&eye_log::Value::Str("aspect_mismatch".to_string()))
        );
        assert_eq!(
            rec.fields.get("edid_h_mm"),
            Some(&eye_log::Value::F64(170.0))
        );
        match rec.fields.get("h_mm") {
            Some(eye_log::Value::F64(v)) => {
                let d = 310.0_f64.hypot(170.0);
                let px_diag = 3840.0_f64.hypot(2160.0);
                assert_abs_diff_eq!(*v, d * 2160.0 / px_diag, epsilon = 1e-9)
            }
            other => panic!("expected h_mm F64, got {other:?}"),
        }

        let mut screen = edp1();
        screen.size_mm = Vector2::new(310.0, 200.0);
        let (_, records) = eye_log::testing::capture_logs(tracing::Level::DEBUG, || {
            nominal_rig(screen, &streams(), &NominalRigConfig::default()).unwrap();
        });
        let rec = records
            .iter()
            .find(|r| r.message == "screen size corrected")
            .expect("no 'screen size corrected' record");
        assert_eq!(
            rec.fields.get("reason"),
            Some(&eye_log::Value::Str("aspect_mismatch".to_string()))
        );
        assert_eq!(
            rec.fields.get("edid_h_mm"),
            Some(&eye_log::Value::F64(200.0))
        );

        let cfg = NominalRigConfig {
            screen_size_mm: Some([300.0, 170.0]),
            ..NominalRigConfig::default()
        };
        let (_, records) = eye_log::testing::capture_logs(tracing::Level::DEBUG, || {
            nominal_rig(edp1(), &streams(), &cfg).unwrap();
        });
        let rec = records
            .iter()
            .find(|r| r.message == "screen size corrected")
            .expect("no 'screen size corrected' record");
        assert_eq!(
            rec.fields.get("reason"),
            Some(&eye_log::Value::Str("config_override".to_string()))
        );
        assert_eq!(rec.fields.get("w_mm"), Some(&eye_log::Value::F64(300.0)));

        let mut consistent = edp1();
        consistent.size_mm = Vector2::new(309.9, 174.3);
        let (_, records) = eye_log::testing::capture_logs(tracing::Level::DEBUG, || {
            nominal_rig(consistent, &streams(), &NominalRigConfig::default()).unwrap();
        });
        assert!(
            !records.iter().any(|r| r.message == "screen size corrected"),
            "{records:?}"
        );
    }

    #[test]
    fn test_config_override_wins() {
        let size = corrected_screen_size_mm(&edp1(), Some([300.0, 170.0]));
        assert_relative_eq!(size.x, 300.0, epsilon = 1e-9);
        assert_relative_eq!(size.y, 170.0, epsilon = 1e-9);

        let cfg = NominalRigConfig {
            screen_size_mm: Some([0.0, 170.0]),
            ..NominalRigConfig::default()
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
                Vector3::new(154.07424315426908, -7.0, 0.0),
                epsilon = 1e-9
            );
        }
    }

    #[test]
    fn test_unknown_camera_in_config_errors() {
        let cfg = NominalRigConfig {
            cameras: vec![NominalCamera {
                id: "depth".into(),
                position_mm: None,
                focal_px: None,
                diag_fov_deg: None,
            }],
            ..NominalRigConfig::default()
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
        let rig = nominal_rig(edp1(), &streams(), &dell_cfg()).unwrap();
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
    fn test_logs_nominal_camera_at_info() {
        let (_, records) = eye_log::testing::capture_logs(tracing::Level::INFO, || {
            nominal_rig(edp1(), &streams(), &dell_cfg()).unwrap();
        });

        let camera_recs: Vec<_> = records
            .iter()
            .filter(|r| r.message == "nominal camera")
            .collect();
        assert_eq!(camera_recs.len(), 2);
        for rec in &camera_recs {
            assert_eq!(rec.level, eye_log::Level::Info);
            match rec.fields.get(field::CAMERA) {
                Some(eye_log::Value::Str(s)) => assert!(s == "rgb" || s == "ir", "{s}"),
                other => panic!("expected camera str, got {other:?}"),
            }
            assert_eq!(
                rec.fields.get("focal_source"),
                Some(&eye_log::Value::Str("default_diag_fov_deg".to_string()))
            );
            assert_eq!(
                rec.fields.get("fov_min_deg"),
                Some(&eye_log::Value::F64(75.8))
            );
            assert_eq!(
                rec.fields.get("fov_max_deg"),
                Some(&eye_log::Value::F64(87.0))
            );
            assert_eq!(
                rec.fields.get("position_source"),
                Some(&eye_log::Value::Str("default".to_string()))
            );
            assert_eq!(rec.fields.get("y_mm"), Some(&eye_log::Value::F64(-7.0)));
        }

        let rig_built = records
            .iter()
            .find(|r| r.message == "nominal rig built")
            .expect("no 'nominal rig built' record");
        assert_eq!(
            rig_built.fields.get("cameras"),
            Some(&eye_log::Value::U64(2))
        );

        let shared = records
            .iter()
            .find(|r| r.message.contains("share a nominal position"))
            .expect("no shared-position record");
        assert!(!shared.message.contains("rgb") && !shared.message.contains("ir"));
        assert_eq!(
            shared.fields.get("camera_a"),
            Some(&eye_log::Value::Str("rgb".to_string()))
        );
        assert_eq!(
            shared.fields.get("camera_b"),
            Some(&eye_log::Value::Str("ir".to_string()))
        );
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
