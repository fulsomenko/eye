use nalgebra::{Isometry3, Vector2};
use serde::{Deserialize, Serialize};

use crate::{CameraId, CoreError, OutputId};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraModel {
    pub id: CameraId,
    pub width: u32,
    pub height: u32,
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    /// OpenCV order: k1, k2, p1, p2, k3.
    pub distortion: [f64; 5],
    /// Maps camera-frame points to the reference (screen) frame, mm.
    pub screen_from_camera: Isometry3<f64>,
}

impl CameraModel {
    pub fn validate(&self) -> Result<(), CoreError> {
        let fail = |reason| {
            Err(CoreError::InvalidCameraModel {
                camera: self.id.clone(),
                reason,
            })
        };
        if self.width == 0 || self.height == 0 {
            return fail("image size must be non-zero");
        }
        if !(self.fx.is_finite() && self.fx > 0.0 && self.fy.is_finite() && self.fy > 0.0) {
            return fail("focal lengths must be positive and finite");
        }
        if !(self.cx.is_finite() && self.cy.is_finite()) {
            return fail("principal point must be finite");
        }
        if !self.distortion.iter().all(|k| k.is_finite()) {
            return fail("distortion must be finite");
        }
        let pose = &self.screen_from_camera;
        if !(pose.translation.vector.iter().all(|v| v.is_finite())
            && pose.rotation.coords.iter().all(|v| v.is_finite()))
        {
            return fail("screen_from_camera must be finite");
        }
        if (pose.rotation.coords.norm() - 1.0).abs() > 1e-9 {
            return fail("screen_from_camera rotation must be a unit quaternion");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScreenModel {
    pub output: OutputId,
    /// Active area width and height, mm.
    pub size_mm: Vector2<f64>,
    /// Physical (pre-scale) pixels.
    pub size_px: (u32, u32),
    /// Logical px = physical px / scale.
    pub scale: f64,
}

impl ScreenModel {
    pub fn validate(&self) -> Result<(), CoreError> {
        let fail = |reason| {
            Err(CoreError::InvalidScreen {
                output: self.output.clone(),
                reason,
            })
        };
        if !self.size_mm.iter().all(|v| v.is_finite() && *v > 0.0) {
            return fail("size_mm must be positive and finite");
        }
        if self.size_px.0 == 0 || self.size_px.1 == 0 {
            return fail("size_px must be non-zero");
        }
        if !(self.scale.is_finite() && self.scale > 0.0) {
            return fail("scale must be positive and finite");
        }
        Ok(())
    }
}

/// Validated camera models (at least one, unique ids) plus the screen they are mounted on.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "RigRepr", into = "RigRepr")]
pub struct Rig {
    cameras: Vec<CameraModel>,
    screen: ScreenModel,
}

#[derive(Serialize, Deserialize)]
struct RigRepr {
    cameras: Vec<CameraModel>,
    screen: ScreenModel,
}

impl Rig {
    pub fn new(cameras: Vec<CameraModel>, screen: ScreenModel) -> Result<Self, CoreError> {
        if cameras.is_empty() {
            return Err(CoreError::EmptyRig);
        }
        for (i, camera) in cameras.iter().enumerate() {
            camera.validate()?;
            if cameras[..i].iter().any(|earlier| earlier.id == camera.id) {
                return Err(CoreError::DuplicateCamera(camera.id.clone()));
            }
        }
        screen.validate()?;
        Ok(Self { cameras, screen })
    }

    pub fn cameras(&self) -> &[CameraModel] {
        &self.cameras
    }

    pub fn camera<Q: AsRef<str> + ?Sized>(&self, id: &Q) -> Option<&CameraModel> {
        let id = id.as_ref();
        self.cameras.iter().find(|c| c.id.as_str() == id)
    }

    pub fn screen(&self) -> &ScreenModel {
        &self.screen
    }

    pub fn into_parts(self) -> (Vec<CameraModel>, ScreenModel) {
        (self.cameras, self.screen)
    }
}

impl TryFrom<RigRepr> for Rig {
    type Error = CoreError;

    fn try_from(repr: RigRepr) -> Result<Self, Self::Error> {
        Self::new(repr.cameras, repr.screen)
    }
}

impl From<Rig> for RigRepr {
    fn from(rig: Rig) -> Self {
        Self {
            cameras: rig.cameras,
            screen: rig.screen,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use approx::assert_relative_eq;
    use nalgebra::{Translation3, UnitQuaternion, Vector3};

    use super::*;
    use crate::CoreError;

    fn nominal_camera() -> CameraModel {
        CameraModel {
            id: CameraId::from("ir"),
            width: 640,
            height: 360,
            fx: 500.0,
            fy: 500.0,
            cx: 320.0,
            cy: 180.0,
            distortion: [0.0; 5],
            screen_from_camera: Isometry3::from_parts(
                Translation3::new(155.0, -8.0, 0.0),
                UnitQuaternion::from_axis_angle(&Vector3::y_axis(), PI),
            ),
        }
    }

    fn nominal_screen() -> ScreenModel {
        ScreenModel {
            output: OutputId::from("eDP-1"),
            size_mm: Vector2::new(310.0, 170.0),
            size_px: (3840, 2160),
            scale: 2.0,
        }
    }

    #[test]
    fn test_nominal_camera_looks_towards_user() {
        let pose = nominal_camera().screen_from_camera;
        let z = pose.rotation * Vector3::z();
        let x = pose.rotation * Vector3::x();
        assert_relative_eq!(z, -Vector3::z(), epsilon = 1e-12);
        assert_relative_eq!(x, -Vector3::x(), epsilon = 1e-12);
        assert_relative_eq!(
            pose.translation.vector,
            Vector3::new(155.0, -8.0, 0.0),
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_rig_rejects_empty_cameras() {
        assert_eq!(
            Rig::new(Vec::new(), nominal_screen()),
            Err(CoreError::EmptyRig)
        );
    }

    #[test]
    fn test_rig_rejects_duplicate_camera_id() {
        let cam = nominal_camera();
        assert_eq!(
            Rig::new(vec![cam.clone(), cam], nominal_screen()),
            Err(CoreError::DuplicateCamera(CameraId::from("ir")))
        );
    }

    #[test]
    fn test_rig_rejects_zero_focal_length() {
        let mut cam = nominal_camera();
        cam.fx = 0.0;
        assert_eq!(
            Rig::new(vec![cam], nominal_screen()),
            Err(CoreError::InvalidCameraModel {
                camera: CameraId::from("ir"),
                reason: "focal lengths must be positive and finite",
            })
        );
    }

    #[test]
    fn test_rig_rejects_zero_scale() {
        let mut screen = nominal_screen();
        screen.scale = 0.0;
        assert!(matches!(
            Rig::new(vec![nominal_camera()], screen),
            Err(CoreError::InvalidScreen { .. })
        ));
    }

    #[test]
    fn test_rig_camera_lookup_by_name() {
        let rig = Rig::new(vec![nominal_camera()], nominal_screen()).unwrap();
        assert_eq!(rig.camera("ir").unwrap().width, 640);
        assert!(rig.camera("rgb").is_none());
    }

    #[test]
    fn test_rig_camera_accepts_str_and_id() {
        let rig = Rig::new(vec![nominal_camera()], nominal_screen()).unwrap();
        assert!(rig.camera(&CameraId::from("ir")).is_some());
    }

    #[test]
    fn test_rig_toml_roundtrip_equal() {
        let rig = Rig::new(vec![nominal_camera()], nominal_screen()).unwrap();
        let text = toml::to_string(&rig).unwrap();
        let back: Rig = toml::from_str(&text).unwrap();
        assert_eq!(back.screen(), rig.screen());
        assert_relative_eq!(
            back.cameras()[0].screen_from_camera,
            rig.cameras()[0].screen_from_camera,
            epsilon = 1e-12
        );
        assert_eq!(back.cameras()[0].fx, 500.0);
    }

    #[test]
    fn test_rig_json_roundtrip_equal() {
        let rig = Rig::new(vec![nominal_camera()], nominal_screen()).unwrap();
        let text = serde_json::to_string(&rig).unwrap();
        let back: Rig = serde_json::from_str(&text).unwrap();
        assert_eq!(back, rig);
    }

    #[test]
    fn test_rig_deserialize_rejects_invalid_rig() {
        let rig = Rig::new(vec![nominal_camera()], nominal_screen()).unwrap();
        let text = toml::to_string(&rig)
            .unwrap()
            .replace("fx = 500.0", "fx = -1.0");
        let err = toml::from_str::<Rig>(&text).unwrap_err();
        assert!(err.to_string().contains("focal lengths"));
    }

    #[test]
    fn test_rig_deserialize_rejects_non_unit_rotation() {
        let rig = Rig::new(vec![nominal_camera()], nominal_screen()).unwrap();
        let text = toml::to_string(&rig)
            .unwrap()
            .replace("rotation = [0.0, 1.0,", "rotation = [0.0, 2.0,");
        let err = toml::from_str::<Rig>(&text).unwrap_err();
        assert!(err.to_string().contains("unit quaternion"));
    }
}
