pub mod hyprland;
pub mod wayland;

use crate::ProbeError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Transform {
    Normal,
    Rot90,
    Rot180,
    Rot270,
    Flipped,
    Flipped90,
    Flipped180,
    Flipped270,
}

impl Transform {
    /// wl_output.transform wire values 0..=7 (Hyprland uses the same numbering).
    pub fn from_wl(value: u32) -> Option<Self> {
        match value {
            0 => Some(Transform::Normal),
            1 => Some(Transform::Rot90),
            2 => Some(Transform::Rot180),
            3 => Some(Transform::Rot270),
            4 => Some(Transform::Flipped),
            5 => Some(Transform::Flipped90),
            6 => Some(Transform::Flipped180),
            7 => Some(Transform::Flipped270),
            _ => None,
        }
    }

    pub fn swaps_axes(self) -> bool {
        matches!(
            self,
            Transform::Rot90 | Transform::Rot270 | Transform::Flipped90 | Transform::Flipped270
        )
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct OutputInfo {
    pub name: String,
    pub make: String,
    pub model: String,
    /// Current mode in hardware pixels, panel orientation (before `transform`).
    pub mode_px: (u32, u32),
    pub refresh_hz: f64,
    /// `None` when the compositor reports 0 x 0 (projectors, virtual outputs).
    pub physical_mm: Option<(u32, u32)>,
    pub scale: f64,
    pub transform: Transform,
    pub logical_position: (i32, i32),
    /// Size in compositor (logical) coordinates, after `transform` and `scale`.
    pub logical_size: (u32, u32),
}

impl OutputInfo {
    pub fn id(&self) -> eye_core::OutputId {
        eye_core::OutputId::from(self.name.as_str())
    }

    pub fn screen_model(&self) -> Result<eye_core::ScreenModel, ProbeError> {
        let (w, h) = self
            .physical_mm
            .ok_or_else(|| ProbeError::MissingPhysicalSize {
                output: self.name.clone(),
            })?;
        Ok(eye_core::ScreenModel {
            output: self.id(),
            size_mm: nalgebra::Vector2::new(w as f64, h as f64),
            size_px: self.mode_px,
            scale: self.scale,
        })
    }
}

pub trait DisplayProbe {
    fn outputs(&self) -> Result<Vec<OutputInfo>, ProbeError>;

    fn output(&self, name: &str) -> Result<OutputInfo, ProbeError> {
        let outputs = self.outputs()?;
        let available = outputs.iter().map(|o| o.name.clone()).collect();
        outputs
            .into_iter()
            .find(|o| o.name == name)
            .ok_or(ProbeError::OutputNotFound {
                name: name.to_owned(),
                available,
            })
    }
}

pub(crate) fn log_output(info: &OutputInfo, backend: &'static str) {
    let (physical_w_mm, physical_h_mm) = info.physical_mm.unwrap_or((0, 0));
    tracing::info!(
        backend,
        output = %info.name,
        make = %info.make,
        model = %info.model,
        mode_w = info.mode_px.0,
        mode_h = info.mode_px.1,
        refresh_hz = info.refresh_hz,
        physical_w_mm,
        physical_h_mm,
        scale = info.scale,
        transform = ?info.transform,
        logical_x = info.logical_position.0,
        logical_y = info.logical_position.1,
        logical_w = info.logical_size.0,
        logical_h = info.logical_size.1,
        "output probed"
    );
}

/// Logical size from mode pixels: divide by scale, round, swap for 90/270 transforms.
pub(crate) fn logical_size(mode_px: (u32, u32), scale: f64, transform: Transform) -> (u32, u32) {
    let (w, h) = mode_px;
    let logical = (
        (w as f64 / scale).round() as u32,
        (h as f64 / scale).round() as u32,
    );
    if transform.swaps_axes() {
        (logical.1, logical.0)
    } else {
        logical
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_logical_size_normal_divides_by_scale() {
        assert_eq!(
            logical_size((3840, 2160), 2.0, Transform::Normal),
            (1920, 1080)
        );
    }

    #[test]
    fn test_logical_size_rot90_swaps_axes() {
        assert_eq!(
            logical_size((2560, 1600), 1.5, Transform::Rot90),
            (1067, 1707)
        );
    }

    #[test]
    fn test_from_wl_maps_all_eight_values() {
        assert_eq!(Transform::from_wl(0), Some(Transform::Normal));
        assert_eq!(Transform::from_wl(1), Some(Transform::Rot90));
        assert_eq!(Transform::from_wl(2), Some(Transform::Rot180));
        assert_eq!(Transform::from_wl(3), Some(Transform::Rot270));
        assert_eq!(Transform::from_wl(4), Some(Transform::Flipped));
        assert_eq!(Transform::from_wl(5), Some(Transform::Flipped90));
        assert_eq!(Transform::from_wl(6), Some(Transform::Flipped180));
        assert_eq!(Transform::from_wl(7), Some(Transform::Flipped270));
        assert_eq!(Transform::from_wl(8), None);
    }

    proptest! {
        #[test]
        fn test_logical_size_roundtrips_integer_scales(w in 1u32..8000, h in 1u32..8000, s in 1u32..=4) {
            let result = logical_size((w * s, h * s), s as f64, Transform::Normal);
            prop_assert_eq!(result, (w, h));
        }
    }
}
