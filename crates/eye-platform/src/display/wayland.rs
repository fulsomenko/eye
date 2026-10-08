use std::time::Duration;

use smithay_client_toolkit::output::{OutputHandler, OutputState};
use smithay_client_toolkit::reexports::client::globals::registry_queue_init;
use smithay_client_toolkit::reexports::client::protocol::wl_output::WlOutput;
use smithay_client_toolkit::reexports::client::{Connection, QueueHandle};
use smithay_client_toolkit::registry::{ProvidesRegistryState, RegistryState};
use smithay_client_toolkit::{delegate_dispatch2, delegate_registry, registry_handlers};

use crate::ProbeError;
use crate::display::hyprland::HyprlandDisplayProbe;
use crate::display::{DisplayProbe, OutputInfo, Transform, logical_size};
use crate::session::SessionInfo;

#[derive(Debug, Default, Clone, Copy)]
pub struct WaylandDisplayProbe;

impl WaylandDisplayProbe {
    pub fn new() -> Self {
        Self
    }
}

/// Plain copy of the sctk fields we use; sctk's `OutputInfo` is `#[non_exhaustive]`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct WlOutputSnapshot {
    pub global_id: u32,
    pub name: Option<String>,
    pub make: String,
    pub model: String,
    pub physical_size: (i32, i32),
    pub transform: u32,
    pub scale_factor: i32,
    pub current_mode: Option<((i32, i32), i32)>,
    pub logical_position: Option<(i32, i32)>,
    pub logical_size: Option<(i32, i32)>,
}

pub(crate) fn snapshot(info: &smithay_client_toolkit::output::OutputInfo) -> WlOutputSnapshot {
    WlOutputSnapshot {
        global_id: info.id,
        name: info.name.clone(),
        make: info.make.clone(),
        model: info.model.clone(),
        physical_size: info.physical_size,
        transform: u32::from(info.transform),
        scale_factor: info.scale_factor,
        current_mode: info
            .modes
            .iter()
            .find(|m| m.current)
            .map(|m| (m.dimensions, m.refresh_rate)),
        logical_position: info.logical_position,
        logical_size: info.logical_size,
    }
}

pub(crate) fn to_output_info(s: &WlOutputSnapshot) -> Result<OutputInfo, ProbeError> {
    let name = s
        .name
        .clone()
        .unwrap_or_else(|| format!("wl_output-{}", s.global_id));
    if s.name.is_none() {
        tracing::warn!(global_id = s.global_id, "wayland output has no name");
    }
    let transform =
        Transform::from_wl(s.transform).ok_or_else(|| ProbeError::InvalidTransform {
            output: name.clone(),
            value: s.transform,
        })?;
    let ((mw, mh), refresh_mhz) = s.current_mode.ok_or_else(|| ProbeError::Wayland {
        reason: format!("output {name} has no current mode"),
    })?;
    let mode_px = (mw as u32, mh as u32);
    let (scale, resolved_logical_size) = match s.logical_size {
        Some((lw, lh)) if lw > 0 && lh > 0 => {
            let tw = if transform.swaps_axes() {
                mode_px.1
            } else {
                mode_px.0
            };
            let raw = f64::from(tw) / f64::from(lw);
            ((raw * 120.0).round() / 120.0, (lw as u32, lh as u32))
        }
        _ => {
            let scale = f64::from(s.scale_factor.max(1));
            (scale, logical_size(mode_px, scale, transform))
        }
    };
    let physical_mm = match s.physical_size {
        (w, h) if w > 0 && h > 0 => Some((w as u32, h as u32)),
        _ => None,
    };
    Ok(OutputInfo {
        name,
        make: s.make.clone(),
        model: s.model.clone(),
        mode_px,
        refresh_hz: f64::from(refresh_mhz) / 1000.0,
        physical_mm,
        scale,
        transform,
        logical_position: s.logical_position.unwrap_or((0, 0)),
        logical_size: resolved_logical_size,
    })
}

struct ProbeState {
    registry: RegistryState,
    outputs: OutputState,
}

impl OutputHandler for ProbeState {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.outputs
    }

    fn new_output(&mut self, _conn: &Connection, _qh: &QueueHandle<Self>, _output: WlOutput) {}

    fn update_output(&mut self, _conn: &Connection, _qh: &QueueHandle<Self>, _output: WlOutput) {}

    fn output_destroyed(&mut self, _conn: &Connection, _qh: &QueueHandle<Self>, _output: WlOutput) {
    }
}

impl ProvidesRegistryState for ProbeState {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry
    }

    registry_handlers!(OutputState);
}

delegate_registry!(ProbeState);
delegate_dispatch2!(ProbeState);

impl DisplayProbe for WaylandDisplayProbe {
    fn outputs(&self) -> Result<Vec<OutputInfo>, ProbeError> {
        let wl = |e: &dyn std::fmt::Display| ProbeError::Wayland {
            reason: e.to_string(),
        };
        let conn = Connection::connect_to_env().map_err(|e| wl(&e))?;
        let (globals, mut queue) = registry_queue_init::<ProbeState>(&conn).map_err(|e| wl(&e))?;
        let qh = queue.handle();
        let mut state = ProbeState {
            registry: RegistryState::new(&globals),
            outputs: OutputState::new(&globals, &qh),
        };
        queue.roundtrip(&mut state).map_err(|e| wl(&e))?;
        queue.roundtrip(&mut state).map_err(|e| wl(&e))?;
        state
            .outputs
            .outputs()
            .filter_map(|o| state.outputs.info(&o))
            .map(|info| to_output_info(&snapshot(&info)))
            .collect()
    }
}

#[derive(Debug)]
pub enum SelectedDisplayProbe {
    Hyprland(HyprlandDisplayProbe),
    Wayland(WaylandDisplayProbe),
}

impl DisplayProbe for SelectedDisplayProbe {
    fn outputs(&self) -> Result<Vec<OutputInfo>, ProbeError> {
        match self {
            Self::Hyprland(p) => p.outputs(),
            Self::Wayland(p) => p.outputs(),
        }
    }
}

pub fn select_display_probe(session: &SessionInfo) -> Result<SelectedDisplayProbe, ProbeError> {
    session.require_wayland()?;
    match session.hyprland_socket() {
        Ok(socket) if socket.exists() => Ok(SelectedDisplayProbe::Hyprland(
            HyprlandDisplayProbe::with_socket(socket, Duration::from_secs(1)),
        )),
        _ => Ok(SelectedDisplayProbe::Wayland(WaylandDisplayProbe::new())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::os::unix::net::UnixListener;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const REAL_FIXTURE: &str =
        include_str!("../../tests/fixtures/hyprland-monitors-latitude7420.json");

    static DIR_COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn temp_dir() -> std::path::PathBuf {
        let n = DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("eye-wayland-test-{}-{}", std::process::id(), n));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn latitude() -> WlOutputSnapshot {
        WlOutputSnapshot {
            global_id: 40,
            name: Some("eDP-1".to_string()),
            make: "AU Optronics".to_string(),
            model: "0x143B".to_string(),
            physical_size: (310, 170),
            transform: 0,
            scale_factor: 2,
            current_mode: Some(((3840, 2160), 60025)),
            logical_position: Some((0, 0)),
            logical_size: Some((1920, 1080)),
        }
    }

    #[test]
    fn test_latitude_snapshot_matches_hyprland_fixture() {
        let mut from_wayland = to_output_info(&latitude()).unwrap();
        let hyprland_outputs = crate::display::hyprland::parse_monitors(REAL_FIXTURE).unwrap();
        let mut from_hyprland = hyprland_outputs.into_iter().next().unwrap();

        assert_relative_eq!(from_wayland.refresh_hz, 60.025, epsilon = 1e-9);
        assert_relative_eq!(from_hyprland.refresh_hz, 60.025, epsilon = 1e-9);
        from_wayland.refresh_hz = 0.0;
        from_hyprland.refresh_hz = 0.0;

        assert_eq!(from_wayland, from_hyprland);
    }

    #[test]
    fn test_fractional_scale_from_xdg_logical_size_rounds_to_120ths() {
        let snapshot = WlOutputSnapshot {
            global_id: 1,
            name: Some("DP-1".to_string()),
            make: String::new(),
            model: String::new(),
            physical_size: (0, 0),
            transform: 0,
            scale_factor: 2,
            current_mode: Some(((2560, 1600), 60000)),
            logical_position: Some((0, 0)),
            logical_size: Some((1707, 1067)),
        };
        let info = to_output_info(&snapshot).unwrap();
        assert_eq!(info.scale, 1.5);
    }

    #[test]
    fn test_rotated_output_uses_transformed_mode_width() {
        let snapshot = WlOutputSnapshot {
            global_id: 1,
            name: Some("DP-1".to_string()),
            make: String::new(),
            model: String::new(),
            physical_size: (0, 0),
            transform: 1,
            scale_factor: 2,
            current_mode: Some(((2560, 1600), 60000)),
            logical_position: Some((0, 0)),
            logical_size: Some((1067, 1707)),
        };
        let info = to_output_info(&snapshot).unwrap();
        assert_eq!(info.transform, Transform::Rot90);
        assert_eq!(info.scale, 1.5);
        assert_eq!(info.logical_size, (1067, 1707));
    }

    #[test]
    fn test_missing_xdg_output_falls_back_to_integer_scale() {
        let mut snapshot = latitude();
        snapshot.logical_size = None;
        snapshot.logical_position = None;
        let info = to_output_info(&snapshot).unwrap();
        assert_eq!(info.scale, 2.0);
        assert_eq!(info.logical_size, (1920, 1080));
        assert_eq!(info.logical_position, (0, 0));
    }

    #[test]
    fn test_zero_physical_size_is_none() {
        let mut snapshot = latitude();
        snapshot.physical_size = (0, 0);
        let info = to_output_info(&snapshot).unwrap();
        assert_eq!(info.physical_mm, None);
    }

    #[test]
    fn test_unnamed_output_gets_global_id_name() {
        let mut snapshot = latitude();
        snapshot.name = None;
        let info = to_output_info(&snapshot).unwrap();
        assert_eq!(info.name, "wl_output-40");
    }

    #[test]
    fn test_no_current_mode_is_error() {
        let mut snapshot = latitude();
        snapshot.current_mode = None;
        assert!(matches!(
            to_output_info(&snapshot),
            Err(ProbeError::Wayland { .. })
        ));
    }

    #[test]
    fn test_select_prefers_hyprland_when_socket_exists() {
        let dir = temp_dir();
        let sig = "some-signature";
        let hypr_dir = dir.join("hypr").join(sig);
        std::fs::create_dir_all(&hypr_dir).unwrap();
        let socket_path = hypr_dir.join(".socket.sock");
        let _listener = UnixListener::bind(&socket_path).unwrap();

        let runtime_dir = dir.to_string_lossy().into_owned();
        let session = SessionInfo::from_lookup(move |k| match k {
            "WAYLAND_DISPLAY" => Some("wayland-1".into()),
            "HYPRLAND_INSTANCE_SIGNATURE" => Some(sig.into()),
            "XDG_RUNTIME_DIR" => Some(runtime_dir.clone().into()),
            _ => None,
        });

        let probe = select_display_probe(&session).unwrap();
        assert!(matches!(probe, SelectedDisplayProbe::Hyprland(_)));
    }

    #[test]
    fn test_select_falls_back_to_wayland_for_stale_signature() {
        let dir = temp_dir();
        let sig = "stale-signature";
        let runtime_dir = dir.to_string_lossy().into_owned();
        let session = SessionInfo::from_lookup(move |k| match k {
            "WAYLAND_DISPLAY" => Some("wayland-1".into()),
            "HYPRLAND_INSTANCE_SIGNATURE" => Some(sig.into()),
            "XDG_RUNTIME_DIR" => Some(runtime_dir.clone().into()),
            _ => None,
        });

        let probe = select_display_probe(&session).unwrap();
        assert!(matches!(probe, SelectedDisplayProbe::Wayland(_)));
    }

    #[test]
    fn test_select_rejects_non_wayland_session() {
        let session = SessionInfo::from_lookup(|_| None);
        assert!(matches!(
            select_display_probe(&session),
            Err(ProbeError::NotWayland)
        ));
    }

    #[test]
    #[ignore = "needs wayland"]
    fn test_live_wayland_probe_matches_hyprland_probe() {
        let session = SessionInfo::from_env();
        let wayland = WaylandDisplayProbe::new().output("eDP-1").unwrap();
        let hyprland = HyprlandDisplayProbe::new(&session)
            .unwrap()
            .output("eDP-1")
            .unwrap();
        assert_eq!(wayland.name, hyprland.name);
        assert_eq!(wayland.mode_px, hyprland.mode_px);
        assert_eq!(wayland.physical_mm, hyprland.physical_mm);
        assert_eq!(wayland.scale, hyprland.scale);
        assert_eq!(wayland.transform, hyprland.transform);
        assert_eq!(wayland.logical_position, hyprland.logical_position);
        assert_eq!(wayland.logical_size, hyprland.logical_size);
    }
}
