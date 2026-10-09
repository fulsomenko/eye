use std::ffi::OsString;
use std::path::PathBuf;

use crate::ProbeError;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionType {
    Wayland,
    X11,
    Tty,
    Other(String),
    Unset,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Compositor {
    Hyprland { instance_signature: String },
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct SessionInfo {
    pub session_type: SessionType,
    pub wayland_display: Option<String>,
    pub compositor: Compositor,
    pub runtime_dir: Option<PathBuf>,
}

impl SessionInfo {
    pub fn from_env() -> Self {
        Self::from_lookup(|k| std::env::var_os(k))
    }

    pub fn from_lookup(lookup: impl Fn(&str) -> Option<OsString>) -> Self {
        let get = |k: &str| {
            lookup(k)
                .map(|v| v.to_string_lossy().into_owned())
                .filter(|v| !v.is_empty())
        };
        let session_type = match get("XDG_SESSION_TYPE").as_deref() {
            None => SessionType::Unset,
            Some("wayland") => SessionType::Wayland,
            Some("x11") => SessionType::X11,
            Some("tty") => SessionType::Tty,
            Some(other) => SessionType::Other(other.to_owned()),
        };
        let compositor = match get("HYPRLAND_INSTANCE_SIGNATURE") {
            Some(instance_signature) => Compositor::Hyprland { instance_signature },
            None => Compositor::Unknown,
        };
        let wayland_display = get("WAYLAND_DISPLAY");
        let runtime_dir = get("XDG_RUNTIME_DIR").map(PathBuf::from);
        tracing::debug!(
            session_type = ?session_type,
            wayland = wayland_display.is_some(),
            compositor = match &compositor {
                Compositor::Hyprland { .. } => "hyprland",
                Compositor::Unknown => "unknown",
            },
            runtime_dir = runtime_dir.is_some(),
            "session detected"
        );
        Self {
            session_type,
            wayland_display,
            compositor,
            runtime_dir,
        }
    }

    /// True when a Wayland socket is advertised; `XDG_SESSION_TYPE` is informational only.
    pub fn is_wayland(&self) -> bool {
        self.wayland_display.is_some()
    }

    pub fn require_wayland(&self) -> Result<(), ProbeError> {
        if self.is_wayland() {
            Ok(())
        } else {
            Err(ProbeError::NotWayland)
        }
    }

    /// `$XDG_RUNTIME_DIR/hypr/<signature>/.socket.sock`; existence is not checked.
    pub fn hyprland_socket(&self) -> Result<PathBuf, ProbeError> {
        let Compositor::Hyprland { instance_signature } = &self.compositor else {
            return Err(ProbeError::NotHyprland);
        };
        let runtime_dir = self.runtime_dir.as_ref().ok_or(ProbeError::NoRuntimeDir)?;
        Ok(runtime_dir
            .join("hypr")
            .join(instance_signature)
            .join(".socket.sock"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<OsString> {
        let map: HashMap<String, OsString> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), OsString::from(v)))
            .collect();
        move |k: &str| map.get(k).cloned()
    }

    #[test]
    fn test_dev_machine_env_detects_hyprland_wayland() {
        let info = SessionInfo::from_lookup(env(&[
            ("XDG_SESSION_TYPE", "wayland"),
            ("WAYLAND_DISPLAY", "wayland-1"),
            (
                "HYPRLAND_INSTANCE_SIGNATURE",
                "78e2f9b6d17c11100f60c1316b8824ef10a10af3_1790868721_1848945805",
            ),
            ("XDG_RUNTIME_DIR", "/run/user/1001"),
        ]));
        assert_eq!(info.session_type, SessionType::Wayland);
        assert_eq!(info.wayland_display, Some("wayland-1".to_string()));
        assert_eq!(
            info.compositor,
            Compositor::Hyprland {
                instance_signature:
                    "78e2f9b6d17c11100f60c1316b8824ef10a10af3_1790868721_1848945805".to_string()
            }
        );
        assert_eq!(
            info.hyprland_socket().unwrap(),
            PathBuf::from(
                "/run/user/1001/hypr/78e2f9b6d17c11100f60c1316b8824ef10a10af3_1790868721_1848945805/.socket.sock"
            )
        );
    }

    #[test]
    fn test_wayland_display_without_session_type_is_wayland() {
        let info = SessionInfo::from_lookup(env(&[("WAYLAND_DISPLAY", "wayland-0")]));
        assert!(info.is_wayland());
        assert_eq!(info.session_type, SessionType::Unset);
        assert!(info.require_wayland().is_ok());
    }

    #[test]
    fn test_x11_session_without_wayland_display_is_rejected() {
        let info = SessionInfo::from_lookup(env(&[("XDG_SESSION_TYPE", "x11")]));
        assert!(matches!(
            info.require_wayland(),
            Err(ProbeError::NotWayland)
        ));
    }

    #[test]
    fn test_empty_signature_is_not_hyprland() {
        let info = SessionInfo::from_lookup(env(&[("HYPRLAND_INSTANCE_SIGNATURE", "")]));
        assert_eq!(info.compositor, Compositor::Unknown);
        assert!(matches!(
            info.hyprland_socket(),
            Err(ProbeError::NotHyprland)
        ));
    }

    #[test]
    fn test_hyprland_without_runtime_dir_errors() {
        let info =
            SessionInfo::from_lookup(env(&[("HYPRLAND_INSTANCE_SIGNATURE", "some-signature")]));
        assert!(matches!(
            info.hyprland_socket(),
            Err(ProbeError::NoRuntimeDir)
        ));
    }

    #[test]
    fn test_unknown_session_type_is_preserved() {
        let info = SessionInfo::from_lookup(env(&[("XDG_SESSION_TYPE", "mir")]));
        assert_eq!(info.session_type, SessionType::Other("mir".to_string()));
    }

    #[test]
    fn test_session_info_serializes_to_toml() {
        let info = SessionInfo::from_lookup(env(&[
            ("XDG_SESSION_TYPE", "wayland"),
            ("WAYLAND_DISPLAY", "wayland-1"),
            (
                "HYPRLAND_INSTANCE_SIGNATURE",
                "78e2f9b6d17c11100f60c1316b8824ef10a10af3_1790868721_1848945805",
            ),
            ("XDG_RUNTIME_DIR", "/run/user/1001"),
        ]));
        let table = toml::Table::try_from(&info).unwrap();
        assert_eq!(table["session_type"].as_str(), Some("wayland"));
        assert_eq!(
            table["compositor"]["hyprland"]["instance_signature"].as_str(),
            Some("78e2f9b6d17c11100f60c1316b8824ef10a10af3_1790868721_1848945805")
        );
    }

    #[test]
    fn test_logs_session_detected_at_debug() {
        use eye_log::Value;

        let (_, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            SessionInfo::from_lookup(env(&[
                ("WAYLAND_DISPLAY", "wayland-1"),
                ("HYPRLAND_INSTANCE_SIGNATURE", "s"),
            ]))
        });
        let record = records
            .iter()
            .find(|r| r.message == "session detected")
            .expect("session detected logged");
        assert_eq!(record.level, eye_log::Level::Debug);
        assert_eq!(record.fields.get("wayland"), Some(&Value::Bool(true)));
        assert_eq!(
            record.fields.get("compositor"),
            Some(&Value::Str("hyprland".to_string()))
        );
    }

    #[test]
    #[ignore = "needs wayland"]
    fn test_live_env_is_hyprland_wayland() {
        let info = SessionInfo::from_env();
        assert!(info.is_wayland());
        assert!(matches!(info.compositor, Compositor::Hyprland { .. }));
        let socket = info.hyprland_socket().unwrap();
        assert!(socket.exists());
    }
}
