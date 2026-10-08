use std::io::{Read, Write};
use std::net::Shutdown;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::session::SessionInfo;
use crate::{DisplayProbe, OutputInfo, ProbeError, Transform};

#[derive(Debug, Clone)]
pub struct HyprlandDisplayProbe {
    socket: PathBuf,
    timeout: Duration,
}

impl HyprlandDisplayProbe {
    pub fn new(session: &SessionInfo) -> Result<Self, ProbeError> {
        Ok(Self::with_socket(
            session.hyprland_socket()?,
            Duration::from_secs(1),
        ))
    }

    pub fn with_socket(socket: PathBuf, timeout: Duration) -> Self {
        Self { socket, timeout }
    }
}

impl DisplayProbe for HyprlandDisplayProbe {
    fn outputs(&self) -> Result<Vec<OutputInfo>, ProbeError> {
        parse_monitors(&request(&self.socket, "j/monitors", self.timeout)?)
    }
}

pub(crate) fn request(
    socket: &Path,
    command: &str,
    timeout: Duration,
) -> Result<String, ProbeError> {
    let io = |source| ProbeError::Io {
        path: socket.to_owned(),
        source,
    };
    let mut stream = UnixStream::connect(socket).map_err(io)?;
    stream.set_read_timeout(Some(timeout)).map_err(io)?;
    stream.set_write_timeout(Some(timeout)).map_err(io)?;
    stream.write_all(command.as_bytes()).map_err(io)?;
    stream.shutdown(Shutdown::Write).map_err(io)?;
    let mut reply = String::new();
    stream.read_to_string(&mut reply).map_err(io)?;
    Ok(reply)
}

pub(crate) fn parse_monitors(json: &str) -> Result<Vec<OutputInfo>, ProbeError> {
    if !json.trim_start().starts_with('[') {
        return Err(ProbeError::Hyprland {
            reply: json.to_owned(),
        });
    }
    let monitors: Vec<HyprMonitor> = serde_json::from_str(json)?;
    monitors
        .into_iter()
        .filter(|m| !m.disabled)
        .map(HyprMonitor::into_output_info)
        .collect()
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct HyprMonitor {
    name: String,
    #[serde(default)]
    make: String,
    #[serde(default)]
    model: String,
    width: u32,
    height: u32,
    #[serde(rename = "physicalWidth")]
    physical_width: u32,
    #[serde(rename = "physicalHeight")]
    physical_height: u32,
    #[serde(rename = "refreshRate")]
    refresh_rate: f64,
    x: i32,
    y: i32,
    scale: f64,
    transform: u32,
    #[serde(default)]
    disabled: bool,
}

impl HyprMonitor {
    fn into_output_info(self) -> Result<OutputInfo, ProbeError> {
        let transform =
            Transform::from_wl(self.transform).ok_or_else(|| ProbeError::InvalidTransform {
                output: self.name.clone(),
                value: self.transform,
            })?;
        let physical_mm = if self.physical_width == 0 || self.physical_height == 0 {
            None
        } else {
            Some((self.physical_width, self.physical_height))
        };
        let mode_px = (self.width, self.height);
        Ok(OutputInfo {
            name: self.name,
            make: self.make,
            model: self.model,
            mode_px,
            refresh_hz: self.refresh_rate,
            physical_mm,
            scale: self.scale,
            transform,
            logical_position: (self.x, self.y),
            logical_size: crate::display::logical_size(mode_px, self.scale, transform),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::display::DisplayProbe;
    use approx::assert_relative_eq;
    use nalgebra::Vector2;
    use std::os::unix::net::UnixListener;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static SOCKET_COUNTER: AtomicUsize = AtomicUsize::new(0);

    const REAL_FIXTURE: &str =
        include_str!("../../tests/fixtures/hyprland-monitors-latitude7420.json");
    const SYNTHETIC_FIXTURE: &str =
        include_str!("../../tests/fixtures/hyprland-monitors-synthetic.json");

    #[test]
    fn test_real_latitude_fixture_parses_edp1() {
        let outputs = parse_monitors(REAL_FIXTURE).unwrap();
        assert_eq!(outputs.len(), 1);
        let o = &outputs[0];
        assert_eq!(o.name, "eDP-1");
        assert_eq!(o.make, "AU Optronics");
        assert_eq!(o.model, "0x143B");
        assert_eq!(o.mode_px, (3840, 2160));
        assert_eq!(o.physical_mm, Some((310, 170)));
        assert_eq!(o.scale, 2.0);
        assert_eq!(o.transform, Transform::Normal);
        assert_eq!(o.logical_position, (0, 0));
        assert_eq!(o.logical_size, (1920, 1080));
        assert_relative_eq!(o.refresh_hz, 60.025, epsilon = 1e-9);
    }

    #[test]
    fn test_integer_scale_deserializes_as_f64() {
        let outputs = parse_monitors(REAL_FIXTURE).unwrap();
        assert_eq!(outputs[0].scale, 2.0);
    }

    #[test]
    fn test_fractional_scale_with_rotation_swaps_logical_size() {
        let outputs = parse_monitors(SYNTHETIC_FIXTURE).unwrap();
        let edp1 = outputs.iter().find(|o| o.name == "eDP-1").unwrap();
        assert_eq!(edp1.transform, Transform::Rot90);
        assert_eq!(edp1.logical_size, (1067, 1707));
    }

    #[test]
    fn test_zero_physical_size_maps_to_none_and_screen_model_errors() {
        let outputs = parse_monitors(SYNTHETIC_FIXTURE).unwrap();
        let dp1 = outputs.iter().find(|o| o.name == "DP-1").unwrap();
        assert_eq!(dp1.physical_mm, None);
        assert!(matches!(
            dp1.screen_model(),
            Err(ProbeError::MissingPhysicalSize { ref output }) if output == "DP-1"
        ));
    }

    #[test]
    fn test_disabled_monitor_is_filtered() {
        let outputs = parse_monitors(SYNTHETIC_FIXTURE).unwrap();
        let names: Vec<&str> = outputs.iter().map(|o| o.name.as_str()).collect();
        assert_eq!(names, vec!["eDP-1", "DP-1"]);
    }

    #[test]
    fn test_unknown_request_reply_is_hyprland_error() {
        assert!(matches!(
            parse_monitors("unknown request"),
            Err(ProbeError::Hyprland { ref reply }) if reply == "unknown request"
        ));
    }

    #[test]
    fn test_invalid_transform_is_rejected() {
        let json = REAL_FIXTURE.replace("\"transform\": 0", "\"transform\": 9");
        assert!(matches!(
            parse_monitors(&json),
            Err(ProbeError::InvalidTransform { value: 9, .. })
        ));
    }

    #[test]
    fn test_screen_model_from_real_fixture() {
        let outputs = parse_monitors(REAL_FIXTURE).unwrap();
        let model = outputs[0].screen_model().unwrap();
        assert_eq!(model.output.as_str(), "eDP-1");
        assert_eq!(model.size_mm, Vector2::new(310.0, 170.0));
        assert_eq!(model.size_px, (3840, 2160));
        assert_eq!(model.scale, 2.0);
    }

    #[test]
    fn test_request_over_fake_socket_sends_command_and_reads_reply() {
        let n = SOCKET_COUNTER.fetch_add(1, Ordering::Relaxed);
        let socket_path =
            std::env::temp_dir().join(format!("eye-hypr-{}-{}.sock", std::process::id(), n));
        let _ = std::fs::remove_file(&socket_path);
        let listener = UnixListener::bind(&socket_path).unwrap();
        let handle = std::thread::spawn({
            let socket_path = socket_path.clone();
            move || {
                let (mut stream, _) = listener.accept().unwrap();
                let mut request = Vec::new();
                stream.read_to_end(&mut request).unwrap();
                stream.write_all(REAL_FIXTURE.as_bytes()).unwrap();
                drop(stream);
                let _ = std::fs::remove_file(&socket_path);
                request
            }
        });

        let probe = HyprlandDisplayProbe::with_socket(socket_path.clone(), Duration::from_secs(1));
        let outputs = probe.outputs().unwrap();

        let received = handle.join().unwrap();
        assert_eq!(received, b"j/monitors");
        assert_eq!(outputs[0].name, "eDP-1");
    }

    #[test]
    fn test_missing_socket_is_io_error_with_path() {
        let probe = HyprlandDisplayProbe::with_socket(
            PathBuf::from("/nonexistent/.socket.sock"),
            Duration::from_secs(1),
        );
        assert!(matches!(
            probe.outputs(),
            Err(ProbeError::Io { ref path, .. }) if path == Path::new("/nonexistent/.socket.sock")
        ));
    }

    struct StubProbe;

    impl DisplayProbe for StubProbe {
        fn outputs(&self) -> Result<Vec<OutputInfo>, ProbeError> {
            parse_monitors(REAL_FIXTURE)
        }
    }

    #[test]
    fn test_output_lookup_by_name_lists_available() {
        let probe = StubProbe;
        assert!(matches!(
            probe.output("HDMI-A-9"),
            Err(ProbeError::OutputNotFound { ref available, .. })
                if available == &["eDP-1".to_string()]
        ));
    }

    #[test]
    #[ignore = "needs wayland"]
    fn test_live_hyprland_reports_edp1() {
        let session = crate::session::SessionInfo::from_env();
        let probe = HyprlandDisplayProbe::new(&session).unwrap();
        let edp1 = probe.output("eDP-1").unwrap();
        assert_eq!(edp1.mode_px, (3840, 2160));
        assert_eq!(edp1.physical_mm, Some((310, 170)));
        assert_eq!(edp1.logical_size, (1920, 1080));
    }
}
