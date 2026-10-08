use std::fmt;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;

use anyhow::Context;
use eye_platform::{
    CameraDevice, CameraKind, CameraProbe, DisplayProbe, DmiInfo, EmitterControl, IrEmitter,
    MsxuIrEmitter, OutputInfo, ProbeError, SelectedDisplayProbe, SessionInfo, SessionType,
    V4l2CameraProbe, builtin_profiles, find_face_auth_control, match_profile, select_display_probe,
};
use serde::Serialize;

use crate::ctx::Ctx;

#[derive(Debug, clap::Args)]
pub struct Args {
    /// Machine-readable output
    #[arg(long)]
    pub json: bool,
}

pub type OpenEmitter =
    Box<dyn Fn(&CameraDevice) -> Result<Box<dyn IrEmitter + Send>, eye_platform::EmitterError>>;

pub struct Probes {
    pub session: SessionInfo,
    pub display: Result<(&'static str, Box<dyn DisplayProbe>), ProbeError>,
    pub cameras: Box<dyn CameraProbe>,
    pub open_emitter: OpenEmitter,
    pub dmi: Result<DmiInfo, ProbeError>,
}

impl fmt::Debug for Probes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Probes")
            .field("session", &self.session)
            .finish_non_exhaustive()
    }
}

impl Probes {
    pub fn system() -> Probes {
        let session = SessionInfo::from_env();
        let display = select_display_probe(&session).map(|selected| {
            let name: &'static str = match &selected {
                SelectedDisplayProbe::Hyprland(_) => "hyprland",
                SelectedDisplayProbe::Wayland(_) => "wayland",
            };
            (name, Box::new(selected) as Box<dyn DisplayProbe>)
        });
        Probes {
            session,
            display,
            cameras: Box::new(V4l2CameraProbe::new()),
            open_emitter: Box::new(|camera: &CameraDevice| {
                MsxuIrEmitter::discover(camera).map(|e| Box::new(e) as Box<dyn IrEmitter + Send>)
            }),
            dmi: DmiInfo::read(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProbeReport {
    pub session: SessionInfo,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_backend: Option<&'static str>,
    pub outputs: Vec<OutputInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_error: Option<String>,
    pub cameras: Vec<CameraDevice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub camera_error: Option<String>,
    pub emitters: Vec<EmitterStatus>,
    pub hardware_profile: Option<HardwareProfileReport>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HardwareProfileReport {
    pub id: String,
    pub verified: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EmitterStatus {
    pub node: PathBuf,
    pub control: EmitterControl,
    pub state: String,
}

pub fn collect(probes: &Probes) -> ProbeReport {
    let (display_backend, outputs, display_error) = match &probes.display {
        Ok((name, probe)) => match probe.outputs() {
            Ok(outputs) => (Some(*name), outputs, None),
            Err(e) => (Some(*name), Vec::new(), Some(e.to_string())),
        },
        Err(e) => (None, Vec::new(), Some(e.to_string())),
    };
    let (cameras, camera_error) = match probes.cameras.cameras() {
        Ok(c) => (c, None),
        Err(e) => (Vec::new(), Some(e.to_string())),
    };
    let emitters = cameras
        .iter()
        .filter_map(|cam| {
            let control = find_face_auth_control(&cam.extension_units)?;
            let state = match (probes.open_emitter)(cam).and_then(|e| e.is_enabled()) {
                Ok(true) => "on".to_owned(),
                Ok(false) => "off".to_owned(),
                Err(e) => format!("error: {e}"),
            };
            Some(EmitterStatus {
                node: cam.node.clone(),
                control,
                state,
            })
        })
        .collect();
    let hardware_profile = probes
        .dmi
        .as_ref()
        .ok()
        .and_then(|dmi| match_profile(builtin_profiles(), dmi))
        .map(|p| HardwareProfileReport {
            id: p.id.clone(),
            verified: p.verified,
        });

    ProbeReport {
        session: probes.session.clone(),
        display_backend,
        outputs,
        display_error,
        cameras,
        camera_error,
        emitters,
        hardware_profile,
    }
}

fn fmt_num(n: f64) -> String {
    if n.fract() == 0.0 {
        format!("{}", n as i64)
    } else {
        n.to_string()
    }
}

fn session_type_label(t: &SessionType) -> String {
    match t {
        SessionType::Wayland => "wayland".to_string(),
        SessionType::X11 => "x11".to_string(),
        SessionType::Tty => "tty".to_string(),
        SessionType::Unset => "unset".to_string(),
        SessionType::Other(s) => s.clone(),
    }
}

pub fn write_human(report: &ProbeReport, out: &mut impl Write) -> io::Result<()> {
    let mut session_parts = vec![session_type_label(&report.session.session_type)];
    if let Some(w) = &report.session.wayland_display {
        session_parts.push(format!("WAYLAND_DISPLAY={w}"));
    }
    if let Some(b) = report.display_backend {
        session_parts.push(b.to_string());
    }
    writeln!(out, "{:<7}  {}", "session", session_parts.join("  "))?;

    if let Some(err) = &report.display_error {
        writeln!(out, "{:<7}  error: {err}", "display")?;
    }

    for o in &report.outputs {
        let mut parts = vec![
            format!("{}x{}", o.mode_px.0, o.mode_px.1),
            format!("scale {}", fmt_num(o.scale)),
            format!("{}x{} logical", o.logical_size.0, o.logical_size.1),
        ];
        if let Some((w, h)) = o.physical_mm {
            parts.push(format!("{w}x{h} mm"));
        }
        parts.push(format!(
            "at {},{}",
            o.logical_position.0, o.logical_position.1
        ));
        parts.push(format!(
            "via {}",
            report.display_backend.unwrap_or("unknown")
        ));
        writeln!(out, "{:<7}  {}  {}", "output", o.name, parts.join("  "))?;
    }

    if let Some(err) = &report.camera_error {
        writeln!(out, "{:<7}  error: {err}", "camera")?;
    }

    for c in &report.cameras {
        let card = c.card.split(':').next().unwrap_or(&c.card);
        let kind = match c.kind {
            CameraKind::Rgb => "rgb",
            CameraKind::Ir => "ir",
            CameraKind::Other => "other",
        };
        let usb = c
            .usb
            .as_ref()
            .map(|u| format!("{:04x}:{:04x}", u.vendor_id, u.product_id))
            .unwrap_or_else(|| "-".to_string());
        let formats = c
            .formats
            .iter()
            .map(|f| {
                let sizes = f
                    .sizes
                    .iter()
                    .map(|s| {
                        format!(
                            "{}x{}@{}",
                            s.width,
                            s.height,
                            fmt_num(s.fps.first().copied().unwrap_or(0.0))
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                format!("{} {sizes}", f.fourcc)
            })
            .collect::<Vec<_>>()
            .join(", ");
        let mut parts = vec![card.to_string(), kind.to_string(), usb, formats];
        if let Some(ctl) = find_face_auth_control(&c.extension_units) {
            parts.push(format!("msxu unit {} selector {}", ctl.unit, ctl.selector));
        }
        writeln!(
            out,
            "{:<7}  {}  {}",
            "camera",
            c.node.display(),
            parts.join("  ")
        )?;
    }

    for e in &report.emitters {
        writeln!(out, "{:<7}  {}  {}", "emitter", e.node.display(), e.state)?;
    }

    match &report.hardware_profile {
        Some(p) => writeln!(
            out,
            "hardware profile: {} ({})",
            p.id,
            if p.verified { "verified" } else { "unverified" }
        )?,
        None => writeln!(out, "hardware profile: none (generic prior)")?,
    }

    Ok(())
}

pub fn run(ctx: &Ctx, args: Args) -> anyhow::Result<()> {
    let report = collect(&Probes::system());
    let mut out: Box<dyn Write> = match &ctx.output {
        Some(path) => Box::new(BufWriter::new(
            File::create(path).with_context(|| format!("creating {}", path.display()))?,
        )),
        None => Box::new(io::stdout().lock()),
    };
    if args.json {
        serde_json::to_writer_pretty(&mut out, &report)?;
        writeln!(out)?;
    } else {
        write_human(&report, &mut out)?;
    }
    out.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use eye_platform::EmitterError;

    use super::*;
    use crate::testing::{FakeCameras, FakeDisplay, FakeEmitter, edp1, latitude_cameras};

    fn no_dmi() -> Result<DmiInfo, ProbeError> {
        Err(ProbeError::Io {
            path: PathBuf::from("/sys/class/dmi/id"),
            source: io::Error::other("fake: no DMI"),
        })
    }

    fn fake_probes(
        display: Result<(&'static str, Box<dyn DisplayProbe>), ProbeError>,
        cameras: Option<Vec<CameraDevice>>,
        open_emitter: OpenEmitter,
    ) -> Probes {
        Probes {
            session: SessionInfo::from_lookup(|_| None),
            display,
            cameras: Box::new(FakeCameras(cameras)),
            open_emitter,
            dmi: no_dmi(),
        }
    }

    #[test]
    fn test_collect_reports_all_sections() {
        let probes = fake_probes(
            Ok((
                "hyprland",
                Box::new(FakeDisplay(Some(vec![edp1()]))) as Box<dyn DisplayProbe>,
            )),
            Some(latitude_cameras()),
            Box::new(|_: &CameraDevice| {
                Ok(Box::new(FakeEmitter::new(false)) as Box<dyn IrEmitter + Send>)
            }),
        );

        let report = collect(&probes);

        assert_eq!(report.outputs.len(), 1);
        assert_eq!(report.cameras.len(), 2);
        assert_eq!(
            report.emitters,
            vec![EmitterStatus {
                node: PathBuf::from("/dev/video2"),
                control: EmitterControl {
                    unit: 4,
                    selector: 6
                },
                state: "off".to_string(),
            }]
        );
    }

    #[test]
    fn test_collect_display_error_is_reported_not_fatal() {
        let probes = fake_probes(
            Err(ProbeError::NotHyprland),
            Some(latitude_cameras()),
            Box::new(|_: &CameraDevice| {
                Ok(Box::new(FakeEmitter::new(false)) as Box<dyn IrEmitter + Send>)
            }),
        );

        let report = collect(&probes);

        assert!(report.display_error.is_some());
        assert_eq!(report.display_backend, None);
        assert_eq!(report.cameras.len(), 2);
        assert_eq!(report.emitters.len(), 1);
    }

    #[test]
    fn test_collect_emitter_error_is_per_device() {
        let probes = fake_probes(
            Ok((
                "hyprland",
                Box::new(FakeDisplay(Some(vec![edp1()]))) as Box<dyn DisplayProbe>,
            )),
            Some(latitude_cameras()),
            Box::new(|cam: &CameraDevice| {
                Err(EmitterError::NoControl {
                    node: cam.node.clone(),
                })
            }),
        );

        let report = collect(&probes);

        assert_eq!(report.display_error, None);
        assert_eq!(report.camera_error, None);
        assert_eq!(report.outputs.len(), 1);
        assert_eq!(report.cameras.len(), 2);
        assert_eq!(report.emitters.len(), 1);
        assert!(report.emitters[0].state.starts_with("error: "));
    }

    fn golden_report() -> ProbeReport {
        ProbeReport {
            session: SessionInfo::from_lookup(|k| match k {
                "XDG_SESSION_TYPE" => Some("wayland".into()),
                "WAYLAND_DISPLAY" => Some("wayland-1".into()),
                _ => None,
            }),
            display_backend: Some("hyprland"),
            outputs: vec![edp1()],
            display_error: None,
            cameras: latitude_cameras(),
            camera_error: None,
            emitters: vec![EmitterStatus {
                node: PathBuf::from("/dev/video2"),
                control: EmitterControl {
                    unit: 4,
                    selector: 6,
                },
                state: "off".to_string(),
            }],
            hardware_profile: Some(HardwareProfileReport {
                id: "dell-latitude-7420".to_string(),
                verified: true,
            }),
        }
    }

    #[test]
    fn test_human_output_golden() {
        let report = golden_report();
        let mut out = Vec::new();
        write_human(&report, &mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert_eq!(
            text,
            "session  wayland  WAYLAND_DISPLAY=wayland-1  hyprland\n\
             output   eDP-1  3840x2160  scale 2  1920x1080 logical  310x170 mm  at 0,0  via hyprland\n\
             camera   /dev/video0  Integrated_Webcam_HD  rgb  0c45:672c  MJPG 1280x720@30 960x540@30 848x480@30 640x480@30 640x360@30, YUYV 640x480@30\n\
             camera   /dev/video2  Integrated_Webcam_HD  ir  0c45:672c  GREY 640x360@30  msxu unit 4 selector 6\n\
             emitter  /dev/video2  off\n\
             hardware profile: dell-latitude-7420 (verified)\n"
        );
    }

    #[test]
    fn test_human_output_reports_no_profile() {
        let mut report = golden_report();
        report.hardware_profile = None;
        let mut out = Vec::new();
        write_human(&report, &mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.ends_with("hardware profile: none (generic prior)\n"));
    }

    #[test]
    fn test_json_report_shape() {
        let report = golden_report();
        let value = serde_json::to_value(&report).unwrap();
        assert_eq!(
            value["outputs"][0]["mode_px"],
            serde_json::json!([3840, 2160])
        );
        assert_eq!(value["outputs"][0]["scale"], serde_json::json!(2.0));
        assert_eq!(value["cameras"][1]["kind"], serde_json::json!("ir"));
        assert!(value.get("display_error").is_none());
        assert!(value.get("camera_error").is_none());
        assert_eq!(
            value["hardware_profile"],
            serde_json::json!({"id": "dell-latitude-7420", "verified": true})
        );

        let mut no_profile = golden_report();
        no_profile.hardware_profile = None;
        let value = serde_json::to_value(&no_profile).unwrap();
        assert_eq!(value["hardware_profile"], serde_json::Value::Null);
    }

    #[test]
    fn test_report_converts_to_toml_table() {
        let report = golden_report();
        toml::Table::try_from(&report).unwrap();
    }
}
