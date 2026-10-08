pub mod opener;

use std::{
    collections::BTreeMap,
    fs, io,
    path::{Path, PathBuf},
    sync::Arc,
};

use eye::config::EnvOverrides;
use eye_capture::{CaptureError, FrameSource, IlluminationMeta};
use eye_platform::{
    CameraDevice, CameraKind, CameraProbe, V4l2CameraProbe,
    emitter::{MODE_OFF, MODE_ON_DEFAULT},
    find_face_auth_control,
};

use crate::{
    mode::{
        ActiveMode, EmitterSetting, LabEmitter, Mode, ModeError, ModeHost, ModeSession, Role,
        StreamTarget, Teardown,
    },
    modes::opener::Opener,
    sequence::{EmitterSel, FormatSel, ModeSpec, StreamFormat, StreamsSel},
};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CameraSelection {
    pub rgb: Option<PathBuf>,
    pub ir: Option<PathBuf>,
}

impl CameraSelection {
    /// A flag wins over `EYE_CAMERA` / `EYE_IR_CAMERA`.
    pub fn resolve(
        rgb_flag: Option<PathBuf>,
        ir_flag: Option<PathBuf>,
        env: &EnvOverrides,
    ) -> Self {
        Self {
            rgb: rgb_flag.or_else(|| env.camera.clone().map(PathBuf::from)),
            ir: ir_flag.or_else(|| env.ir_camera.clone().map(PathBuf::from)),
        }
    }
}

/// Explicit node: that device (any kind) or `NoSuchCamera`. Otherwise the first device of `kind`.
pub fn select(
    cameras: &[CameraDevice],
    explicit: Option<&Path>,
    kind: CameraKind,
) -> Result<Option<CameraDevice>, ModeError> {
    match explicit {
        Some(node) => cameras
            .iter()
            .find(|c| c.node == node)
            .cloned()
            .map(Some)
            .ok_or_else(|| ModeError::NoSuchCamera(node.to_path_buf())),
        None => Ok(cameras.iter().find(|c| c.kind == kind).cloned()),
    }
}

/// Every capturable (fourcc, size, fps), in probe order; fps rounded to u32, consecutive duplicates removed.
pub fn capturable_formats(camera: &CameraDevice) -> Vec<StreamFormat> {
    let mut out: Vec<StreamFormat> = camera
        .formats
        .iter()
        .filter(|f| opener::pixel_format(&f.fourcc).is_some())
        .flat_map(|f| {
            f.sizes.iter().flat_map(move |s| {
                s.fps.iter().map(move |fps| StreamFormat {
                    fourcc: f.fourcc.clone(),
                    width: s.width,
                    height: s.height,
                    fps: fps.round() as u32,
                })
            })
        })
        .collect();
    out.dedup();
    out
}

/// Largest area, then highest fps; the last one wins a full tie.
pub fn default_format(formats: &[StreamFormat]) -> Option<&StreamFormat> {
    formats
        .iter()
        .max_by_key(|f| (u64::from(f.width) * u64::from(f.height), f.fps))
}

#[derive(Debug)]
pub struct LiveHost {
    rgb: Option<CameraDevice>,
    ir: Option<CameraDevice>,
    opener: Arc<dyn Opener>,
    probe_error: Option<String>,
}

impl LiveHost {
    /// `select` for both roles (an explicit node that is not in `cameras` is `NoSuchCamera`).
    pub fn new(
        cameras: Vec<CameraDevice>,
        selection: &CameraSelection,
        opener: Arc<dyn Opener>,
    ) -> Result<Self, ModeError> {
        let rgb = select(&cameras, selection.rgb.as_deref(), CameraKind::Rgb)?;
        let ir = select(&cameras, selection.ir.as_deref(), CameraKind::Ir)?;
        Ok(Self {
            rgb,
            ir,
            opener,
            probe_error: None,
        })
    }

    /// `V4l2CameraProbe::new().cameras()` with `V4l2Opener`; a probe error is kept for the report (`probe_error`)
    /// and leaves the host with no cameras, so camera-free suites still run.
    pub fn probe(selection: &CameraSelection) -> Result<Self, ModeError> {
        match V4l2CameraProbe::new().cameras() {
            Ok(cameras) => Self::new(cameras, selection, Arc::new(opener::V4l2Opener)),
            Err(e) => {
                let mut host = Self::new(Vec::new(), selection, Arc::new(opener::V4l2Opener))?;
                host.probe_error = Some(e.to_string());
                Ok(host)
            }
        }
    }

    pub fn camera(&self, role: Role) -> Option<&CameraDevice> {
        match role {
            Role::Rgb => self.rgb.as_ref(),
            Role::Ir => self.ir.as_ref(),
        }
    }

    pub fn matrix(&self) -> Vec<MatrixRow> {
        let mut rows = Vec::new();
        for streams in [StreamsSel::Rgb, StreamsSel::Ir, StreamsSel::Dual] {
            let spec = ModeSpec {
                streams,
                emitter: EmitterSel::Each,
                rgb: FormatSel::All,
                ir: FormatSel::All,
            };
            let streams_label = streams_label(streams);
            match self.expand(&spec) {
                Ok(modes) => {
                    for mode in modes {
                        rows.push(MatrixRow {
                            mode: mode.to_string(),
                            streams: streams_label,
                            emitter: emitter_label(mode.emitter),
                            rgb: mode.rgb.as_ref().map(|t| t.format.to_string()),
                            ir: mode.ir.as_ref().map(|t| t.format.to_string()),
                            runnable: true,
                            note: None,
                        });
                    }
                }
                Err(e) => rows.push(MatrixRow {
                    mode: "-".to_owned(),
                    streams: streams_label,
                    emitter: "keep",
                    rgb: None,
                    ir: None,
                    runnable: false,
                    note: Some(e.to_string()),
                }),
            }
        }

        for (role, camera) in [(Role::Rgb, &self.rgb), (Role::Ir, &self.ir)] {
            let Some(camera) = camera else { continue };
            let capturable = capturable_formats(camera);
            for format in &camera.formats {
                for size in &format.sizes {
                    for fps in &size.fps {
                        let candidate = StreamFormat {
                            fourcc: format.fourcc.clone(),
                            width: size.width,
                            height: size.height,
                            fps: fps.round() as u32,
                        };
                        if capturable.contains(&candidate) {
                            continue;
                        }
                        let role_label = match role {
                            Role::Rgb => "rgb",
                            Role::Ir => "ir",
                        };
                        rows.push(MatrixRow {
                            mode: format!("{role_label} {candidate}"),
                            streams: role_label,
                            emitter: "keep",
                            rgb: (role == Role::Rgb).then(|| candidate.to_string()),
                            ir: (role == Role::Ir).then(|| candidate.to_string()),
                            runnable: false,
                            note: Some(format!("V4l2Source cannot capture {}", candidate.fourcc)),
                        });
                    }
                }
            }
        }

        rows
    }

    /// `emitter: {node} byte 2 = 0x0N` | `emitter: none` (no IR camera, or no MSXU control) | `emitter: {node} error: {e}`.
    pub fn emitter_status(&self) -> String {
        let Some(ir) = &self.ir else {
            return "emitter: none".to_owned();
        };
        if find_face_auth_control(&ir.extension_units).is_none() {
            return "emitter: none".to_owned();
        }
        match self
            .opener
            .open_emitter(ir)
            .map_err(ModeError::from)
            .and_then(|e| e.read_mode().map_err(ModeError::from))
        {
            Ok(b) => format!("emitter: {} byte 2 = {b:#04x}", ir.node.display()),
            Err(e) => format!("emitter: {} error: {e}", ir.node.display()),
        }
    }

    /// `metadata: {node}` | `metadata: none`.
    pub fn metadata_status(&self) -> String {
        match self.ir.as_ref().and_then(|c| c.metadata_node.as_ref()) {
            Some(node) => format!("metadata: {}", node.display()),
            None => "metadata: none".to_owned(),
        }
    }

    fn targets(&self, role: Role, sel: &FormatSel) -> Result<Vec<StreamTarget>, ModeError> {
        let camera = self.camera(role).ok_or(ModeError::NoCamera(role))?;
        let formats = capturable_formats(camera);
        let selected: Vec<StreamFormat> = match sel {
            FormatSel::Default => vec![
                default_format(&formats)
                    .cloned()
                    .ok_or_else(|| ModeError::NoCapturableFormat(camera.node.clone()))?,
            ],
            FormatSel::All => {
                if formats.is_empty() {
                    return Err(ModeError::NoCapturableFormat(camera.node.clone()));
                }
                formats.clone()
            }
            FormatSel::Exact(f) => {
                if formats.contains(f) {
                    vec![f.clone()]
                } else if opener::pixel_format(&f.fourcc).is_none() {
                    return Err(ModeError::NotCapturable {
                        node: camera.node.clone(),
                        format: f.clone(),
                    });
                } else {
                    return Err(ModeError::Unsupported {
                        node: camera.node.clone(),
                        format: f.clone(),
                    });
                }
            }
        };
        Ok(selected
            .into_iter()
            .map(|format| StreamTarget {
                node: camera.node.clone(),
                format,
            })
            .collect())
    }
}

fn streams_label(streams: StreamsSel) -> &'static str {
    match streams {
        StreamsSel::Rgb => "rgb",
        StreamsSel::Ir => "ir",
        StreamsSel::Dual => "dual",
        StreamsSel::None => "none",
        StreamsSel::All => "*",
    }
}

fn emitter_label(setting: EmitterSetting) -> &'static str {
    match setting {
        EmitterSetting::Keep => "keep",
        EmitterSetting::On => "on",
        EmitterSetting::Off => "off",
    }
}

impl ModeHost for LiveHost {
    fn expand(&self, spec: &ModeSpec) -> Result<Vec<Mode>, ModeError> {
        let streams: &[(bool, bool)] = match spec.streams {
            StreamsSel::None => &[(false, false)],
            StreamsSel::Rgb => &[(true, false)],
            StreamsSel::Ir => &[(false, true)],
            StreamsSel::Dual => &[(true, true)],
            StreamsSel::All => &[(true, false), (false, true), (true, true)],
        };
        let mut modes = Vec::new();
        for &(has_rgb, has_ir) in streams {
            let rgb: Vec<Option<StreamTarget>> = if has_rgb {
                self.targets(Role::Rgb, &spec.rgb)?
                    .into_iter()
                    .map(Some)
                    .collect()
            } else {
                vec![None]
            };
            let ir: Vec<Option<StreamTarget>> = if has_ir {
                self.targets(Role::Ir, &spec.ir)?
                    .into_iter()
                    .map(Some)
                    .collect()
            } else {
                vec![None]
            };
            let emitters: &[EmitterSetting] = match (spec.emitter, has_ir) {
                (EmitterSel::Each, true) => &[EmitterSetting::Off, EmitterSetting::On],
                (EmitterSel::Each, false) | (EmitterSel::Keep, _) => &[EmitterSetting::Keep],
                (EmitterSel::On, _) => &[EmitterSetting::On],
                (EmitterSel::Off, _) => &[EmitterSetting::Off],
            };
            for r in &rgb {
                for i in &ir {
                    for &emitter in emitters {
                        modes.push(Mode {
                            emitter,
                            rgb: r.clone(),
                            ir: i.clone(),
                        });
                    }
                }
            }
        }
        Ok(modes)
    }

    fn enter(&mut self, mode: &Mode) -> Result<ActiveMode, ModeError> {
        if mode.ir.is_some()
            && let Some(node) = self.ir.as_ref().and_then(|c| c.metadata_node.as_deref())
        {
            match self.opener.prepare_meta(node) {
                Ok(fourcc) if fourcc == eye_platform::META_FORMAT_UVCM => {}
                Ok(fourcc) => tracing::warn!(
                    node = %node.display(),
                    fourcc = %String::from_utf8_lossy(&fourcc),
                    "metadata node is not UVCM; IR frames are tagged by brightness"
                ),
                Err(err) => tracing::warn!(
                    node = %node.display(),
                    %err,
                    "could not select UVCM; IR frames are tagged by brightness"
                ),
            }
        }
        let teardown: Option<Teardown> = match mode.emitter {
            EmitterSetting::Keep => None,
            setting => {
                let ir = self.ir.as_ref().ok_or(ModeError::NoCamera(Role::Ir))?;
                let target = if setting == EmitterSetting::On {
                    MODE_ON_DEFAULT
                } else {
                    MODE_OFF
                };
                Some(self.opener.guard_emitter(ir, target)?)
            }
        };
        let session = LiveSession {
            mode: mode.clone(),
            rgb: self.rgb.clone(),
            ir: self.ir.clone(),
            opener: Arc::clone(&self.opener),
        };
        Ok(ActiveMode {
            session: Arc::new(session),
            teardown,
        })
    }

    fn environment(&self) -> BTreeMap<String, String> {
        let mut env = BTreeMap::new();
        if let Some(v) = read_sysfs_param("clock") {
            env.insert("uvcvideo.clock".to_owned(), v);
        }
        if let Some(v) = read_sysfs_param("hwtimestamps") {
            env.insert("uvcvideo.hwtimestamps".to_owned(), v);
        }
        if let Some(rgb) = &self.rgb {
            env.insert("camera.rgb".to_owned(), camera_env_value(rgb));
        }
        if let Some(ir) = &self.ir {
            env.insert("camera.ir".to_owned(), camera_env_value(ir));
            if let Some(node) = &ir.metadata_node {
                env.insert("camera.ir.metadata".to_owned(), node.display().to_string());
            }
        }
        if let Some(err) = &self.probe_error {
            env.insert("camera.probe_error".to_owned(), err.clone());
        }
        env
    }
}

fn read_sysfs_param(name: &str) -> Option<String> {
    let path = PathBuf::from("/sys/module/uvcvideo/parameters").join(name);
    fs::read_to_string(path).ok().map(|s| s.trim().to_owned())
}

fn camera_env_value(camera: &CameraDevice) -> String {
    let mut value = match &camera.usb {
        Some(usb) => {
            let bcd = fs::read_to_string(usb.sysfs_device.join("bcdDevice"))
                .map(|s| s.trim().to_owned())
                .unwrap_or_else(|_| "?".to_owned());
            format!(
                "{} {:04x}:{:04x} bcdDevice {bcd}",
                camera.node.display(),
                usb.vendor_id,
                usb.product_id
            )
        }
        None => camera.node.display().to_string(),
    };
    if let Some(control) = find_face_auth_control(&camera.extension_units) {
        value.push_str(&format!(" msxu unit {}", control.unit));
    }
    value
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct MatrixRow {
    pub mode: String,
    pub streams: &'static str,
    pub emitter: &'static str,
    pub rgb: Option<String>,
    pub ir: Option<String>,
    pub runnable: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

#[derive(Debug)]
struct LiveSession {
    mode: Mode,
    rgb: Option<CameraDevice>,
    ir: Option<CameraDevice>,
    opener: Arc<dyn Opener>,
}

impl ModeSession for LiveSession {
    fn open(&self, role: Role) -> Result<Box<dyn FrameSource>, CaptureError> {
        let target = self.mode.target(role).ok_or_else(|| CaptureError::Open {
            path: PathBuf::from("<none>"),
            source: io::Error::new(
                io::ErrorKind::NotFound,
                format!("mode has no {role:?} stream"),
            ),
        })?;
        self.opener.open_source(role, target)
    }

    fn open_meta(&self) -> Result<Option<Box<dyn IlluminationMeta>>, CaptureError> {
        if self.mode.ir.is_none() {
            return Ok(None);
        }
        let Some(node) = self.ir.as_ref().and_then(|c| c.metadata_node.as_ref()) else {
            return Ok(None);
        };
        Ok(Some(self.opener.open_meta(node)?))
    }

    fn emitter(&self) -> Result<Box<dyn LabEmitter>, ModeError> {
        let ir = self.ir.as_ref().ok_or(ModeError::NoCamera(Role::Ir))?;
        Ok(self.opener.open_emitter(ir)?)
    }

    fn device(&self, role: Role) -> Option<CameraDevice> {
        match role {
            Role::Rgb => self.rgb.clone(),
            Role::Ir => self.ir.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{FakeOpener, SharedFakeXu, dell_cameras, fake_live_host};

    #[test]
    fn test_capturable_formats_drop_yuyv() {
        let cams = dell_cameras();
        let rgb = cams.iter().find(|c| c.kind == CameraKind::Rgb).unwrap();
        let formats: Vec<String> = capturable_formats(rgb)
            .iter()
            .map(StreamFormat::to_string)
            .collect();
        assert_eq!(
            formats,
            [
                "MJPG 1280x720@30",
                "MJPG 960x540@30",
                "MJPG 848x480@30",
                "MJPG 640x480@30",
                "MJPG 640x360@30",
            ]
        );
    }

    #[test]
    fn test_default_format_is_largest_then_fastest() {
        let cams = dell_cameras();
        let rgb = cams.iter().find(|c| c.kind == CameraKind::Rgb).unwrap();
        let ir = cams.iter().find(|c| c.kind == CameraKind::Ir).unwrap();
        assert_eq!(
            default_format(&capturable_formats(rgb))
                .unwrap()
                .to_string(),
            "MJPG 1280x720@30"
        );
        assert_eq!(
            default_format(&capturable_formats(ir)).unwrap().to_string(),
            "GREY 640x360@30"
        );

        let synthetic = [
            "MJPG 640x480@60".parse().unwrap(),
            "MJPG 1280x720@15".parse().unwrap(),
            "MJPG 1280x720@30".parse().unwrap(),
        ];
        assert_eq!(
            default_format(&synthetic).unwrap().to_string(),
            "MJPG 1280x720@30"
        );
    }

    #[test]
    fn test_expand_default_dual_on() {
        let opener = Arc::new(FakeOpener::new(SharedFakeXu::with_mode(1)));
        let host = fake_live_host(Arc::clone(&opener));
        let spec = ModeSpec {
            streams: StreamsSel::Dual,
            emitter: EmitterSel::On,
            ..ModeSpec::default()
        };
        let modes = host.expand(&spec).unwrap();
        assert_eq!(modes.len(), 1);
        assert_eq!(
            modes[0].to_string(),
            "dual MJPG 1280x720@30 + GREY 640x360@30, emitter on"
        );
        assert_eq!(
            modes[0].rgb.as_ref().unwrap().node,
            PathBuf::from("/dev/video0")
        );
        assert_eq!(
            modes[0].ir.as_ref().unwrap().node,
            PathBuf::from("/dev/video2")
        );
    }

    #[test]
    fn test_expand_full_matrix_counts() {
        let opener = Arc::new(FakeOpener::new(SharedFakeXu::with_mode(1)));
        let host = fake_live_host(opener);
        let spec = ModeSpec {
            streams: StreamsSel::All,
            emitter: EmitterSel::Each,
            rgb: FormatSel::All,
            ir: FormatSel::All,
        };
        let modes = host.expand(&spec).unwrap();
        assert_eq!(modes.len(), 17);
        let rgb_only = modes
            .iter()
            .filter(|m| m.rgb.is_some() && m.ir.is_none())
            .count();
        let ir_only = modes
            .iter()
            .filter(|m| m.rgb.is_none() && m.ir.is_some())
            .count();
        let dual = modes
            .iter()
            .filter(|m| m.rgb.is_some() && m.ir.is_some())
            .count();
        assert_eq!(rgb_only, 5);
        assert_eq!(ir_only, 2);
        assert_eq!(dual, 10);
        assert!(
            modes
                .iter()
                .filter(|m| m.rgb.is_some() && m.ir.is_none())
                .all(|m| m.emitter == EmitterSetting::Keep)
        );
        let ir_only_emitters: Vec<_> = modes
            .iter()
            .filter(|m| m.rgb.is_none() && m.ir.is_some())
            .map(|m| m.emitter)
            .collect();
        assert_eq!(ir_only_emitters, [EmitterSetting::Off, EmitterSetting::On]);
    }

    #[test]
    fn test_expand_exact_format_errors() {
        let opener = Arc::new(FakeOpener::new(SharedFakeXu::with_mode(1)));
        let host = fake_live_host(opener);
        let spec = ModeSpec {
            streams: StreamsSel::Rgb,
            rgb: FormatSel::Exact("MJPG 1920x1080@30".parse().unwrap()),
            ..ModeSpec::default()
        };
        assert!(matches!(
            host.expand(&spec),
            Err(ModeError::Unsupported { .. })
        ));

        let spec = ModeSpec {
            streams: StreamsSel::Rgb,
            rgb: FormatSel::Exact("YUYV 640x480@30".parse().unwrap()),
            ..ModeSpec::default()
        };
        assert!(matches!(
            host.expand(&spec),
            Err(ModeError::NotCapturable { .. })
        ));
    }

    #[test]
    fn test_expand_without_ir_camera_is_no_camera() {
        let opener = Arc::new(FakeOpener::new(SharedFakeXu::with_mode(1)));
        let cams = vec![
            dell_cameras()
                .into_iter()
                .find(|c| c.kind == CameraKind::Rgb)
                .unwrap(),
        ];
        let host = LiveHost::new(cams, &CameraSelection::default(), opener).unwrap();

        let ir_spec = ModeSpec {
            streams: StreamsSel::Ir,
            ..ModeSpec::default()
        };
        assert!(matches!(
            host.expand(&ir_spec),
            Err(ModeError::NoCamera(Role::Ir))
        ));

        let rgb_spec = ModeSpec {
            streams: StreamsSel::Rgb,
            ..ModeSpec::default()
        };
        assert!(host.expand(&rgb_spec).is_ok());
    }

    #[test]
    fn test_matrix_has_17_runnable_rows_and_yuyv_last() {
        let opener = Arc::new(FakeOpener::new(SharedFakeXu::with_mode(1)));
        let host = fake_live_host(opener);
        let rows = host.matrix();
        assert_eq!(rows.len(), 18);
        assert_eq!(rows.iter().filter(|r| r.runnable).count(), 17);
        let last = rows.last().unwrap();
        assert_eq!(last.mode, "rgb YUYV 640x480@30");
        assert!(!last.runnable);
        assert!(last.note.as_deref().unwrap().contains("YUYV"));
        assert_eq!(last.rgb.as_deref(), Some("YUYV 640x480@30"));
        assert!(last.ir.is_none());
    }

    #[test]
    fn test_select_explicit_node_and_missing_node() {
        let cams = dell_cameras();
        let video2 = cams
            .iter()
            .find(|c| c.kind == CameraKind::Ir)
            .unwrap()
            .clone();
        let found = select(&cams, Some(Path::new("/dev/video2")), CameraKind::Rgb)
            .unwrap()
            .unwrap();
        assert_eq!(found.node, video2.node);

        let err = select(&cams, Some(Path::new("/dev/video9")), CameraKind::Rgb).unwrap_err();
        assert!(matches!(err, ModeError::NoSuchCamera(p) if p == Path::new("/dev/video9")));

        let found = select(&cams, None, CameraKind::Ir).unwrap().unwrap();
        assert_eq!(found.node, video2.node);
    }

    #[test]
    fn test_camera_selection_flag_beats_env() {
        let env = EnvOverrides {
            camera: Some("/dev/video0".to_owned()),
            ir_camera: None,
            output: None,
        };
        let sel = CameraSelection::resolve(Some(PathBuf::from("/dev/video5")), None, &env);
        assert_eq!(sel.rgb, Some(PathBuf::from("/dev/video5")));

        let sel = CameraSelection::resolve(None, None, &env);
        assert_eq!(sel.rgb, Some(PathBuf::from("/dev/video0")));
    }

    #[test]
    fn test_enter_keep_writes_nothing() {
        let opener = Arc::new(FakeOpener::new(SharedFakeXu::with_mode(3)));
        let mut host = fake_live_host(Arc::clone(&opener));
        let mode = &host
            .expand(&ModeSpec {
                streams: StreamsSel::Ir,
                emitter: EmitterSel::Keep,
                ..ModeSpec::default()
            })
            .unwrap()[0];
        let active = host.enter(mode).unwrap();
        assert!(active.teardown.is_none());
        assert_eq!(opener.xu.mode(), 3);
        assert_eq!(opener.opens.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[test]
    fn test_enter_off_then_drop_restores_exact_prior_0x03() {
        let opener = Arc::new(FakeOpener::new(SharedFakeXu::with_mode(3)));
        let mut host = fake_live_host(Arc::clone(&opener));
        let mode = &host
            .expand(&ModeSpec {
                streams: StreamsSel::Ir,
                emitter: EmitterSel::Off,
                ..ModeSpec::default()
            })
            .unwrap()[0];
        let active = host.enter(mode).unwrap();
        assert_eq!(opener.xu.mode(), 0x01);
        drop(active);
        assert_eq!(opener.xu.mode(), 0x03);
        let writes = opener.xu.0.lock().unwrap().writes.clone();
        assert_eq!(
            writes,
            vec![[1, 3, 1, 0, 0, 0, 0, 0, 0], [1, 3, 3, 0, 0, 0, 0, 0, 0],]
        );
    }

    #[test]
    fn test_enter_with_ir_selects_uvcm_on_metadata_node() {
        let opener = Arc::new(FakeOpener::new(SharedFakeXu::with_mode(1)));
        let mut host = fake_live_host(Arc::clone(&opener));
        let ir_mode = host
            .expand(&ModeSpec {
                streams: StreamsSel::Ir,
                ..ModeSpec::default()
            })
            .unwrap()[0]
            .clone();
        host.enter(&ir_mode).unwrap();
        assert_eq!(
            *opener.prepared_meta.lock().unwrap(),
            vec![PathBuf::from("/dev/video3")]
        );

        let rgb_mode = host
            .expand(&ModeSpec {
                streams: StreamsSel::Rgb,
                ..ModeSpec::default()
            })
            .unwrap()[0]
            .clone();
        opener.prepared_meta.lock().unwrap().clear();
        host.enter(&rgb_mode).unwrap();
        assert!(opener.prepared_meta.lock().unwrap().is_empty());
    }

    #[test]
    fn test_live_session_open_meta_follows_mode() {
        use crate::testkit::FakeMetaStream;

        let opener = Arc::new(FakeOpener::new(SharedFakeXu::with_mode(1)));
        opener
            .meta
            .lock()
            .unwrap()
            .push_back(Box::new(FakeMetaStream));
        let mut host = fake_live_host(Arc::clone(&opener));

        let ir_mode = host
            .expand(&ModeSpec {
                streams: StreamsSel::Ir,
                ..ModeSpec::default()
            })
            .unwrap()[0]
            .clone();
        let active = host.enter(&ir_mode).unwrap();
        assert!(active.session.open_meta().unwrap().is_some());

        let rgb_mode = host
            .expand(&ModeSpec {
                streams: StreamsSel::Rgb,
                ..ModeSpec::default()
            })
            .unwrap()[0]
            .clone();
        let active = host.enter(&rgb_mode).unwrap();
        assert!(active.session.open_meta().unwrap().is_none());
    }

    #[test]
    fn test_enter_on_with_camera_lacking_control_is_emitter_error() {
        let cams: Vec<CameraDevice> = dell_cameras()
            .into_iter()
            .map(|mut c| {
                c.metadata_node = None;
                c
            })
            .collect();
        let selection = CameraSelection {
            ir: Some(PathBuf::from("/dev/video0")),
            ..CameraSelection::default()
        };
        let mut host = LiveHost::new(cams, &selection, Arc::new(opener::V4l2Opener)).unwrap();
        let mode = Mode {
            emitter: EmitterSetting::On,
            rgb: None,
            ir: Some(StreamTarget {
                node: PathBuf::from("/dev/video0"),
                format: "MJPG 640x360@30".parse().unwrap(),
            }),
        };
        let err = host.enter(&mode).unwrap_err();
        assert!(matches!(
            err,
            ModeError::Emitter(eye_platform::EmitterError::NoControl { .. })
        ));
    }

    #[test]
    fn test_emitter_status_reads_byte() {
        let opener = Arc::new(FakeOpener::new(SharedFakeXu::with_mode(3)));
        let host = fake_live_host(opener);
        assert_eq!(host.emitter_status(), "emitter: /dev/video2 byte 2 = 0x03");

        let cams = vec![
            dell_cameras()
                .into_iter()
                .find(|c| c.kind == CameraKind::Rgb)
                .unwrap(),
        ];
        let opener2 = Arc::new(FakeOpener::new(SharedFakeXu::with_mode(3)));
        let host2 = LiveHost::new(cams, &CameraSelection::default(), opener2).unwrap();
        assert_eq!(host2.emitter_status(), "emitter: none");
    }

    #[test]
    #[ignore = "needs hardware"]
    fn test_live_matrix_on_dev_machine() {
        let host = LiveHost::probe(&CameraSelection::default()).unwrap();
        let rows = host.matrix();
        assert_eq!(rows.iter().filter(|r| r.runnable).count(), 17);
    }

    #[test]
    #[ignore = "needs hardware"]
    fn test_live_enter_restores_prior_payload() {
        use eye_platform::MsxuIrEmitter;

        let selection = CameraSelection::default();
        let mut host = LiveHost::probe(&selection).unwrap();
        let ir = host.camera(Role::Ir).unwrap().clone();
        let before = MsxuIrEmitter::discover(&ir).unwrap().payload().unwrap();

        let mode = host
            .expand(&ModeSpec {
                streams: StreamsSel::Ir,
                emitter: EmitterSel::On,
                ..ModeSpec::default()
            })
            .unwrap()[0]
            .clone();
        let active = host.enter(&mode).unwrap();
        assert_eq!(
            MsxuIrEmitter::discover(&ir).unwrap().mode().unwrap(),
            MODE_ON_DEFAULT
        );
        drop(active);
        assert_eq!(
            MsxuIrEmitter::discover(&ir).unwrap().payload().unwrap(),
            before
        );
        assert_eq!(
            eye_platform::meta_format(Path::new("/dev/video3")).unwrap(),
            eye_platform::META_FORMAT_UVCM
        );
    }
}
