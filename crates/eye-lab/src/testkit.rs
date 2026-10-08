use std::{
    collections::{HashMap, VecDeque},
    fmt,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU8, Ordering},
    },
    time::Duration,
};

use eye_capture::{CaptureError, FrameSource, IlluminationMeta};
use eye_core::{CameraId, CameraInfo, Frame, FrameHeader, Illumination, PixelFormat, Timestamp};
use eye_platform::{CameraDevice, EmitterError};

use crate::{
    case::{Needs, ParamError, RunOptions, TestCase, TestCtx, TestError, TestOutput, TestRegistry},
    mode::{
        ActiveMode, EmitterSetting, LabEmitter, Mode, ModeError, ModeHost, ModeSession, Role,
        StreamTarget,
    },
    runner::PlannedStep,
    sequence::{EmitterSel, ModeSpec, StreamsSel},
};

pub(crate) const RGB_TARGET: &str = "MJPG 64x36@30";
pub(crate) const IR_TARGET: &str = "GREY 64x36@30";

/// Expands like the real host but with fixed tiny targets: rgb on /dev/video0, ir on /dev/video2.
/// `streams = "*"` -> [rgb, ir, dual]; `emitter = "*"` -> [off, on] when the mode has ir, else [keep].
/// `rgb`/`ir` selectors are ignored (always the fixed target).
#[derive(Debug)]
pub(crate) struct FakeHost {
    pub log: Arc<Mutex<Vec<String>>>,
    pub fail_enter: bool,
    pub session: Arc<dyn ModeSession>,
}

impl FakeHost {
    pub(crate) fn new(session: Arc<dyn ModeSession>) -> Self {
        Self {
            log: Arc::new(Mutex::new(Vec::new())),
            fail_enter: false,
            session,
        }
    }
}

impl ModeHost for FakeHost {
    fn expand(&self, spec: &ModeSpec) -> Result<Vec<Mode>, ModeError> {
        let rgb_target = StreamTarget {
            node: PathBuf::from("/dev/video0"),
            format: RGB_TARGET.parse().expect("valid rgb target"),
        };
        let ir_target = StreamTarget {
            node: PathBuf::from("/dev/video2"),
            format: IR_TARGET.parse().expect("valid ir target"),
        };

        let base: Vec<(Option<StreamTarget>, Option<StreamTarget>)> = match spec.streams {
            StreamsSel::None => vec![(None, None)],
            StreamsSel::Rgb => vec![(Some(rgb_target.clone()), None)],
            StreamsSel::Ir => vec![(None, Some(ir_target.clone()))],
            StreamsSel::Dual => vec![(Some(rgb_target.clone()), Some(ir_target.clone()))],
            StreamsSel::All => vec![
                (Some(rgb_target.clone()), None),
                (None, Some(ir_target.clone())),
                (Some(rgb_target.clone()), Some(ir_target.clone())),
            ],
        };

        let mut modes = Vec::new();
        for (rgb, ir) in base {
            let has_ir = ir.is_some();
            let emitters: Vec<EmitterSetting> = match spec.emitter {
                EmitterSel::Keep => vec![EmitterSetting::Keep],
                EmitterSel::On => vec![EmitterSetting::On],
                EmitterSel::Off => vec![EmitterSetting::Off],
                EmitterSel::Each => {
                    if has_ir {
                        vec![EmitterSetting::Off, EmitterSetting::On]
                    } else {
                        vec![EmitterSetting::Keep]
                    }
                }
            };
            for emitter in emitters {
                modes.push(Mode {
                    emitter,
                    rgb: rgb.clone(),
                    ir: ir.clone(),
                });
            }
        }
        Ok(modes)
    }

    fn enter(&mut self, mode: &Mode) -> Result<ActiveMode, ModeError> {
        if self.fail_enter {
            return Err(ModeError::Unavailable("fake host configured to fail enter"));
        }
        self.log.lock().unwrap().push(format!("enter {mode}"));
        Ok(ActiveMode {
            session: Arc::clone(&self.session),
            teardown: Some(Box::new(LogOnDrop(
                Arc::clone(&self.log),
                format!("teardown {mode}"),
            ))),
        })
    }
}

/// Pushes "teardown {mode}" when dropped; `enter` returns it as the teardown.
#[derive(Debug)]
pub(crate) struct LogOnDrop(pub Arc<Mutex<Vec<String>>>, pub String);

impl Drop for LogOnDrop {
    fn drop(&mut self) {
        self.0.lock().unwrap().push(self.1.clone());
    }
}

/// `open(role)` pops the next scripted source for that role (an EndOfStream-only source when empty);
/// `open_meta()` pops the next scripted metadata stream (`Ok(None)` when empty).
pub(crate) struct FakeSession {
    pub sources: Mutex<HashMap<Role, VecDeque<Box<dyn FrameSource>>>>,
    pub meta: Mutex<VecDeque<Box<dyn IlluminationMeta>>>,
    pub emitter: Option<FakeEmitter>,
    pub devices: HashMap<Role, CameraDevice>,
}

impl fmt::Debug for FakeSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("FakeSession")
    }
}

impl FakeSession {
    pub(crate) fn empty() -> Self {
        Self {
            sources: Mutex::new(HashMap::new()),
            meta: Mutex::new(VecDeque::new()),
            emitter: None,
            devices: HashMap::new(),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn with_source(self, role: Role, s: Box<dyn FrameSource>) -> Self {
        self.sources
            .lock()
            .unwrap()
            .entry(role)
            .or_default()
            .push_back(s);
        self
    }
}

impl ModeSession for FakeSession {
    fn open(&self, role: Role) -> Result<Box<dyn FrameSource>, CaptureError> {
        let mut sources = self.sources.lock().unwrap();
        if let Some(source) = sources.get_mut(&role).and_then(VecDeque::pop_front) {
            return Ok(source);
        }
        Ok(Box::new(FakeSource {
            info: camera_info(
                match role {
                    Role::Rgb => "rgb",
                    Role::Ir => "ir",
                },
                PixelFormat::Gray8,
                1,
                1,
            ),
            frames: VecDeque::new(),
        }))
    }

    fn open_meta(&self) -> Result<Option<Box<dyn IlluminationMeta>>, CaptureError> {
        Ok(self.meta.lock().unwrap().pop_front())
    }

    fn emitter(&self) -> Result<Box<dyn LabEmitter>, ModeError> {
        match &self.emitter {
            Some(e) => Ok(Box::new(e.clone())),
            None => Err(ModeError::NoCamera(Role::Ir)),
        }
    }

    fn device(&self, role: Role) -> Option<CameraDevice> {
        self.devices.get(&role).cloned()
    }
}

#[derive(Debug, Clone)]
pub(crate) struct FakeEmitter {
    pub mode: Arc<AtomicU8>,
    pub ignore_writes: bool,
    pub writes: Arc<Mutex<Vec<u8>>>,
}

impl LabEmitter for FakeEmitter {
    fn read_mode(&self) -> Result<u8, EmitterError> {
        Ok(self.mode.load(Ordering::SeqCst))
    }

    fn write_mode(&mut self, mode: u8) -> Result<(), EmitterError> {
        self.writes.lock().unwrap().push(mode);
        if !self.ignore_writes {
            self.mode.store(mode, Ordering::SeqCst);
        }
        Ok(())
    }
}

/// Pops scripted frames, then `Err(CaptureError::EndOfStream)`.
#[derive(Debug)]
pub(crate) struct FakeSource {
    pub info: CameraInfo,
    pub frames: VecDeque<Frame>,
}

impl FrameSource for FakeSource {
    fn camera(&self) -> &CameraInfo {
        &self.info
    }

    fn next_frame(&mut self) -> Result<Frame, CaptureError> {
        self.frames.pop_front().ok_or(CaptureError::EndOfStream)
    }
}

pub(crate) fn camera_info(id: &str, format: PixelFormat, width: u32, height: u32) -> CameraInfo {
    CameraInfo {
        id: CameraId::from(id),
        format,
        width,
        height,
        frame_interval: Duration::from_nanos(33_333_333),
    }
}

#[allow(dead_code)]
pub(crate) fn gray_frame(
    camera: &str,
    seq: u64,
    t_ns: u64,
    width: u32,
    height: u32,
    value: u8,
) -> Frame {
    Frame::new(
        FrameHeader {
            camera: CameraId::from(camera),
            seq,
            timestamp: Timestamp::from_nanos(t_ns),
            width,
            height,
            format: PixelFormat::Gray8,
            illumination: Illumination::Unknown,
        },
        Arc::from(vec![value; (width * height) as usize]),
    )
    .expect("valid synthetic frame")
}

#[allow(dead_code)]
pub(crate) fn mjpeg_frame(camera: &str, seq: u64, t_ns: u64, width: u32, height: u32) -> Frame {
    Frame::new(
        FrameHeader {
            camera: CameraId::from(camera),
            seq,
            timestamp: Timestamp::from_nanos(t_ns),
            width,
            height,
            format: PixelFormat::Mjpeg,
            illumination: Illumination::Ambient,
        },
        Arc::from(vec![0xFF, 0xD8, 0x00, 0x00, 0xFF, 0xD9]),
    )
    .expect("valid synthetic frame")
}

pub(crate) fn ctx(session: Arc<dyn ModeSession>, mode: Mode) -> TestCtx {
    TestCtx::new(
        session,
        mode,
        Arc::new(std::sync::atomic::AtomicBool::new(false)),
        Arc::new(std::sync::atomic::AtomicBool::new(false)),
        Duration::from_secs(30),
        Arc::new(RunOptions::default()),
    )
}

/// Always passes; reports the given needs. Tests register it as "needs-ir" / "needs-rgb" via
/// `fn build_needs_ir(_: &toml::Table) -> Result<Box<dyn TestCase>, ParamError>`.
#[derive(Debug)]
pub(crate) struct NeedsCase(pub Needs);

impl TestCase for NeedsCase {
    fn name(&self) -> &'static str {
        "needs"
    }

    fn needs(&self) -> Needs {
        self.0
    }

    fn default_timeout(&self) -> Duration {
        Duration::from_secs(10)
    }

    fn run(&self, _ctx: &TestCtx) -> Result<TestOutput, TestError> {
        Ok(TestOutput::default())
    }
}

fn build_needs_ir(_: &toml::Table) -> Result<Box<dyn TestCase>, ParamError> {
    Ok(Box::new(NeedsCase(Needs {
        ir: true,
        ..Needs::default()
    })))
}

fn build_needs_rgb(_: &toml::Table) -> Result<Box<dyn TestCase>, ParamError> {
    Ok(Box::new(NeedsCase(Needs {
        rgb: true,
        ..Needs::default()
    })))
}

fn testkit_registry() -> TestRegistry {
    let mut r = TestRegistry::builtin();
    r.register("needs-ir", "testkit: needs ir", build_needs_ir);
    r.register("needs-rgb", "testkit: needs rgb", build_needs_rgb);
    r
}

fn parse_mode(mode: &str) -> ModeSpec {
    if mode.trim().is_empty() {
        return ModeSpec::default();
    }
    let wrapped = format!("mode = {{ {mode} }}");
    let mut table: toml::Table =
        toml::from_str(&wrapped).unwrap_or_else(|e| panic!("invalid testkit mode {mode:?}: {e}"));
    let value = table.remove("mode").expect("mode key present");
    let text = toml::to_string(&value)
        .unwrap_or_else(|e| panic!("serializing testkit mode {mode:?}: {e}"));
    toml::from_str(&text).unwrap_or_else(|e| panic!("invalid testkit mode {mode:?}: {e}"))
}

fn parse_params(params: &str) -> toml::Table {
    if params.trim().is_empty() {
        return toml::Table::new();
    }
    let wrapped = format!("params = {{ {params} }}");
    let mut table: toml::Table = toml::from_str(&wrapped)
        .unwrap_or_else(|e| panic!("invalid testkit params {params:?}: {e}"));
    match table.remove("params") {
        Some(toml::Value::Table(t)) => t,
        other => panic!("testkit: params did not parse to a table: {other:?}"),
    }
}

/// `params`/`mode` are TOML inline-table bodies, e.g. planned("selftest-check", "value = 10.0", "streams = \"none\"").
pub(crate) fn planned(test: &str, params: &str, mode: &str) -> PlannedStep {
    let registry = testkit_registry();
    let info = registry
        .get(test)
        .unwrap_or_else(|| panic!("testkit: unknown test {test:?}"));
    let params_table = parse_params(params);
    let case = (info.factory)(&params_table)
        .unwrap_or_else(|e| panic!("testkit: bad params for {test:?}: {e}"));
    let timeout = case.default_timeout();
    PlannedStep {
        origin: "testkit".to_owned(),
        label: test.to_owned(),
        test: test.to_owned(),
        mode: parse_mode(mode),
        case: Arc::from(case),
        timeout,
    }
}

/// `planned` with `step.timeout` replaced.
pub(crate) fn planned_with_timeout(
    test: &str,
    params: &str,
    mode: &str,
    timeout: Duration,
) -> PlannedStep {
    PlannedStep {
        timeout,
        ..planned(test, params, mode)
    }
}
