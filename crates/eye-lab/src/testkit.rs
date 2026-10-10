use std::{
    collections::{HashMap, VecDeque},
    fmt,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU8, AtomicUsize, Ordering},
    },
    time::Duration,
};

use eye_capture::{CaptureError, FrameSource, IlluminationMeta, MetaRecord};
use eye_core::{
    CameraId, CameraInfo, Ellipse2, EyeObservation, FaceObservation, Frame, FrameHeader, FrameSet,
    Illumination, Measured, Observations, PixelFormat, Side, Timestamp,
    observation::SCHEME_IR_PUPIL_PAIR,
    stage::{Detector, StageError},
};
use eye_platform::{
    CameraDevice, CameraKind, EmitterError, ExtensionUnit, FormatInfo, FrameSizeInfo,
    MsxuIrEmitter, UsbIdentity, XuError, XuOpener, XuQuery, XuTransport, emitter::MSXU_GUID,
    find_face_auth_control,
};
use nalgebra::Point2;

use crate::{
    case::{Needs, ParamError, RunOptions, TestCase, TestCtx, TestError, TestOutput, TestRegistry},
    mode::{
        ActiveMode, EmitterSetting, LabEmitter, Mode, ModeError, ModeHost, ModeSession, Role,
        StreamTarget, Teardown,
    },
    modes::{CameraSelection, LiveHost, opener::Opener},
    pipeline::DetectorFactory,
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

    #[allow(dead_code)]
    pub(crate) fn with_meta(self, m: Box<dyn IlluminationMeta>) -> Self {
        self.meta.lock().unwrap().push_back(m);
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

fn build_read_emitter(_: &toml::Table) -> Result<Box<dyn TestCase>, ParamError> {
    Ok(Box::new(ReadEmitterCase))
}

#[allow(dead_code)]
pub(crate) const P: u64 = 33_333_333;

/// 64x36 Gray8 frames, seq `start_seq + k`, timestamp `start_ns + k * period_ns`, gray value `value(seq)`.
pub(crate) fn synth_gray(
    camera: &str,
    n: usize,
    start_seq: u64,
    start_ns: u64,
    period_ns: u64,
    value: impl Fn(u64) -> u8,
) -> Vec<Frame> {
    (0..n as u64)
        .map(|k| {
            let seq = start_seq + k;
            gray_frame(camera, seq, start_ns + k * period_ns, 64, 36, value(seq))
        })
        .collect()
}

/// MJPG frames (`FF D8 00 00 FF D9`) at the given timestamps, seqs from 0.
pub(crate) fn mjpeg_at(camera: &str, t_ns: &[u64], width: u32, height: u32) -> Vec<Frame> {
    t_ns.iter()
        .enumerate()
        .map(|(seq, &t)| mjpeg_frame(camera, seq as u64, t, width, height))
        .collect()
}

pub(crate) fn synth_mjpeg(
    camera: &str,
    n: usize,
    start_seq: u64,
    start_ns: u64,
    period_ns: u64,
) -> Vec<Frame> {
    (0..n as u64)
        .map(|k| mjpeg_frame(camera, start_seq + k, start_ns + k * period_ns, 64, 36))
        .collect()
}

/// `FakeSource` with `camera_info` from the first frame.
pub(crate) fn boxed(frames: Vec<Frame>) -> Box<dyn FrameSource> {
    let info = frames
        .first()
        .map(|f| {
            let h = f.header();
            camera_info(h.camera.as_str(), h.format, h.width, h.height)
        })
        .unwrap_or_else(|| camera_info("none", PixelFormat::Gray8, 1, 1));
    Box::new(FakeSource {
        info,
        frames: frames.into(),
    })
}

/// Rewrites each frame's timestamp to `Timestamp::now() - age` when it is handed out.
#[derive(Debug)]
pub(crate) struct LiveStampedSource {
    pub inner: FakeSource,
    pub age: Duration,
}

impl FrameSource for LiveStampedSource {
    fn camera(&self) -> &CameraInfo {
        self.inner.camera()
    }

    fn next_frame(&mut self) -> Result<Frame, CaptureError> {
        let frame = self.inner.next_frame()?;
        let (mut header, data) = frame.into_parts();
        header.timestamp = Timestamp(Timestamp::now().0.saturating_sub(self.age));
        Frame::new(header, data).map_err(|e| CaptureError::Io {
            camera: self.inner.info.id.clone(),
            source: std::io::Error::other(e.to_string()),
        })
    }
}

/// Endless (until `remaining` hits 0) 64x36 IR source that alternates 0/46 (odd seqs lit) while
/// `mode >= 2`, else constant 3; 66_666_666 ns period (15 fps).
#[derive(Debug)]
pub(crate) struct EmitterAwareSource {
    pub mode: Arc<AtomicU8>,
    pub seq: u64,
    pub remaining: usize,
}

impl FrameSource for EmitterAwareSource {
    fn camera(&self) -> &CameraInfo {
        unreachable!("EmitterAwareSource.camera() is unused by the cases that consume it")
    }

    fn next_frame(&mut self) -> Result<Frame, CaptureError> {
        if self.remaining == 0 {
            return Err(CaptureError::EndOfStream);
        }
        self.remaining -= 1;
        let seq = self.seq;
        self.seq += 1;
        let on = self.mode.load(Ordering::SeqCst) >= 2;
        let value = if on {
            if seq % 2 == 1 { 46 } else { 0 }
        } else {
            3
        };
        Ok(gray_frame("ir", seq, seq * 66_666_666, 64, 36, value))
    }
}

/// Wraps an `EmitterAwareSource`; on its first `next_frame()` stores `MODE_OFF` into `mode`
/// (a firmware that resets selector 6 on STREAMON).
#[derive(Debug)]
pub(crate) struct ResetOnStreamOn {
    pub inner: EmitterAwareSource,
    pub mode: Arc<AtomicU8>,
    pub started: bool,
}

impl FrameSource for ResetOnStreamOn {
    fn camera(&self) -> &CameraInfo {
        self.inner.camera()
    }

    fn next_frame(&mut self) -> Result<Frame, CaptureError> {
        if !self.started {
            self.started = true;
            self.mode
                .store(eye_platform::emitter::MODE_OFF, Ordering::SeqCst);
        }
        self.inner.next_frame()
    }
}

/// `FakeSession` whose devices map has an IR `CameraDevice` with `usb.sysfs_device = dir`
/// (a temp dir holding `idVendor` and `power/runtime_status`).
pub(crate) fn session_with_usb_dir(
    dir: &Path,
    emitter: FakeEmitter,
    ir_sources: Vec<Box<dyn FrameSource>>,
) -> FakeSession {
    let usb = UsbIdentity {
        vendor_id: 0x0c45,
        product_id: 0x672c,
        interface: 2,
        sysfs_device: dir.to_path_buf(),
    };
    let device = CameraDevice {
        node: PathBuf::from("/dev/video2"),
        card: "Dell IR".to_owned(),
        driver: "uvcvideo".to_owned(),
        bus: "usb-0000:00:14.0-6".to_owned(),
        kind: CameraKind::Ir,
        formats: vec![],
        usb: Some(usb),
        extension_units: vec![],
        metadata_node: None,
    };
    let mut session = FakeSession {
        emitter: Some(emitter),
        ..FakeSession::empty()
    };
    session.devices.insert(Role::Ir, device);
    for s in ir_sources {
        session = session.with_source(Role::Ir, s);
    }
    session
}

/// Metadata stream that pops scripted records, then `Err(CaptureError::EndOfStream)`.
#[derive(Debug)]
pub(crate) struct FakeMeta {
    pub records: VecDeque<MetaRecord>,
}

impl IlluminationMeta for FakeMeta {
    fn next_record(&mut self) -> Result<MetaRecord, CaptureError> {
        self.records.pop_front().ok_or(CaptureError::EndOfStream)
    }
}

fn target(s: &str) -> StreamTarget {
    StreamTarget {
        node: PathBuf::from(if s == RGB_TARGET {
            "/dev/video0"
        } else {
            "/dev/video2"
        }),
        format: s.parse().expect("valid testkit target"),
    }
}

/// rgb `RGB_TARGET` on `/dev/video0`.
pub(crate) fn mode_rgb() -> Mode {
    Mode {
        emitter: EmitterSetting::Keep,
        rgb: Some(target(RGB_TARGET)),
        ir: None,
    }
}

/// ir `IR_TARGET` on `/dev/video2`.
pub(crate) fn mode_ir(emitter: EmitterSetting) -> Mode {
    Mode {
        emitter,
        rgb: None,
        ir: Some(target(IR_TARGET)),
    }
}

pub(crate) fn mode_dual(emitter: EmitterSetting) -> Mode {
    Mode {
        emitter,
        rgb: Some(target(RGB_TARGET)),
        ir: Some(target(IR_TARGET)),
    }
}

fn testkit_registry() -> TestRegistry {
    let mut r = TestRegistry::builtin();
    r.register("needs-ir", "testkit: needs ir", build_needs_ir);
    r.register("needs-rgb", "testkit: needs rgb", build_needs_rgb);
    r.register(
        "read-emitter",
        "testkit: reads the emitter mode",
        build_read_emitter,
    );
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

/// XuTransport over shared state; behaves like the IR MSXU control in EYE-2: unit 4 selector 6,
/// len 9, info 0x03, max [1,3,3,0,..], def [1,3,1,0,..]. Other unit/selector -> NotFound.
/// A buffer of the wrong length -> Ioctl EINVAL. SetCur with `fail_set` -> Ioctl EIO.
#[derive(Debug, Clone)]
pub(crate) struct SharedFakeXu(pub Arc<Mutex<FakeXuState>>);

#[derive(Debug)]
pub(crate) struct FakeXuState {
    pub cur: [u8; 9],
    pub writes: Vec<[u8; 9]>,
    pub ignore_writes: bool,
    pub fail_set: bool,
}

impl SharedFakeXu {
    pub(crate) fn with_mode(mode: u8) -> Self {
        Self(Arc::new(Mutex::new(FakeXuState {
            cur: [1, 3, mode, 0, 0, 0, 0, 0, 0],
            writes: Vec::new(),
            ignore_writes: false,
            fail_set: false,
        })))
    }

    pub(crate) fn mode(&self) -> u8 {
        self.0.lock().unwrap().cur[2]
    }
}

impl XuTransport for SharedFakeXu {
    fn query(
        &self,
        unit: u8,
        selector: u8,
        query: XuQuery,
        data: &mut [u8],
    ) -> Result<(), XuError> {
        if (unit, selector) != (4, 6) {
            return Err(XuError::NotFound { unit, selector });
        }
        let ioctl = |errno| XuError::Ioctl {
            unit,
            selector,
            query,
            errno,
        };
        let mut s = self.0.lock().unwrap();
        match query {
            XuQuery::GetLen if data.len() == 2 => data.copy_from_slice(&9u16.to_le_bytes()),
            XuQuery::GetInfo if data.len() == 1 => data[0] = 0x03,
            XuQuery::GetCur if data.len() == 9 => data.copy_from_slice(&s.cur),
            XuQuery::GetMax if data.len() == 9 => {
                data.copy_from_slice(&[1, 3, 3, 0, 0, 0, 0, 0, 0])
            }
            XuQuery::GetDef if data.len() == 9 => {
                data.copy_from_slice(&[1, 3, 1, 0, 0, 0, 0, 0, 0])
            }
            XuQuery::GetMin | XuQuery::GetRes if data.len() == 9 => data.fill(0),
            XuQuery::SetCur if data.len() == 9 => {
                if s.fail_set {
                    return Err(ioctl(nix::errno::Errno::EIO));
                }
                let mut w = [0u8; 9];
                w.copy_from_slice(data);
                s.writes.push(w);
                if !s.ignore_writes {
                    s.cur = w;
                }
            }
            _ => return Err(ioctl(nix::errno::Errno::EINVAL)),
        }
        Ok(())
    }
}

/// Opener over SharedFakeXu plus scripted sources and metadata streams.
pub(crate) struct FakeOpener {
    pub xu: SharedFakeXu,
    /// Emitter opens (open_emitter calls and guard re-opens), shared with the guard's opener closure.
    pub opens: Arc<AtomicUsize>,
    pub sources: Mutex<HashMap<Role, VecDeque<Box<dyn FrameSource>>>>,
    pub prepared_meta: Mutex<Vec<PathBuf>>,
    pub meta: Mutex<VecDeque<Box<dyn IlluminationMeta>>>,
}

impl fmt::Debug for FakeOpener {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("FakeOpener")
    }
}

impl FakeOpener {
    pub(crate) fn new(xu: SharedFakeXu) -> Self {
        Self {
            xu,
            opens: Arc::new(AtomicUsize::new(0)),
            sources: Mutex::new(HashMap::new()),
            prepared_meta: Mutex::new(Vec::new()),
            meta: Mutex::new(VecDeque::new()),
        }
    }
}

impl Opener for FakeOpener {
    fn open_source(
        &self,
        role: Role,
        _target: &StreamTarget,
    ) -> Result<Box<dyn FrameSource>, CaptureError> {
        let mut sources = self.sources.lock().unwrap();
        sources
            .get_mut(&role)
            .and_then(VecDeque::pop_front)
            .ok_or(CaptureError::EndOfStream)
    }

    fn open_emitter(&self, camera: &CameraDevice) -> Result<Box<dyn LabEmitter>, EmitterError> {
        self.opens.fetch_add(1, Ordering::SeqCst);
        let control = find_face_auth_control(&camera.extension_units).ok_or_else(|| {
            EmitterError::NoControl {
                node: camera.node.clone(),
            }
        })?;
        Ok(Box::new(MsxuIrEmitter::with_transport(
            self.xu.clone(),
            control,
        )?))
    }

    fn guard_emitter(&self, camera: &CameraDevice, mode: u8) -> Result<Teardown, EmitterError> {
        let control = find_face_auth_control(&camera.extension_units).ok_or_else(|| {
            EmitterError::NoControl {
                node: camera.node.clone(),
            }
        })?;
        let (xu, opens) = (self.xu.clone(), Arc::clone(&self.opens));
        let open: XuOpener<SharedFakeXu> = Box::new(move |_: &Path| {
            opens.fetch_add(1, Ordering::SeqCst);
            Ok(xu.clone())
        });
        Ok(Box::new(eye_platform::EmitterGuard::with_opener(
            camera.node.clone(),
            control,
            mode,
            open,
        )?))
    }

    fn prepare_meta(&self, node: &Path) -> Result<[u8; 4], XuError> {
        self.prepared_meta.lock().unwrap().push(node.to_path_buf());
        Ok(eye_platform::META_FORMAT_UVCM)
    }

    fn open_meta(&self, _node: &Path) -> Result<Box<dyn IlluminationMeta>, CaptureError> {
        self.meta
            .lock()
            .unwrap()
            .pop_front()
            .ok_or(CaptureError::EndOfStream)
    }
}

/// A scripted `IlluminationMeta` with no records; used only to prove `open_meta` returns `Some`.
#[derive(Debug)]
pub(crate) struct FakeMetaStream;

impl IlluminationMeta for FakeMetaStream {
    fn next_record(&mut self) -> Result<MetaRecord, CaptureError> {
        Err(CaptureError::EndOfStream)
    }
}

/// `/dev/video0` (Rgb, MJPG 1280x720/960x540/848x480/640x480/640x360 @30 + YUYV 640x480@30, usb interface 0, no XUs,
/// metadata_node /dev/video1) and `/dev/video2` (Ir, GREY 640x360@30, usb interface 2,
/// XU { interface: 2, unit_id: 4, guid: MSXU_GUID, num_controls: 16, selectors: vec![6, 9] }, metadata_node /dev/video3);
/// both with `usb.sysfs_device` = `/sys/devices/pci0000:00/0000:00:14.0/usb3/3-6`, vendor 0x0c45, product 0x672c.
pub(crate) fn dell_cameras() -> Vec<CameraDevice> {
    let sysfs_device = PathBuf::from("/sys/devices/pci0000:00/0000:00:14.0/usb3/3-6");
    let usb = |interface: u8| UsbIdentity {
        vendor_id: 0x0c45,
        product_id: 0x672c,
        interface,
        sysfs_device: sysfs_device.clone(),
    };
    let size = |width: u32, height: u32| FrameSizeInfo {
        width,
        height,
        fps: vec![30.0],
    };
    let rgb = CameraDevice {
        node: PathBuf::from("/dev/video0"),
        card: "Dell RGB".to_owned(),
        driver: "uvcvideo".to_owned(),
        bus: "usb-0000:00:14.0-6".to_owned(),
        kind: CameraKind::Rgb,
        formats: vec![
            FormatInfo {
                fourcc: "MJPG".to_owned(),
                sizes: vec![
                    size(1280, 720),
                    size(960, 540),
                    size(848, 480),
                    size(640, 480),
                    size(640, 360),
                ],
            },
            FormatInfo {
                fourcc: "YUYV".to_owned(),
                sizes: vec![size(640, 480)],
            },
        ],
        usb: Some(usb(0)),
        extension_units: vec![],
        metadata_node: Some(PathBuf::from("/dev/video1")),
    };
    let ir = CameraDevice {
        node: PathBuf::from("/dev/video2"),
        card: "Dell IR".to_owned(),
        driver: "uvcvideo".to_owned(),
        bus: "usb-0000:00:14.0-6".to_owned(),
        kind: CameraKind::Ir,
        formats: vec![FormatInfo {
            fourcc: "GREY".to_owned(),
            sizes: vec![size(640, 360)],
        }],
        usb: Some(usb(2)),
        extension_units: vec![ExtensionUnit {
            interface: 2,
            unit_id: 4,
            guid: MSXU_GUID,
            num_controls: 16,
            selectors: vec![6, 9],
        }],
        metadata_node: Some(PathBuf::from("/dev/video3")),
    };
    vec![rgb, ir]
}

pub(crate) fn fake_live_host(opener: Arc<FakeOpener>) -> LiveHost {
    LiveHost::new(dell_cameras(), &CameraSelection::default(), opener)
        .expect("fixture cameras resolve")
}

/// Test-only case (needs `emitter`): one measurement `Measurement::info("mode", f64::from(ctx.session().emitter()?.read_mode()?), "")`.
#[derive(Debug)]
pub(crate) struct ReadEmitterCase;

impl TestCase for ReadEmitterCase {
    fn name(&self) -> &'static str {
        "read-emitter"
    }

    fn needs(&self) -> Needs {
        Needs {
            emitter: true,
            ..Needs::default()
        }
    }

    fn default_timeout(&self) -> Duration {
        Duration::from_secs(10)
    }

    fn run(&self, ctx: &TestCtx) -> Result<TestOutput, TestError> {
        let mut out = TestOutput::default();
        let mode = ctx.session().emitter()?.read_mode()?;
        out.push(crate::case::Measurement::info("mode", f64::from(mode), ""));
        Ok(out)
    }
}

/// Accepts Gray8 frames whose illumination is in `accepts` (empty = all); finds one Left eye with a
/// pupil circle of radius 3 at `(100.0 + dx(seq), 50.0)` (sigma 0.5) in a `FaceObservation { scheme: SCHEME_IR_PUPIL_PAIR, landmarks: vec![], eyes }`
/// when the frame's first byte is > 20, else returns `Observations::empty`; every `fail_every`-th call returns
/// `Err(StageError::Failed("fake".into()))`.
#[derive(Debug)]
pub(crate) struct FakeDetector {
    pub accepts: Vec<Illumination>,
    pub fail_every: Option<usize>,
    pub dx: fn(u64) -> f64,
    pub calls: usize,
}

impl Detector for FakeDetector {
    fn name(&self) -> &'static str {
        "fake"
    }

    fn accepts(&self, format: PixelFormat, illumination: Illumination) -> bool {
        format == PixelFormat::Gray8
            && (self.accepts.is_empty() || self.accepts.contains(&illumination))
    }

    fn detect(&mut self, frames: &FrameSet) -> Result<Vec<Observations>, StageError> {
        self.calls += 1;
        if let Some(n) = self.fail_every
            && n != 0
            && self.calls.is_multiple_of(n)
        {
            return Err(StageError::Failed("fake".into()));
        }
        Ok(frames
            .frames()
            .iter()
            .map(|f| {
                let h = f.header();
                if f.data().first().is_some_and(|&b| b > 20) {
                    let mut eye = EyeObservation::new(Side::Left);
                    eye.pupil = Some(
                        Measured::new(
                            Ellipse2::circle(Point2::new(100.0 + (self.dx)(h.seq), 50.0), 3.0)
                                .expect("valid fake pupil circle"),
                            0.5,
                        )
                        .expect("valid fake sigma"),
                    );
                    Observations {
                        camera: h.camera.clone(),
                        timestamp: h.timestamp,
                        face: Some(FaceObservation {
                            scheme: SCHEME_IR_PUPIL_PAIR,
                            landmarks: vec![],
                            eyes: vec![eye],
                        }),
                    }
                } else {
                    Observations::empty(h.camera.clone(), h.timestamp)
                }
            })
            .collect())
    }
}

pub(crate) fn fake_detectors(
    accepts: Vec<Illumination>,
    fail_every: Option<usize>,
    dx: fn(u64) -> f64,
) -> DetectorFactory {
    Arc::new(move |_section, _rig| {
        Ok(Box::new(FakeDetector {
            accepts: accepts.clone(),
            fail_every,
            dx,
            calls: 0,
        }) as Box<dyn Detector>)
    })
}
