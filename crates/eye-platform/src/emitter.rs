use std::path::{Path, PathBuf};

use crate::ProbeError;
use crate::camera::usb_desc::{ExtensionUnit, Guid};
use crate::camera::{CameraDevice, CameraProbe, V4l2CameraProbe};
use crate::uvc::{UvcXuDevice, XuError, XuQuery, XuTransport};

pub const MSXU_GUID: Guid = Guid::from_fields(
    0x0f3f95dc,
    0x2632,
    0x4c4e,
    [0x92, 0xc9, 0xa0, 0x47, 0x82, 0xf4, 0x3b, 0xc8],
);
pub const MSXU_FACE_AUTH_SELECTOR: u8 = 6;
pub const MODE_BYTE: usize = 2;
pub const MODE_OFF: u8 = 0x01;
/// R10: the single alternating-illumination flag; EYE-25 may switch it to 0x03.
pub const MODE_ON_DEFAULT: u8 = 0x02;

pub trait IrEmitter {
    fn set_enabled(&mut self, on: bool) -> Result<(), EmitterError>;
    fn is_enabled(&self) -> Result<bool, EmitterError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct EmitterControl {
    pub unit: u8,
    pub selector: u8,
}

/// The first MSXU unit that advertises the face-authentication selector.
pub fn find_face_auth_control(units: &[ExtensionUnit]) -> Option<EmitterControl> {
    units
        .iter()
        .find(|u| u.guid == MSXU_GUID && u.selectors.contains(&MSXU_FACE_AUTH_SELECTOR))
        .map(|u| EmitterControl {
            unit: u.unit_id,
            selector: MSXU_FACE_AUTH_SELECTOR,
        })
}

#[derive(Debug, thiserror::Error)]
pub enum EmitterError {
    #[error("no MSXU face-authentication control on {node}")]
    NoControl { node: PathBuf },
    #[error(transparent)]
    Xu(#[from] XuError),
    #[error(transparent)]
    Probe(#[from] ProbeError),
    #[error("{node} is not a V4L2 capture camera")]
    NoSuchCamera { node: PathBuf },
    #[error("control {control:?} has length {len}, expected at least 3")]
    UnexpectedLength { control: EmitterControl, len: u16 },
    #[error("control {control:?} is not get+set (GET_INFO {info:#04x})")]
    NotSettable { control: EmitterControl, info: u8 },
    #[error("mode {requested:#04x} exceeds the device maximum {max:#04x}")]
    ModeOutOfRange { requested: u8, max: u8 },
    #[error("device reports mode {actual:#04x} after writing {requested:#04x}")]
    NotApplied { requested: u8, actual: u8 },
    #[error("device reports payload {actual:02x?} after restoring {expected:02x?}")]
    NotRestored { expected: Vec<u8>, actual: Vec<u8> },
}

#[derive(Debug)]
pub struct MsxuIrEmitter<T: XuTransport = UvcXuDevice> {
    xu: T,
    control: EmitterControl,
    on_mode: u8,
}

impl MsxuIrEmitter<UvcXuDevice> {
    /// Finds the control in `camera.extension_units` (no I/O when there is none), then opens `camera.node`.
    pub fn discover(camera: &CameraDevice) -> Result<Self, EmitterError> {
        let control = control_of(camera)?;
        Self::with_transport(UvcXuDevice::open(&camera.node)?, control)
    }

    /// Convenience for `eye emitter --device /dev/video2`: probes cameras, finds `node`, then `discover`.
    pub fn open(node: &Path) -> Result<Self, EmitterError> {
        let wanted = std::fs::canonicalize(node).unwrap_or_else(|_| node.to_owned());
        let camera = V4l2CameraProbe::new()
            .cameras()?
            .into_iter()
            .find(|c| std::fs::canonicalize(&c.node).unwrap_or_else(|_| c.node.clone()) == wanted)
            .ok_or_else(|| EmitterError::NoSuchCamera {
                node: node.to_owned(),
            })?;
        Self::discover(&camera)
    }
}

impl<T: XuTransport> MsxuIrEmitter<T> {
    pub fn with_transport(xu: T, control: EmitterControl) -> Result<Self, EmitterError> {
        let len = xu.len(control.unit, control.selector)?;
        if usize::from(len) <= MODE_BYTE {
            return Err(EmitterError::UnexpectedLength { control, len });
        }
        let info = xu.info(control.unit, control.selector)?;
        if !(info.supports_get() && info.supports_set()) {
            return Err(EmitterError::NotSettable {
                control,
                info: info.0,
            });
        }
        Ok(Self {
            xu,
            control,
            on_mode: MODE_ON_DEFAULT,
        })
    }

    pub fn with_on_mode(self, mode: u8) -> Self {
        Self {
            on_mode: mode,
            ..self
        }
    }

    pub fn control(&self) -> EmitterControl {
        self.control
    }

    /// Byte 2 of GET_CUR.
    pub fn mode(&self) -> Result<u8, EmitterError> {
        Ok(self.payload()?[MODE_BYTE])
    }

    /// The whole GET_CUR payload (9 bytes on this camera).
    pub fn payload(&self) -> Result<Vec<u8>, EmitterError> {
        Ok(self
            .xu
            .get(self.control.unit, self.control.selector, XuQuery::GetCur)?)
    }

    /// Writes mode byte 2 (bounded by GET_MAX, other bytes preserved from GET_CUR, verified by read-back).
    pub fn set_mode(&mut self, mode: u8) -> Result<(), EmitterError> {
        let EmitterControl { unit, selector } = self.control;
        let max = self.xu.get(unit, selector, XuQuery::GetMax)?[MODE_BYTE];
        if mode > max {
            return Err(EmitterError::ModeOutOfRange {
                requested: mode,
                max,
            });
        }
        let mut payload = self.xu.get(unit, selector, XuQuery::GetCur)?;
        payload[MODE_BYTE] = mode;
        self.xu.set_cur(unit, selector, &payload)?;
        let actual = self.mode()?;
        if actual != mode {
            return Err(EmitterError::NotApplied {
                requested: mode,
                actual,
            });
        }
        Ok(())
    }

    /// Writes `payload` verbatim with SET_CUR and verifies the read-back equals it.
    pub fn restore_payload(&mut self, payload: &[u8]) -> Result<(), EmitterError> {
        let EmitterControl { unit, selector } = self.control;
        self.xu.set_cur(unit, selector, payload)?;
        let actual = self.payload()?;
        if actual != payload {
            return Err(EmitterError::NotRestored {
                expected: payload.to_vec(),
                actual,
            });
        }
        Ok(())
    }
}

impl<T: XuTransport> IrEmitter for MsxuIrEmitter<T> {
    fn set_enabled(&mut self, on: bool) -> Result<(), EmitterError> {
        self.set_mode(if on { self.on_mode } else { MODE_OFF })
    }

    fn is_enabled(&self) -> Result<bool, EmitterError> {
        Ok(self.mode()? >= 0x02)
    }
}

fn control_of(camera: &CameraDevice) -> Result<EmitterControl, EmitterError> {
    find_face_auth_control(&camera.extension_units).ok_or_else(|| EmitterError::NoControl {
        node: camera.node.clone(),
    })
}

/// Re-opens the control transport for a node. `EmitterGuard` calls it on apply, `reapply` and drop.
pub type XuOpener<T> = Box<dyn Fn(&Path) -> Result<T, XuError> + Send + Sync>;

/// Applies an emitter mode and, on drop, writes back the exact payload that was there before.
/// Holds no open device between apply and drop: it re-opens the node to restore.
pub struct EmitterGuard<T: XuTransport = UvcXuDevice> {
    node: PathBuf,
    control: EmitterControl,
    mode: u8,
    prior: Vec<u8>,
    open: XuOpener<T>,
}

impl EmitterGuard<UvcXuDevice> {
    /// `apply(camera, MODE_ON_DEFAULT)`.
    pub fn enable(camera: &CameraDevice) -> Result<Self, EmitterError> {
        Self::apply(camera, MODE_ON_DEFAULT)
    }

    pub fn apply(camera: &CameraDevice, mode: u8) -> Result<Self, EmitterError> {
        let open: XuOpener<UvcXuDevice> = Box::new(|node: &Path| UvcXuDevice::open(node));
        Self::with_opener(camera.node.clone(), control_of(camera)?, mode, open)
    }
}

impl<T: XuTransport> EmitterGuard<T> {
    pub fn with_opener(
        node: PathBuf,
        control: EmitterControl,
        mode: u8,
        open: XuOpener<T>,
    ) -> Result<Self, EmitterError> {
        let mut emitter = MsxuIrEmitter::with_transport(open(&node)?, control)?;
        let prior = emitter.payload()?;
        let guard = Self {
            node,
            control,
            mode,
            prior,
            open,
        };
        emitter.set_mode(mode)?;
        Ok(guard)
    }

    pub fn node(&self) -> &Path {
        &self.node
    }

    pub fn control(&self) -> EmitterControl {
        self.control
    }

    /// The mode this guard applied.
    pub fn mode(&self) -> u8 {
        self.mode
    }

    /// The payload restored on drop.
    pub fn prior(&self) -> &[u8] {
        &self.prior
    }

    /// Writes the applied mode again (R5 `on_stream_start` hook, if EYE-25 finds the setting resets per stream).
    pub fn reapply(&self) -> Result<(), EmitterError> {
        self.emitter()?.set_mode(self.mode)
    }

    fn emitter(&self) -> Result<MsxuIrEmitter<T>, EmitterError> {
        MsxuIrEmitter::with_transport((self.open)(&self.node)?, self.control)
    }
}

impl<T: XuTransport> std::fmt::Debug for EmitterGuard<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmitterGuard")
            .field("node", &self.node)
            .field("control", &self.control)
            .field("mode", &self.mode)
            .field("prior", &self.prior)
            .finish_non_exhaustive()
    }
}

impl<T: XuTransport> Drop for EmitterGuard<T> {
    fn drop(&mut self) {
        if let Err(err) = self
            .emitter()
            .and_then(|mut e| e.restore_payload(&self.prior))
        {
            tracing::warn!(node = %self.node.display(), prior = ?self.prior, %err, "could not restore the IR emitter");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::camera::CameraKind;
    use crate::uvc::{FakeControl, FakeXu};

    fn ir_fake() -> FakeXu {
        FakeXu::default()
            .with_control(
                4,
                6,
                FakeControl {
                    info: 0x03,
                    cur: vec![1, 3, 1, 0, 0, 0, 0, 0, 0],
                    min: vec![0; 9],
                    max: vec![1, 3, 3, 0, 0, 0, 0, 0, 0],
                    res: vec![0; 9],
                    def: vec![1, 3, 1, 0, 0, 0, 0, 0, 0],
                    ..FakeControl::default()
                },
            )
            .with_control(
                4,
                9,
                FakeControl {
                    info: 0x03,
                    cur: vec![1, 0, 0, 0],
                    min: vec![0; 4],
                    max: vec![1, 0, 0, 0],
                    res: vec![0; 4],
                    def: vec![1, 0, 0, 0],
                    ..FakeControl::default()
                },
            )
    }

    const IR: EmitterControl = EmitterControl {
        unit: 4,
        selector: 6,
    };

    fn opener(xu: &FakeXu) -> XuOpener<FakeXu> {
        let shared = xu.clone();
        Box::new(move |_: &Path| Ok(shared.clone()))
    }

    fn fixture() -> Vec<u8> {
        std::fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/usb-0c45-672c.descriptors"),
        )
        .unwrap()
    }

    #[test]
    fn test_fixture_ir_interface_yields_unit_4_selector_6() {
        let units = crate::camera::usb_desc::extension_units(&fixture()).unwrap();
        let ir_units: Vec<_> = units.iter().filter(|u| u.interface == 2).cloned().collect();
        assert_eq!(
            find_face_auth_control(&ir_units),
            Some(EmitterControl {
                unit: 4,
                selector: 6
            })
        );
    }

    #[test]
    fn test_rgb_msxu_without_controls_is_not_used() {
        let units = crate::camera::usb_desc::extension_units(&fixture()).unwrap();
        let rgb_units: Vec<_> = units.iter().filter(|u| u.interface == 0).cloned().collect();
        assert_eq!(find_face_auth_control(&rgb_units), None);
    }

    #[test]
    fn test_enable_sets_byte_2_to_2_and_preserves_rest() -> Result<(), EmitterError> {
        let xu = ir_fake();
        let mut e = MsxuIrEmitter::with_transport(xu.clone(), IR)?;
        e.set_enabled(true)?;
        assert_eq!(
            xu.control(4, 6).writes,
            vec![vec![1, 3, 2, 0, 0, 0, 0, 0, 0]]
        );
        assert!(e.is_enabled()?);
        Ok(())
    }

    #[test]
    fn test_disable_sets_byte_2_to_1() -> Result<(), EmitterError> {
        let xu = ir_fake();
        xu.update(4, 6, |c| c.cur[2] = 2);
        let mut e = MsxuIrEmitter::with_transport(xu.clone(), IR)?;
        e.set_enabled(false)?;
        assert_eq!(
            xu.control(4, 6).writes,
            vec![vec![1, 3, 1, 0, 0, 0, 0, 0, 0]]
        );
        assert!(!e.is_enabled()?);
        Ok(())
    }

    #[test]
    fn test_enable_reapplies_when_already_on() -> Result<(), EmitterError> {
        let xu = ir_fake();
        xu.update(4, 6, |c| c.cur[2] = 2);
        let mut e = MsxuIrEmitter::with_transport(xu.clone(), IR)?;
        e.set_enabled(true)?;
        assert_eq!(xu.control(4, 6).writes.len(), 1);
        Ok(())
    }

    #[test]
    fn test_custom_on_mode_3_is_written() -> Result<(), EmitterError> {
        let xu = ir_fake();
        let mut e = MsxuIrEmitter::with_transport(xu.clone(), IR)?.with_on_mode(0x03);
        e.set_enabled(true)?;
        assert_eq!(xu.control(4, 6).cur[2], 0x03);
        assert!(e.is_enabled()?);
        Ok(())
    }

    #[test]
    fn test_mode_above_max_is_rejected_without_write() -> Result<(), EmitterError> {
        let xu = ir_fake();
        let mut e = MsxuIrEmitter::with_transport(xu.clone(), IR)?.with_on_mode(0x04);
        let result = e.set_enabled(true);
        assert!(matches!(
            result,
            Err(EmitterError::ModeOutOfRange {
                requested: 4,
                max: 3
            })
        ));
        assert_eq!(xu.control(4, 6).writes.len(), 0);
        Ok(())
    }

    #[test]
    fn test_ignored_write_is_not_applied() -> Result<(), EmitterError> {
        let xu = ir_fake();
        xu.update(4, 6, |c| c.ignore_writes = true);
        let mut e = MsxuIrEmitter::with_transport(xu.clone(), IR)?;
        let result = e.set_enabled(true);
        assert!(matches!(
            result,
            Err(EmitterError::NotApplied {
                requested: 2,
                actual: 1
            })
        ));
        Ok(())
    }

    #[test]
    fn test_get_only_control_is_not_settable() {
        let xu = ir_fake();
        xu.update(4, 6, |c| c.info = 0x01);
        let result = MsxuIrEmitter::with_transport(xu, IR);
        assert!(matches!(
            result,
            Err(EmitterError::NotSettable { info: 1, .. })
        ));
    }

    #[test]
    fn test_short_control_is_rejected() {
        let xu = ir_fake();
        xu.update(4, 6, |c| {
            c.cur = vec![1, 3];
            c.min = vec![0, 0];
            c.max = vec![1, 3];
            c.res = vec![0, 0];
            c.def = vec![1, 3];
        });
        let result = MsxuIrEmitter::with_transport(xu, IR);
        assert!(matches!(
            result,
            Err(EmitterError::UnexpectedLength { len: 2, .. })
        ));
    }

    #[test]
    fn test_discover_without_control_names_node() {
        let units = crate::camera::usb_desc::extension_units(&fixture()).unwrap();
        let rgb_units: Vec<_> = units.into_iter().filter(|u| u.interface == 0).collect();
        let cam = CameraDevice {
            node: PathBuf::from("/dev/video0"),
            card: String::new(),
            driver: String::new(),
            bus: String::new(),
            kind: CameraKind::Rgb,
            formats: vec![],
            usb: None,
            extension_units: rgb_units,
            metadata_node: None,
        };
        let result = MsxuIrEmitter::discover(&cam);
        assert!(matches!(
            result,
            Err(EmitterError::NoControl { ref node }) if node == Path::new("/dev/video0")
        ));
    }

    #[test]
    fn test_guard_restores_exact_prior_payload() -> Result<(), EmitterError> {
        let xu = ir_fake();
        xu.update(4, 6, |c| {
            c.cur = vec![1, 3, 1, 7, 0, 0, 0, 0, 5];
            c.def = vec![1, 3, 1, 7, 0, 0, 0, 0, 5];
        });
        {
            let guard = EmitterGuard::with_opener(
                PathBuf::from("/dev/video2"),
                IR,
                MODE_ON_DEFAULT,
                opener(&xu),
            )?;
            assert_eq!(xu.control(4, 6).cur, vec![1, 3, 2, 7, 0, 0, 0, 0, 5]);
            assert_eq!(guard.prior(), &[1, 3, 1, 7, 0, 0, 0, 0, 5][..]);
            assert_eq!(xu.handles(), 2);
        }
        assert_eq!(xu.control(4, 6).cur, vec![1, 3, 1, 7, 0, 0, 0, 0, 5]);
        assert_eq!(
            xu.control(4, 6).writes,
            vec![
                vec![1, 3, 2, 7, 0, 0, 0, 0, 5],
                vec![1, 3, 1, 7, 0, 0, 0, 0, 5]
            ]
        );
        assert_eq!(xu.handles(), 1);
        Ok(())
    }

    #[test]
    fn test_guard_keeps_on_when_it_was_on() -> Result<(), EmitterError> {
        let xu = ir_fake();
        xu.update(4, 6, |c| {
            c.cur = vec![1, 3, 3, 0, 0, 0, 0, 0, 0];
            c.def = vec![1, 3, 3, 0, 0, 0, 0, 0, 0];
        });
        {
            let _guard = EmitterGuard::with_opener(
                PathBuf::from("/dev/video2"),
                IR,
                MODE_ON_DEFAULT,
                opener(&xu),
            )?;
        }
        assert_eq!(xu.control(4, 6).cur, vec![1, 3, 3, 0, 0, 0, 0, 0, 0]);
        assert_eq!(xu.control(4, 6).writes.len(), 2);
        Ok(())
    }

    #[test]
    fn test_guard_failed_apply_still_restores() -> Result<(), EmitterError> {
        let xu = ir_fake();
        let result = EmitterGuard::with_opener(PathBuf::from("/dev/video2"), IR, 0x04, opener(&xu));
        assert!(matches!(
            result,
            Err(EmitterError::ModeOutOfRange {
                requested: 4,
                max: 3
            })
        ));
        assert_eq!(
            xu.control(4, 6).writes,
            vec![vec![1, 3, 1, 0, 0, 0, 0, 0, 0]]
        );
        Ok(())
    }

    #[test]
    fn test_guard_restore_failure_does_not_panic() -> Result<(), EmitterError> {
        let xu = ir_fake();
        let guard = EmitterGuard::with_opener(
            PathBuf::from("/dev/video2"),
            IR,
            MODE_ON_DEFAULT,
            opener(&xu),
        )?;
        xu.update(4, 6, |c| c.ignore_writes = true);
        drop(guard);
        assert_eq!(xu.control(4, 6).cur[2], 2);
        Ok(())
    }

    #[test]
    fn test_guard_reapply_writes_mode_again() -> Result<(), EmitterError> {
        let xu = ir_fake();
        let guard = EmitterGuard::with_opener(
            PathBuf::from("/dev/video2"),
            IR,
            MODE_ON_DEFAULT,
            opener(&xu),
        )?;
        xu.update(4, 6, |c| c.cur[2] = 1);
        guard.reapply()?;
        assert_eq!(xu.control(4, 6).cur[2], 2);
        assert_eq!(xu.control(4, 6).writes.len(), 2);
        Ok(())
    }

    #[test]
    fn test_guard_and_emitter_are_send() {
        fn f<T: Send>() {}
        f::<EmitterGuard>();
        f::<EmitterGuard<FakeXu>>();
        f::<MsxuIrEmitter>();
    }

    #[test]
    #[ignore = "needs hardware"]
    fn test_open_metadata_node_is_no_such_camera() {
        let result = MsxuIrEmitter::open(Path::new("/dev/video1"));
        assert!(matches!(result, Err(EmitterError::NoSuchCamera { .. })));
        let e = MsxuIrEmitter::open(Path::new("/dev/video2")).unwrap();
        assert_eq!(
            e.control(),
            EmitterControl {
                unit: 4,
                selector: 6
            }
        );
    }

    #[test]
    #[ignore = "needs hardware"]
    fn test_live_guard_restores_prior_payload() -> Result<(), EmitterError> {
        let cameras = V4l2CameraProbe::new().cameras()?;
        let ir = cameras
            .into_iter()
            .find(|c| c.kind == CameraKind::Ir)
            .expect("an IR camera");
        let before = MsxuIrEmitter::discover(&ir)?.payload()?;
        let g = EmitterGuard::enable(&ir)?;
        assert_eq!(MsxuIrEmitter::discover(&ir)?.mode()?, MODE_ON_DEFAULT);
        drop(g);
        assert_eq!(MsxuIrEmitter::discover(&ir)?.payload()?, before);
        Ok(())
    }
}
