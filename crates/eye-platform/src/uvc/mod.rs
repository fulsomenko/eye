#[allow(unsafe_code)]
mod sys;

use std::os::fd::AsFd;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum XuQuery {
    SetCur = 0x01,
    GetCur = 0x81,
    GetMin = 0x82,
    GetMax = 0x83,
    GetRes = 0x84,
    GetLen = 0x85,
    GetInfo = 0x86,
    GetDef = 0x87,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XuInfo(pub u8);

impl XuInfo {
    pub fn supports_get(self) -> bool {
        self.0 & 0x01 != 0
    }
    pub fn supports_set(self) -> bool {
        self.0 & 0x02 != 0
    }
}

#[derive(Debug, thiserror::Error)]
pub enum XuError {
    #[error("open {path}: {source}")]
    Open {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("unit {unit} selector {selector} does not exist")]
    NotFound { unit: u8, selector: u8 },
    #[error("{query:?} on unit {unit} selector {selector}: {errno}")]
    Ioctl {
        unit: u8,
        selector: u8,
        query: XuQuery,
        errno: nix::errno::Errno,
    },
    #[error("payload of {len} bytes does not fit the u16 size field")]
    TooLong { len: usize },
    #[error("metadata format on {path}: {errno}")]
    MetaFormat {
        path: PathBuf,
        errno: nix::errno::Errno,
    },
}

/// The UVC metadata format that carries the Microsoft per-frame items (FrameIllumination).
pub const META_FORMAT_UVCM: [u8; 4] = *b"UVCM";

/// `VIDIOC_G_FMT` on a metadata node: the current dataformat fourcc.
pub fn meta_format(node: &Path) -> Result<[u8; 4], XuError> {
    meta_format_io(node, None)
}

/// `VIDIOC_S_FMT` on a metadata node; returns the fourcc the driver now reports. EBUSY while it streams.
pub fn set_meta_format(node: &Path, fourcc: [u8; 4]) -> Result<[u8; 4], XuError> {
    meta_format_io(node, Some(fourcc))
}

fn meta_format_io(node: &Path, set: Option<[u8; 4]>) -> Result<[u8; 4], XuError> {
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(node)
        .map_err(|source| XuError::Open {
            path: node.to_owned(),
            source,
        })?;
    sys::meta_format(file.as_fd(), set.map(u32::from_le_bytes))
        .map(u32::to_le_bytes)
        .map_err(|errno| XuError::MetaFormat {
            path: node.to_owned(),
            errno,
        })
}

pub trait XuTransport {
    fn query(&self, unit: u8, selector: u8, query: XuQuery, data: &mut [u8])
    -> Result<(), XuError>;

    fn len(&self, unit: u8, selector: u8) -> Result<u16, XuError> {
        let mut b = [0u8; 2];
        self.query(unit, selector, XuQuery::GetLen, &mut b)?;
        Ok(u16::from_le_bytes(b))
    }

    fn info(&self, unit: u8, selector: u8) -> Result<XuInfo, XuError> {
        let mut b = [0u8; 1];
        self.query(unit, selector, XuQuery::GetInfo, &mut b)?;
        Ok(XuInfo(b[0]))
    }

    fn get(&self, unit: u8, selector: u8, query: XuQuery) -> Result<Vec<u8>, XuError> {
        let mut buf = vec![0u8; usize::from(self.len(unit, selector)?)];
        self.query(unit, selector, query, &mut buf)?;
        Ok(buf)
    }

    fn set_cur(&self, unit: u8, selector: u8, data: &[u8]) -> Result<(), XuError> {
        let mut buf = data.to_vec();
        self.query(unit, selector, XuQuery::SetCur, &mut buf)
    }
}

#[derive(Debug)]
pub struct UvcXuDevice {
    file: std::fs::File,
    path: PathBuf,
}

impl UvcXuDevice {
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, XuError> {
        let path = path.into();
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|source| XuError::Open {
                path: path.clone(),
                source,
            })?;
        Ok(Self { file, path })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl XuTransport for UvcXuDevice {
    fn query(
        &self,
        unit: u8,
        selector: u8,
        query: XuQuery,
        data: &mut [u8],
    ) -> Result<(), XuError> {
        let len = data.len();
        sys::ctrl_query(self.file.as_fd(), unit, selector, query as u8, data).map_err(|errno| {
            match errno {
                nix::errno::Errno::ENOENT => XuError::NotFound { unit, selector },
                nix::errno::Errno::EOVERFLOW => XuError::TooLong { len },
                errno => XuError::Ioctl {
                    unit,
                    selector,
                    query,
                    errno,
                },
            }
        })
    }
}

#[cfg(test)]
#[derive(Debug, Clone, Default)]
pub(crate) struct FakeControl {
    pub info: u8,
    pub cur: Vec<u8>,
    pub min: Vec<u8>,
    pub max: Vec<u8>,
    pub res: Vec<u8>,
    pub def: Vec<u8>,
    pub writes: Vec<Vec<u8>>,
    pub ignore_writes: bool,
}

#[cfg(test)]
#[derive(Debug, Clone, Default)]
pub(crate) struct FakeXu {
    controls: std::sync::Arc<std::sync::Mutex<std::collections::HashMap<(u8, u8), FakeControl>>>,
}

#[cfg(test)]
impl FakeXu {
    pub(crate) fn with_control(self, unit: u8, selector: u8, control: FakeControl) -> Self {
        self.controls
            .lock()
            .unwrap()
            .insert((unit, selector), control);
        self
    }

    /// Snapshot of one control (panics if absent; test-only).
    pub(crate) fn control(&self, unit: u8, selector: u8) -> FakeControl {
        self.controls.lock().unwrap()[&(unit, selector)].clone()
    }

    #[allow(dead_code)]
    pub(crate) fn update(&self, unit: u8, selector: u8, f: impl FnOnce(&mut FakeControl)) {
        f(self
            .controls
            .lock()
            .unwrap()
            .get_mut(&(unit, selector))
            .unwrap());
    }

    /// Number of live clones sharing this device state (the test and every open transport).
    pub(crate) fn handles(&self) -> usize {
        std::sync::Arc::strong_count(&self.controls)
    }
}

#[cfg(test)]
impl XuTransport for FakeXu {
    fn query(
        &self,
        unit: u8,
        selector: u8,
        query: XuQuery,
        data: &mut [u8],
    ) -> Result<(), XuError> {
        let mut controls = self.controls.lock().unwrap();
        let c = controls
            .get_mut(&(unit, selector))
            .ok_or(XuError::NotFound { unit, selector })?;
        let fail = |errno| XuError::Ioctl {
            unit,
            selector,
            query,
            errno,
        };
        let expected = match query {
            XuQuery::GetLen => 2,
            XuQuery::GetInfo => 1,
            _ => c.cur.len(),
        };
        if data.len() != expected {
            return Err(fail(nix::errno::Errno::EINVAL));
        }
        match query {
            XuQuery::GetLen => data.copy_from_slice(&(c.cur.len() as u16).to_le_bytes()),
            XuQuery::GetInfo => data[0] = c.info,
            XuQuery::SetCur if c.info & 0x02 == 0 => return Err(fail(nix::errno::Errno::EIO)),
            XuQuery::SetCur => {
                c.writes.push(data.to_vec());
                if !c.ignore_writes {
                    c.cur = data.to_vec();
                }
            }
            XuQuery::GetCur => data.copy_from_slice(&c.cur),
            XuQuery::GetMin => data.copy_from_slice(&c.min),
            XuQuery::GetMax => data.copy_from_slice(&c.max),
            XuQuery::GetRes => data.copy_from_slice(&c.res),
            XuQuery::GetDef => data.copy_from_slice(&c.def),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ir_fake() -> FakeXu {
        FakeXu::default().with_control(
            4,
            6,
            FakeControl {
                info: 0x03,
                cur: vec![1, 3, 1, 0, 0, 0, 0, 0, 0],
                min: vec![0; 9],
                max: vec![1, 3, 3, 0, 0, 0, 0, 0, 0],
                res: vec![0; 9],
                def: vec![1, 3, 1, 0, 0, 0, 0, 0, 0],
                writes: vec![],
                ignore_writes: false,
            },
        )
    }

    #[test]
    fn test_xu_query_codes_match_uvc_spec() {
        assert_eq!(XuQuery::SetCur as u8, 0x01);
        assert_eq!(XuQuery::GetCur as u8, 0x81);
        assert_eq!(XuQuery::GetLen as u8, 0x85);
        assert_eq!(XuQuery::GetInfo as u8, 0x86);
        assert_eq!(XuQuery::GetDef as u8, 0x87);
    }

    #[test]
    fn test_xu_info_bits_decode_get_and_set() {
        assert!(XuInfo(0x03).supports_get());
        assert!(XuInfo(0x03).supports_set());
        assert!(XuInfo(0x01).supports_get());
        assert!(!XuInfo(0x01).supports_set());
    }

    #[test]
    fn test_get_uses_len_then_reads_payload() {
        let fake = ir_fake();
        assert_eq!(
            fake.get(4, 6, XuQuery::GetCur).unwrap(),
            vec![1, 3, 1, 0, 0, 0, 0, 0, 0]
        );
    }

    #[test]
    fn test_unknown_selector_is_not_found() {
        let fake = ir_fake();
        let result = fake.get(3, 1, XuQuery::GetCur);
        assert!(matches!(
            result,
            Err(XuError::NotFound {
                unit: 3,
                selector: 1
            })
        ));
    }

    #[test]
    fn test_set_cur_records_write() {
        let fake = ir_fake();
        fake.set_cur(4, 6, &[1, 3, 2, 0, 0, 0, 0, 0, 0]).unwrap();
        assert_eq!(
            fake.get(4, 6, XuQuery::GetCur).unwrap(),
            vec![1, 3, 2, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(fake.control(4, 6).writes.len(), 1);
    }

    #[test]
    fn test_wrong_length_is_einval() {
        let fake = ir_fake();
        let result = fake.query(4, 6, XuQuery::GetCur, &mut [0u8; 4]);
        assert!(matches!(
            result,
            Err(XuError::Ioctl {
                errno: nix::errno::Errno::EINVAL,
                ..
            })
        ));
    }

    #[test]
    fn test_clone_shares_device_state() {
        let a = ir_fake();
        let b = a.clone();
        b.set_cur(4, 6, &[1, 3, 2, 0, 0, 0, 0, 0, 0]).unwrap();
        assert_eq!(a.get(4, 6, XuQuery::GetCur).unwrap()[2], 2);
        assert_eq!(a.handles(), 2);
    }

    #[test]
    fn test_meta_format_on_non_v4l2_file_is_meta_format_error() {
        let result = meta_format(Path::new("/dev/null"));
        assert!(matches!(result, Err(XuError::MetaFormat { .. })));

        let result = meta_format(Path::new("/nonexistent"));
        assert!(matches!(result, Err(XuError::Open { .. })));
    }

    #[test]
    #[ignore = "needs hardware"]
    fn test_live_metadata_node_reports_a_uvc_format() {
        let fourcc = meta_format(Path::new("/dev/video3")).unwrap();
        assert!(fourcc == *b"UVCM" || fourcc == *b"UVCH");
    }

    #[test]
    #[ignore = "needs hardware"]
    fn test_live_set_meta_format_uvcm_roundtrips() {
        assert_eq!(
            set_meta_format(Path::new("/dev/video3"), META_FORMAT_UVCM).unwrap(),
            *b"UVCM"
        );
        assert_eq!(meta_format(Path::new("/dev/video3")).unwrap(), *b"UVCM");
    }

    #[test]
    fn test_oversized_buffer_is_too_long() {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open("/dev/null")
            .unwrap();
        let mut data = vec![0u8; 70000];
        let result = sys::ctrl_query(file.as_fd(), 0, 0, 0, &mut data);
        assert_eq!(result, Err(nix::errno::Errno::EOVERFLOW));
    }

    #[test]
    #[ignore = "needs hardware"]
    fn test_live_ir_msxu_face_auth_reads_match_inv() {
        let dev = UvcXuDevice::open("/dev/video2").unwrap();
        assert_eq!(dev.len(4, 6).unwrap(), 9);
        assert_eq!(dev.info(4, 6).unwrap(), XuInfo(0x03));
        assert_eq!(
            dev.get(4, 6, XuQuery::GetMax).unwrap(),
            vec![1, 3, 3, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(
            dev.get(4, 6, XuQuery::GetDef).unwrap(),
            vec![1, 3, 1, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(dev.len(4, 9).unwrap(), 4);
        let result = dev.get(3, 1, XuQuery::GetCur);
        assert!(matches!(result, Err(XuError::NotFound { .. })));
    }
}
