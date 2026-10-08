use std::os::fd::{AsRawFd, BorrowedFd};

#[repr(C)]
pub(super) struct UvcXuControlQuery {
    unit: u8,
    selector: u8,
    query: u8,
    size: u16,
    data: *mut u8,
}

#[cfg(target_pointer_width = "64")]
const _: () = assert!(std::mem::size_of::<UvcXuControlQuery>() == 16);

nix::ioctl_readwrite!(uvcioc_ctrl_query, b'u', 0x21, UvcXuControlQuery);

pub(super) fn ctrl_query(
    fd: BorrowedFd<'_>,
    unit: u8,
    selector: u8,
    query: u8,
    data: &mut [u8],
) -> nix::Result<()> {
    let size = u16::try_from(data.len()).map_err(|_| nix::errno::Errno::EOVERFLOW)?;
    let mut q = UvcXuControlQuery {
        unit,
        selector,
        query,
        size,
        data: data.as_mut_ptr(),
    };
    // SAFETY: `q.data` points to `size` writable bytes borrowed for the whole call,
    // and `fd` is a live descriptor borrowed from the caller.
    unsafe { uvcioc_ctrl_query(fd.as_raw_fd(), &mut q) }.map(drop)
}

pub(super) const V4L2_BUF_TYPE_META_CAPTURE: u32 = 13;

/// `struct v4l2_format` with the `fmt.meta` arm spelled out (x86_64 layout).
#[repr(C)]
pub(super) struct V4l2MetaFormat {
    type_: u32,
    _union_align: u32,
    dataformat: u32,
    buffersize: u32,
    _rest: [u8; 192],
}

#[cfg(target_pointer_width = "64")]
const _: () = assert!(std::mem::size_of::<V4l2MetaFormat>() == 208);

nix::ioctl_readwrite!(vidioc_g_fmt, b'V', 4, V4l2MetaFormat);
nix::ioctl_readwrite!(vidioc_s_fmt, b'V', 5, V4l2MetaFormat);

pub(super) fn meta_format(fd: BorrowedFd<'_>, set: Option<u32>) -> nix::Result<u32> {
    let mut f = V4l2MetaFormat {
        type_: V4L2_BUF_TYPE_META_CAPTURE,
        _union_align: 0,
        dataformat: set.unwrap_or(0),
        buffersize: 0,
        _rest: [0; 192],
    };
    // SAFETY: `f` is a complete 208-byte `struct v4l2_format` whose `fmt.meta` arm is initialised;
    // the kernel reads and writes only inside it, and `fd` is a live descriptor borrowed from the caller.
    unsafe {
        match set {
            Some(_) => vidioc_s_fmt(fd.as_raw_fd(), &mut f),
            None => vidioc_g_fmt(fd.as_raw_fd(), &mut f),
        }
    }?;
    Ok(f.dataformat)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_code_matches_kernel_uvcioc_ctrl_query() {
        #[cfg(target_pointer_width = "64")]
        assert_eq!(
            nix::request_code_readwrite!(b'u', 0x21, std::mem::size_of::<UvcXuControlQuery>())
                as u32,
            0xC010_7521
        );
    }

    #[test]
    fn test_query_struct_field_offsets_match_c_layout() {
        assert_eq!(std::mem::offset_of!(UvcXuControlQuery, size), 4);
        assert_eq!(std::mem::offset_of!(UvcXuControlQuery, data), 8);
    }

    #[test]
    fn test_fmt_request_codes_match_kernel() {
        assert_eq!(
            nix::request_code_readwrite!(b'V', 4, std::mem::size_of::<V4l2MetaFormat>()) as u32,
            0xC0D0_5604
        );
        assert_eq!(
            nix::request_code_readwrite!(b'V', 5, std::mem::size_of::<V4l2MetaFormat>()) as u32,
            0xC0D0_5605
        );
        assert_eq!(std::mem::offset_of!(V4l2MetaFormat, dataformat), 8);
    }
}
