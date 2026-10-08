use std::time::{Duration, Instant};

use eye_capture::{FrameSource, TaggedSource, TaggerConfig};
use eye_core::{Frame, Illumination, PixelFormat, Timestamp};

use crate::{
    case::{TestCtx, TestError, TestRegistry},
    mode::{Mode, Role},
};

pub mod dual;
pub mod ir;
pub mod stream;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameMeta {
    pub seq: u64,
    pub t: Timestamp,
    pub received: Timestamp,
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    pub illumination: Illumination,
    /// Gray8 only (subsampled, step 4); NaN otherwise.
    pub mean: f64,
    pub payload_ok: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Grab {
    pub frames: Vec<FrameMeta>,
    pub first_frame: Duration,
}

/// Reads `n` frames and keeps metadata only: no pixel data outlives this call (D11).
pub fn grab(ctx: &TestCtx, source: &mut dyn FrameSource, n: usize) -> Result<Grab, TestError> {
    let start = Instant::now();
    let mut first_frame = None;
    let mut frames = Vec::with_capacity(n);
    while frames.len() < n {
        ctx.check()?;
        let frame = source.next_frame()?;
        let received = Timestamp::now();
        first_frame.get_or_insert_with(|| start.elapsed());
        frames.push(meta(&frame, received));
    }
    Ok(Grab {
        frames,
        first_frame: first_frame.unwrap_or_default(),
    })
}

pub fn meta(frame: &Frame, received: Timestamp) -> FrameMeta {
    let h = frame.header();
    let data = frame.data();
    let pixels = h.width as usize * h.height as usize;
    let payload_ok = match h.format {
        PixelFormat::Gray8 => data.len() == pixels,
        PixelFormat::Mjpeg => data.starts_with(&[0xFF, 0xD8]),
        PixelFormat::Rgb8 => data.len() == pixels * 3,
    };
    let mean = if h.format == PixelFormat::Gray8 && payload_ok {
        eye_capture::mean_brightness(data, h.width, h.height, 4)
    } else {
        f64::NAN
    };
    FrameMeta {
        seq: h.seq,
        t: h.timestamp,
        received,
        width: h.width,
        height: h.height,
        format: h.format,
        illumination: h.illumination,
        mean,
        payload_ok,
    }
}

/// How IR frames get their lit/dark tag. `Auto` = the pipeline's way: FrameIllumination metadata
/// when the session offers it, brightness otherwise (R30). `Brightness` = physical evidence only.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tagging {
    #[default]
    Auto,
    Brightness,
}

/// Opens the IR stream wrapped in the default tagger.
pub fn tagged_ir(
    ctx: &TestCtx,
    tagging: Tagging,
) -> Result<TaggedSource<Box<dyn FrameSource>>, TestError> {
    let raw = ctx.session().open(Role::Ir)?;
    let meta = match tagging {
        Tagging::Auto => ctx.session().open_meta()?,
        Tagging::Brightness => None,
    };
    Ok(match meta {
        Some(meta) => TaggedSource::with_metadata(raw, meta, TaggerConfig::default())?,
        None => TaggedSource::new(raw, TaggerConfig::default())?,
    })
}

/// Which streams of the mode a single-stream case examines.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RoleSel {
    Rgb,
    Ir,
    #[default]
    All,
}

impl RoleSel {
    pub fn needs(self) -> crate::case::Needs {
        use crate::case::Needs;
        match self {
            RoleSel::Rgb => Needs {
                rgb: true,
                ..Needs::default()
            },
            RoleSel::Ir => Needs {
                ir: true,
                ..Needs::default()
            },
            RoleSel::All => Needs {
                any_stream: true,
                ..Needs::default()
            },
        }
    }

    /// The selected roles present in `mode`, RGB first.
    pub fn roles(self, mode: &Mode) -> Vec<Role> {
        [Role::Rgb, Role::Ir]
            .into_iter()
            .filter(|&r| mode.target(r).is_some())
            .filter(|&r| {
                matches!(
                    (self, r),
                    (RoleSel::All, _) | (RoleSel::Rgb, Role::Rgb) | (RoleSel::Ir, Role::Ir)
                )
            })
            .collect()
    }
}

pub fn prefix(role: Role) -> &'static str {
    match role {
        Role::Rgb => "rgb",
        Role::Ir => "ir",
    }
}

pub fn register(r: &mut TestRegistry) {
    r.register(
        "opens",
        "every selected stream opens and delivers frames",
        stream::build_opens,
    );
    r.register(
        "measured_fps",
        "frame rate from buffer timestamps, interval jitter, stalls",
        stream::build_measured_fps,
    );
    r.register(
        "monotonic_timestamps",
        "buffer timestamps increase and are recent CLOCK_MONOTONIC",
        stream::build_monotonic_timestamps,
    );
    r.register(
        "frame_format",
        "frame size, format and payload match the mode",
        stream::build_frame_format,
    );
    r.register(
        "ir_alternation",
        "IR lit/dark alternation and contrast with emitter on, none with off",
        ir::build_ir_alternation,
    );
    r.register(
        "emitter_toggle",
        "emitter on/off round trip with read-back and stream check",
        ir::build_emitter_toggle,
    );
    r.register(
        "dual_sync",
        "RGB vs IR timestamp offset, dark pairing, rates and drift while both stream",
        dual::build_dual_sync,
    );
}
