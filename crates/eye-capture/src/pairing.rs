//! Pairs IR and RGB frames for the fused estimator (R30).

use std::time::Duration;

use eye_core::{CameraId, CameraInfo, Frame, FrameSet, Illumination, PixelFormat, Timestamp};

use crate::CaptureError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PairingConfig {
    /// Added to secondary timestamps before every comparison. R30 measured RGB = dark IR - 3.0 ms;
    /// `[capture] rgb_offset_ns` feeds it, and the bracket rule does not depend on it.
    pub offset_secondary_ns: i64,
    /// A held secondary frame pairs with the first lit primary frame at most this long after it.
    pub bracket_window: Duration,
}

impl Default for PairingConfig {
    /// offset 0, window 100 ms (the next lit IR frame is ~71 ms after an RGB frame; the one after that is ~200 ms).
    fn default() -> Self {
        Self {
            offset_secondary_ns: 0,
            bracket_window: Duration::from_millis(100),
        }
    }
}

#[derive(Debug)]
pub struct Pairer {
    cameras: Vec<CameraId>,
    cfg: Option<PairingConfig>,
    held: Option<Frame>,
    last_lit: Option<Timestamp>,
}

impl Pairer {
    /// 1 camera: every frame is emitted at once. 2 cameras: `[primary, secondary]`, the primary must be the
    /// Gray8 (IR) camera and the ids distinct; else `CaptureError::Config`. `PairingConfig::default()`.
    pub fn new(cameras: &[CameraInfo]) -> Result<Self, CaptureError> {
        Self::with_config(cameras, PairingConfig::default())
    }

    pub fn with_config(cameras: &[CameraInfo], cfg: PairingConfig) -> Result<Self, CaptureError> {
        let ids = || {
            cameras
                .iter()
                .map(|c| c.id.to_string())
                .collect::<Vec<_>>()
                .join(",")
        };
        match cameras {
            [one] => Ok(Self {
                cameras: vec![one.id.clone()],
                cfg: None,
                held: None,
                last_lit: None,
            }),
            [primary, secondary]
                if primary.format == PixelFormat::Gray8 && primary.id != secondary.id =>
            {
                Ok(Self {
                    cameras: vec![primary.id.clone(), secondary.id.clone()],
                    cfg: Some(cfg),
                    held: None,
                    last_lit: None,
                })
            }
            [_, _] => Err(CaptureError::Config {
                camera: ids(),
                reason: "a camera pair needs a Gray8 primary and two distinct ids".into(),
            }),
            _ => Err(CaptureError::Config {
                camera: ids(),
                reason: "Pairer supports 1 or 2 cameras".into(),
            }),
        }
    }

    /// Accepts the next frame in arrival order; returns the sets completed by it, oldest first (0, 1 or 2).
    pub fn push(&mut self, frame: Frame) -> Vec<FrameSet> {
        let Some(side) = self
            .cameras
            .iter()
            .position(|id| *id == frame.header().camera)
        else {
            tracing::warn!(camera = %frame.header().camera, "frame from a camera the pairer does not know");
            return vec![FrameSet::single(frame)];
        };
        let Some(cfg) = self.cfg else {
            return vec![FrameSet::single(frame)];
        };
        let mut out = Vec::new();
        if side == 1 {
            if let Some(old) = self.held.take() {
                out.push(FrameSet::single(old));
            }
            let t = shifted(&frame, cfg.offset_secondary_ns);
            if self.last_lit.is_some_and(|lit| lit.as_nanos() as i128 > t) {
                out.push(FrameSet::single(frame));
            } else {
                self.held = Some(frame);
            }
            return out;
        }
        let t = frame.header().timestamp;
        let lit = frame.header().illumination == Illumination::IrLit;
        if let Some(held) = self.held.take() {
            let d = t.as_nanos() as i128 - shifted(&held, cfg.offset_secondary_ns);
            if lit && d > 0 && d <= cfg.bracket_window.as_nanos() as i128 {
                self.last_lit = Some(self.last_lit.map_or(t, |l| l.max(t)));
                out.push(
                    FrameSet::new(vec![frame, held]).expect("a pair holds two distinct cameras"),
                );
                return out;
            }
            if d > cfg.bracket_window.as_nanos() as i128 {
                out.push(FrameSet::single(held));
            } else {
                self.held = Some(held);
            }
        }
        if lit {
            self.last_lit = Some(self.last_lit.map_or(t, |l| l.max(t)));
        }
        out.push(FrameSet::single(frame));
        out
    }

    /// Emits the held secondary frame alone (end of stream, or the caller decided a camera stalled).
    pub fn flush(&mut self) -> Option<FrameSet> {
        self.held.take().map(FrameSet::single)
    }
}

fn shifted(frame: &Frame, offset_ns: i64) -> i128 {
    frame.header().timestamp.as_nanos() as i128 + i128::from(offset_ns)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::testing::{camera_info, gray_frame, mjpeg_frame};

    const T: u64 = 68_000_000;

    fn ir(seq: u64, t_ns: u64, illumination: Illumination) -> Frame {
        let mut frame = gray_frame("ir", seq, t_ns, 4, 2, 0);
        frame.set_illumination(illumination);
        frame
    }

    fn rgb(seq: u64, t_ns: u64) -> Frame {
        mjpeg_frame("rgb", seq, t_ns, &[0xff, 0xd8, 0xff, 0xd9])
    }

    fn infos() -> [CameraInfo; 2] {
        [
            camera_info("ir", PixelFormat::Gray8, 4, 2),
            camera_info("rgb", PixelFormat::Mjpeg, 1280, 720),
        ]
    }

    fn drive(mut pairer: Pairer, frames: Vec<Frame>) -> Vec<Vec<(String, u64)>> {
        let mut sets = Vec::new();
        for frame in frames {
            sets.extend(pairer.push(frame));
        }
        if let Some(set) = pairer.flush() {
            sets.push(set);
        }
        sets.iter().map(set_seqs).collect()
    }

    fn set_seqs(set: &FrameSet) -> Vec<(String, u64)> {
        set.frames()
            .iter()
            .map(|f| (f.header().camera.to_string(), f.header().seq))
            .collect()
    }

    /// IR seq `s` in `1..=n_ir` at `s * T`, `IrDark` for `s <= 3` (warm-up) and for odd `s`,
    /// `IrLit` for even `s >= 4`; RGB `j` (via `.enumerate()` over `(3..=n_ir).step_by(2)`) at
    /// `(2j + 3) * T - 3_000_000`; sorted by timestamp.
    fn r30(n_ir: u64) -> Vec<Frame> {
        let mut frames: Vec<Frame> = (1..=n_ir)
            .map(|s| {
                let illumination = if s <= 3 || s % 2 == 1 {
                    Illumination::IrDark
                } else {
                    Illumination::IrLit
                };
                ir(s, s * T, illumination)
            })
            .collect();
        for (j, _) in (3..=n_ir).step_by(2).enumerate() {
            let j = j as u64;
            frames.push(rgb(j, (2 * j + 3) * T - 3_000_000));
        }
        frames.sort_by_key(|f| f.header().timestamp);
        frames
    }

    fn seqs(sets: &[&str]) -> Vec<Vec<(String, u64)>> {
        sets.iter()
            .map(|s| {
                s.split(',')
                    .filter(|p| !p.is_empty())
                    .map(|p| {
                        let (camera, seq) = p.split_once(':').unwrap();
                        (camera.to_string(), seq.parse().unwrap())
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_rgb_pairs_with_next_lit_ir_frame() {
        let sets = drive(Pairer::new(&infos()).unwrap(), r30(9));
        let expected = seqs(&[
            "ir:1",
            "ir:2",
            "ir:3",
            "ir:4,rgb:0",
            "ir:5",
            "ir:6,rgb:1",
            "ir:7",
            "ir:8,rgb:2",
            "ir:9",
            "rgb:3",
        ]);
        assert_eq!(sets, expected);
    }

    #[test]
    fn test_single_camera_emits_immediately() {
        let mut pairer = Pairer::new(&infos()[..1]).unwrap();
        let out = pairer.push(ir(1, T, Illumination::Unknown));
        assert_eq!(out.len(), 1);
        assert!(pairer.flush().is_none());
    }

    #[test]
    fn test_untagged_primary_never_pairs_and_releases_stale_rgb() {
        let frames = vec![
            ir(1, T, Illumination::Ambient),
            rgb(0, T - 3_000_000),
            ir(2, 2 * T, Illumination::Ambient),
            ir(3, 3 * T, Illumination::Ambient),
        ];
        let sets = drive(Pairer::new(&infos()).unwrap(), frames);
        let expected = seqs(&["ir:1", "ir:2", "rgb:0", "ir:3"]);
        assert_eq!(sets, expected);
    }

    #[test]
    fn test_rgb_after_its_next_lit_frame_is_emitted_alone() {
        let mut pairer = Pairer::new(&infos()).unwrap();

        let ir4 = pairer.push(ir(4, 4 * T, Illumination::IrLit));
        assert_eq!(
            ir4.iter().map(set_seqs).collect::<Vec<_>>(),
            seqs(&["ir:4"])
        );

        let rgb0 = pairer.push(rgb(0, 3 * T - 3_000_000));
        assert_eq!(
            rgb0.iter().map(set_seqs).collect::<Vec<_>>(),
            seqs(&["rgb:0"])
        );

        assert!(pairer.flush().is_none());
    }

    #[test]
    fn test_second_rgb_releases_first() {
        let frames = vec![rgb(0, T), rgb(1, 2 * T)];
        let sets = drive(Pairer::new(&infos()).unwrap(), frames);
        let expected = seqs(&["rgb:0", "rgb:1"]);
        assert_eq!(sets, expected);
    }

    #[test]
    fn test_offset_is_applied_to_secondary() {
        let cfg = PairingConfig {
            offset_secondary_ns: -40_000_000,
            ..Default::default()
        };
        let frames = vec![rgb(0, 3 * T - 3_000_000), ir(4, 4 * T, Illumination::IrLit)];
        let sets = drive(Pairer::with_config(&infos(), cfg).unwrap(), frames.clone());
        let expected = seqs(&["rgb:0", "ir:4"]);
        assert_eq!(sets, expected);

        let sets = drive(Pairer::new(&infos()).unwrap(), frames);
        let expected = seqs(&["ir:4,rgb:0"]);
        assert_eq!(sets, expected);
    }

    #[test]
    fn test_rgb_primary_is_rejected() {
        let rgb_info = camera_info("rgb", PixelFormat::Mjpeg, 1280, 720);
        let ir_info = camera_info("ir", PixelFormat::Gray8, 4, 2);
        assert!(matches!(
            Pairer::new(&[rgb_info, ir_info.clone()]),
            Err(CaptureError::Config { .. })
        ));
        assert!(matches!(Pairer::new(&[]), Err(CaptureError::Config { .. })));
        let dup = camera_info("ir", PixelFormat::Gray8, 4, 2);
        assert!(matches!(
            Pairer::new(&[ir_info, dup]),
            Err(CaptureError::Config { .. })
        ));
    }

    #[test]
    fn test_unknown_camera_is_emitted_alone() {
        let mut pairer = Pairer::new(&infos()).unwrap();

        let out = pairer.push(rgb(0, 3 * T - 3_000_000));
        assert!(out.is_empty());

        let mut depth_frame = gray_frame("depth", 0, 3 * T, 4, 2, 0);
        depth_frame.set_illumination(Illumination::Unknown);
        let out = pairer.push(depth_frame);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].frames().len(), 1);
        assert_eq!(out[0].frames()[0].header().camera.as_str(), "depth");

        let out = pairer.push(ir(4, 4 * T, Illumination::IrLit));
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].frames().len(), 2);
    }

    proptest! {
        #[test]
        fn prop_every_frame_in_exactly_one_set(
            n in 4u64..60,
            drop_mask in proptest::collection::vec(proptest::bool::weighted(0.1), 60),
            lit_even in proptest::bool::ANY,
            jitter in proptest::collection::vec(0u64..10_000_000, 120),
        ) {
            let mut input: Vec<Frame> = Vec::new();
            let mut jitter_iter = jitter.into_iter();
            for k in 0..n {
                let dropped = drop_mask.get(k as usize).copied().unwrap_or(false);
                if dropped {
                    continue;
                }
                let lit = (k % 2 == 0) == lit_even;
                let illumination = if lit { Illumination::IrLit } else { Illumination::IrDark };
                let t_ir = (k + 1) * T;
                input.push(ir(k, t_ir, illumination));
                if !lit {
                    input.push(rgb(k, t_ir - 3_000_000));
                }
            }

            let mut keyed: Vec<(u128, Frame)> = input
                .into_iter()
                .map(|f| {
                    let jitter_ns = jitter_iter.next().unwrap_or(0);
                    let key = f.header().timestamp.as_nanos() as u128 + jitter_ns as u128;
                    (key, f)
                })
                .collect();
            keyed.sort_by_key(|(key, _)| *key);
            let push_order: Vec<Frame> = keyed.into_iter().map(|(_, f)| f).collect();

            let mut expected: Vec<(String, u64)> = push_order
                .iter()
                .map(|f| (f.header().camera.to_string(), f.header().seq))
                .collect();
            expected.sort();
            let lit_timestamps: Vec<Timestamp> = push_order
                .iter()
                .filter(|f| {
                    f.header().camera.as_str() == "ir"
                        && f.header().illumination == Illumination::IrLit
                })
                .map(|f| f.header().timestamp)
                .collect();

            type Emitted = (String, u64, Timestamp, Illumination);
            let to_emitted = |set: &FrameSet| -> Vec<Emitted> {
                set.frames()
                    .iter()
                    .map(|f| {
                        let header = f.header();
                        (
                            header.camera.to_string(),
                            header.seq,
                            header.timestamp,
                            header.illumination,
                        )
                    })
                    .collect()
            };

            let mut pairer = Pairer::new(&infos()).unwrap();
            let mut emitted: Vec<Vec<Emitted>> = Vec::new();
            for frame in push_order {
                for set in pairer.push(frame) {
                    emitted.push(to_emitted(&set));
                }
            }
            if let Some(set) = pairer.flush() {
                emitted.push(to_emitted(&set));
            }

            let mut actual: Vec<(String, u64)> = emitted
                .iter()
                .flatten()
                .map(|(camera, seq, _, _)| (camera.clone(), *seq))
                .collect();
            actual.sort();
            prop_assert_eq!(actual, expected);

            for set in &emitted {
                if set.len() == 2 {
                    let (ir_camera, _, t_ir, ir_illumination) = &set[0];
                    let (rgb_camera, _, t_rgb, _) = &set[1];
                    prop_assert_eq!(ir_camera.as_str(), "ir");
                    prop_assert_eq!(*ir_illumination, Illumination::IrLit);
                    prop_assert_eq!(rgb_camera.as_str(), "rgb");
                    let d = t_ir.nanos_since(*t_rgb);
                    prop_assert!(d > 0 && d <= 100_000_000);
                    let between = lit_timestamps
                        .iter()
                        .any(|&t| t.nanos_since(*t_rgb) > 0 && t.nanos_since(*t_ir) < 0);
                    prop_assert!(!between);
                }
            }
        }
    }
}
