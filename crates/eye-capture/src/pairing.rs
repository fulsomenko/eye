//! Pairs IR and RGB frames for the fused estimator (R30).

use std::time::Duration;

use eye_core::{
    CameraId, CameraInfo, Frame, FrameSet, Illumination, PixelFormat, Timestamp, log::field,
};

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
    /// Gray8 (IR) camera and the ids distinct; else `CaptureError::Pairing`. `PairingConfig::default()`.
    pub fn new(cameras: &[CameraInfo]) -> Result<Self, CaptureError> {
        Self::with_config(cameras, PairingConfig::default())
    }

    pub fn with_config(cameras: &[CameraInfo], cfg: PairingConfig) -> Result<Self, CaptureError> {
        let ids = || {
            cameras
                .iter()
                .map(|c| c.id.clone())
                .collect::<Vec<CameraId>>()
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
            [_, _] => Err(CaptureError::Pairing {
                cameras: ids(),
                reason: "a camera pair needs a Gray8 primary and two distinct ids",
            }),
            _ => Err(CaptureError::Pairing {
                cameras: ids(),
                reason: "Pairer supports 1 or 2 cameras",
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
            tracing::warn!(
                { field::CAMERA } = frame.header().camera.as_str(),
                { field::SEQ } = frame.header().seq,
                "frame from a camera the pairer does not know"
            );
            return vec![FrameSet::single(frame)];
        };
        let Some(cfg) = self.cfg else {
            log_alone(&frame, "single_camera");
            return vec![FrameSet::single(frame)];
        };
        let mut out = Vec::new();
        if side == 1 {
            if let Some(old) = self.held.take() {
                log_alone(&old, "superseded");
                out.push(FrameSet::single(old));
            }
            let t = shifted(&frame, cfg.offset_secondary_ns);
            if self.last_lit.is_some_and(|lit| lit.as_nanos() as i128 > t) {
                log_alone(&frame, "lit_primary_passed");
                out.push(FrameSet::single(frame));
            } else {
                let h = frame.header();
                tracing::debug!(
                    { field::CAMERA } = h.camera.as_str(),
                    { field::SEQ } = h.seq,
                    "frame held"
                );
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
                tracing::debug!(
                    { field::CAMERA } = frame.header().camera.as_str(),
                    { field::SEQ } = frame.header().seq,
                    secondary_camera = held.header().camera.as_str(),
                    secondary_seq = held.header().seq,
                    delta_ns = d as i64,
                    "frames paired"
                );
                out.push(
                    FrameSet::new(vec![frame, held]).expect("a pair holds two distinct cameras"),
                );
                return out;
            }
            if d > cfg.bracket_window.as_nanos() as i128 {
                log_alone(&held, "bracket_window_expired");
                out.push(FrameSet::single(held));
            } else {
                self.held = Some(held);
            }
        }
        if lit {
            self.last_lit = Some(self.last_lit.map_or(t, |l| l.max(t)));
        }
        log_alone(&frame, "unpaired_primary");
        out.push(FrameSet::single(frame));
        out
    }

    /// Emits the held secondary frame alone (end of stream, or the caller decided a camera stalled).
    pub fn flush(&mut self) -> Option<FrameSet> {
        self.held.take().map(|frame| {
            log_alone(&frame, "flush");
            FrameSet::single(frame)
        })
    }
}

fn log_alone(frame: &Frame, reason: &'static str) {
    let h = frame.header();
    tracing::debug!(
        { field::CAMERA } = h.camera.as_str(),
        { field::SEQ } = h.seq,
        { field::REASON } = reason,
        "frame emitted alone"
    );
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
            Err(CaptureError::Pairing { .. })
        ));
        assert!(matches!(
            Pairer::new(&[]),
            Err(CaptureError::Pairing { .. })
        ));
        let dup = camera_info("ir", PixelFormat::Gray8, 4, 2);
        assert!(matches!(
            Pairer::new(&[ir_info, dup]),
            Err(CaptureError::Pairing { .. })
        ));
    }

    #[test]
    fn test_pairing_error_names_both_cameras() {
        let ir_info = camera_info("ir", PixelFormat::Mjpeg, 1280, 720);
        let rgb_info = camera_info("rgb", PixelFormat::Mjpeg, 1280, 720);
        let err = Pairer::new(&[ir_info, rgb_info]).unwrap_err();
        assert!(matches!(
            &err,
            CaptureError::Pairing { cameras, reason }
                if *cameras == vec![CameraId::from("ir"), CameraId::from("rgb")]
                    && *reason == "a camera pair needs a Gray8 primary and two distinct ids"
        ));
        assert_eq!(
            err.to_string(),
            "cameras [ir, rgb]: a camera pair needs a Gray8 primary and two distinct ids"
        );
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

    #[test]
    fn test_logs_unknown_camera_at_warn_with_seq() {
        let (_, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            let mut pairer = Pairer::new(&infos()).unwrap();
            pairer.push(rgb(0, 3 * T - 3_000_000));
            let mut depth_frame = gray_frame("depth", 0, 3 * T, 4, 2, 0);
            depth_frame.set_illumination(Illumination::Unknown);
            pairer.push(depth_frame);
        });
        let warn: Vec<_> = records
            .iter()
            .filter(|r| r.message == "frame from a camera the pairer does not know")
            .collect();
        assert_eq!(warn.len(), 1);
        assert_eq!(warn[0].level, eye_log::Level::Warn);
        assert_eq!(warn[0].fields[field::SEQ], eye_log::Value::U64(0));
    }

    #[test]
    fn test_logs_frames_paired_at_debug() {
        let (_, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            drive(Pairer::new(&infos()).unwrap(), r30(9))
        });
        let paired: Vec<_> = records
            .iter()
            .filter(|r| r.message == "frames paired")
            .collect();
        assert_eq!(paired.len(), 3);
        for rec in &paired {
            assert_eq!(rec.level, eye_log::Level::Debug);
            assert_eq!(
                rec.fields["secondary_camera"],
                eye_log::Value::Str("rgb".to_string())
            );
            match rec.fields["delta_ns"] {
                eye_log::Value::I64(d) => assert!(d > 0),
                ref other => panic!("expected I64, got {other:?}"),
            }
        }
        let held_indices: Vec<usize> = records
            .iter()
            .enumerate()
            .filter(|(_, r)| r.message == "frame held")
            .map(|(i, _)| i)
            .collect();
        let paired_indices: Vec<usize> = records
            .iter()
            .enumerate()
            .filter(|(_, r)| r.message == "frames paired")
            .map(|(i, _)| i)
            .collect();
        assert!(held_indices.len() >= paired_indices.len());
        let mut remaining_held = held_indices.clone();
        for &paired_idx in &paired_indices {
            let pos = remaining_held
                .iter()
                .position(|&h| h < paired_idx)
                .expect("a `frame held` record precedes each `frames paired`");
            remaining_held.remove(pos);
        }
    }

    #[test]
    fn test_logs_frame_emitted_alone_at_debug_with_reason() {
        let frames = vec![
            ir(1, T, Illumination::Ambient),
            rgb(0, T - 3_000_000),
            ir(2, 2 * T, Illumination::Ambient),
            ir(3, 3 * T, Illumination::Ambient),
        ];
        let (flushed, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            let mut pairer = Pairer::new(&infos()).unwrap();
            for frame in frames {
                pairer.push(frame);
            }
            pairer.flush()
        });
        assert!(flushed.is_none());
        let alone: Vec<_> = records
            .iter()
            .filter(|r| r.message == "frame emitted alone")
            .collect();
        for rec in &alone {
            assert_eq!(rec.level, eye_log::Level::Debug);
        }
        let reasons: Vec<eye_log::Value> = alone
            .iter()
            .map(|r| r.fields[field::REASON].clone())
            .collect();
        assert_eq!(
            reasons,
            vec![
                eye_log::Value::Str("unpaired_primary".to_string()),
                eye_log::Value::Str("unpaired_primary".to_string()),
                eye_log::Value::Str("bracket_window_expired".to_string()),
                eye_log::Value::Str("unpaired_primary".to_string()),
            ]
        );
    }

    #[test]
    fn test_logs_flush_emits_held_secondary_alone_at_debug() {
        let (flushed, records) = eye_log::testing::capture_logs(tracing::Level::TRACE, || {
            let mut pairer = Pairer::new(&infos()).unwrap();
            pairer.push(rgb(0, 3 * T - 3_000_000));
            pairer.flush()
        });
        assert!(flushed.is_some());
        let alone: Vec<_> = records
            .iter()
            .filter(|r| r.message == "frame emitted alone")
            .collect();
        assert_eq!(alone.len(), 1);
        assert_eq!(alone[0].level, eye_log::Level::Debug);
        assert_eq!(
            alone[0].fields[field::REASON],
            eye_log::Value::Str("flush".to_string())
        );
        assert_eq!(
            alone[0].fields[field::CAMERA],
            eye_log::Value::Str("rgb".to_string())
        );
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
