use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::JoinHandle;

use crossbeam_channel::{Receiver, Sender, TrySendError};
use eye_capture::{CaptureError, FrameSource};
use eye_core::log::field;
use eye_core::{CameraId, Frame};

use crate::tracker::TrackerError;

#[derive(Debug)]
pub(crate) enum CaptureMsg {
    Frame(Frame),
    Failed(CaptureError),
}

#[derive(Debug)]
pub(crate) struct CaptureThread {
    pub(crate) camera: CameraId,
    pub(crate) rx: Receiver<CaptureMsg>,
    pub(crate) dropped: Arc<AtomicU64>,
    pub(crate) handle: JoinHandle<()>,
}

pub(crate) fn spawn_capture(
    mut source: Box<dyn FrameSource>,
    capacity: usize,
    stop: Arc<AtomicBool>,
) -> Result<CaptureThread, TrackerError> {
    let camera = source.camera().id.clone();
    let (tx, rx) = crossbeam_channel::bounded(capacity.max(1));
    let evict = rx.clone();
    let dropped = Arc::new(AtomicU64::new(0));
    let counter = Arc::clone(&dropped);
    let parent = tracing::Span::current();
    let handle = std::thread::Builder::new()
        .name(format!("eye-cap-{camera}"))
        .spawn(move || {
            let _parent = parent.entered();
            capture_loop(source.as_mut(), &tx, &evict, &stop, &counter);
        })
        .map_err(TrackerError::Spawn)?;
    Ok(CaptureThread {
        camera,
        rx,
        dropped,
        handle,
    })
}

pub(crate) fn capture_loop(
    source: &mut dyn FrameSource,
    tx: &Sender<CaptureMsg>,
    evict: &Receiver<CaptureMsg>,
    stop: &AtomicBool,
    dropped: &AtomicU64,
) {
    let camera = source.camera().id.clone();
    while !stop.load(Ordering::Acquire) {
        let msg = match source.next_frame() {
            Ok(frame) => CaptureMsg::Frame(frame),
            Err(CaptureError::Timeout { .. }) => continue,
            Err(CaptureError::EndOfStream) => {
                tracing::debug!(
                    { field::CAMERA } = camera.as_str(),
                    { field::REASON } = "end_of_stream",
                    "capture ended"
                );
                break;
            }
            Err(e) => {
                tracing::error!(
                    { field::CAMERA } = camera.as_str(),
                    error = %e,
                    "capture failed"
                );
                send_drop_oldest(tx, evict, CaptureMsg::Failed(e), dropped);
                break;
            }
        };
        send_drop_oldest(tx, evict, msg, dropped);
    }
}

pub(crate) fn send_drop_oldest(
    tx: &Sender<CaptureMsg>,
    evict: &Receiver<CaptureMsg>,
    msg: CaptureMsg,
    dropped: &AtomicU64,
) {
    let msg = match tx.try_send(msg) {
        Ok(()) | Err(TrySendError::Disconnected(_)) => return,
        Err(TrySendError::Full(msg)) => msg,
    };
    if let Ok(CaptureMsg::Frame(evicted)) = evict.try_recv() {
        let total = dropped.fetch_add(1, Ordering::Relaxed) + 1;
        let h = evicted.header();
        tracing::warn!(
            { field::CAMERA } = h.camera.as_str(),
            { field::SEQ } = h.seq,
            dropped = total,
            "frame dropped: capture channel full"
        );
    }
    let _ = tx.try_send(msg);
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicU64;

    use eye_core::Illumination;

    use super::*;
    use crate::testkit;

    #[test]
    fn test_send_drop_oldest_evicts_oldest_and_counts() {
        let (tx, rx) = crossbeam_channel::bounded(2);
        let evict = rx.clone();
        let dropped = AtomicU64::new(0);
        for seq in 0..5u64 {
            let frame = testkit::frame("ir", seq, seq, Illumination::IrLit);
            send_drop_oldest(&tx, &evict, CaptureMsg::Frame(frame), &dropped);
        }
        let remaining: Vec<u64> = rx
            .try_iter()
            .map(|msg| match msg {
                CaptureMsg::Frame(frame) => frame.header().timestamp.as_nanos() / 1_000_000,
                CaptureMsg::Failed(_) => unreachable!("only frames were sent"),
            })
            .collect();
        assert_eq!(remaining, vec![3, 4]);
        assert_eq!(dropped.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn test_logs_frame_dropped_at_warn_with_seq() {
        let (_, records) = eye_log::testing::capture_logs(tracing::Level::WARN, || {
            let (tx, rx) = crossbeam_channel::bounded(2);
            let evict = rx.clone();
            let dropped = AtomicU64::new(0);
            for seq in 0..5u64 {
                let frame = testkit::frame("ir", seq, seq, Illumination::IrLit);
                send_drop_oldest(&tx, &evict, CaptureMsg::Frame(frame), &dropped);
            }
        });
        let warns: Vec<_> = records
            .iter()
            .filter(|r| r.message == "frame dropped: capture channel full")
            .collect();
        assert_eq!(warns.len(), 3);
        let seqs: Vec<&eye_log::Value> = warns
            .iter()
            .map(|r| r.fields.get(field::SEQ).expect("seq present"))
            .collect();
        assert_eq!(
            seqs,
            vec![
                &eye_log::Value::U64(0),
                &eye_log::Value::U64(1),
                &eye_log::Value::U64(2),
            ]
        );
        let dropped_totals: Vec<&eye_log::Value> = warns
            .iter()
            .map(|r| r.fields.get("dropped").expect("dropped present"))
            .collect();
        assert_eq!(
            dropped_totals,
            vec![
                &eye_log::Value::U64(1),
                &eye_log::Value::U64(2),
                &eye_log::Value::U64(3),
            ]
        );
        for rec in &warns {
            assert_eq!(rec.level, eye_log::Level::Warn);
        }
    }

    #[test]
    fn test_logs_capture_ended_at_debug_and_capture_failed_at_error() {
        let (_, ended_records) = eye_log::testing::capture_logs(tracing::Level::DEBUG, || {
            let (tx, rx) = crossbeam_channel::bounded(8);
            let evict = rx.clone();
            let dropped = AtomicU64::new(0);
            let stop = AtomicBool::new(false);
            let mut source = testkit::script_source(vec![]);
            capture_loop(&mut source, &tx, &evict, &stop, &dropped);
        });
        let ended = ended_records
            .iter()
            .find(|r| r.message == "capture ended")
            .expect("debug record present");
        assert_eq!(ended.level, eye_log::Level::Debug);
        assert_eq!(
            ended.fields.get(field::REASON),
            Some(&eye_log::Value::Str("end_of_stream".to_string()))
        );

        let (_, failed_records) = eye_log::testing::capture_logs(tracing::Level::ERROR, || {
            let (tx, rx) = crossbeam_channel::bounded(8);
            let evict = rx.clone();
            let dropped = AtomicU64::new(0);
            let stop = AtomicBool::new(false);
            let mut source = testkit::script_source(vec![Err(CaptureError::Disconnected {
                camera: "ir".to_string(),
            })]);
            capture_loop(&mut source, &tx, &evict, &stop, &dropped);
        });
        let failed = failed_records
            .iter()
            .find(|r| r.message == "capture failed")
            .expect("error record present");
        assert_eq!(failed.level, eye_log::Level::Error);
        assert_eq!(
            failed.fields.get(field::CAMERA),
            Some(&eye_log::Value::Str("ir".to_string()))
        );
    }
}
