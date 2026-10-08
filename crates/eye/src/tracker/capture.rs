use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::JoinHandle;

use crossbeam_channel::{Receiver, Sender, TrySendError};
use eye_capture::{CaptureError, FrameSource};
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
    let handle = std::thread::Builder::new()
        .name(format!("eye-cap-{camera}"))
        .spawn(move || {
            while !stop.load(Ordering::Acquire) {
                let msg = match source.next_frame() {
                    Ok(frame) => CaptureMsg::Frame(frame),
                    Err(CaptureError::Timeout { .. }) => continue,
                    Err(CaptureError::EndOfStream) => break,
                    Err(e) => {
                        send_drop_oldest(&tx, &evict, CaptureMsg::Failed(e), &counter);
                        break;
                    }
                };
                send_drop_oldest(&tx, &evict, msg, &counter);
            }
        })
        .map_err(TrackerError::Spawn)?;
    Ok(CaptureThread {
        camera,
        rx,
        dropped,
        handle,
    })
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
    if let Ok(CaptureMsg::Frame(_)) = evict.try_recv() {
        dropped.fetch_add(1, Ordering::Relaxed);
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
}
