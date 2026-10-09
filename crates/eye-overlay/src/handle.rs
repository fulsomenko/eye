//! A join-and-message handle to the overlay thread.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::JoinHandle;

use std::sync::mpsc::TrySendError;

use smithay_client_toolkit::reexports::calloop;

use crate::error::OverlayError;

pub struct OverlayHandle<M: Send + 'static> {
    pub(crate) tx: Option<calloop::channel::SyncSender<M>>,
    pub(crate) thread: Option<JoinHandle<Result<(), OverlayError>>>,
    pub(crate) dropped: Arc<AtomicU64>,
    pub(crate) logical_size: (u32, u32),
}

impl<M: Send + 'static> fmt::Debug for OverlayHandle<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OverlayHandle")
            .field("dropped", &self.dropped.load(Ordering::Relaxed))
            .field("logical_size", &self.logical_size)
            .finish_non_exhaustive()
    }
}

impl<M: Send + 'static> OverlayHandle<M> {
    /// Never blocks. A full queue drops `msg` (counted); `Err(Closed)` once the thread has exited.
    pub fn send(&self, msg: M) -> Result<(), OverlayError> {
        let Some(tx) = &self.tx else {
            return Err(OverlayError::Closed);
        };
        match tx.try_send(msg) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_)) => {
                let dropped_total = self.dropped.fetch_add(1, Ordering::Relaxed) + 1;
                tracing::warn!(dropped_total, "gaze point dropped, overlay queue full");
                Ok(())
            }
            Err(TrySendError::Disconnected(_)) => Err(OverlayError::Closed),
        }
    }

    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    pub fn logical_size(&self) -> (u32, u32) {
        self.logical_size
    }

    /// Drops the sender (the loop sees `Closed`) and joins, returning the thread's result.
    pub fn close(mut self) -> Result<(), OverlayError> {
        self.tx.take();
        match self.thread.take() {
            Some(thread) => thread.join().unwrap_or(Err(OverlayError::Panicked)),
            None => Ok(()),
        }
    }

    /// A handle with no thread, for tests of code that sends to an overlay.
    #[cfg(test)]
    pub(crate) fn detached(tx: calloop::channel::SyncSender<M>) -> Self {
        Self {
            tx: Some(tx),
            thread: None,
            dropped: Arc::new(AtomicU64::new(0)),
            logical_size: (0, 0),
        }
    }
}

impl<M: Send + 'static> Drop for OverlayHandle<M> {
    fn drop(&mut self) {
        self.tx.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use eye_log::Value;
    use eye_log::testing::capture_logs;
    use smithay_client_toolkit::reexports::calloop;

    use super::*;

    #[test]
    fn test_handle_full_queue_drops_newest_and_counts() {
        let (tx, _rx) = calloop::channel::sync_channel::<u32>(2);
        let handle = OverlayHandle::detached(tx);
        for i in 0..5 {
            assert!(handle.send(i).is_ok());
        }
        assert_eq!(handle.dropped(), 3);
    }

    #[test]
    fn test_logs_gaze_point_dropped_at_warn() {
        let (tx, _rx) = calloop::channel::sync_channel::<u32>(2);
        let handle = OverlayHandle::detached(tx);

        let (_, records) = capture_logs(tracing::Level::TRACE, || {
            for i in 0..5 {
                handle.send(i).expect("never errors on a full queue");
            }
        });

        let warns: Vec<_> = records
            .iter()
            .filter(|r| r.level == eye_log::Level::Warn)
            .collect();
        assert_eq!(warns.len(), 3, "{records:?}");
        assert_eq!(warns.last().unwrap().fields["dropped_total"], Value::U64(3));
    }

    #[test]
    fn test_handle_send_after_receiver_gone_returns_closed() {
        let (tx, rx) = calloop::channel::sync_channel::<u32>(2);
        let handle = OverlayHandle::detached(tx);
        drop(rx);
        assert!(matches!(handle.send(1), Err(OverlayError::Closed)));
    }
}
