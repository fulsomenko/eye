use crossbeam_channel::Receiver;
use signal_hook::iterator::{Handle, Signals};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    Soft,
    Force,
}

#[derive(Debug, Default)]
pub struct ShutdownPolicy {
    requested: bool,
}

impl ShutdownPolicy {
    pub fn on_signal(&mut self) -> Decision {
        if std::mem::replace(&mut self.requested, true) {
            Decision::Force
        } else {
            Decision::Soft
        }
    }
}

/// After this drops, SIGINT and SIGTERM stay ignored for the rest of the
/// process: `signal-hook-registry` cannot restore the previous (default)
/// handler on unregister, so callers must keep the handle alive for as long
/// as they want the signals handled.
#[derive(Debug)]
pub struct ShutdownHandle {
    rx: Receiver<()>,
    signals: Handle,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl ShutdownHandle {
    pub fn receiver(&self) -> &Receiver<()> {
        &self.rx
    }
}

impl Drop for ShutdownHandle {
    fn drop(&mut self) {
        self.signals.close();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

pub fn install() -> anyhow::Result<ShutdownHandle> {
    use signal_hook::consts::signal::{SIGINT, SIGTERM};

    let mut signals = Signals::new([SIGINT, SIGTERM])?;
    let handle = signals.handle();
    let (tx, rx) = crossbeam_channel::bounded(1);
    let thread = eye_core::log::spawn_in_current_span("eye-signals", move || {
        let mut policy = ShutdownPolicy::default();
        for signal in signals.forever() {
            match policy.on_signal() {
                Decision::Force => std::process::exit(130),
                Decision::Soft => {
                    tracing::info!(signal, "shutting down; send the signal again to force");
                    let _ = tx.try_send(());
                }
            }
        }
    })?;
    Ok(ShutdownHandle {
        rx,
        signals: handle,
        thread: Some(thread),
    })
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use signal_hook::consts::signal::SIGTERM;

    use super::*;

    #[test]
    #[ignore = "raises a real SIGTERM in the test process and signal-hook keeps its handler installed afterwards"]
    fn test_sigterm_delivers_shutdown() {
        let handle = install().unwrap();
        signal_hook::low_level::raise(SIGTERM).unwrap();
        assert_eq!(
            handle.receiver().recv_timeout(Duration::from_secs(2)),
            Ok(())
        );
    }

    #[test]
    fn test_policy_first_signal_soft_then_force() {
        let mut p = ShutdownPolicy::default();
        assert_eq!(p.on_signal(), Decision::Soft);
        assert_eq!(p.on_signal(), Decision::Force);
        assert_eq!(p.on_signal(), Decision::Force);
    }

    #[test]
    fn test_handle_drop_joins_thread() {
        let (done_tx, done_rx) = crossbeam_channel::bounded(1);
        std::thread::spawn(move || {
            let handle = install().unwrap();
            let rx = handle.receiver().clone();
            drop(handle);
            let _ = done_tx.send(rx.try_recv());
        });
        let result = done_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("dropping ShutdownHandle did not join the signal thread within 2 s");
        assert!(matches!(
            result,
            Err(crossbeam_channel::TryRecvError::Disconnected)
        ));
    }
}
