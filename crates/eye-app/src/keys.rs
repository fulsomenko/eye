//! Reads `r`/`s` from the launching terminal while a target sequence is shown, so the operator
//! can flag a bad presentation without touching the click-through, non-focusable overlay. A
//! no-op (disconnected receiver, no thread, terminal untouched) when stdin is not a TTY, so
//! scripts and tests are unaffected.

use std::io::IsTerminal;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crossbeam_channel::{Receiver, Sender};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::terminal;
use signal_hook::consts::signal::SIGINT;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReviewKey {
    Retake,
    Skip,
}

/// What a raw key read becomes: a review key for the pump, a signal to raise because raw mode
/// (which clears ISIG) stops the terminal from generating it itself, or nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KeyAction {
    Review(ReviewKey),
    RaiseSigint,
    Ignore,
}

fn decide(key: KeyEvent) -> KeyAction {
    match key.code {
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            KeyAction::RaiseSigint
        }
        KeyCode::Char('r') => KeyAction::Review(ReviewKey::Retake),
        KeyCode::Char('s') => KeyAction::Review(ReviewKey::Skip),
        _ => KeyAction::Ignore,
    }
}

/// Whether `watch_loop` keeps polling after a key action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LoopStep {
    Continue,
    Stop,
}

/// Side effects are passed in as closures so a test can assert ordering without a real TTY or
/// signal: `RaiseSigint` disables raw mode before raising, since nothing runs after `Stop`.
fn apply_action(
    action: KeyAction,
    tx: &Sender<ReviewKey>,
    disable_raw_mode: impl FnOnce(),
    raise_sigint: impl FnOnce(),
) -> LoopStep {
    match action {
        KeyAction::Review(msg) => {
            let _ = tx.send(msg);
            LoopStep::Continue
        }
        KeyAction::RaiseSigint => {
            disable_raw_mode();
            raise_sigint();
            LoopStep::Stop
        }
        KeyAction::Ignore => LoopStep::Continue,
    }
}

const POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Watches stdin on a dedicated thread for the lifetime of this value; disables raw mode and
/// joins the thread on drop, from any exit path. `enable_raw_mode` clears ISIG, so a Ctrl-C
/// keypress would otherwise never reach the process as SIGINT while this is active; `watch_loop`
/// detects it itself and raises SIGINT, so the existing `shutdown::install` handle still fires.
#[derive(Debug)]
pub struct KeyWatcher {
    rx: Receiver<ReviewKey>,
    active: bool,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl KeyWatcher {
    pub fn spawn() -> Self {
        Self::spawn_if(std::io::stdin().is_terminal())
    }

    /// `spawn` with the TTY decision passed in, so tests can exercise both branches without
    /// depending on the process's actual stdin.
    pub fn spawn_if(is_tty: bool) -> Self {
        let (tx, rx) = crossbeam_channel::unbounded();
        if !is_tty {
            return Self::inactive(rx);
        }
        if let Err(error) = terminal::enable_raw_mode() {
            tracing::warn!(%error, "failed to enable raw mode; retake/skip keys disabled");
            return Self::inactive(rx);
        }
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = Arc::clone(&stop);
        let thread = std::thread::spawn(move || watch_loop(&tx, &thread_stop));
        Self {
            rx,
            active: true,
            stop,
            thread: Some(thread),
        }
    }

    fn inactive(rx: Receiver<ReviewKey>) -> Self {
        Self {
            rx,
            active: false,
            stop: Arc::new(AtomicBool::new(false)),
            thread: None,
        }
    }

    pub fn receiver(&self) -> Receiver<ReviewKey> {
        self.rx.clone()
    }

    pub fn is_active(&self) -> bool {
        self.active
    }
}

fn watch_loop(tx: &Sender<ReviewKey>, stop: &AtomicBool) {
    while !stop.load(Ordering::Relaxed) {
        match event::poll(POLL_INTERVAL) {
            Ok(true) => match event::read() {
                Ok(Event::Key(key)) => {
                    let step = apply_action(
                        decide(key),
                        tx,
                        || {
                            if let Err(error) = terminal::disable_raw_mode() {
                                tracing::warn!(%error, "failed to restore terminal raw mode");
                            }
                        },
                        || {
                            if let Err(error) = signal_hook::low_level::raise(SIGINT) {
                                tracing::warn!(%error, "failed to raise SIGINT for ctrl-c");
                            }
                        },
                    );
                    if step == LoopStep::Stop {
                        return;
                    }
                }
                Ok(_) => {}
                Err(error) => {
                    tracing::warn!(%error, "key read failed; retake/skip keys disabled");
                    return;
                }
            },
            Ok(false) => {}
            Err(error) => {
                tracing::warn!(%error, "key poll failed; retake/skip keys disabled");
                return;
            }
        }
    }
}

impl Drop for KeyWatcher {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        if let Err(error) = terminal::disable_raw_mode() {
            tracing::warn!(%error, "failed to restore terminal raw mode");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keys_ignored_when_stdin_is_not_a_tty() {
        let watcher = KeyWatcher::spawn_if(false);

        assert!(!watcher.is_active());
        assert_eq!(
            watcher.receiver().recv_timeout(Duration::from_millis(50)),
            Err(crossbeam_channel::RecvTimeoutError::Disconnected)
        );
    }

    #[test]
    fn test_decide_maps_ctrl_c_to_raise_sigint() {
        let key = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert_eq!(decide(key), KeyAction::RaiseSigint);
    }

    #[test]
    fn test_decide_maps_plain_c_to_ignore() {
        let key = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::NONE);
        assert_eq!(decide(key), KeyAction::Ignore);
    }

    #[test]
    fn test_decide_maps_r_and_s_to_review_keys() {
        assert_eq!(
            decide(KeyEvent::new(KeyCode::Char('r'), KeyModifiers::NONE)),
            KeyAction::Review(ReviewKey::Retake)
        );
        assert_eq!(
            decide(KeyEvent::new(KeyCode::Char('s'), KeyModifiers::NONE)),
            KeyAction::Review(ReviewKey::Skip)
        );
    }

    #[test]
    fn test_raise_sigint_disables_raw_mode_before_raising_and_stops_loop() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let order = std::cell::RefCell::new(Vec::new());

        let step = apply_action(
            KeyAction::RaiseSigint,
            &tx,
            || order.borrow_mut().push("disable_raw_mode"),
            || order.borrow_mut().push("raise_sigint"),
        );

        assert_eq!(*order.borrow(), vec!["disable_raw_mode", "raise_sigint"]);
        assert_eq!(step, LoopStep::Stop);
    }

    #[test]
    fn test_review_key_continues_loop_without_touching_terminal() {
        let (tx, rx) = crossbeam_channel::unbounded();

        let step = apply_action(
            KeyAction::Review(ReviewKey::Retake),
            &tx,
            || panic!("must not disable raw mode for a review key"),
            || panic!("must not raise SIGINT for a review key"),
        );

        assert_eq!(step, LoopStep::Continue);
        assert_eq!(rx.try_recv(), Ok(ReviewKey::Retake));
    }

    #[test]
    fn test_decide_ignores_other_keys() {
        assert_eq!(
            decide(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE)),
            KeyAction::Ignore
        );
    }
}
