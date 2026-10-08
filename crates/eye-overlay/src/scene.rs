//! What the overlay thread draws each frame, and when it should draw next.

use std::time::Instant;

use eye_core::Timestamp;

use crate::canvas::Canvas;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Schedule {
    Idle,
    NextFrame,
    At(Instant),
    Exit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresentedAt {
    Presentation(Timestamp),
    Commit(Timestamp),
}

pub trait Scene: Send + 'static {
    type Msg: Send + 'static;

    fn on_msg(&mut self, msg: Self::Msg, now: Instant);

    /// Called once per compositor frame callback. `true` keeps the callback chain alive
    /// (another `render` is needed); scenes with nothing to animate keep the default `false`.
    fn step(&mut self, _now: Instant) -> bool {
        false
    }

    /// The canvas has already been cleared where this buffer was last drawn.
    fn render(&mut self, canvas: &mut Canvas<'_>, now: Instant) -> Schedule;

    /// Asked right after `render`. `Some(mark)` requests presentation feedback for this commit.
    fn presentation_mark(&self) -> Option<u64> {
        None
    }

    fn on_presented(&mut self, _mark: u64, _at: PresentedAt, _now: Instant) -> Schedule {
        Schedule::Idle
    }
}
