//! Shows a sequence of dot targets and timestamps when each became visible and was replaced.

use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use eye_core::log::field;
use eye_core::session::TargetClock;
use eye_core::session::TargetTiming;
use eye_core::{OutputId, Timestamp};
use nalgebra::{Matrix2, Point2};

use crate::canvas::{Canvas, Rgba};
use crate::ellipse::{K95, confidence_ellipse};
use crate::error::OverlayError;
use crate::handle::OverlayHandle;
use crate::scene::{PresentedAt, Scene, Schedule};
use crate::surface::{SurfaceOptions, spawn};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TargetSpec {
    pub px_logical: Point2<f64>,
    pub timing: TargetTiming,
    /// Set on a target re-presented via `TargetDisplay::append` after an online rejection; draws
    /// with an amber settle ring instead of white so the user knows this dot is a retry.
    pub retry: bool,
}

impl From<(Point2<f64>, TargetTiming)> for TargetSpec {
    fn from((px_logical, timing): (Point2<f64>, TargetTiming)) -> Self {
        Self {
            px_logical,
            timing,
            retry: false,
        }
    }
}

impl From<(Point2<f64>, Duration)> for TargetSpec {
    fn from((px_logical, dwell): (Point2<f64>, Duration)) -> Self {
        Self {
            px_logical,
            timing: TargetTiming {
                settle: dwell * 2 / 5,
                window: dwell * 3 / 5,
                dwell,
            },
            retry: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TargetShown {
    pub index: usize,
    pub output: OutputId,
    pub px_logical: Point2<f64>,
    pub shown_at: Timestamp,
    pub clock: TargetClock,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TargetEvent {
    Shown(TargetShown),
    Hidden {
        index: usize,
        at: Timestamp,
        clock: TargetClock,
    },
    Finished,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Feedback {
    pub px_logical: Point2<f64>,
    pub cov_px: Matrix2<f64>,
    /// `false` while drawn from the nominal (uncalibrated) mapping; `true` once a live refit applied.
    pub calibrated: bool,
    pub at: Timestamp,
}

#[derive(Debug, Clone)]
pub struct FeedbackSender(crossbeam_channel::Sender<Feedback>);

impl FeedbackSender {
    pub fn new(sender: crossbeam_channel::Sender<Feedback>) -> Self {
        Self(sender)
    }

    /// Never blocks (the channel is unbounded); a disconnected (closed) receiver is silently ignored.
    pub fn send(&self, feedback: Feedback) {
        let _ = self.0.send(feedback);
    }
}

/// A message on the channel a running `TargetDisplay` drains to learn about online-rejection
/// retries. `Settled` marks that the app has fully reacted to one `Hidden` event (whether or not
/// it appended a retry), so a scene waiting to finish can tell "no more appends are coming" apart
/// from "none have arrived yet".
#[derive(Debug, Clone, PartialEq)]
pub enum AppendMsg {
    Append(TargetSpec),
    Settled,
}

/// A cloneable handle for appending targets to a running `TargetDisplay` from another thread,
/// e.g. a `PumpObserver` that does not hold the `TargetDisplay` itself.
#[derive(Debug, Clone)]
pub struct AppendSender(crossbeam_channel::Sender<AppendMsg>);

impl AppendSender {
    pub fn new(sender: crossbeam_channel::Sender<AppendMsg>) -> Self {
        Self(sender)
    }

    pub fn append(&self, spec: TargetSpec) -> Result<(), TargetsError> {
        validate(std::slice::from_ref(&spec))?;
        let _ = self.0.send(AppendMsg::Append(spec));
        Ok(())
    }

    /// Call exactly once per `Hidden` event received, whether or not it led to an `append`.
    pub fn settle(&self) {
        let _ = self.0.send(AppendMsg::Settled);
    }
}

#[derive(Debug, thiserror::Error)]
pub enum TargetsError {
    #[error("no targets")]
    Empty,
    #[error("target {index} has a non-finite position or zero dwell")]
    InvalidTarget { index: usize },
    #[error("target {index} at {px:?} lies outside the {size:?} output")]
    OutOfBounds {
        index: usize,
        px: Point2<f64>,
        size: (u32, u32),
    },
    #[error(transparent)]
    Overlay(#[from] OverlayError),
}

#[derive(Debug)]
pub struct TargetDisplay {
    events: crossbeam_channel::Receiver<TargetEvent>,
    handle: OverlayHandle<std::convert::Infallible>,
    failure: Arc<Mutex<Option<TargetsError>>>,
    appends: crossbeam_channel::Sender<AppendMsg>,
}

impl TargetDisplay {
    /// Validates (before connecting), connects on `output` with namespace "eye-targets", shows `lead_in` of blank,
    /// then the targets in order.
    pub fn spawn(
        output: &OutputId,
        lead_in: Duration,
        targets: Vec<TargetSpec>,
        track_append: bool,
    ) -> Result<Self, TargetsError> {
        let (display, _feedback) =
            Self::spawn_inner(output, lead_in, targets, false, track_append)?;
        Ok(display)
    }

    /// Like `spawn`, but the returned `FeedbackSender` lets the caller push live `Feedback` for
    /// the scene to draw under the current target.
    pub fn spawn_with_feedback(
        output: &OutputId,
        lead_in: Duration,
        targets: Vec<TargetSpec>,
        track_append: bool,
    ) -> Result<(Self, FeedbackSender), TargetsError> {
        let (display, feedback) = Self::spawn_inner(output, lead_in, targets, true, track_append)?;
        Ok((display, feedback.expect("requested with_feedback")))
    }

    fn spawn_inner(
        output: &OutputId,
        lead_in: Duration,
        targets: Vec<TargetSpec>,
        with_feedback: bool,
        track_append: bool,
    ) -> Result<(Self, Option<FeedbackSender>), TargetsError> {
        validate(&targets)?;
        let target_count = targets.len();
        let (events_tx, events) = crossbeam_channel::unbounded();
        let (feedback_rx, feedback_tx) = if with_feedback {
            let (tx, rx) = crossbeam_channel::unbounded();
            (Some(rx), Some(FeedbackSender::new(tx)))
        } else {
            (None, None)
        };
        let (appends_tx, appends_rx) = crossbeam_channel::unbounded();
        let failure = Arc::new(Mutex::new(None));
        let scene = TargetScene {
            output: output.clone(),
            targets,
            lead_in: (!lead_in.is_zero()).then_some(lead_in),
            lead_in_until: None,
            current: 0,
            first_frame: None,
            confirmed_at: None,
            events: events_tx,
            failure: Arc::clone(&failure),
            feedback: feedback_rx,
            latest_feedback: None,
            appends: appends_rx,
            track_settle: track_append,
            hidden_sent: 0,
            settled_received: 0,
            awaiting_finish: false,
            last_hidden: None,
        };
        let handle = spawn(
            SurfaceOptions {
                output: Some(output.as_str().to_owned()),
                namespace: "eye-targets",
            },
            scene,
        )?;
        tracing::info!(
            output = %output,
            targets = target_count,
            lead_in_ms = lead_in.as_millis() as u64,
            "target sequence started"
        );
        Ok((
            Self {
                events,
                handle,
                failure,
                appends: appends_tx,
            },
            feedback_tx,
        ))
    }

    /// Unbounded: the overlay thread never blocks on a slow reader. Disconnects when the overlay thread ends.
    pub fn events(&self) -> &crossbeam_channel::Receiver<TargetEvent> {
        &self.events
    }

    /// Pushes `spec` onto the running sequence; the scene renders it after the current targets
    /// and does not emit `Finished` until it too has been shown.
    pub fn append(&self, spec: TargetSpec) -> Result<(), TargetsError> {
        validate(std::slice::from_ref(&spec))?;
        let _ = self.appends.send(AppendMsg::Append(spec));
        Ok(())
    }

    /// A cloneable handle equivalent to `append`, for a caller (e.g. a `PumpObserver`) that does
    /// not hold this `TargetDisplay`.
    pub fn appender(&self) -> AppendSender {
        AppendSender::new(self.appends.clone())
    }

    /// Blocks until the sequence ended (drains and DISCARDS unread events), closes, and returns the scene's failure if any.
    /// Callers that need the events read `events()` until it disconnects first.
    pub fn wait(self) -> Result<(), TargetsError> {
        while self.events.recv().is_ok() {}
        self.finish()
    }

    /// Aborts the sequence early: closes the surface, joins, returns the scene's failure if any.
    pub fn close(self) -> Result<(), TargetsError> {
        self.finish()
    }

    fn finish(self) -> Result<(), TargetsError> {
        let Self {
            handle, failure, ..
        } = self;
        handle.close()?;
        match failure
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
        {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

pub(crate) fn validate(targets: &[TargetSpec]) -> Result<(), TargetsError> {
    if targets.is_empty() {
        return Err(TargetsError::Empty);
    }
    for (index, t) in targets.iter().enumerate() {
        if !t.px_logical.x.is_finite() || !t.px_logical.y.is_finite() || t.timing.dwell.is_zero() {
            return Err(TargetsError::InvalidTarget { index });
        }
    }
    Ok(())
}

#[derive(Debug)]
pub(crate) struct TargetScene {
    output: OutputId,
    targets: Vec<TargetSpec>,
    lead_in: Option<Duration>,
    lead_in_until: Option<Instant>,
    current: usize,
    first_frame: Option<Instant>,
    confirmed_at: Option<Instant>,
    events: crossbeam_channel::Sender<TargetEvent>,
    failure: Arc<Mutex<Option<TargetsError>>>,
    feedback: Option<crossbeam_channel::Receiver<Feedback>>,
    latest_feedback: Option<Feedback>,
    appends: crossbeam_channel::Receiver<AppendMsg>,
    /// When set, `Finished` is held back until `settled_received` catches up with `hidden_sent`,
    /// giving a `Hidden` observer a chance to append a retry before the sequence is declared done.
    track_settle: bool,
    hidden_sent: u64,
    settled_received: u64,
    /// Set once `on_presented` has sent the final `Hidden` and is waiting on `settled_received`
    /// before emitting `Finished`; checked by `render` so a mid-loop `advance()` past the last
    /// target (before that `Hidden` is actually sent) never finishes prematurely.
    awaiting_finish: bool,
    /// Index of the last target a `Hidden` was sent for; guards against sending it twice when
    /// `confirmed_at` is reset for a retry without `current` moving.
    last_hidden: Option<usize>,
}

impl TargetScene {
    fn advance(&mut self) {
        self.current += 1;
        self.first_frame = None;
        self.confirmed_at = None;
    }
}

impl Scene for TargetScene {
    type Msg = std::convert::Infallible;
    fn on_msg(&mut self, msg: Self::Msg, _: Instant) {
        match msg {}
    }

    fn render(&mut self, canvas: &mut Canvas<'_>, now: Instant) -> Schedule {
        if let Some(lead_in) = self.lead_in {
            let until = *self.lead_in_until.get_or_insert(now + lead_in);
            if now < until {
                return Schedule::At(until);
            }
            self.lead_in = None;
        }
        if let Some(rx) = &self.feedback {
            for fb in rx.try_iter() {
                self.latest_feedback = Some(fb);
            }
        }
        let had_current = self.targets.get(self.current).is_some();
        for msg in self.appends.try_iter() {
            match msg {
                AppendMsg::Append(spec) => self.targets.push(spec),
                AppendMsg::Settled => self.settled_received += 1,
            }
        }
        if !had_current && self.targets.get(self.current).is_some() {
            self.confirmed_at = None;
            self.first_frame = None;
            self.awaiting_finish = false;
        }
        let schedule = loop {
            let Some(t) = self.targets.get(self.current) else {
                if self.awaiting_finish && self.settled_received >= self.hidden_sent {
                    self.awaiting_finish = false;
                    tracing::info!(targets = self.targets.len(), "target sequence finished");
                    let _ = self.events.send(TargetEvent::Finished);
                    return Schedule::Exit;
                }
                break Schedule::NextFrame;
            };
            if let Some(confirmed) = self.confirmed_at
                && now >= confirmed + t.timing.dwell
            {
                tracing::debug!(
                    index = self.current,
                    dwell_ms = t.timing.dwell.as_millis() as u64,
                    "target dwell elapsed"
                );
                self.advance();
                continue;
            }
            let size = canvas.logical_size();
            if !in_bounds(t.px_logical, size) {
                tracing::error!(
                    index = self.current,
                    x = t.px_logical.x,
                    y = t.px_logical.y,
                    width = size.0,
                    height = size.1,
                    "target outside output"
                );
                *self.failure.lock().unwrap_or_else(PoisonError::into_inner) =
                    Some(TargetsError::OutOfBounds {
                        index: self.current,
                        px: t.px_logical,
                        size,
                    });
                return Schedule::Exit;
            }
            if let Some(fb) = &self.latest_feedback {
                draw_feedback(canvas, fb);
            }
            let first = *self.first_frame.get_or_insert(now);
            let elapsed = now - first;
            draw_target(canvas, t.px_logical, elapsed, t.timing, t.retry);
            break match self.confirmed_at {
                Some(c) if elapsed >= t.timing.settle + t.timing.window => {
                    Schedule::At(c + t.timing.dwell)
                }
                _ => Schedule::NextFrame,
            };
        };
        match (schedule, self.latest_feedback.is_some()) {
            (Schedule::At(_), true) => Schedule::NextFrame,
            _ => schedule,
        }
    }

    fn presentation_mark(&self) -> Option<u64> {
        (self.lead_in.is_none() && self.confirmed_at.is_none()).then_some(self.current as u64)
    }

    fn on_presented(&mut self, mark: u64, at: PresentedAt, now: Instant) -> Schedule {
        if mark != self.current as u64 {
            tracing::trace!(
                mark,
                current = self.current,
                { field::REASON } = "stale_mark",
                "presentation mark ignored"
            );
            return Schedule::Idle;
        }
        if self.confirmed_at.is_some() {
            tracing::trace!(
                mark,
                current = self.current,
                { field::REASON } = "already_confirmed",
                "presentation mark ignored"
            );
            return Schedule::Idle;
        }
        self.confirmed_at = Some(now);
        let (ts, clock) = match at {
            PresentedAt::Presentation(t) => (t, TargetClock::Presentation),
            PresentedAt::Commit(t) => (t, TargetClock::Commit),
        };
        if self.current > 0 && self.last_hidden != Some(self.current - 1) {
            tracing::debug!(
                index = self.current - 1,
                { field::TS_NS } = ts.as_nanos(),
                clock = at.clock_name(),
                "target hidden"
            );
            let _ = self.events.send(TargetEvent::Hidden {
                index: self.current - 1,
                at: ts,
                clock,
            });
            self.hidden_sent += 1;
            self.last_hidden = Some(self.current - 1);
        }
        match self.targets.get(self.current) {
            Some(t) => {
                let confirm_us = self
                    .first_frame
                    .map_or(0, |f| now.saturating_duration_since(f).as_micros() as u64);
                tracing::debug!(
                    index = self.current,
                    x = t.px_logical.x,
                    y = t.px_logical.y,
                    { field::TS_NS } = ts.as_nanos(),
                    clock = at.clock_name(),
                    confirm_us,
                    "target shown"
                );
                let shown = TargetShown {
                    index: self.current,
                    output: self.output.clone(),
                    px_logical: t.px_logical,
                    shown_at: ts,
                    clock,
                };
                let _ = self.events.send(TargetEvent::Shown(shown));
                Schedule::Idle
            }
            None if self.track_settle => {
                self.awaiting_finish = true;
                Schedule::NextFrame
            }
            None => {
                tracing::info!(targets = self.targets.len(), "target sequence finished");
                let _ = self.events.send(TargetEvent::Finished);
                Schedule::Exit
            }
        }
    }
}

const WHITE: Rgba = Rgba {
    r: 255,
    g: 255,
    b: 255,
    a: 255,
};
const GREEN: Rgba = Rgba {
    r: 64,
    g: 220,
    b: 96,
    a: 255,
};
const AMBER: Rgba = Rgba {
    r: 255,
    g: 176,
    b: 32,
    a: 255,
};
const COUNTDOWN_RADIUS: f64 = 16.0;
const COUNTDOWN_WIDTH: f64 = 2.0;
const FEEDBACK_UNCALIBRATED: Rgba = Rgba {
    r: 160,
    g: 160,
    b: 160,
    a: 255,
};
const FEEDBACK_CALIBRATED: Rgba = Rgba {
    r: 64,
    g: 128,
    b: 240,
    a: 255,
};
const FEEDBACK_RADIUS: f64 = 6.0;
const FEEDBACK_ELLIPSE_WIDTH: f64 = 1.0;

fn animation_len(dwell: Duration) -> Duration {
    Duration::from_millis(500).min(dwell / 2)
}

fn ring_radius(e: Duration, dwell: Duration) -> f64 {
    4.0 + 12.0 * (1.0 - e.as_secs_f64() / animation_len(dwell).as_secs_f64()).max(0.0)
}

fn lerp_channel(a: u8, b: u8, f: f64) -> u8 {
    (f64::from(a) + (f64::from(b) - f64::from(a)) * f).round() as u8
}

fn phase_color(e: Duration, t: TargetTiming, retry: bool) -> Rgba {
    let start = if retry { AMBER } else { WHITE };
    let f = (e.saturating_sub(t.settle).as_secs_f64() / t.window.as_secs_f64()).clamp(0.0, 1.0);
    Rgba {
        r: lerp_channel(start.r, GREEN.r, f),
        g: lerp_channel(start.g, GREEN.g, f),
        b: lerp_channel(start.b, GREEN.b, f),
        a: 255,
    }
}

fn countdown_sweep(e: Duration, settle: Duration) -> f64 {
    (e.as_secs_f64() / settle.as_secs_f64()).clamp(0.0, 1.0) * std::f64::consts::TAU
}

fn in_bounds(p: Point2<f64>, (w, h): (u32, u32)) -> bool {
    p.x >= 0.0 && p.y >= 0.0 && p.x < f64::from(w) && p.y < f64::from(h)
}

fn draw_target(
    c: &mut Canvas<'_>,
    p: Point2<f64>,
    elapsed: Duration,
    timing: TargetTiming,
    retry: bool,
) {
    let color = phase_color(elapsed, timing, retry);
    let sweep = countdown_sweep(elapsed, timing.settle);
    if sweep > 0.0 {
        c.stroke_arc(
            p,
            COUNTDOWN_RADIUS,
            -std::f64::consts::FRAC_PI_2,
            sweep,
            COUNTDOWN_WIDTH,
            color,
        );
    }
    let ring = ring_radius(elapsed, timing.dwell);
    c.stroke_ellipse(p, (ring, ring), 0.0, 2.0, color);
    c.fill_circle(p, 3.0, color);
}

fn draw_feedback(c: &mut Canvas<'_>, fb: &Feedback) {
    let color = if fb.calibrated {
        FEEDBACK_CALIBRATED
    } else {
        FEEDBACK_UNCALIBRATED
    };
    let (w, h) = c.logical_size();
    let max_axis = f64::from(w).hypot(f64::from(h));
    if let Some(e) = confidence_ellipse(fb.px_logical, &fb.cov_px, K95, max_axis) {
        c.stroke_ellipse(
            e.center,
            e.semi_axes,
            e.angle,
            FEEDBACK_ELLIPSE_WIDTH,
            color,
        );
    }
    c.fill_circle(fb.px_logical, FEEDBACK_RADIUS, color);
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use eye_log::Value;
    use eye_log::testing::capture_logs;

    use super::*;
    use crate::canvas::bgra;

    fn scene(
        targets: Vec<TargetSpec>,
        lead_in: Duration,
    ) -> (TargetScene, crossbeam_channel::Receiver<TargetEvent>) {
        let (tx, rx) = crossbeam_channel::unbounded();
        let (_appends_tx, appends_rx) = crossbeam_channel::unbounded();
        let scene = TargetScene {
            output: OutputId::from("eDP-1"),
            targets,
            lead_in: (!lead_in.is_zero()).then_some(lead_in),
            lead_in_until: None,
            current: 0,
            first_frame: None,
            confirmed_at: None,
            events: tx,
            failure: Arc::new(Mutex::new(None)),
            feedback: None,
            latest_feedback: None,
            appends: appends_rx,
            track_settle: false,
            hidden_sent: 0,
            settled_received: 0,
            awaiting_finish: false,
            last_hidden: None,
        };
        (scene, rx)
    }

    fn render_into(
        scene: &mut TargetScene,
        buf: &mut [u8],
        size: (u32, u32),
        now: Instant,
    ) -> Schedule {
        buf.fill(0);
        let mut canvas = Canvas::new(buf, size, 1).expect("valid buffer");
        scene.render(&mut canvas, now)
    }

    #[test]
    fn test_scene_waits_dwell_then_advances() {
        let timing = TargetTiming {
            settle: Duration::from_millis(300),
            window: Duration::from_millis(300),
            dwell: Duration::from_secs(1),
        };
        let targets = vec![
            TargetSpec {
                px_logical: Point2::new(50.0, 50.0),
                timing,
                retry: false,
            },
            TargetSpec {
                px_logical: Point2::new(150.0, 50.0),
                timing,
                retry: false,
            },
        ];
        let (mut scene, rx) = scene(targets, Duration::ZERO);
        let size = (200u32, 200u32);
        let mut buf = vec![0u8; (size.0 * size.1 * 4) as usize];
        let t0 = Instant::now();

        let schedule = render_into(&mut scene, &mut buf, size, t0);
        assert_eq!(schedule, Schedule::NextFrame);

        let t1 = Timestamp::from_nanos(1);
        let _ = scene.on_presented(0, PresentedAt::Commit(t1), t0);

        let schedule = render_into(&mut scene, &mut buf, size, t0 + Duration::from_millis(600));
        assert_eq!(schedule, Schedule::At(t0 + Duration::from_secs(1)));

        let schedule = render_into(&mut scene, &mut buf, size, t0 + Duration::from_secs(1));
        assert_eq!(schedule, Schedule::NextFrame);
        assert_eq!(bgra(&buf, size.0, 50, 50)[3], 0);
        assert_eq!(scene.presentation_mark(), Some(1));

        let t2 = Timestamp::from_nanos(2);
        let schedule = scene.on_presented(1, PresentedAt::Commit(t2), t0 + Duration::from_secs(1));
        assert_eq!(schedule, Schedule::Idle);

        let events: Vec<_> = rx.try_iter().collect();
        assert_eq!(
            events,
            vec![
                TargetEvent::Shown(TargetShown {
                    index: 0,
                    output: OutputId::from("eDP-1"),
                    px_logical: Point2::new(50.0, 50.0),
                    shown_at: t1,
                    clock: TargetClock::Commit,
                }),
                TargetEvent::Hidden {
                    index: 0,
                    at: t2,
                    clock: TargetClock::Commit,
                },
                TargetEvent::Shown(TargetShown {
                    index: 1,
                    output: OutputId::from("eDP-1"),
                    px_logical: Point2::new(150.0, 50.0),
                    shown_at: t2,
                    clock: TargetClock::Commit,
                }),
            ]
        );
    }

    #[test]
    fn test_validate_rejects_empty_and_invalid() {
        assert!(matches!(validate(&[]), Err(TargetsError::Empty)));

        let targets = vec![
            TargetSpec::from((Point2::new(1.0, 1.0), Duration::from_secs(1))),
            TargetSpec::from((Point2::new(f64::NAN, 1.0), Duration::from_secs(1))),
        ];
        assert!(matches!(
            validate(&targets),
            Err(TargetsError::InvalidTarget { index: 1 })
        ));

        let targets = vec![TargetSpec::from((Point2::new(1.0, 1.0), Duration::ZERO))];
        assert!(matches!(
            validate(&targets),
            Err(TargetsError::InvalidTarget { index: 0 })
        ));
    }

    #[test]
    fn test_tuple_into_target_spec() {
        let spec: TargetSpec = (Point2::new(960.0, 540.0), Duration::from_millis(1500)).into();
        assert_eq!(
            spec,
            TargetSpec {
                px_logical: Point2::new(960.0, 540.0),
                timing: TargetTiming {
                    settle: Duration::from_millis(600),
                    window: Duration::from_millis(900),
                    dwell: Duration::from_millis(1500),
                },
                retry: false,
            }
        );
    }

    #[test]
    fn test_bare_dwell_maps_to_two_fifths_settle() {
        let spec: TargetSpec = (Point2::new(0.0, 0.0), Duration::from_millis(1000)).into();
        assert_eq!(spec.timing.settle, Duration::from_millis(400));
        assert_eq!(spec.timing.window, Duration::from_millis(600));
        assert_eq!(spec.timing.dwell, Duration::from_millis(1000));
    }

    #[test]
    fn test_scene_lead_in_is_blank_without_marks() {
        let targets = vec![TargetSpec::from((
            Point2::new(50.0, 50.0),
            Duration::from_secs(1),
        ))];
        let (mut scene, _rx) = scene(targets, Duration::from_secs(1));
        let size = (200u32, 200u32);
        let mut buf = vec![0u8; (size.0 * size.1 * 4) as usize];
        let t0 = Instant::now();

        let schedule = render_into(&mut scene, &mut buf, size, t0);
        assert_eq!(schedule, Schedule::At(t0 + Duration::from_secs(1)));
        assert!(buf.iter().all(|&b| b == 0));
        assert_eq!(scene.presentation_mark(), None);

        let schedule = render_into(&mut scene, &mut buf, size, t0 + Duration::from_secs(1));
        assert_eq!(schedule, Schedule::NextFrame);
        assert_eq!(bgra(&buf, size.0, 50, 50)[3], 255);
        assert_eq!(scene.presentation_mark(), Some(0));
    }

    #[test]
    fn test_ring_radius_profile() {
        assert_abs_diff_eq!(
            ring_radius(Duration::ZERO, Duration::from_secs(2)),
            16.0,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            ring_radius(Duration::from_millis(250), Duration::from_secs(2)),
            10.0,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            ring_radius(Duration::from_millis(500), Duration::from_secs(2)),
            4.0,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            ring_radius(Duration::from_secs(5), Duration::from_secs(2)),
            4.0,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            ring_radius(Duration::from_millis(300), Duration::from_millis(600)),
            4.0,
            epsilon = 1e-12
        );
    }

    fn timing_for_fade_tests() -> TargetTiming {
        TargetTiming {
            settle: Duration::from_millis(1000),
            window: Duration::from_millis(600),
            dwell: Duration::from_millis(2000),
        }
    }

    #[test]
    fn test_target_is_white_during_settle() {
        let timing = timing_for_fade_tests();
        let targets = vec![TargetSpec {
            px_logical: Point2::new(50.0, 50.0),
            timing,
            retry: false,
        }];
        let (mut scene, _rx) = scene(targets, Duration::ZERO);
        let size = (200u32, 200u32);
        let mut buf = vec![0u8; (size.0 * size.1 * 4) as usize];
        let t0 = Instant::now();

        render_into(&mut scene, &mut buf, size, t0);
        render_into(&mut scene, &mut buf, size, t0 + timing.settle / 2);
        assert_eq!(bgra(&buf, size.0, 50, 50), [255, 255, 255, 255]);
    }

    #[test]
    fn test_target_fades_to_green_over_window() {
        let timing = timing_for_fade_tests();
        let targets = vec![TargetSpec {
            px_logical: Point2::new(50.0, 50.0),
            timing,
            retry: false,
        }];
        let (mut scene, _rx) = scene(targets, Duration::ZERO);
        let size = (200u32, 200u32);
        let mut buf = vec![0u8; (size.0 * size.1 * 4) as usize];
        let t0 = Instant::now();
        render_into(&mut scene, &mut buf, size, t0);

        let mid = timing.settle + timing.window / 2;
        render_into(&mut scene, &mut buf, size, t0 + mid);
        let px = bgra(&buf, size.0, 50, 50);
        // midpoint of WHITE and GREEN
        assert_abs_diff_eq!(f64::from(px[0]), 176.0, epsilon = 2.0);
        assert_abs_diff_eq!(f64::from(px[1]), 238.0, epsilon = 2.0);
        assert_abs_diff_eq!(f64::from(px[2]), 160.0, epsilon = 2.0);
        assert_eq!(px[3], 255);

        let after = timing.settle + timing.window;
        render_into(&mut scene, &mut buf, size, t0 + after);
        let px2 = bgra(&buf, size.0, 50, 50);
        assert_eq!(px2, [GREEN.b, GREEN.g, GREEN.r, GREEN.a]);
    }

    #[test]
    fn test_countdown_arc_sweeps_during_settle() {
        let timing = timing_for_fade_tests();
        let targets = vec![TargetSpec {
            px_logical: Point2::new(50.0, 50.0),
            timing,
            retry: false,
        }];
        let (mut scene, _rx) = scene(targets, Duration::ZERO);
        let size = (200u32, 200u32);
        let mut buf = vec![0u8; (size.0 * size.1 * 4) as usize];
        let t0 = Instant::now();
        render_into(&mut scene, &mut buf, size, t0);

        render_into(&mut scene, &mut buf, size, t0 + timing.settle / 4);
        assert_eq!(bgra(&buf, size.0, 34, 50)[3], 0);

        render_into(&mut scene, &mut buf, size, t0 + timing.settle * 3 / 4);
        assert!(bgra(&buf, size.0, 34, 50)[3] > 0);
    }

    #[test]
    fn test_schedule_is_next_frame_while_fading() {
        let timing = timing_for_fade_tests();
        let targets = vec![TargetSpec {
            px_logical: Point2::new(50.0, 50.0),
            timing,
            retry: false,
        }];
        let (mut scene, _rx) = scene(targets, Duration::ZERO);
        let size = (200u32, 200u32);
        let mut buf = vec![0u8; (size.0 * size.1 * 4) as usize];
        let t0 = Instant::now();
        render_into(&mut scene, &mut buf, size, t0);
        let _ = scene.on_presented(0, PresentedAt::Commit(Timestamp::from_nanos(1)), t0);

        let schedule = render_into(
            &mut scene,
            &mut buf,
            size,
            t0 + timing.settle + timing.window / 2,
        );
        assert_eq!(schedule, Schedule::NextFrame);
    }

    #[test]
    fn test_scene_shows_first_target_and_requests_mark() {
        let targets = vec![TargetSpec::from((
            Point2::new(960.5, 540.5),
            Duration::from_secs(1),
        ))];
        let (mut scene, _rx) = scene(targets, Duration::ZERO);
        let size = (1920u32, 1080u32);
        let mut buf = vec![0u8; (size.0 * size.1 * 4) as usize];
        let t0 = Instant::now();

        let schedule = render_into(&mut scene, &mut buf, size, t0);
        assert_eq!(bgra(&buf, size.0, 960, 540)[3], 255);
        assert_eq!(schedule, Schedule::NextFrame);
        assert_eq!(scene.presentation_mark(), Some(0));
    }

    #[test]
    fn test_scene_on_presented_emits_shown_once() {
        let targets = vec![TargetSpec::from((
            Point2::new(50.0, 50.0),
            Duration::from_secs(1),
        ))];
        let (mut scene, rx) = scene(targets, Duration::ZERO);
        let t0 = Instant::now();
        let t = Timestamp::from_nanos(42);

        let schedule = scene.on_presented(0, PresentedAt::Presentation(t), t0);
        assert_eq!(schedule, Schedule::Idle);
        let schedule = scene.on_presented(0, PresentedAt::Presentation(t), t0);
        assert_eq!(schedule, Schedule::Idle);

        let events: Vec<_> = rx.try_iter().collect();
        assert_eq!(
            events,
            vec![TargetEvent::Shown(TargetShown {
                index: 0,
                output: OutputId::from("eDP-1"),
                px_logical: Point2::new(50.0, 50.0),
                shown_at: t,
                clock: TargetClock::Presentation,
            })]
        );
        assert_eq!(scene.presentation_mark(), None);
    }

    #[test]
    fn test_scene_unconfirmed_target_keeps_requesting_frames() {
        let targets = vec![TargetSpec::from((
            Point2::new(50.0, 50.0),
            Duration::from_secs(1),
        ))];
        let (mut scene, _rx) = scene(targets, Duration::ZERO);
        let size = (200u32, 200u32);
        let mut buf = vec![0u8; (size.0 * size.1 * 4) as usize];
        let t0 = Instant::now();

        render_into(&mut scene, &mut buf, size, t0);

        let schedule = render_into(&mut scene, &mut buf, size, t0 + Duration::from_secs(2));
        assert_eq!(schedule, Schedule::NextFrame);
        assert_eq!(bgra(&buf, size.0, 50, 50)[3], 255);
        assert_eq!(scene.presentation_mark(), Some(0));
    }

    #[test]
    fn test_scene_final_blank_frame_emits_finished_and_exits() {
        let targets = vec![
            TargetSpec::from((Point2::new(50.0, 50.0), Duration::from_millis(100))),
            TargetSpec::from((Point2::new(150.0, 50.0), Duration::from_millis(100))),
        ];
        let (mut scene, rx) = scene(targets, Duration::ZERO);
        let size = (200u32, 200u32);
        let mut buf = vec![0u8; (size.0 * size.1 * 4) as usize];
        let t0 = Instant::now();

        render_into(&mut scene, &mut buf, size, t0);
        let t1 = Timestamp::from_nanos(1);
        scene.on_presented(0, PresentedAt::Commit(t1), t0);

        let t_advance = t0 + Duration::from_millis(100);
        render_into(&mut scene, &mut buf, size, t_advance);
        let t2 = Timestamp::from_nanos(2);
        scene.on_presented(1, PresentedAt::Commit(t2), t_advance);

        let t_final = t_advance + Duration::from_millis(100);
        let schedule = render_into(&mut scene, &mut buf, size, t_final);
        assert_eq!(schedule, Schedule::NextFrame);
        assert!(buf.iter().all(|&b| b == 0));
        assert_eq!(scene.presentation_mark(), Some(2));

        let t3 = Timestamp::from_nanos(3);
        let schedule = scene.on_presented(2, PresentedAt::Commit(t3), t_final);
        assert_eq!(schedule, Schedule::Exit);

        let events: Vec<_> = rx.try_iter().collect();
        assert_eq!(
            events,
            vec![
                TargetEvent::Shown(TargetShown {
                    index: 0,
                    output: OutputId::from("eDP-1"),
                    px_logical: Point2::new(50.0, 50.0),
                    shown_at: t1,
                    clock: TargetClock::Commit,
                }),
                TargetEvent::Hidden {
                    index: 0,
                    at: t2,
                    clock: TargetClock::Commit,
                },
                TargetEvent::Shown(TargetShown {
                    index: 1,
                    output: OutputId::from("eDP-1"),
                    px_logical: Point2::new(150.0, 50.0),
                    shown_at: t2,
                    clock: TargetClock::Commit,
                }),
                TargetEvent::Hidden {
                    index: 1,
                    at: t3,
                    clock: TargetClock::Commit,
                },
                TargetEvent::Finished,
            ]
        );
    }

    #[test]
    fn test_scene_out_of_bounds_target_exits_with_error() {
        let targets = vec![TargetSpec::from((
            Point2::new(2000.0, 10.0),
            Duration::from_secs(1),
        ))];
        let (mut scene, rx) = scene(targets, Duration::ZERO);
        let size = (1920u32, 1080u32);
        let mut buf = vec![0u8; (size.0 * size.1 * 4) as usize];
        let t0 = Instant::now();
        let failure = Arc::clone(&scene.failure);

        let schedule = render_into(&mut scene, &mut buf, size, t0);
        assert_eq!(schedule, Schedule::Exit);
        assert!(rx.try_iter().all(|e| e != TargetEvent::Finished));
        match failure
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
        {
            Some(TargetsError::OutOfBounds { index: 0, .. }) => {}
            other => panic!("expected OutOfBounds, got {other:?}"),
        }
    }

    #[test]
    fn test_scene_subpixel_centre_preserved() {
        let targets = vec![TargetSpec::from((
            Point2::new(100.25, 100.75),
            Duration::from_secs(1),
        ))];
        let (mut scene, _rx) = scene(targets, Duration::ZERO);
        let size = (200u32, 200u32);
        let mut buf = vec![0u8; (size.0 * size.1 * 4) as usize];
        let t0 = Instant::now();

        render_into(&mut scene, &mut buf, size, t0);

        let (cx, cy) = (100.25f64, 100.75f64);
        let mut sum_w = 0.0;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        for y in 92..=109u32 {
            for x in 92..=109u32 {
                let px_cx = f64::from(x) + 0.5;
                let px_cy = f64::from(y) + 0.5;
                if (px_cx - cx).hypot(px_cy - cy) <= 8.0 {
                    let alpha = f64::from(bgra(&buf, size.0, x, y)[3]);
                    sum_w += alpha;
                    sum_x += alpha * px_cx;
                    sum_y += alpha * px_cy;
                }
            }
        }
        assert!(sum_w > 0.0);
        let centroid_x = sum_x / sum_w;
        let centroid_y = sum_y / sum_w;
        assert_abs_diff_eq!(centroid_x, cx, epsilon = 0.05);
        assert_abs_diff_eq!(centroid_y, cy, epsilon = 0.05);
    }

    #[test]
    fn test_feedback_point_is_drawn_grey_then_blue() {
        let timing = timing_for_fade_tests();
        let targets = vec![TargetSpec {
            px_logical: Point2::new(50.0, 50.0),
            timing,
            retry: false,
        }];
        let (mut scene, _rx) = scene(targets, Duration::ZERO);
        let (fb_tx, fb_rx) = crossbeam_channel::unbounded();
        scene.feedback = Some(fb_rx);
        let size = (200u32, 200u32);
        let mut buf = vec![0u8; (size.0 * size.1 * 4) as usize];
        let t0 = Instant::now();

        fb_tx
            .send(Feedback {
                px_logical: Point2::new(150.0, 50.0),
                cov_px: Matrix2::identity() * 4.0,
                calibrated: false,
                at: Timestamp::from_nanos(1),
            })
            .unwrap();
        render_into(&mut scene, &mut buf, size, t0);
        assert_eq!(
            bgra(&buf, size.0, 150, 50),
            [
                FEEDBACK_UNCALIBRATED.b,
                FEEDBACK_UNCALIBRATED.g,
                FEEDBACK_UNCALIBRATED.r,
                FEEDBACK_UNCALIBRATED.a,
            ]
        );

        fb_tx
            .send(Feedback {
                px_logical: Point2::new(150.0, 50.0),
                cov_px: Matrix2::identity() * 4.0,
                calibrated: true,
                at: Timestamp::from_nanos(2),
            })
            .unwrap();
        render_into(&mut scene, &mut buf, size, t0 + Duration::from_millis(10));
        assert_eq!(
            bgra(&buf, size.0, 150, 50),
            [
                FEEDBACK_CALIBRATED.b,
                FEEDBACK_CALIBRATED.g,
                FEEDBACK_CALIBRATED.r,
                FEEDBACK_CALIBRATED.a,
            ]
        );
    }

    #[test]
    fn test_feedback_without_targets_is_ignored() {
        let dwell = Duration::from_millis(100);
        let targets = vec![TargetSpec::from((Point2::new(50.0, 50.0), dwell))];
        let (mut scene, _rx) = scene(targets, Duration::ZERO);
        let (fb_tx, fb_rx) = crossbeam_channel::unbounded();
        scene.feedback = Some(fb_rx);
        let size = (200u32, 200u32);
        let mut buf = vec![0u8; (size.0 * size.1 * 4) as usize];
        let t0 = Instant::now();

        render_into(&mut scene, &mut buf, size, t0);
        scene.on_presented(0, PresentedAt::Commit(Timestamp::from_nanos(1)), t0);

        // Advance past the only target's dwell so no target is current any more.
        let t_after = t0 + dwell + Duration::from_millis(10);
        fb_tx
            .send(Feedback {
                px_logical: Point2::new(150.0, 50.0),
                cov_px: Matrix2::identity() * 4.0,
                calibrated: false,
                at: Timestamp::from_nanos(2),
            })
            .unwrap();
        let schedule = render_into(&mut scene, &mut buf, size, t_after);
        assert_eq!(schedule, Schedule::NextFrame);
        assert_eq!(bgra(&buf, size.0, 150, 50)[3], 0);
    }

    #[test]
    fn test_append_extends_sequence_before_finished() {
        let timing = TargetTiming {
            settle: Duration::from_millis(10),
            window: Duration::from_millis(10),
            dwell: Duration::from_millis(100),
        };
        let (events_tx, rx) = crossbeam_channel::unbounded();
        let (appends_tx, appends_rx) = crossbeam_channel::unbounded();
        let mut scene = TargetScene {
            output: OutputId::from("eDP-1"),
            targets: vec![TargetSpec {
                px_logical: Point2::new(50.0, 50.0),
                timing,
                retry: false,
            }],
            lead_in: None,
            lead_in_until: None,
            current: 0,
            first_frame: None,
            confirmed_at: None,
            events: events_tx,
            failure: Arc::new(Mutex::new(None)),
            feedback: None,
            latest_feedback: None,
            appends: appends_rx,
            track_settle: false,
            hidden_sent: 0,
            settled_received: 0,
            awaiting_finish: false,
            last_hidden: None,
        };
        let size = (200u32, 200u32);
        let mut buf = vec![0u8; (size.0 * size.1 * 4) as usize];
        let t0 = Instant::now();

        render_into(&mut scene, &mut buf, size, t0);
        scene.on_presented(0, PresentedAt::Commit(Timestamp::from_nanos(1)), t0);

        appends_tx
            .send(AppendMsg::Append(TargetSpec {
                px_logical: Point2::new(150.0, 50.0),
                timing,
                retry: true,
            }))
            .unwrap();

        let t1 = t0 + timing.dwell;
        let schedule = render_into(&mut scene, &mut buf, size, t1);
        assert_eq!(schedule, Schedule::NextFrame);
        assert_eq!(scene.presentation_mark(), Some(1));

        let t2 = Timestamp::from_nanos(2);
        let schedule = scene.on_presented(1, PresentedAt::Commit(t2), t1);
        assert_eq!(schedule, Schedule::Idle);

        let events: Vec<_> = rx.try_iter().collect();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, TargetEvent::Shown(s) if s.index == 1)),
            "{events:?}"
        );
        assert!(!events.contains(&TargetEvent::Finished), "{events:?}");
    }

    #[test]
    fn test_rejecting_the_last_target_still_shows_its_retry() {
        let timing = TargetTiming {
            settle: Duration::from_millis(10),
            window: Duration::from_millis(10),
            dwell: Duration::from_millis(100),
        };
        let (events_tx, rx) = crossbeam_channel::unbounded();
        let (appends_tx, appends_rx) = crossbeam_channel::unbounded();
        let mut scene = TargetScene {
            output: OutputId::from("eDP-1"),
            targets: vec![TargetSpec {
                px_logical: Point2::new(50.0, 50.0),
                timing,
                retry: false,
            }],
            lead_in: None,
            lead_in_until: None,
            current: 0,
            first_frame: None,
            confirmed_at: None,
            events: events_tx,
            failure: Arc::new(Mutex::new(None)),
            feedback: None,
            latest_feedback: None,
            appends: appends_rx,
            track_settle: true,
            hidden_sent: 0,
            settled_received: 0,
            awaiting_finish: false,
            last_hidden: None,
        };
        let size = (200u32, 200u32);
        let mut buf = vec![0u8; (size.0 * size.1 * 4) as usize];
        let t0 = Instant::now();
        let mut all_events = Vec::new();

        render_into(&mut scene, &mut buf, size, t0);
        scene.on_presented(0, PresentedAt::Commit(Timestamp::from_nanos(1)), t0);

        let t1 = t0 + timing.dwell;
        let schedule = render_into(&mut scene, &mut buf, size, t1);
        assert_eq!(schedule, Schedule::NextFrame);
        assert_eq!(scene.presentation_mark(), Some(1));

        let t2 = Timestamp::from_nanos(2);
        let schedule = scene.on_presented(1, PresentedAt::Commit(t2), t1);
        assert_eq!(schedule, Schedule::NextFrame, "waits instead of finishing");

        let events_so_far: Vec<_> = rx.try_iter().collect();
        assert!(
            events_so_far.contains(&TargetEvent::Hidden {
                index: 0,
                at: t2,
                clock: TargetClock::Commit,
            }),
            "{events_so_far:?}"
        );
        assert!(
            !events_so_far.contains(&TargetEvent::Finished),
            "must not finish before the observer has reacted to the final Hidden: {events_so_far:?}"
        );
        all_events.extend(events_so_far);

        let schedule = render_into(&mut scene, &mut buf, size, t1 + Duration::from_millis(1));
        assert_eq!(
            schedule,
            Schedule::NextFrame,
            "still waiting: no append or settle has arrived yet"
        );
        assert!(rx.try_iter().all(|e| e != TargetEvent::Finished));

        appends_tx
            .send(AppendMsg::Append(TargetSpec {
                px_logical: Point2::new(150.0, 50.0),
                timing,
                retry: true,
            }))
            .unwrap();
        appends_tx.send(AppendMsg::Settled).unwrap();

        let t3 = t1 + Duration::from_millis(2);
        let schedule = render_into(&mut scene, &mut buf, size, t3);
        assert_eq!(schedule, Schedule::NextFrame);
        assert_eq!(
            scene.presentation_mark(),
            Some(1),
            "the retry is now current and awaiting its own presentation mark"
        );
        assert!(
            rx.try_iter().all(|e| e != TargetEvent::Finished),
            "the retry must be shown before Finished"
        );

        let t4 = Timestamp::from_nanos(4);
        let schedule = scene.on_presented(1, PresentedAt::Commit(t4), t3);
        assert_eq!(schedule, Schedule::Idle);
        let events: Vec<_> = rx.try_iter().collect();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, TargetEvent::Shown(s) if s.index == 1 && s.px_logical == Point2::new(150.0, 50.0))),
            "{events:?}"
        );
        all_events.extend(events);

        let hidden_zero_count = all_events
            .iter()
            .filter(|e| matches!(e, TargetEvent::Hidden { index: 0, .. }))
            .count();
        assert_eq!(
            hidden_zero_count, 1,
            "expected exactly one Hidden{{index: 0}} across the whole run, got {all_events:?}"
        );
    }

    #[test]
    fn test_finishes_immediately_when_not_tracking_settle() {
        let targets = vec![TargetSpec::from((
            Point2::new(50.0, 50.0),
            Duration::from_millis(100),
        ))];
        let (mut scene, rx) = scene(targets, Duration::ZERO);
        let size = (200u32, 200u32);
        let mut buf = vec![0u8; (size.0 * size.1 * 4) as usize];
        let t0 = Instant::now();

        render_into(&mut scene, &mut buf, size, t0);
        scene.on_presented(0, PresentedAt::Commit(Timestamp::from_nanos(1)), t0);

        let t1 = t0 + Duration::from_millis(100);
        render_into(&mut scene, &mut buf, size, t1);
        let schedule = scene.on_presented(1, PresentedAt::Commit(Timestamp::from_nanos(2)), t1);
        assert_eq!(schedule, Schedule::Exit);
        let events: Vec<_> = rx.try_iter().collect();
        assert!(events.contains(&TargetEvent::Finished), "{events:?}");
    }

    #[test]
    fn test_appended_target_renders_amber_during_settle() {
        let timing = timing_for_fade_tests();
        let targets = vec![TargetSpec {
            px_logical: Point2::new(50.0, 50.0),
            timing,
            retry: true,
        }];
        let (mut scene, _rx) = scene(targets, Duration::ZERO);
        let size = (200u32, 200u32);
        let mut buf = vec![0u8; (size.0 * size.1 * 4) as usize];
        let t0 = Instant::now();

        render_into(&mut scene, &mut buf, size, t0 + timing.settle / 2);
        assert_eq!(
            bgra(&buf, size.0, 50, 50),
            [AMBER.b, AMBER.g, AMBER.r, AMBER.a]
        );
    }

    #[test]
    fn test_logs_target_shown_at_debug() {
        let targets = vec![TargetSpec::from((
            Point2::new(50.0, 50.0),
            Duration::from_secs(1),
        ))];
        let (mut scene, _rx) = scene(targets, Duration::ZERO);
        let size = (200u32, 200u32);
        let mut buf = vec![0u8; (size.0 * size.1 * 4) as usize];
        let t0 = Instant::now();
        render_into(&mut scene, &mut buf, size, t0);

        let (_, records) = capture_logs(tracing::Level::TRACE, || {
            scene.on_presented(0, PresentedAt::Presentation(Timestamp::from_nanos(42)), t0)
        });
        let shown = records
            .iter()
            .find(|r| r.message == "target shown")
            .expect("target shown record");
        assert_eq!(shown.level, eye_log::Level::Debug);
        assert_eq!(shown.target, "eye_overlay::targets");
        assert_eq!(shown.fields[field::TS_NS], Value::U64(42));
        assert_eq!(
            shown.fields["clock"],
            Value::Str("presentation".to_string())
        );
        assert_eq!(shown.fields["index"], Value::U64(0));
        assert!(matches!(shown.fields["x"], Value::F64(_)));
        assert!(matches!(shown.fields["y"], Value::F64(_)));

        let (_, records2) = capture_logs(tracing::Level::TRACE, || {
            scene.on_presented(0, PresentedAt::Presentation(Timestamp::from_nanos(42)), t0)
        });
        let ignored = records2
            .iter()
            .find(|r| r.message == "presentation mark ignored")
            .expect("ignored record");
        assert_eq!(ignored.level, eye_log::Level::Trace);
        assert_eq!(
            ignored.fields[field::REASON],
            Value::Str("already_confirmed".to_string())
        );
    }

    #[test]
    fn test_logs_target_hidden_at_debug() {
        let timing = TargetTiming {
            settle: Duration::from_millis(10),
            window: Duration::from_millis(10),
            dwell: Duration::from_millis(100),
        };
        let targets = vec![
            TargetSpec {
                px_logical: Point2::new(50.0, 50.0),
                timing,
                retry: false,
            },
            TargetSpec {
                px_logical: Point2::new(150.0, 50.0),
                timing,
                retry: false,
            },
        ];
        let (mut scene, _rx) = scene(targets, Duration::ZERO);
        let size = (200u32, 200u32);
        let mut buf = vec![0u8; (size.0 * size.1 * 4) as usize];
        let t0 = Instant::now();
        render_into(&mut scene, &mut buf, size, t0);
        scene.on_presented(0, PresentedAt::Commit(Timestamp::from_nanos(1)), t0);

        let t1 = t0 + timing.dwell;
        render_into(&mut scene, &mut buf, size, t1);

        let (_, records) = capture_logs(tracing::Level::TRACE, || {
            scene.on_presented(1, PresentedAt::Commit(Timestamp::from_nanos(2)), t1)
        });
        let hidden_idx = records
            .iter()
            .position(|r| r.message == "target hidden")
            .expect("hidden record");
        let shown_idx = records
            .iter()
            .position(|r| r.message == "target shown")
            .expect("shown record");
        assert!(hidden_idx < shown_idx, "{records:?}");
        assert_eq!(records[hidden_idx].level, eye_log::Level::Debug);
        assert_eq!(records[hidden_idx].fields["index"], Value::U64(0));
        assert_eq!(records[shown_idx].fields["index"], Value::U64(1));
    }

    #[test]
    fn test_logs_target_sequence_finished_at_info() {
        let targets = vec![
            TargetSpec::from((Point2::new(50.0, 50.0), Duration::from_millis(100))),
            TargetSpec::from((Point2::new(150.0, 50.0), Duration::from_millis(100))),
        ];
        let (mut scene, _rx) = scene(targets, Duration::ZERO);
        let size = (200u32, 200u32);
        let mut buf = vec![0u8; (size.0 * size.1 * 4) as usize];
        let t0 = Instant::now();

        render_into(&mut scene, &mut buf, size, t0);
        scene.on_presented(0, PresentedAt::Commit(Timestamp::from_nanos(1)), t0);

        let t_advance = t0 + Duration::from_millis(100);
        render_into(&mut scene, &mut buf, size, t_advance);
        scene.on_presented(1, PresentedAt::Commit(Timestamp::from_nanos(2)), t_advance);

        let t_final = t_advance + Duration::from_millis(100);
        render_into(&mut scene, &mut buf, size, t_final);

        let (_, records) = capture_logs(tracing::Level::TRACE, || {
            scene.on_presented(2, PresentedAt::Commit(Timestamp::from_nanos(3)), t_final)
        });

        let finished = records
            .iter()
            .rfind(|r| r.message == "target sequence finished")
            .expect("finished record");
        assert_eq!(finished.level, eye_log::Level::Info);
        assert_eq!(finished.fields["targets"], Value::U64(2));
    }

    #[test]
    fn test_logs_target_outside_output_at_error() {
        let targets = vec![TargetSpec::from((
            Point2::new(2000.0, 10.0),
            Duration::from_secs(1),
        ))];
        let (mut scene, _rx) = scene(targets, Duration::ZERO);
        let size = (1920u32, 1080u32);
        let mut buf = vec![0u8; (size.0 * size.1 * 4) as usize];
        let t0 = Instant::now();

        let (_, records) = capture_logs(tracing::Level::TRACE, || {
            render_into(&mut scene, &mut buf, size, t0)
        });

        let errs: Vec<_> = records
            .iter()
            .filter(|r| r.level == eye_log::Level::Error)
            .collect();
        assert_eq!(errs.len(), 1, "{records:?}");
        assert_eq!(errs[0].message, "target outside output");
        assert_eq!(errs[0].fields["index"], Value::U64(0));
        assert_eq!(errs[0].fields["width"], Value::U64(1920));
        assert_eq!(errs[0].fields["height"], Value::U64(1080));
    }

    #[test]
    fn test_logs_presentation_mark_ignored_at_trace() {
        let targets = vec![TargetSpec::from((
            Point2::new(50.0, 50.0),
            Duration::from_secs(1),
        ))];
        let (mut scene, _rx) = scene(targets, Duration::ZERO);
        let t0 = Instant::now();

        let (_, records) = capture_logs(tracing::Level::TRACE, || {
            scene.on_presented(5, PresentedAt::Commit(Timestamp::from_nanos(1)), t0)
        });
        let rec = records
            .iter()
            .find(|r| r.message == "presentation mark ignored")
            .expect("ignored record");
        assert_eq!(rec.level, eye_log::Level::Trace);
        assert_eq!(rec.fields["mark"], Value::U64(5));
        assert_eq!(rec.fields["current"], Value::U64(0));
        assert_eq!(rec.fields["reason"], Value::Str("stale_mark".to_string()));

        scene.on_presented(0, PresentedAt::Commit(Timestamp::from_nanos(2)), t0);
        let (_, records2) = capture_logs(tracing::Level::TRACE, || {
            scene.on_presented(0, PresentedAt::Commit(Timestamp::from_nanos(3)), t0)
        });
        let rec2 = records2
            .iter()
            .find(|r| r.message == "presentation mark ignored")
            .expect("ignored record 2");
        assert_eq!(
            rec2.fields["reason"],
            Value::Str("already_confirmed".to_string())
        );
    }

    #[test]
    #[ignore = "needs wayland"]
    fn test_live_targets_use_monotonic_presentation_clock() {
        let output = std::env::var("EYE_OUTPUT").unwrap_or_else(|_| "eDP-1".to_string());
        let output = OutputId::from(output.as_str());
        let targets = vec![
            TargetSpec::from((Point2::new(480.0, 270.0), Duration::from_millis(400))),
            TargetSpec::from((Point2::new(960.0, 540.0), Duration::from_millis(400))),
            TargetSpec::from((Point2::new(1440.0, 810.0), Duration::from_millis(400))),
        ];
        let display = TargetDisplay::spawn(&output, Duration::from_millis(200), targets, false)
            .expect("spawn succeeds");

        let mut shown = Vec::new();
        let mut hidden = Vec::new();
        let mut finished = false;
        let mut timestamps = Vec::new();
        while let Ok(event) = display.events().recv() {
            let received = Timestamp::now();
            match event {
                TargetEvent::Shown(s) => {
                    println!("Shown index {} clock {:?}", s.index, s.clock);
                    assert!(received.as_nanos() - s.shown_at.as_nanos() <= 250_000_000);
                    timestamps.push(s.shown_at.as_nanos());
                    shown.push(s);
                }
                TargetEvent::Hidden { index, at, clock } => {
                    println!("Hidden index {index} clock {clock:?}");
                    timestamps.push(at.as_nanos());
                    hidden.push((index, at, clock));
                }
                TargetEvent::Finished => {
                    finished = true;
                }
            }
        }

        assert_eq!(shown.len(), 3);
        assert_eq!(hidden.len(), 3);
        assert!(finished);
        assert!(timestamps.windows(2).all(|w| w[0] <= w[1]));
        for (i, h) in hidden.iter().enumerate() {
            assert_eq!(h.0, i);
        }
        for i in 0..2 {
            assert_eq!(hidden[i].1, shown[i + 1].shown_at);
        }

        display.close().expect("close succeeds");
    }
}
