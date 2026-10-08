//! Shows a sequence of dot targets and timestamps when each became visible and was replaced.

use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use eye_core::session::TargetClock;
use eye_core::{OutputId, Timestamp};
use nalgebra::Point2;

use crate::canvas::{Canvas, Rgba};
use crate::error::OverlayError;
use crate::handle::OverlayHandle;
use crate::scene::{PresentedAt, Scene, Schedule};
use crate::surface::{SurfaceOptions, spawn};

#[derive(Debug, Clone, PartialEq)]
pub struct TargetSpec {
    pub px_logical: Point2<f64>,
    pub dwell: Duration,
}

impl From<(Point2<f64>, Duration)> for TargetSpec {
    fn from((px_logical, dwell): (Point2<f64>, Duration)) -> Self {
        Self { px_logical, dwell }
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
}

impl TargetDisplay {
    /// Validates (before connecting), connects on `output` with namespace "eye-targets", shows `lead_in` of blank,
    /// then the targets in order.
    pub fn spawn(
        output: &OutputId,
        lead_in: Duration,
        targets: Vec<TargetSpec>,
    ) -> Result<Self, TargetsError> {
        validate(&targets)?;
        let (events_tx, events) = crossbeam_channel::unbounded();
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
        };
        let handle = spawn(
            SurfaceOptions {
                output: Some(output.as_str().to_owned()),
                namespace: "eye-targets",
            },
            scene,
        )?;
        Ok(Self {
            events,
            handle,
            failure,
        })
    }

    /// Unbounded: the overlay thread never blocks on a slow reader. Disconnects when the overlay thread ends.
    pub fn events(&self) -> &crossbeam_channel::Receiver<TargetEvent> {
        &self.events
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
        if !t.px_logical.x.is_finite() || !t.px_logical.y.is_finite() || t.dwell.is_zero() {
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
        loop {
            let Some(t) = self.targets.get(self.current) else {
                return Schedule::NextFrame;
            };
            if let Some(confirmed) = self.confirmed_at
                && now >= confirmed + t.dwell
            {
                self.advance();
                continue;
            }
            let size = canvas.logical_size();
            if !in_bounds(t.px_logical, size) {
                *self.failure.lock().unwrap_or_else(PoisonError::into_inner) =
                    Some(TargetsError::OutOfBounds {
                        index: self.current,
                        px: t.px_logical,
                        size,
                    });
                return Schedule::Exit;
            }
            let first = *self.first_frame.get_or_insert(now);
            let elapsed = now - first;
            draw_target(canvas, t.px_logical, ring_radius(elapsed, t.dwell));
            return match self.confirmed_at {
                Some(c) if elapsed >= animation_len(t.dwell) => Schedule::At(c + t.dwell),
                _ => Schedule::NextFrame,
            };
        }
    }

    fn presentation_mark(&self) -> Option<u64> {
        (self.lead_in.is_none() && self.confirmed_at.is_none()).then_some(self.current as u64)
    }

    fn on_presented(&mut self, mark: u64, at: PresentedAt, now: Instant) -> Schedule {
        if mark != self.current as u64 || self.confirmed_at.is_some() {
            return Schedule::Idle;
        }
        self.confirmed_at = Some(now);
        let (ts, clock) = match at {
            PresentedAt::Presentation(t) => (t, TargetClock::Presentation),
            PresentedAt::Commit(t) => (t, TargetClock::Commit),
        };
        if self.current > 0 {
            let _ = self.events.send(TargetEvent::Hidden {
                index: self.current - 1,
                at: ts,
                clock,
            });
        }
        match self.targets.get(self.current) {
            Some(t) => {
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
            None => {
                let _ = self.events.send(TargetEvent::Finished);
                Schedule::Exit
            }
        }
    }
}

fn animation_len(dwell: Duration) -> Duration {
    Duration::from_millis(500).min(dwell / 2)
}

fn ring_radius(e: Duration, dwell: Duration) -> f64 {
    4.0 + 12.0 * (1.0 - e.as_secs_f64() / animation_len(dwell).as_secs_f64()).max(0.0)
}

fn in_bounds(p: Point2<f64>, (w, h): (u32, u32)) -> bool {
    p.x >= 0.0 && p.y >= 0.0 && p.x < f64::from(w) && p.y < f64::from(h)
}

fn draw_target(c: &mut Canvas<'_>, p: Point2<f64>, ring: f64) {
    c.stroke_ellipse(
        p,
        (ring, ring),
        0.0,
        2.0,
        Rgba {
            r: 255,
            g: 255,
            b: 255,
            a: 255,
        },
    );
    c.fill_circle(
        p,
        3.0,
        Rgba {
            r: 255,
            g: 255,
            b: 255,
            a: 255,
        },
    );
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;
    use crate::canvas::bgra;

    fn scene(
        targets: Vec<TargetSpec>,
        lead_in: Duration,
    ) -> (TargetScene, crossbeam_channel::Receiver<TargetEvent>) {
        let (tx, rx) = crossbeam_channel::unbounded();
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
        let targets = vec![
            TargetSpec::from((Point2::new(50.0, 50.0), Duration::from_secs(1))),
            TargetSpec::from((Point2::new(150.0, 50.0), Duration::from_secs(1))),
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
                dwell: Duration::from_millis(1500),
            }
        );
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
    #[ignore = "needs wayland"]
    fn test_live_targets_use_monotonic_presentation_clock() {
        let output = std::env::var("EYE_OUTPUT").unwrap_or_else(|_| "eDP-1".to_string());
        let output = OutputId::from(output.as_str());
        let targets = vec![
            TargetSpec::from((Point2::new(480.0, 270.0), Duration::from_millis(400))),
            TargetSpec::from((Point2::new(960.0, 540.0), Duration::from_millis(400))),
            TargetSpec::from((Point2::new(1440.0, 810.0), Duration::from_millis(400))),
        ];
        let display = TargetDisplay::spawn(&output, Duration::from_millis(200), targets)
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
