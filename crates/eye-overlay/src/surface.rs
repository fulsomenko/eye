//! Spawns the overlay thread: Wayland connection, layer surface, event loop.

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::thread;
use std::time::Instant;

use eye_core::Timestamp;
use eye_core::log::field;
use smithay_client_toolkit::compositor::{
    CompositorHandler, CompositorState, FrameCallbackData, Region,
};
use smithay_client_toolkit::output::{OutputHandler, OutputState};
use smithay_client_toolkit::presentation_time::{
    PresentTime, PresentationTimeHandler, PresentationTimeState,
};
use smithay_client_toolkit::reexports::calloop;
use smithay_client_toolkit::reexports::calloop::timer::{TimeoutAction, Timer};
use smithay_client_toolkit::reexports::calloop::{EventLoop, LoopHandle, RegistrationToken};
use smithay_client_toolkit::reexports::calloop_wayland_source::WaylandSource;
use smithay_client_toolkit::reexports::client::globals::registry_queue_init;
use smithay_client_toolkit::reexports::client::protocol::{wl_output, wl_shm, wl_surface};
use smithay_client_toolkit::reexports::client::{Connection, QueueHandle, WEnum};
use smithay_client_toolkit::reexports::protocols::wp::presentation_time::client::wp_presentation_feedback;
use smithay_client_toolkit::registry::{ProvidesRegistryState, RegistryState};
use smithay_client_toolkit::shell::WaylandSurface;
use smithay_client_toolkit::shell::wlr_layer::{
    Anchor, KeyboardInteractivity, Layer, LayerShell, LayerShellHandler, LayerSurface,
    LayerSurfaceConfigure,
};
use smithay_client_toolkit::shm::slot::{Buffer, SlotPool};
use smithay_client_toolkit::shm::{Shm, ShmHandler};
use smithay_client_toolkit::{delegate_registry, registry_handlers};
use tiny_skia::IntRect;

use crate::canvas::Canvas;
use crate::error::OverlayError;
use crate::handle::OverlayHandle;
use crate::scene::{PresentedAt, Scene, Schedule};

#[derive(Debug, Clone)]
pub struct SurfaceOptions {
    /// `wl_output` name (e.g. "eDP-1"); `None` = the first output.
    pub output: Option<String>,
    pub namespace: &'static str,
}

pub fn spawn<S: Scene>(
    options: SurfaceOptions,
    scene: S,
) -> Result<OverlayHandle<S::Msg>, OverlayError> {
    let (startup_tx, startup_rx) =
        crossbeam_channel::bounded::<Result<(u32, u32), OverlayError>>(1);
    let (tx, channel) = calloop::channel::sync_channel::<S::Msg>(64);

    let parent = tracing::Span::current();
    let thread = thread::spawn(move || {
        let _parent = parent.entered();
        run(options, scene, channel, startup_tx)
    });

    match startup_rx.recv() {
        Ok(Ok(logical)) => Ok(OverlayHandle {
            tx: Some(tx),
            thread: Some(thread),
            dropped: Arc::new(AtomicU64::new(0)),
            logical_size: logical,
        }),
        Ok(Err(e)) => {
            let _ = thread.join();
            Err(e)
        }
        Err(_) => match thread.join().unwrap_or(Err(OverlayError::Panicked)) {
            Ok(()) => Err(OverlayError::Closed),
            Err(e) => Err(e),
        },
    }
}

struct Slot {
    buffer: Buffer,
    size: (u32, u32),
    drawn: Vec<IntRect>,
}

struct State<S: Scene> {
    registry: RegistryState,
    outputs: OutputState,
    compositor: CompositorState,
    shm: Shm,
    presentation: PresentationTimeState,
    layer: Option<LayerSurface>,
    pool: SlotPool,
    buffers: Vec<Slot>,
    on_screen: Vec<IntRect>,
    scene: S,
    logical: (u32, u32),
    scale: u32,
    configured: bool,
    frame_pending: bool,
    needs_redraw: bool,
    exit: Option<Result<(), OverlayError>>,
    qh: QueueHandle<State<S>>,
    loop_handle: Option<LoopHandle<'static, State<S>>>,
    timer: Option<RegistrationToken>,
    our_output: Option<wl_output::WlOutput>,
    feedbacks: Vec<(
        wp_presentation_feedback::WpPresentationFeedback,
        u64,
        Timestamp,
    )>,
}

impl<S: Scene> State<S> {
    fn request_redraw(&mut self) {
        if !self.configured || self.frame_pending || self.loop_handle.is_none() {
            self.needs_redraw = true;
        } else {
            self.draw();
        }
    }

    fn new_buffer(&mut self, bw: u32, bh: u32) -> Result<usize, OverlayError> {
        if self.buffers.len() >= 2
            && let Some(i) = self
                .buffers
                .iter()
                .position(|b| b.buffer.canvas(&mut self.pool).is_some())
        {
            self.buffers.remove(i);
        }
        let (buffer, canvas) = self
            .pool
            .create_buffer(
                bw as i32,
                bh as i32,
                bw as i32 * 4,
                wl_shm::Format::Argb8888,
            )
            .map_err(|e| OverlayError::Shm(e.to_string()))?;
        canvas.fill(0);
        self.buffers.push(Slot {
            buffer,
            size: (bw, bh),
            drawn: Vec::new(),
        });
        tracing::debug!(
            buffer_w = bw,
            buffer_h = bh,
            buffers = self.buffers.len(),
            "shm buffer created"
        );
        Ok(self.buffers.len() - 1)
    }

    fn draw(&mut self) {
        let Some(layer) = self.layer.clone() else {
            return;
        };
        self.needs_redraw = false;
        let (w, h) = self.logical;
        let (bw, bh) = (w * self.scale, h * self.scale);
        let idx = match self
            .buffers
            .iter()
            .position(|b| b.size == (bw, bh) && b.buffer.canvas(&mut self.pool).is_some())
        {
            Some(i) => i,
            None => match self.new_buffer(bw, bh) {
                Ok(i) => i,
                Err(e) => {
                    self.exit = Some(Err(e));
                    return;
                }
            },
        };
        let slot = &mut self.buffers[idx];
        let bytes = slot
            .buffer
            .canvas(&mut self.pool)
            .expect("checked free above");
        let mut canvas =
            Canvas::new(bytes, (w, h), self.scale).expect("buffer sized from logical * scale");
        for r in slot.drawn.drain(..) {
            canvas.clear_px(r);
        }
        let schedule = self.scene.render(&mut canvas, Instant::now());
        let drawn = canvas.take_damage();
        let mark = self.scene.presentation_mark();
        log_drawn(drawn.len(), (bw, bh), schedule, mark.is_some());
        let surface = layer.wl_surface();
        for r in drawn.iter().chain(&self.on_screen) {
            surface.damage_buffer(r.x(), r.y(), r.width() as i32, r.height() as i32);
        }
        slot.drawn = drawn.clone();
        self.on_screen = drawn;
        surface.frame(&self.qh, FrameCallbackData(surface.clone()));
        self.frame_pending = true;
        slot.buffer.attach_to(surface).expect("buffer is free");
        let feedback = mark.map(|_| self.presentation.feedback(surface, &self.qh));
        layer.commit();
        let commit_time = Timestamp::now();
        match (mark, feedback) {
            (Some(m), Some(Ok(fb))) => self.feedbacks.push((fb, m, commit_time)),
            (Some(m), Some(Err(_))) => {
                tracing::debug!(
                    mark = m,
                    { field::REASON } = "feedback_request_failed",
                    "presentation feedback unavailable, using commit time"
                );
                let s =
                    self.scene
                        .on_presented(m, PresentedAt::Commit(commit_time), Instant::now());
                self.apply(s);
            }
            _ => {}
        }
        self.apply(schedule);
    }

    fn apply(&mut self, schedule: Schedule) {
        match schedule {
            Schedule::Idle => {}
            Schedule::NextFrame => self.needs_redraw = true,
            Schedule::At(t) => {
                tracing::trace!(
                    in_us = t.saturating_duration_since(Instant::now()).as_micros() as u64,
                    "redraw scheduled"
                );
                let Some(handle) = self.loop_handle.clone() else {
                    return;
                };
                if let Some(old) = self.timer.take() {
                    handle.remove(old);
                }
                match handle.insert_source(Timer::from_deadline(t), |_, _, state: &mut State<S>| {
                    state.timer = None;
                    state.request_redraw();
                    TimeoutAction::Drop
                }) {
                    Ok(token) => self.timer = Some(token),
                    Err(e) => self.exit = Some(Err(OverlayError::EventLoop(e.to_string()))),
                }
            }
            Schedule::Exit => {
                tracing::debug!("scene requested exit");
                self.exit = Some(Ok(()));
            }
        }
    }
}

impl<S: Scene> CompositorHandler for State<S> {
    fn scale_factor_changed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        new_factor: i32,
    ) {
        self.scale = new_factor.max(1) as u32;
        tracing::info!(scale = self.scale, "buffer scale changed");
        if let Some(layer) = &self.layer {
            let _ = layer.set_buffer_scale(self.scale);
        }
        self.buffers.clear();
        self.on_screen.clear();
        self.request_redraw();
    }

    fn transform_changed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _new_transform: wl_output::Transform,
    ) {
    }

    fn frame(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        time: u32,
    ) {
        self.frame_pending = false;
        let moving = self.scene.step(Instant::now());
        tracing::trace!(
            compositor_ms = time,
            redraw = self.needs_redraw,
            "frame callback"
        );
        if self.needs_redraw || moving {
            self.draw();
        }
    }

    fn surface_enter(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _output: &wl_output::WlOutput,
    ) {
    }

    fn surface_leave(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _output: &wl_output::WlOutput,
    ) {
    }
}

impl<S: Scene> OutputHandler for State<S> {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.outputs
    }

    fn new_output(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _output: wl_output::WlOutput,
    ) {
    }

    fn update_output(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _output: wl_output::WlOutput,
    ) {
    }

    fn output_destroyed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        output: wl_output::WlOutput,
    ) {
        if self.our_output.as_ref() == Some(&output) {
            let name = self
                .outputs
                .info(&output)
                .and_then(|info| info.name)
                .unwrap_or_default();
            self.exit = Some(Err(OverlayError::OutputGone(name)));
        }
    }
}

impl<S: Scene> LayerShellHandler for State<S> {
    fn closed(&mut self, _conn: &Connection, _qh: &QueueHandle<Self>, _layer: &LayerSurface) {
        self.exit = Some(Err(OverlayError::SurfaceClosed));
    }

    fn configure(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _layer: &LayerSurface,
        configure: LayerSurfaceConfigure,
        _serial: u32,
    ) {
        let new_size = configure.new_size;
        let changed = new_size != self.logical || !self.configured;
        if new_size != self.logical {
            self.buffers.clear();
            self.on_screen.clear();
        }
        if let Some(output_logical) = self
            .our_output
            .as_ref()
            .and_then(|o| self.outputs.info(o))
            .and_then(|i| i.logical_size)
            && logical_size_mismatch(output_logical, new_size)
        {
            tracing::warn!(
                output_w = output_logical.0,
                output_h = output_logical.1,
                width = new_size.0,
                height = new_size.1,
                "configure size differs from output logical size"
            );
        }
        log_configured(new_size, self.scale, changed);
        self.logical = new_size;
        self.configured = true;
        self.request_redraw();
    }
}

fn logical_size_mismatch(output_logical: (i32, i32), configured: (u32, u32)) -> bool {
    output_logical != (configured.0 as i32, configured.1 as i32)
}

impl<S: Scene> ShmHandler for State<S> {
    fn shm_state(&mut self) -> &mut Shm {
        &mut self.shm
    }
}

impl<S: Scene> PresentationTimeHandler for State<S> {
    fn presentation_time_state(&mut self) -> &mut PresentationTimeState {
        &mut self.presentation
    }

    fn presented(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        feedback: &wp_presentation_feedback::WpPresentationFeedback,
        _surface: &wl_surface::WlSurface,
        _outputs: Vec<wl_output::WlOutput>,
        time: PresentTime,
        _refresh: u32,
        _seq: u64,
        _flags: WEnum<wp_presentation_feedback::Kind>,
    ) {
        let Some(pos) = self.feedbacks.iter().position(|(fb, _, _)| fb == feedback) else {
            return;
        };
        let (_, mark, commit_time) = self.feedbacks.swap_remove(pos);
        let at = presented_at(&time, commit_time);
        let ts = match at {
            PresentedAt::Presentation(t) | PresentedAt::Commit(t) => t,
        };
        tracing::debug!(
            mark,
            clock = at.clock_name(),
            { field::TS_NS } = ts.as_nanos(),
            commit_ns = commit_time.as_nanos(),
            "frame presented"
        );
        let schedule = self.scene.on_presented(mark, at, Instant::now());
        self.apply(schedule);
    }

    fn discarded(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        feedback: &wp_presentation_feedback::WpPresentationFeedback,
        _surface: &wl_surface::WlSurface,
    ) {
        self.feedbacks.retain(|(fb, _, _)| fb != feedback);
        tracing::debug!(
            pending = self.feedbacks.len(),
            "presentation feedback discarded"
        );
    }
}

impl<S: Scene> ProvidesRegistryState for State<S> {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry
    }
    registry_handlers![OutputState];
}

delegate_registry!(@<S: Scene> State<S>);
smithay_client_toolkit::delegate_dispatch2!(@<S: Scene> State<S>);

#[allow(clippy::type_complexity)]
fn run<S: Scene>(
    options: SurfaceOptions,
    scene: S,
    channel: calloop::channel::Channel<S::Msg>,
    startup: crossbeam_channel::Sender<Result<(u32, u32), OverlayError>>,
) -> Result<(), OverlayError> {
    let setup = (|| -> Result<_, OverlayError> {
        let conn = Connection::connect_to_env()?;
        let (globals, mut queue) = registry_queue_init::<State<S>>(&conn)?;
        let qh = queue.handle();

        let compositor =
            CompositorState::bind(&globals, &qh).map_err(|source| OverlayError::MissingGlobal {
                interface: "wl_compositor",
                source,
            })?;
        let layer_shell =
            LayerShell::bind(&globals, &qh).map_err(|source| OverlayError::MissingGlobal {
                interface: "zwlr_layer_shell_v1",
                source,
            })?;
        let shm = Shm::bind(&globals, &qh).map_err(|source| OverlayError::MissingGlobal {
            interface: "wl_shm",
            source,
        })?;
        let pool = SlotPool::new(4096, &shm).map_err(|e| OverlayError::Shm(e.to_string()))?;
        let outputs = OutputState::new(&globals, &qh);
        let registry = RegistryState::new(&globals);
        let presentation = PresentationTimeState::bind(&globals, &qh);

        let mut state = State {
            registry,
            outputs,
            compositor,
            shm,
            presentation,
            layer: None,
            pool,
            buffers: Vec::new(),
            on_screen: Vec::new(),
            scene,
            logical: (0, 0),
            scale: 1,
            configured: false,
            frame_pending: false,
            needs_redraw: false,
            exit: None,
            qh: qh.clone(),
            loop_handle: None,
            timer: None,
            our_output: None,
            feedbacks: Vec::new(),
        };

        queue
            .roundtrip(&mut state)
            .map_err(|e| OverlayError::Dispatch(e.to_string()))?;
        queue
            .roundtrip(&mut state)
            .map_err(|e| OverlayError::Dispatch(e.to_string()))?;

        let named: Vec<(Option<String>, wl_output::WlOutput)> = state
            .outputs
            .outputs()
            .map(|output| {
                (
                    state.outputs.info(&output).and_then(|info| info.name),
                    output,
                )
            })
            .collect();
        let output = select_output(&named, options.output.as_deref())?;

        let surface = state.compositor.create_surface(&qh);
        let layer = layer_shell.create_layer_surface(
            &qh,
            surface,
            Layer::Overlay,
            Some(options.namespace),
            Some(&output),
        );
        layer.set_anchor(Anchor::TOP | Anchor::BOTTOM | Anchor::LEFT | Anchor::RIGHT);
        layer.set_size(0, 0);
        layer.set_exclusive_zone(-1);
        layer.set_keyboard_interactivity(KeyboardInteractivity::None);
        let region = Region::new(&state.compositor)?;
        layer.set_input_region(Some(region.wl_region()));
        layer.commit();
        state.layer = Some(layer);
        state.our_output = Some(output);

        while !state.configured && state.exit.is_none() {
            queue
                .blocking_dispatch(&mut state)
                .map_err(|e| OverlayError::Dispatch(e.to_string()))?;
        }
        if let Some(Err(e)) = state.exit.take() {
            return Err(e);
        }

        Ok((state, conn, queue))
    })();

    let (mut state, conn, queue) = match setup {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(
                error = %e,
                namespace = options.namespace,
                output = options.output.as_deref().unwrap_or(""),
                "overlay setup failed"
            );
            let _ = startup.send(Err(e));
            return Ok(());
        }
    };

    if startup.send(Ok(state.logical)).is_err() {
        return Ok(());
    }

    let mut event_loop: EventLoop<'static, State<S>> =
        EventLoop::try_new().map_err(|e| OverlayError::EventLoop(e.to_string()))?;
    state.loop_handle = Some(event_loop.handle());

    WaylandSource::new(conn, queue)
        .insert(event_loop.handle())
        .map_err(|e| OverlayError::EventLoop(e.to_string()))?;

    event_loop
        .handle()
        .insert_source(channel, |event, _, state: &mut State<S>| match event {
            calloop::channel::Event::Msg(msg) => {
                state.scene.on_msg(msg, Instant::now());
                state.request_redraw();
            }
            calloop::channel::Event::Closed => {
                tracing::info!("overlay close requested");
                state.exit = Some(Ok(()));
            }
        })
        .map_err(|e| OverlayError::EventLoop(e.to_string()))?;

    state.request_redraw();

    while state.exit.is_none() {
        event_loop
            .dispatch(None, &mut state)
            .map_err(|e| OverlayError::EventLoop(e.to_string()))?;
    }

    let result = state.exit.take().unwrap_or(Ok(()));
    log_exit(&result);
    result
}

pub(crate) fn log_configured(new_size: (u32, u32), scale: u32, changed: bool) {
    if changed {
        tracing::info!(
            width = new_size.0,
            height = new_size.1,
            scale,
            "layer surface configured"
        );
    } else {
        tracing::debug!(
            width = new_size.0,
            height = new_size.1,
            "layer surface reconfigured"
        );
    }
}

pub(crate) fn schedule_name(s: Schedule) -> &'static str {
    match s {
        Schedule::Idle => "idle",
        Schedule::NextFrame => "next_frame",
        Schedule::At(_) => "at",
        Schedule::Exit => "exit",
    }
}

pub(crate) fn log_drawn(
    damage_rects: usize,
    buffer: (u32, u32),
    schedule: Schedule,
    feedback: bool,
) {
    tracing::trace!(
        damage_rects,
        buffer_w = buffer.0,
        buffer_h = buffer.1,
        schedule = schedule_name(schedule),
        feedback,
        "frame drawn"
    );
}

pub(crate) fn log_exit(result: &Result<(), OverlayError>) {
    match result {
        Ok(()) => tracing::info!("overlay exited"),
        Err(e) => tracing::error!(error = %e, "overlay exited with error"),
    }
}

pub(crate) fn select_output<T: Clone>(
    outputs: &[(Option<String>, T)],
    wanted: Option<&str>,
) -> Result<T, OverlayError> {
    match wanted {
        None => outputs
            .first()
            .map(|(_, value)| value.clone())
            .ok_or_else(|| OverlayError::OutputNotFound {
                wanted: String::new(),
                available: Vec::new(),
            }),
        Some(name) => outputs
            .iter()
            .find(|(output_name, _)| output_name.as_deref() == Some(name))
            .map(|(_, value)| value.clone())
            .ok_or_else(|| OverlayError::OutputNotFound {
                wanted: name.to_string(),
                available: outputs.iter().filter_map(|(n, _)| n.clone()).collect(),
            }),
    }
}

/// `Presentation` only when the compositor's presentation clock is `CLOCK_MONOTONIC`.
pub(crate) fn presented_at(time: &PresentTime, commit: Timestamp) -> PresentedAt {
    if time.clk_id == nix::time::ClockId::CLOCK_MONOTONIC.as_raw() as u32 {
        PresentedAt::Presentation(Timestamp(std::time::Duration::new(
            time.tv_sec,
            time.tv_nsec,
        )))
    } else {
        PresentedAt::Commit(commit)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    use eye_log::Value;
    use eye_log::testing::capture_logs;

    use super::*;

    #[test]
    fn test_logs_layer_surface_configured_at_info() {
        let (_, records) = capture_logs(tracing::Level::TRACE, || {
            log_configured((1920, 1080), 2, true);
        });
        assert_eq!(records.len(), 1);
        let rec = &records[0];
        assert_eq!(rec.level, eye_log::Level::Info);
        assert_eq!(rec.message, "layer surface configured");
        assert_eq!(rec.fields["width"], Value::U64(1920));
        assert_eq!(rec.fields["height"], Value::U64(1080));
        assert_eq!(rec.fields["scale"], Value::U64(2));

        let (_, records) = capture_logs(tracing::Level::TRACE, || {
            log_configured((1920, 1080), 2, false);
        });
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].level, eye_log::Level::Debug);
        assert_eq!(records[0].message, "layer surface reconfigured");
    }

    #[test]
    fn test_logs_frame_drawn_at_trace() {
        let (_, records) = capture_logs(tracing::Level::TRACE, || {
            log_drawn(3, (3840, 2160), Schedule::NextFrame, true);
        });
        assert_eq!(records.len(), 1);
        let rec = &records[0];
        assert_eq!(rec.level, eye_log::Level::Trace);
        assert_eq!(rec.message, "frame drawn");
        assert_eq!(rec.fields["damage_rects"], Value::U64(3));
        assert_eq!(rec.fields["schedule"], Value::Str("next_frame".to_string()));
        assert_eq!(rec.fields["feedback"], Value::Bool(true));
    }

    #[test]
    fn test_logs_overlay_exited_at_error() {
        let (_, records) = capture_logs(tracing::Level::TRACE, || {
            log_exit(&Err(OverlayError::SurfaceClosed));
        });
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].level, eye_log::Level::Error);
        match &records[0].fields["error"] {
            Value::Str(s) => assert!(s.contains("closed the layer surface")),
            other => panic!("expected Str, got {other:?}"),
        }

        let (_, records) = capture_logs(tracing::Level::TRACE, || {
            log_exit(&Ok(()));
        });
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].level, eye_log::Level::Info);
    }

    #[test]
    fn test_logical_size_mismatch_detects_difference() {
        assert!(logical_size_mismatch((1920, 1080), (1920, 1051)));
        assert!(!logical_size_mismatch((1920, 1080), (1920, 1080)));
    }

    #[test]
    fn test_select_output_by_name() {
        let outputs = [
            (Some("HDMI-A-1".to_string()), 1),
            (Some("eDP-1".to_string()), 2),
        ];
        assert_eq!(select_output(&outputs, Some("eDP-1")).expect("found"), 2);
        assert_eq!(select_output(&outputs, None).expect("default is first"), 1);
        match select_output(&outputs, Some("DP-3")) {
            Err(OverlayError::OutputNotFound { wanted, available }) => {
                assert_eq!(wanted, "DP-3");
                assert_eq!(available, vec!["HDMI-A-1".to_string(), "eDP-1".to_string()]);
            }
            other => panic!("expected OutputNotFound, got {other:?}"),
        }
    }

    #[test]
    fn test_presented_at_requires_monotonic_clock() {
        let commit = Timestamp::from_nanos(999);
        let monotonic = PresentTime {
            clk_id: nix::time::ClockId::CLOCK_MONOTONIC.as_raw() as u32,
            tv_sec: 12,
            tv_nsec: 5,
        };
        assert_eq!(
            presented_at(&monotonic, commit),
            PresentedAt::Presentation(Timestamp(Duration::new(12, 5)))
        );

        let realtime = PresentTime {
            clk_id: nix::time::ClockId::CLOCK_REALTIME.as_raw() as u32,
            tv_sec: 12,
            tv_nsec: 5,
        };
        assert_eq!(presented_at(&realtime, commit), PresentedAt::Commit(commit));
    }

    struct CountingScene {
        count: Arc<AtomicUsize>,
    }

    impl Scene for CountingScene {
        type Msg = ();

        fn on_msg(&mut self, _msg: (), _now: Instant) {}

        fn render(
            &mut self,
            _canvas: &mut crate::canvas::Canvas<'_>,
            _now: Instant,
        ) -> crate::scene::Schedule {
            self.count.fetch_add(1, Ordering::SeqCst);
            crate::scene::Schedule::At(Instant::now() + Duration::from_millis(100))
        }
    }

    struct CircleScene;

    impl Scene for CircleScene {
        type Msg = ();

        fn on_msg(&mut self, _msg: (), _now: Instant) {}

        fn render(
            &mut self,
            canvas: &mut crate::canvas::Canvas<'_>,
            _now: Instant,
        ) -> crate::scene::Schedule {
            let (w, h) = canvas.logical_size();
            canvas.fill_circle(
                nalgebra::Point2::new(w as f64 / 2.0, h as f64 / 2.0),
                20.0,
                crate::canvas::Rgba {
                    r: 255,
                    g: 0,
                    b: 0,
                    a: 255,
                },
            );
            crate::scene::Schedule::Idle
        }
    }

    #[test]
    #[ignore = "needs wayland"]
    fn test_overlay_configures_full_output_size() {
        let output = std::env::var("EYE_OUTPUT").unwrap_or_else(|_| "eDP-1".to_string());
        let handle = spawn(
            SurfaceOptions {
                output: Some(output),
                namespace: "eye-overlay",
            },
            CircleScene,
        )
        .expect("spawn succeeds");
        // 1920x1080 is the dev-machine eDP-1 logical size; EYE_OUTPUT can point elsewhere.
        assert_eq!(handle.logical_size(), (1920, 1080));
        std::thread::sleep(Duration::from_secs(3));
        assert!(handle.close().is_ok());
    }

    #[test]
    #[ignore = "needs wayland"]
    fn test_unknown_output_fails_synchronously() {
        let count = Arc::new(AtomicUsize::new(0));
        let result = spawn(
            SurfaceOptions {
                output: Some("NOPE-1".to_string()),
                namespace: "eye-overlay",
            },
            CountingScene { count },
        );
        assert!(matches!(result, Err(OverlayError::OutputNotFound { .. })));
    }

    struct MovingScene {
        remaining_steps: Arc<AtomicUsize>,
        draws: Arc<AtomicUsize>,
    }

    impl Scene for MovingScene {
        type Msg = ();

        fn on_msg(&mut self, _msg: (), _now: Instant) {}

        fn step(&mut self, _now: Instant) -> bool {
            self.remaining_steps
                .try_update(Ordering::SeqCst, Ordering::SeqCst, |n| {
                    (n > 0).then(|| n - 1)
                })
                .is_ok()
        }

        fn render(
            &mut self,
            _canvas: &mut crate::canvas::Canvas<'_>,
            _now: Instant,
        ) -> crate::scene::Schedule {
            self.draws.fetch_add(1, Ordering::SeqCst);
            crate::scene::Schedule::Idle
        }
    }

    #[test]
    #[ignore = "needs wayland"]
    fn test_frame_callback_draws_only_while_moving() {
        let output = std::env::var("EYE_OUTPUT").unwrap_or_else(|_| "eDP-1".to_string());
        let remaining_steps = Arc::new(AtomicUsize::new(5));
        let draws = Arc::new(AtomicUsize::new(0));
        let handle = spawn(
            SurfaceOptions {
                output: Some(output),
                namespace: "eye-overlay",
            },
            MovingScene {
                remaining_steps: remaining_steps.clone(),
                draws: draws.clone(),
            },
        )
        .expect("spawn succeeds");
        std::thread::sleep(Duration::from_secs(1));
        let settled = draws.load(Ordering::SeqCst);
        assert!(settled >= 6);
        std::thread::sleep(Duration::from_millis(500));
        assert_eq!(draws.load(Ordering::SeqCst), settled);
        assert!(handle.close().is_ok());
    }

    #[test]
    #[ignore = "needs wayland"]
    fn test_first_render_schedule_is_applied() {
        let output = std::env::var("EYE_OUTPUT").unwrap_or_else(|_| "eDP-1".to_string());
        let count = Arc::new(AtomicUsize::new(0));
        let handle = spawn(
            SurfaceOptions {
                output: Some(output),
                namespace: "eye-overlay",
            },
            CountingScene {
                count: count.clone(),
            },
        )
        .expect("spawn succeeds");
        std::thread::sleep(Duration::from_secs(1));
        assert!(count.load(Ordering::SeqCst) >= 2);
        assert!(handle.close().is_ok());
    }
}
