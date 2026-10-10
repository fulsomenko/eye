use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use eye::config::{CameraConfig, CameraFormat, Config};
use eye_calibration::checkerboard::{BoardObservation, DetectorConfig, detect_board};
use eye_calibration::corners::BoardSpec;
use eye_calibration::intrinsics::{
    IntrinsicsConfig, IntrinsicsFit, apply_intrinsics, calibrate_intrinsics,
};
use eye_calibration::stereo::{
    StereoConfig, StereoFit, StereoView, apply_stereo, calibrate_stereo, is_static,
};
use eye_calibration::store::{ProfileStore, write_rig};
use eye_capture::{CaptureError, FrameSource};
use eye_core::image::GrayView;
use eye_core::{CameraId, Frame, Illumination, PixelFormat, Rig};
use eye_detect::mjpeg::decode_mjpeg_gray;
use eye_geometry::camera::Intrinsics;
use eye_platform::EmitterGuard;

use crate::commands::emitter::emitter_guards;
use crate::commands::probe::{ProbeReport, Probes, collect};
use crate::ctx::Ctx;
use crate::rig::{self, RigSource};

#[derive(Debug, clap::Args)]
pub struct Args {
    #[command(subcommand)]
    pub mode: Mode,
}

#[derive(Debug, clap::Subcommand)]
pub enum Mode {
    /// Calibrate one camera's intrinsics and distortion
    Intrinsics(IntrinsicsArgs),
    /// Calibrate the RGB-IR stereo pose (needs calibrated intrinsics for both cameras)
    Stereo(StereoArgs),
}

#[derive(Debug, clap::Args)]
pub struct IntrinsicsArgs {
    /// Camera id from [[camera]] (rgb or ir)
    #[arg(long)]
    pub camera: String,
    /// Views to collect
    #[arg(long, default_value_t = 15)]
    pub views: usize,
    #[command(flatten)]
    pub board: BoardArgs,
}

#[derive(Debug, clap::Args)]
pub struct StereoArgs {
    #[arg(long, default_value = "rgb")]
    pub anchor: String,
    #[arg(long, default_value = "ir")]
    pub other: String,
    #[arg(long, default_value_t = 8)]
    pub views: usize,
    #[command(flatten)]
    pub board: BoardArgs,
}

#[derive(Debug, clap::Args)]
pub struct BoardArgs {
    /// a4: laser-printed assets/calibration/a4-laser.pdf; phone: assets/calibration/phone-nothing-2.png (RGB only)
    #[arg(long, value_enum, default_value_t = Board::A4)]
    pub board: Board,
    /// Square size override in mm (a print not at 100 %)
    #[arg(long, value_name = "MM")]
    pub square_mm: Option<f64>,
    /// Give up collecting after this many seconds and fit what was collected
    #[arg(long, default_value_t = 300)]
    pub timeout_s: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Board {
    A4,
    Phone,
}

pub const A4_SQUARE_MM: f64 = 25.0;
pub const PHONE_SQUARE_MM: f64 = 8.629;

impl Board {
    /// 9x6 inner corners; `square_mm` overrides the board's size.
    pub fn spec(self, square_mm: Option<f64>) -> BoardSpec {
        let default_mm = match self {
            Board::A4 => A4_SQUARE_MM,
            Board::Phone => PHONE_SQUARE_MM,
        };
        BoardSpec {
            inner_cols: 9,
            inner_rows: 6,
            square_mm: square_mm.unwrap_or(default_mm),
        }
    }
}

pub fn check_board(board: Board, ir_involved: bool) -> anyhow::Result<()> {
    if board == Board::Phone && ir_involved {
        anyhow::bail!(
            "the phone board is not visible in IR (an OLED emits no near-IR); print assets/calibration/a4-laser.pdf on a laser printer and use --board a4"
        );
    }
    Ok(())
}

pub fn check_stereo_source(source: RigSource, anchor: &str, other: &str) -> anyhow::Result<()> {
    if source == RigSource::Nominal {
        anyhow::bail!(
            "stereo needs calibrated intrinsics: run `eye calibrate-camera intrinsics --camera {anchor}` and `--camera {other}` first"
        );
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CollectConfig {
    pub target_views: usize,
    /// 3 consecutive detections.
    pub static_frames: usize,
    /// `is_static` tolerance.
    pub static_tol_px: f64,
    /// Mean corner distance to every accepted view.
    pub min_novelty_px: f64,
}

#[derive(Debug)]
pub enum Detection {
    Skipped,
    NoBoard,
    Board(BoardObservation),
}

/// IR dark frames are `Skipped` (they never break a static streak); Gray8 frames are detected
/// directly, MJPG frames after `decode_mjpeg_gray`; other formats error.
pub fn detect_frame(
    frame: &Frame,
    spec: &BoardSpec,
    cfg: &DetectorConfig,
) -> anyhow::Result<Detection> {
    let header = frame.header();
    let observation = match header.format {
        PixelFormat::Gray8 if header.illumination == Illumination::IrDark => {
            return Ok(Detection::Skipped);
        }
        PixelFormat::Gray8 => {
            let view = GrayView::from_frame(frame)?;
            detect_board(&view, spec, cfg)?
        }
        PixelFormat::Mjpeg => {
            let image = decode_mjpeg_gray(frame.data())?;
            detect_board(&image.view(), spec, cfg)?
        }
        PixelFormat::Rgb8 => anyhow::bail!("unsupported pixel format"),
    };
    Ok(match observation {
        Some(obs) => Detection::Board(obs),
        None => Detection::NoBoard,
    })
}

fn mean_corner_distance(a: &BoardObservation, b: &BoardObservation) -> f64 {
    let sum: f64 = a
        .corners
        .iter()
        .zip(&b.corners)
        .map(|(ca, cb)| (ca.image_px - cb.image_px).norm())
        .sum();
    sum / a.corners.len() as f64
}

fn update_streak(
    last: &mut Option<BoardObservation>,
    streak: &mut usize,
    obs: Option<BoardObservation>,
    tol_px: f64,
) {
    match obs {
        None => {
            *last = None;
            *streak = 0;
        }
        Some(cur) => {
            *streak = match last {
                Some(prev) if is_static(prev, &cur, tol_px) => *streak + 1,
                _ => 1,
            };
            *last = Some(cur);
        }
    }
}

#[derive(Debug)]
pub struct ViewCollector {
    cfg: CollectConfig,
    last: Option<BoardObservation>,
    streak: usize,
    views: Vec<BoardObservation>,
}

impl ViewCollector {
    pub fn new(cfg: CollectConfig) -> Self {
        Self {
            cfg,
            last: None,
            streak: 0,
            views: Vec::new(),
        }
    }

    /// Feed one detection (`None` = no board); true when this call accepted a view.
    pub fn push(&mut self, obs: Option<BoardObservation>) -> bool {
        let is_none = obs.is_none();
        update_streak(
            &mut self.last,
            &mut self.streak,
            obs,
            self.cfg.static_tol_px,
        );
        if is_none {
            return false;
        }
        let cur = self.last.clone().expect("just set to Some");
        if self.streak < self.cfg.static_frames
            || self.done()
            || !self
                .views
                .iter()
                .all(|v| mean_corner_distance(&cur, v) >= self.cfg.min_novelty_px)
        {
            return false;
        }
        self.views.push(cur);
        self.streak = 0;
        true
    }

    pub fn done(&self) -> bool {
        self.views.len() >= self.cfg.target_views
    }

    pub fn views(&self) -> &[BoardObservation] {
        &self.views
    }
}

#[derive(Debug)]
pub struct StereoCollector {
    cfg: CollectConfig,
    a_last: Option<BoardObservation>,
    a_streak: usize,
    b_last: Option<BoardObservation>,
    b_streak: usize,
    views: Vec<StereoView>,
}

impl StereoCollector {
    pub fn new(cfg: CollectConfig) -> Self {
        Self {
            cfg,
            a_last: None,
            a_streak: 0,
            b_last: None,
            b_streak: 0,
            views: Vec::new(),
        }
    }

    pub fn push_a(&mut self, obs: Option<BoardObservation>) -> bool {
        update_streak(
            &mut self.a_last,
            &mut self.a_streak,
            obs,
            self.cfg.static_tol_px,
        );
        self.try_accept()
    }

    pub fn push_b(&mut self, obs: Option<BoardObservation>) -> bool {
        update_streak(
            &mut self.b_last,
            &mut self.b_streak,
            obs,
            self.cfg.static_tol_px,
        );
        self.try_accept()
    }

    fn try_accept(&mut self) -> bool {
        if self.a_streak < self.cfg.static_frames || self.b_streak < self.cfg.static_frames {
            return false;
        }
        let (Some(a), Some(b)) = (self.a_last.clone(), self.b_last.clone()) else {
            return false;
        };
        let novel = self
            .views
            .iter()
            .all(|v| mean_corner_distance(&a, &v.a) >= self.cfg.min_novelty_px);
        if !novel {
            return false;
        }
        self.views.push(StereoView { a, b });
        self.a_streak = 0;
        self.b_streak = 0;
        true
    }

    pub fn done(&self) -> bool {
        self.views.len() >= self.cfg.target_views
    }

    pub fn views(&self) -> &[StereoView] {
        &self.views
    }
}

pub fn intrinsics_summary(camera: &str, fit: &IntrinsicsFit, rejected: usize) -> String {
    let intr = &fit.intrinsics;
    let d = &intr.distortion;
    let sigma_fx = fit.cov[(0, 0)].sqrt();
    let module = match fit.module {
        eye_calibration::intrinsics::LensModule::Fov75_8 => "75.8 deg (2.7 mm)",
        eye_calibration::intrinsics::LensModule::Fov87 => "87 deg (6 mm)",
        eye_calibration::intrinsics::LensModule::Unknown => "unknown (outside both Dell modules)",
    };
    let sign = if fit.prior_z >= 0.0 { "+" } else { "-" };
    format!(
        "camera {camera} {w}x{h}: fx {fx:.2} fy {fy:.2} cx {cx:.2} cy {cy:.2} px (sigma fx {sigma_fx:.2})\n\
         distortion k1 {k1:.4} k2 {k2:.4} p1 {p1:.4} p2 {p2:.4} k3 {k3:.4}\n\
         reprojection rms {rms:.2} px over {views} views ({rejected} rejected)\n\
         Dell module: {module}, prior z {sign}{prior_z:.2}",
        camera = camera,
        w = intr.width,
        h = intr.height,
        fx = intr.fx,
        fy = intr.fy,
        cx = intr.cx,
        cy = intr.cy,
        sigma_fx = sigma_fx,
        k1 = d.k1,
        k2 = d.k2,
        p1 = d.p1,
        p2 = d.p2,
        k3 = d.k3,
        rms = fit.rms_px,
        views = fit.views_used.len(),
        rejected = rejected,
        module = module,
        sign = sign,
        prior_z = fit.prior_z.abs(),
    )
}

pub fn stereo_summary(fit: &StereoFit) -> String {
    let baseline = fit.b_from_a.translation.vector.norm();
    let rotation = fit.b_from_a.rotation.angle().to_degrees();
    format!(
        "baseline {baseline:.2} mm, rotation {rotation:.2} deg, rms {rms:.2} px over {views} views",
        rms = fit.rms_px,
        views = fit.views_used.len(),
    )
}

fn require_camera<'a>(config: &'a Config, id: &str) -> anyhow::Result<&'a CameraConfig> {
    config.camera(id).ok_or_else(|| {
        let ids = config
            .cameras
            .iter()
            .map(|c| c.id.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        anyhow::anyhow!("camera {id} is not in [[camera]]; configured: {ids}")
    })
}

fn spawn_captures(
    config: &Config,
    routes: &[(CameraId, crossbeam_channel::Sender<Frame>)],
    stop: &Arc<AtomicBool>,
) -> anyhow::Result<Vec<std::thread::JoinHandle<()>>> {
    let sources = eye::tracker::open_sources(config)?;
    let mut handles = Vec::with_capacity(sources.len());
    for mut source in sources {
        let id = source.camera().id.clone();
        let tx = routes
            .iter()
            .find(|(route_id, _)| route_id == &id)
            .map(|(_, tx)| tx.clone());
        let stop = Arc::clone(stop);
        let handle = eye_core::log::spawn_in_current_span(format!("eye-cal-{id}"), move || {
            while !stop.load(Ordering::Relaxed) {
                match source.next_frame() {
                    Ok(frame) => {
                        if let Some(tx) = &tx {
                            let _ = tx.try_send(frame);
                        }
                    }
                    Err(CaptureError::Timeout { .. }) => continue,
                    Err(err) => {
                        tracing::warn!(camera = %id, error = %err, "capture ended");
                        break;
                    }
                }
            }
        })?;
        handles.push(handle);
    }
    Ok(handles)
}

fn join_captures(stop: &Arc<AtomicBool>, handles: Vec<std::thread::JoinHandle<()>>) {
    stop.store(true, Ordering::Relaxed);
    for handle in handles {
        let _ = handle.join();
    }
}

fn collect_intrinsics(
    config: &Config,
    camera: &CameraConfig,
    shutdown: &crossbeam_channel::Receiver<()>,
    args: &IntrinsicsArgs,
) -> anyhow::Result<Vec<BoardObservation>> {
    let spec = args.board.board.spec(args.board.square_mm);
    let cfg = CollectConfig {
        target_views: args.views,
        static_frames: 3,
        static_tol_px: 0.5,
        min_novelty_px: 20.0,
    };
    let mut collector = ViewCollector::new(cfg);

    let (tx, rx) = crossbeam_channel::bounded(4);
    let stop = Arc::new(AtomicBool::new(false));
    let handles = spawn_captures(config, &[(camera.id.clone(), tx)], &stop)?;
    let timeout = crossbeam_channel::after(Duration::from_secs(args.board.timeout_s));

    while !collector.done() {
        crossbeam_channel::select! {
            recv(shutdown) -> _ => break,
            recv(timeout) -> _ => break,
            recv(rx) -> msg => match msg {
                Ok(frame) => handle_detection(&mut collector, &frame, &spec, camera.id.as_str())?,
                Err(_) => break,
            },
        }
    }

    join_captures(&stop, handles);
    Ok(collector.views().to_vec())
}

fn handle_detection(
    collector: &mut ViewCollector,
    frame: &Frame,
    spec: &BoardSpec,
    camera_id: &str,
) -> anyhow::Result<()> {
    match detect_frame(frame, spec, &DetectorConfig::default())? {
        Detection::Skipped => {}
        Detection::NoBoard => {
            collector.push(None);
        }
        Detection::Board(obs) => {
            if collector.push(Some(obs)) {
                eprintln!(
                    "view {}/{} accepted ({camera_id})",
                    collector.views().len(),
                    collector.cfg.target_views
                );
            }
        }
    }
    Ok(())
}

fn collect_stereo(
    config: &Config,
    anchor: &CameraConfig,
    other: &CameraConfig,
    shutdown: &crossbeam_channel::Receiver<()>,
    args: &StereoArgs,
) -> anyhow::Result<Vec<StereoView>> {
    let spec = args.board.board.spec(args.board.square_mm);
    let cfg = CollectConfig {
        target_views: args.views,
        static_frames: 3,
        static_tol_px: 0.5,
        min_novelty_px: 20.0,
    };
    let mut collector = StereoCollector::new(cfg);

    let (tx_a, rx_a) = crossbeam_channel::bounded(4);
    let (tx_b, rx_b) = crossbeam_channel::bounded(4);
    let stop = Arc::new(AtomicBool::new(false));
    let handles = spawn_captures(
        config,
        &[(anchor.id.clone(), tx_a), (other.id.clone(), tx_b)],
        &stop,
    )?;
    let timeout = crossbeam_channel::after(Duration::from_secs(args.board.timeout_s));

    while !collector.done() {
        crossbeam_channel::select! {
            recv(shutdown) -> _ => break,
            recv(timeout) -> _ => break,
            recv(rx_a) -> msg => match msg {
                Ok(frame) => handle_stereo_detection(&mut collector, true, &frame, &spec, anchor.id.as_str())?,
                Err(_) => break,
            },
            recv(rx_b) -> msg => match msg {
                Ok(frame) => handle_stereo_detection(&mut collector, false, &frame, &spec, other.id.as_str())?,
                Err(_) => break,
            },
        }
    }

    join_captures(&stop, handles);
    Ok(collector.views().to_vec())
}

fn handle_stereo_detection(
    collector: &mut StereoCollector,
    is_a: bool,
    frame: &Frame,
    spec: &BoardSpec,
    camera_id: &str,
) -> anyhow::Result<()> {
    let detection = detect_frame(frame, spec, &DetectorConfig::default())?;
    let obs = match detection {
        Detection::Skipped => return Ok(()),
        Detection::NoBoard => None,
        Detection::Board(obs) => Some(obs),
    };
    let accepted = if is_a {
        collector.push_a(obs)
    } else {
        collector.push_b(obs)
    };
    if accepted {
        eprintln!(
            "view {}/{} accepted ({camera_id})",
            collector.views().len(),
            collector.cfg.target_views
        );
    }
    Ok(())
}

fn fit_and_save_intrinsics(
    ctx: &Ctx,
    store: &ProfileStore,
    rig: Rig,
    camera: &CameraConfig,
    views: &[BoardObservation],
) -> anyhow::Result<String> {
    let collected = views.len();
    let fit = calibrate_intrinsics(
        camera.size[0],
        camera.size[1],
        views,
        &IntrinsicsConfig::default(),
    )?;
    let rejected = collected - fit.views_used.len();
    let summary = intrinsics_summary(camera.id.as_str(), &fit, rejected);
    let rig = apply_intrinsics(rig, &camera.id, &fit)?;
    let path = save_rig(ctx, store, &rig)?;
    Ok(format!("{summary}\nrig saved to {}", path.display()))
}

fn fit_and_save_stereo(
    ctx: &Ctx,
    store: &ProfileStore,
    rig: Rig,
    anchor: &CameraId,
    other: &CameraId,
    views: &[StereoView],
) -> anyhow::Result<String> {
    let intr_a = Intrinsics::from_camera_model(
        rig.camera(anchor.as_str())
            .ok_or_else(|| anyhow::anyhow!("camera {anchor} is not in the rig"))?,
    );
    let intr_b = Intrinsics::from_camera_model(
        rig.camera(other.as_str())
            .ok_or_else(|| anyhow::anyhow!("camera {other} is not in the rig"))?,
    );
    let fit = calibrate_stereo(&intr_a, &intr_b, views, &StereoConfig::default())?;
    let summary = stereo_summary(&fit);
    let rig = apply_stereo(rig, anchor, other, &fit)?;
    let path = save_rig(ctx, store, &rig)?;
    Ok(format!("{summary}\nrig saved to {}", path.display()))
}

fn save_rig(ctx: &Ctx, store: &ProfileStore, rig: &Rig) -> anyhow::Result<std::path::PathBuf> {
    Ok(match &ctx.output {
        Some(path) => {
            write_rig(path, rig)?;
            path.clone()
        }
        None => store.save_rig(rig)?,
    })
}

pub fn run(ctx: &Ctx, args: Args) -> anyhow::Result<()> {
    let config = Config::load(ctx.config_path.as_deref())?;
    let shutdown = crate::shutdown::install()?;
    let report: ProbeReport = collect(&Probes::system());
    let output = rig::target_output(&config, &report.outputs)?;
    let store = ProfileStore::open_default()?;
    let (current_rig, source) = rig::resolve_rig(&config, output, &store)?;

    let summary = match args.mode {
        Mode::Intrinsics(a) => {
            let camera = require_camera(&config, &a.camera)?.clone();
            let ir_involved = camera.format == CameraFormat::Gray;
            check_board(a.board.board, ir_involved)?;
            let _guards = emitter_guards(
                &rig::msxu_cameras(&config),
                &report.cameras,
                EmitterGuard::enable,
            )?;
            let views = collect_intrinsics(&config, &camera, shutdown.receiver(), &a)?;
            fit_and_save_intrinsics(ctx, &store, current_rig, &camera, &views)?
        }
        Mode::Stereo(a) => {
            let anchor = require_camera(&config, &a.anchor)?.clone();
            let other = require_camera(&config, &a.other)?.clone();
            let ir_involved =
                anchor.format == CameraFormat::Gray || other.format == CameraFormat::Gray;
            check_board(a.board.board, ir_involved)?;
            check_stereo_source(source, &a.anchor, &a.other)?;
            let _guards = emitter_guards(
                &rig::msxu_cameras(&config),
                &report.cameras,
                EmitterGuard::enable,
            )?;
            let views = collect_stereo(&config, &anchor, &other, shutdown.receiver(), &a)?;
            fit_and_save_stereo(ctx, &store, current_rig, &anchor.id, &other.id, &views)?
        }
    };

    println!("{summary}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use eye_core::{FrameHeader, Timestamp};
    use nalgebra::{
        Isometry3, Matrix6, Point2, Point3, SMatrix, Translation3, UnitQuaternion, Vector3,
    };

    use super::*;
    use eye_calibration::checkerboard::BoardCorner;
    use eye_calibration::intrinsics::LensModule;

    fn board_spec() -> BoardSpec {
        BoardSpec {
            inner_cols: 9,
            inner_rows: 6,
            square_mm: 25.0,
        }
    }

    fn obs(dx: f64) -> BoardObservation {
        let spec = board_spec();
        let mut corners = Vec::new();
        for j in 0..spec.inner_rows {
            for i in 0..spec.inner_cols {
                corners.push(BoardCorner {
                    grid: (i, j),
                    object_mm: Point3::new(25.0 * f64::from(i), 25.0 * f64::from(j), 0.0),
                    image_px: Point2::new(
                        100.0 + 30.0 * f64::from(i) + dx,
                        80.0 + 30.0 * f64::from(j),
                    ),
                });
            }
        }
        BoardObservation { spec, corners }
    }

    fn board_frame(illumination: Illumination) -> Frame {
        let (w, h) = (640u32, 360u32);
        let mut data = vec![0u8; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                let px = f64::from(x) + 0.5;
                let py = f64::from(y) + 0.5;
                let a = ((px - 140.0) / 40.0).floor() as i64 + 1;
                let b = ((py - 80.0) / 40.0).floor() as i64 + 1;
                let value = if (0..=9).contains(&a) && (0..=6).contains(&b) && (a + b) % 2 == 0 {
                    30u8
                } else {
                    220u8
                };
                data[(y * w + x) as usize] = value;
            }
        }
        Frame::new(
            FrameHeader {
                camera: CameraId::from("ir"),
                seq: 0,
                timestamp: Timestamp::from_nanos(0),
                width: w,
                height: h,
                format: PixelFormat::Gray8,
                illumination,
            },
            data.into(),
        )
        .expect("valid gray frame")
    }

    #[test]
    fn test_view_collector_accepts_after_three_static_frames() {
        let cfg = CollectConfig {
            target_views: 15,
            static_frames: 3,
            static_tol_px: 0.5,
            min_novelty_px: 20.0,
        };
        let mut collector = ViewCollector::new(cfg);
        assert!(!(collector.push(Some(obs(0.0)))));
        assert!(!(collector.push(Some(obs(0.0)))));
        assert!(collector.push(Some(obs(0.0))));
        assert_eq!(collector.views().len(), 1);
    }

    #[test]
    fn test_view_collector_resets_on_motion_and_on_miss() {
        let cfg = CollectConfig {
            target_views: 15,
            static_frames: 3,
            static_tol_px: 0.5,
            min_novelty_px: 20.0,
        };
        let mut collector = ViewCollector::new(cfg);
        assert!(!(collector.push(Some(obs(0.0)))));
        assert!(!(collector.push(Some(obs(0.0)))));
        assert!(!(collector.push(None)));
        assert!(!(collector.push(Some(obs(0.0)))));
        assert!(!(collector.push(Some(obs(0.0)))));
        assert_eq!(collector.views().len(), 0);

        let mut collector = ViewCollector::new(cfg);
        assert!(!(collector.push(Some(obs(0.0)))));
        assert!(!(collector.push(Some(obs(1.0)))));
        assert!(!(collector.push(Some(obs(1.0)))));
        assert!(collector.push(Some(obs(1.0))));
        assert_eq!(collector.views().len(), 1);
    }

    #[test]
    fn test_view_collector_rejects_duplicate_pose() {
        let cfg = CollectConfig {
            target_views: 15,
            static_frames: 3,
            static_tol_px: 0.5,
            min_novelty_px: 20.0,
        };
        let mut collector = ViewCollector::new(cfg);
        collector.push(Some(obs(0.0)));
        collector.push(Some(obs(0.0)));
        assert!(collector.push(Some(obs(0.0))));
        assert_eq!(collector.views().len(), 1);

        assert!(!(collector.push(Some(obs(5.0)))));
        assert!(!(collector.push(Some(obs(5.0)))));
        assert!(!(collector.push(Some(obs(5.0)))));
        assert_eq!(collector.views().len(), 1);

        assert!(!(collector.push(Some(obs(30.0)))));
        assert!(!(collector.push(Some(obs(30.0)))));
        assert!(collector.push(Some(obs(30.0))));
        assert_eq!(collector.views().len(), 2);
    }

    #[test]
    fn test_view_collector_stops_at_target() {
        let cfg = CollectConfig {
            target_views: 2,
            static_frames: 3,
            static_tol_px: 0.5,
            min_novelty_px: 20.0,
        };
        let mut collector = ViewCollector::new(cfg);
        for _ in 0..3 {
            collector.push(Some(obs(0.0)));
        }
        for _ in 0..3 {
            collector.push(Some(obs(30.0)));
        }
        assert!(collector.done());
        assert_eq!(collector.views().len(), 2);

        for _ in 0..3 {
            collector.push(Some(obs(60.0)));
        }
        assert_eq!(collector.views().len(), 2);
    }

    #[test]
    fn test_stereo_collector_needs_both_static() {
        let cfg = CollectConfig {
            target_views: 8,
            static_frames: 3,
            static_tol_px: 0.5,
            min_novelty_px: 20.0,
        };
        let mut collector = StereoCollector::new(cfg);
        assert!(!(collector.push_a(Some(obs(0.0)))));
        assert!(!(collector.push_a(Some(obs(0.0)))));
        assert!(!(collector.push_a(Some(obs(0.0)))));
        assert!(!(collector.push_b(Some(obs(0.0)))));
        assert!(!(collector.push_b(Some(obs(0.0)))));
        assert!(collector.push_b(Some(obs(0.0))));
        assert_eq!(collector.views().len(), 1);

        assert!(!(collector.push_a(Some(obs(0.0)))));
        assert_eq!(collector.a_streak, 1);
        assert!(!(collector.push_b(None)));
        assert_eq!(collector.b_streak, 0);
        assert_eq!(collector.a_streak, 1);
        assert!(collector.b_last.is_none());

        let mut collector = StereoCollector::new(cfg);
        assert!(!(collector.push_b(Some(obs(0.0)))));
        assert!(!(collector.push_b(Some(obs(0.0)))));
        assert!(!(collector.push_b(Some(obs(0.0)))));
        assert!(!(collector.push_a(Some(obs(0.0)))));
        assert!(!(collector.push_a(Some(obs(0.0)))));
        assert!(collector.push_a(Some(obs(0.0))));
        assert_eq!(collector.views().len(), 1);
    }

    #[test]
    fn test_detect_frame_skips_ir_dark_and_finds_lit_board() {
        let spec = board_spec();
        let cfg = DetectorConfig::default();

        let dark = board_frame(Illumination::IrDark);
        assert!(matches!(
            detect_frame(&dark, &spec, &cfg).expect("ok"),
            Detection::Skipped
        ));

        let lit = board_frame(Illumination::IrLit);
        match detect_frame(&lit, &spec, &cfg).expect("ok") {
            Detection::Board(obs) => {
                assert_eq!(obs.corners.len(), 54);
                let d = (obs.corners[0].image_px - Point2::new(140.0, 80.0)).norm();
                assert!(d < 0.5, "d={d}");
            }
            other => panic!("expected a board, got {other:?}"),
        }

        let uniform = Frame::new(
            FrameHeader {
                camera: CameraId::from("ir"),
                seq: 0,
                timestamp: Timestamp::from_nanos(0),
                width: 640,
                height: 360,
                format: PixelFormat::Gray8,
                illumination: Illumination::IrLit,
            },
            vec![128u8; 640 * 360].into(),
        )
        .expect("valid frame");
        assert!(matches!(
            detect_frame(&uniform, &spec, &cfg).expect("ok"),
            Detection::NoBoard
        ));
    }

    #[test]
    fn test_phone_board_rejected_for_ir() {
        let err = check_board(Board::Phone, true).expect_err("phone + ir errors");
        assert_eq!(
            err.to_string(),
            "the phone board is not visible in IR (an OLED emits no near-IR); print assets/calibration/a4-laser.pdf on a laser printer and use --board a4"
        );
        assert!(check_board(Board::Phone, false).is_ok());
        assert!(check_board(Board::A4, true).is_ok());
    }

    #[test]
    fn test_board_specs() {
        assert_eq!(
            Board::A4.spec(None),
            BoardSpec {
                inner_cols: 9,
                inner_rows: 6,
                square_mm: 25.0,
            }
        );
        assert_eq!(
            Board::Phone.spec(None),
            BoardSpec {
                inner_cols: 9,
                inner_rows: 6,
                square_mm: 8.629,
            }
        );
        assert_eq!(Board::A4.spec(Some(24.8)).square_mm, 24.8);
    }

    #[test]
    fn test_stereo_requires_stored_rig() {
        let err = check_stereo_source(RigSource::Nominal, "rgb", "ir").expect_err("nominal errors");
        assert_eq!(
            err.to_string(),
            "stereo needs calibrated intrinsics: run `eye calibrate-camera intrinsics --camera rgb` and `--camera ir` first"
        );
        assert!(check_stereo_source(RigSource::Stored, "rgb", "ir").is_ok());
    }

    #[test]
    fn test_intrinsics_summary_format() {
        let fit = IntrinsicsFit {
            intrinsics: Intrinsics {
                width: 640,
                height: 360,
                fx: 471.2,
                fy: 471.05,
                cx: 321.4,
                cy: 179.8,
                distortion: eye_geometry::camera::Distortion {
                    k1: 0.0812,
                    k2: -0.149,
                    p1: 0.001,
                    p2: -0.0005,
                    k3: 0.0,
                },
            },
            cov: {
                let mut cov = SMatrix::<f64, 9, 9>::zeros();
                cov[(0, 0)] = 0.0961;
                cov
            },
            rms_px: 0.21,
            per_view_rms_px: Vec::new(),
            views_used: (0..15).collect(),
            camera_from_board: Vec::new(),
            module: LensModule::Fov75_8,
            prior_z: 0.99,
        };

        assert_eq!(
            intrinsics_summary("ir", &fit, 3),
            "camera ir 640x360: fx 471.20 fy 471.05 cx 321.40 cy 179.80 px (sigma fx 0.31)\n\
             distortion k1 0.0812 k2 -0.1490 p1 0.0010 p2 -0.0005 k3 0.0000\n\
             reprojection rms 0.21 px over 15 views (3 rejected)\n\
             Dell module: 75.8 deg (2.7 mm), prior z +0.99"
        );
    }

    #[test]
    fn test_stereo_summary_format() {
        let fit = StereoFit {
            b_from_a: Isometry3::from_parts(
                Translation3::new(-25.03, 0.0, 0.0),
                UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.51f64.to_radians()),
            ),
            cov: Matrix6::zeros(),
            rms_px: 0.19,
            views_used: (0..8).collect(),
        };

        assert_eq!(
            stereo_summary(&fit),
            "baseline 25.03 mm, rotation 0.51 deg, rms 0.19 px over 8 views"
        );
    }
}
