//! The overlay as an `eye_core::GazeSink`: spawns the layer-shell surface and
//! feeds it gaze points.

use eye_core::grid::Grid;
use eye_core::{GazePoint, GazeSink, OutputId, ScreenModel, SinkError};
use nalgebra::Vector2;

use crate::ellipse::logical_px_per_mm;
use crate::error::OverlayError;
use crate::handle::OverlayHandle;
use crate::point::{Easing, HideRules, PointScene};
use crate::region::RegionScene;
use crate::surface::{SurfaceOptions, spawn};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverlayMode {
    Point,
    Region { cols: u32, rows: u32 },
}

#[derive(Debug, Clone, PartialEq)]
pub struct OverlayOptions {
    pub output: OutputId,
    pub mode: OverlayMode,
    pub px_per_mm: Vector2<f64>,
    pub color: [u8; 3],
    pub easing: Easing,
    pub hide: HideRules,
}

impl OverlayOptions {
    pub fn new(output: OutputId, mode: OverlayMode, screen: &ScreenModel) -> Self {
        Self {
            output,
            mode,
            px_per_mm: logical_px_per_mm(screen),
            color: [255, 64, 64],
            easing: Easing::default(),
            hide: HideRules::default(),
        }
    }
}

#[derive(Debug)]
pub struct LayerShellOverlay {
    handle: OverlayHandle<GazePoint>,
    output: OutputId,
}

impl LayerShellOverlay {
    pub fn spawn(options: OverlayOptions) -> Result<Self, OverlayError> {
        let surface = SurfaceOptions {
            output: Some(options.output.as_str().to_owned()),
            namespace: "eye-overlay",
        };
        let handle = match options.mode {
            OverlayMode::Point => spawn(
                surface,
                PointScene::with_easing(options.px_per_mm, options.color, options.easing)
                    .with_hide_rules(options.hide),
            )?,
            OverlayMode::Region { cols, rows } => {
                let grid = Grid::new(cols, rows).ok_or(OverlayError::InvalidGrid { cols, rows })?;
                spawn(
                    surface,
                    RegionScene::new(grid, options.px_per_mm, options.color),
                )?
            }
        };
        Ok(Self {
            handle,
            output: options.output,
        })
    }

    /// Closes the surface and joins the overlay thread.
    pub fn shutdown(self) -> Result<(), OverlayError> {
        self.handle.close()
    }

    #[cfg(test)]
    pub(crate) fn from_parts(handle: OverlayHandle<GazePoint>, output: OutputId) -> Self {
        Self { handle, output }
    }
}

impl GazeSink for LayerShellOverlay {
    fn name(&self) -> &'static str {
        "layer-shell"
    }

    fn push(&mut self, point: &GazePoint) -> Result<(), SinkError> {
        if point.output != self.output {
            tracing::trace!(output = %point.output, "gaze point for another output ignored");
            return Ok(());
        }
        self.handle
            .send(point.clone())
            .map_err(|_| SinkError::Closed)
    }
}

#[cfg(test)]
mod tests {
    use eye_core::Timestamp;
    use nalgebra::{Matrix2, Point2, Vector2};
    use smithay_client_toolkit::reexports::calloop;

    use super::*;

    fn edp1() -> ScreenModel {
        ScreenModel {
            output: OutputId::from("eDP-1"),
            size_mm: Vector2::new(310.0, 170.0),
            size_px: (3840, 2160),
            scale: 2.0,
        }
    }

    fn point_for(output: &str) -> GazePoint {
        GazePoint {
            timestamp: Timestamp::from_nanos(0),
            output: OutputId::from(output),
            mm: Point2::new(0.0, 0.0),
            px_physical: Point2::new(0.0, 0.0),
            px_logical: Point2::new(0.0, 0.0),
            cov_mm: Matrix2::zeros(),
            confidence: 1.0,
        }
    }

    #[test]
    fn test_options_new_takes_scale_from_screen() {
        let options = OverlayOptions::new(OutputId::from("eDP-1"), OverlayMode::Point, &edp1());
        assert_eq!(
            options.px_per_mm,
            Vector2::new(1920.0 / 310.0, 1080.0 / 170.0)
        );
        assert_eq!(options.color, [255, 64, 64]);
    }

    #[test]
    fn test_sink_name_is_layer_shell() {
        let (tx, rx) = calloop::channel::sync_channel::<GazePoint>(1);
        drop(rx);
        let overlay =
            LayerShellOverlay::from_parts(OverlayHandle::detached(tx), OutputId::from("eDP-1"));
        assert_eq!(overlay.name(), "layer-shell");
    }

    #[test]
    fn test_push_other_output_is_ignored() {
        let (tx, rx) = calloop::channel::sync_channel::<GazePoint>(1);
        drop(rx);
        let mut overlay =
            LayerShellOverlay::from_parts(OverlayHandle::detached(tx), OutputId::from("eDP-1"));
        assert!(overlay.push(&point_for("HDMI-A-1")).is_ok());
    }

    #[test]
    fn test_push_after_overlay_exit_returns_closed() {
        let (tx, rx) = calloop::channel::sync_channel::<GazePoint>(1);
        drop(rx);
        let mut overlay =
            LayerShellOverlay::from_parts(OverlayHandle::detached(tx), OutputId::from("eDP-1"));
        assert!(matches!(
            overlay.push(&point_for("eDP-1")),
            Err(SinkError::Closed)
        ));
    }

    #[test]
    fn test_zero_grid_is_invalid() {
        let options = OverlayOptions::new(
            OutputId::from("eDP-1"),
            OverlayMode::Region { cols: 0, rows: 4 },
            &edp1(),
        );
        assert!(matches!(
            LayerShellOverlay::spawn(options),
            Err(OverlayError::InvalidGrid { cols: 0, rows: 4 })
        ));
    }

    #[test]
    #[ignore = "needs wayland"]
    fn test_live_point_overlay_smoke() {
        use std::f64::consts::TAU;
        use std::thread;
        use std::time::Duration;

        let output = std::env::var("EYE_OUTPUT").unwrap_or_else(|_| "eDP-1".to_string());
        let screen = ScreenModel {
            output: OutputId::from(output.as_str()),
            size_mm: Vector2::new(310.0, 170.0),
            size_px: (3840, 2160),
            scale: 2.0,
        };
        let options =
            OverlayOptions::new(OutputId::from(output.as_str()), OverlayMode::Point, &screen);
        let mut overlay = LayerShellOverlay::spawn(options).expect("spawn succeeds");

        let (cx, cy) = (960.0, 540.0);
        let radius = 300.0;
        for i in 0..90 {
            let t = f64::from(i) / 90.0 * TAU;
            let px_logical = Point2::new(cx + radius * t.cos(), cy + radius * t.sin());
            let spread = 10.0 + f64::from(i) * 2.0;
            let point = GazePoint {
                timestamp: Timestamp::now(),
                output: OutputId::from(output.as_str()),
                mm: Point2::new(0.0, 0.0),
                px_physical: px_logical,
                px_logical,
                cov_mm: Matrix2::new(spread, 0.0, 0.0, spread),
                confidence: 1.0,
            };
            overlay.push(&point).expect("push succeeds");
            thread::sleep(Duration::from_millis(33));
        }
        thread::sleep(Duration::from_secs(1));
        assert!(overlay.shutdown().is_ok());
    }

    #[test]
    #[ignore = "needs wayland"]
    fn test_live_region_overlay_smoke() {
        use std::thread;
        use std::time::Duration;

        let output = std::env::var("EYE_OUTPUT").unwrap_or_else(|_| "eDP-1".to_string());
        let screen = ScreenModel {
            output: OutputId::from(output.as_str()),
            size_mm: Vector2::new(310.0, 170.0),
            size_px: (3840, 2160),
            scale: 2.0,
        };
        let options = OverlayOptions::new(
            OutputId::from(output.as_str()),
            OverlayMode::Region { cols: 4, rows: 4 },
            &screen,
        );
        let mut overlay = LayerShellOverlay::spawn(options).expect("spawn succeeds");

        let (w, h) = (1920.0, 1080.0);
        for i in 0..90 {
            let x = (f64::from(i) / 90.0) * w;
            let y = (f64::from(i) / 90.0) * h;
            let point = GazePoint {
                timestamp: Timestamp::now(),
                output: OutputId::from(output.as_str()),
                mm: Point2::new(0.0, 0.0),
                px_physical: Point2::new(x, y),
                px_logical: Point2::new(x, y),
                cov_mm: Matrix2::new(10.0, 0.0, 0.0, 10.0),
                confidence: 1.0,
            };
            overlay.push(&point).expect("push succeeds");
            thread::sleep(Duration::from_millis(33));
        }
        thread::sleep(Duration::from_secs(1));
        assert!(overlay.shutdown().is_ok());
    }
}
