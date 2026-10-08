//! A buffer-backed drawing surface: logical-pixel shapes rasterized into a
//! scale-aware, premultiplied Argb8888 byte buffer, with damage tracking.

use std::fmt;

use nalgebra::Point2;
use tiny_skia::{FillRule, IntRect, Paint, Path, PathBuilder, PixmapMut, Rect, Transform};

/// Straight (non-premultiplied) alpha color.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rgba {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogicalRect {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

pub struct Canvas<'a> {
    pixmap: PixmapMut<'a>,
    logical: (u32, u32),
    scale: u32,
    damage: Vec<IntRect>,
}

impl fmt::Debug for Canvas<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Canvas")
            .field("logical", &self.logical)
            .field("scale", &self.scale)
            .field("damage_len", &self.damage.len())
            .finish()
    }
}

impl<'a> Canvas<'a> {
    /// `data` holds `(w*scale) * (h*scale) * 4` bytes, Argb8888 little-endian (B, G, R, A in memory).
    pub fn new(data: &'a mut [u8], logical: (u32, u32), scale: u32) -> Option<Self> {
        let (w, h) = logical;
        let pixmap = PixmapMut::from_bytes(data, w * scale, h * scale)?;
        Some(Self {
            pixmap,
            logical,
            scale,
            damage: Vec::new(),
        })
    }

    pub fn logical_size(&self) -> (u32, u32) {
        self.logical
    }

    pub fn scale(&self) -> u32 {
        self.scale
    }

    pub fn fill_ellipse(
        &mut self,
        center: Point2<f64>,
        semi_axes: (f64, f64),
        angle_rad: f64,
        color: Rgba,
    ) {
        let Some((path, ts)) = self.ellipse_path(center, semi_axes, angle_rad) else {
            return;
        };
        self.record(&path, ts, 0.0);
        self.pixmap
            .fill_path(&path, &Self::paint(color), FillRule::Winding, ts, None);
    }

    pub fn stroke_ellipse(
        &mut self,
        center: Point2<f64>,
        semi_axes: (f64, f64),
        angle_rad: f64,
        width: f64,
        color: Rgba,
    ) {
        let Some((path, ts)) = self.ellipse_path(center, semi_axes, angle_rad) else {
            return;
        };
        let stroke_px = width as f32 * self.scale as f32;
        self.record(&path, ts, stroke_px);
        let stroke = tiny_skia::Stroke {
            width: width as f32,
            ..Default::default()
        };
        self.pixmap
            .stroke_path(&path, &Self::paint(color), &stroke, ts, None);
    }

    pub fn fill_circle(&mut self, center: Point2<f64>, radius: f64, color: Rgba) {
        self.fill_ellipse(center, (radius, radius), 0.0, color);
    }

    pub fn fill_rect(&mut self, rect: LogicalRect, color: Rgba) {
        let Some((path, ts)) = self.rect_path(rect) else {
            return;
        };
        self.record(&path, ts, 0.0);
        self.pixmap
            .fill_path(&path, &Self::paint(color), FillRule::Winding, ts, None);
    }

    pub fn stroke_rect(&mut self, rect: LogicalRect, width: f64, color: Rgba) {
        let Some((path, ts)) = self.rect_path(rect) else {
            return;
        };
        let stroke_px = width as f32 * self.scale as f32;
        self.record(&path, ts, stroke_px);
        let stroke = tiny_skia::Stroke {
            width: width as f32,
            ..Default::default()
        };
        self.pixmap
            .stroke_path(&path, &Self::paint(color), &stroke, ts, None);
    }

    /// Zeroes the buffer-px rect; records no damage.
    pub fn clear_px(&mut self, rect: IntRect) {
        let Some(rect) = rect.intersect(&self.full_rect()) else {
            return;
        };
        let width = self.pixmap.width() as usize;
        let data = self.pixmap.data_mut();
        for y in rect.y()..rect.y() + rect.height() as i32 {
            let row_start = y as usize * width * 4 + rect.x() as usize * 4;
            let row_len = rect.width() as usize * 4;
            data[row_start..row_start + row_len].fill(0);
        }
    }

    pub fn clear_all(&mut self) {
        self.pixmap.data_mut().fill(0);
    }

    /// Buffer-px rects covering every pixel drawn since the last call.
    pub fn take_damage(&mut self) -> Vec<IntRect> {
        std::mem::take(&mut self.damage)
    }

    fn full_rect(&self) -> IntRect {
        IntRect::from_xywh(0, 0, self.pixmap.width(), self.pixmap.height())
            .expect("buffer has a positive size")
    }

    fn rect_path(&self, rect: LogicalRect) -> Option<(Path, Transform)> {
        let r = Rect::from_xywh(rect.x as f32, rect.y as f32, rect.w as f32, rect.h as f32)?;
        let path = PathBuilder::from_rect(r);
        let ts = Transform::from_scale(self.scale as f32, self.scale as f32);
        Some((path, ts))
    }

    fn ellipse_path(
        &self,
        center: Point2<f64>,
        semi_axes: (f64, f64),
        angle_rad: f64,
    ) -> Option<(Path, Transform)> {
        let (cx, cy) = (center.x as f32, center.y as f32);
        let (a, b) = semi_axes;
        let rect = Rect::from_xywh(cx - a as f32, cy - b as f32, 2.0 * a as f32, 2.0 * b as f32)?;
        let path = PathBuilder::from_oval(rect)?;
        let ts = Transform::from_rotate_at(angle_rad.to_degrees() as f32, cx, cy)
            .post_scale(self.scale as f32, self.scale as f32);
        Some((path, ts))
    }

    fn paint(color: Rgba) -> Paint<'static> {
        let mut paint = Paint::default();
        // Argb8888 stores B, G, R, A in memory; tiny-skia writes R, G, B, A, so swap R/B here.
        paint.set_color_rgba8(color.b, color.g, color.r, color.a);
        paint.anti_alias = true;
        paint
    }

    fn record(&mut self, path: &Path, ts: Transform, stroke_px: f32) {
        let Some(bounds) = path.clone().transform(ts).map(|p| p.bounds()) else {
            return;
        };
        let pad = 2.0 + stroke_px;
        if let Some(r) = Rect::from_ltrb(
            bounds.left() - pad,
            bounds.top() - pad,
            bounds.right() + pad,
            bounds.bottom() + pad,
        )
        .and_then(|r| r.round_out())
        .and_then(|r| r.intersect(&self.full_rect()))
        {
            self.damage.push(r);
        }
    }
}

/// Reads the raw BGRA bytes at buffer-pixel `(x, y)` of a buffer with the given `width_px`.
#[cfg(test)]
pub(crate) fn bgra(data: &[u8], width_px: u32, x: u32, y: u32) -> [u8; 4] {
    let stride = width_px as usize * 4;
    let offset = y as usize * stride + x as usize * 4;
    [
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;

    use super::*;

    fn red() -> Rgba {
        Rgba {
            r: 255,
            g: 0,
            b: 0,
            a: 255,
        }
    }

    #[test]
    fn test_canvas_writes_argb8888_byte_order() {
        let mut data = vec![0u8; 20 * 20 * 4];
        let mut canvas = Canvas::new(&mut data, (20, 20), 1).expect("valid buffer");
        canvas.fill_rect(
            LogicalRect {
                x: 0.0,
                y: 0.0,
                w: 20.0,
                h: 20.0,
            },
            red(),
        );
        assert_eq!(bgra(&data, 20, 10, 10), [0, 0, 255, 255]);
    }

    #[test]
    fn test_canvas_premultiplies_alpha() {
        let mut data = vec![0u8; 20 * 20 * 4];
        let mut canvas = Canvas::new(&mut data, (20, 20), 1).expect("valid buffer");
        canvas.fill_rect(
            LogicalRect {
                x: 0.0,
                y: 0.0,
                w: 20.0,
                h: 20.0,
            },
            Rgba {
                r: 255,
                g: 128,
                b: 0,
                a: 128,
            },
        );
        let px = bgra(&data, 20, 10, 10);
        let expected = [0u8, 64, 128, 128];
        for i in 0..4 {
            assert_abs_diff_eq!(f64::from(px[i]), f64::from(expected[i]), epsilon = 1.0);
        }
    }

    #[test]
    fn test_canvas_scale_maps_logical_to_buffer_px() {
        let mut data = vec![0u8; 40 * 40 * 4];
        let mut canvas = Canvas::new(&mut data, (20, 20), 2).expect("valid buffer");
        canvas.fill_rect(
            LogicalRect {
                x: 10.0,
                y: 10.0,
                w: 5.0,
                h: 5.0,
            },
            red(),
        );
        assert_eq!(bgra(&data, 40, 21, 21)[3], 255);
        assert_eq!(bgra(&data, 40, 29, 29)[3], 255);
        assert_eq!(bgra(&data, 40, 18, 18)[3], 0);
        assert_eq!(bgra(&data, 40, 31, 31)[3], 0);
    }

    #[test]
    fn test_canvas_fill_ellipse_rotation() {
        let mut data = vec![0u8; 40 * 40 * 4];
        let mut canvas = Canvas::new(&mut data, (40, 40), 1).expect("valid buffer");
        canvas.fill_ellipse(
            Point2::new(20.0, 20.0),
            (8.0, 2.0),
            std::f64::consts::FRAC_PI_2,
            red(),
        );
        assert!(bgra(&data, 40, 20, 26)[3] > 0);
        assert_eq!(bgra(&data, 40, 26, 20)[3], 0);
    }

    proptest! {
        #[test]
        fn test_canvas_damage_covers_every_touched_pixel(
            cx in 0.0..64.0f64,
            cy in 0.0..64.0f64,
            radius in 0.5..20.0f64,
            scale in 1u32..=2u32,
        ) {
            let (w, h) = (64u32, 64u32);
            let mut data = vec![0u8; (w * scale * h * scale * 4) as usize];
            let mut canvas = Canvas::new(&mut data, (w, h), scale).expect("valid buffer");
            canvas.fill_circle(Point2::new(cx, cy), radius, Rgba { r: 255, g: 255, b: 255, a: 255 });
            let damage = canvas.take_damage();
            let bw = w * scale;
            for y in 0..(h * scale) {
                for x in 0..bw {
                    if bgra(&data, bw, x, y)[3] != 0 {
                        let inside = damage.iter().any(|r| {
                            x as i32 >= r.x()
                                && (x as i32) < r.x() + r.width() as i32
                                && y as i32 >= r.y()
                                && (y as i32) < r.y() + r.height() as i32
                        });
                        prop_assert!(inside, "pixel ({x}, {y}) not covered by damage {damage:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn test_canvas_clear_px_zeroes_only_rect() {
        let mut data = vec![0u8; 10 * 10 * 4];
        let mut canvas = Canvas::new(&mut data, (10, 10), 1).expect("valid buffer");
        canvas.fill_rect(
            LogicalRect {
                x: 0.0,
                y: 0.0,
                w: 10.0,
                h: 10.0,
            },
            red(),
        );
        canvas.clear_px(IntRect::from_xywh(2, 2, 3, 3).expect("valid rect"));
        let mut zero_count = 0;
        for y in 0..10 {
            for x in 0..10 {
                if bgra(&data, 10, x, y)[3] == 0 {
                    zero_count += 1;
                }
            }
        }
        assert_eq!(zero_count, 9);
    }

    #[test]
    fn test_canvas_clear_does_not_record_damage() {
        let mut data = vec![0u8; 10 * 10 * 4];
        let mut canvas = Canvas::new(&mut data, (10, 10), 1).expect("valid buffer");
        canvas.fill_rect(
            LogicalRect {
                x: 0.0,
                y: 0.0,
                w: 10.0,
                h: 10.0,
            },
            red(),
        );
        assert_eq!(canvas.take_damage().len(), 1);
        canvas.clear_px(IntRect::from_xywh(0, 0, 5, 5).expect("valid rect"));
        canvas.clear_all();
        assert_eq!(canvas.take_damage().len(), 0);
    }

    #[test]
    fn test_canvas_new_rejects_short_buffer() {
        let mut data = vec![0u8; 10];
        assert!(Canvas::new(&mut data, (2, 2), 1).is_none());
    }
}
