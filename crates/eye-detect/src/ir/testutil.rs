//! Synthetic IR lit/dark frame renderer shared by the IR detector tests.

use eye_core::image::GrayImage;
use nalgebra::Point2;

use crate::image::Roi;

const FACE_CENTER: (f64, f64) = (320.0, 180.0);
const FACE_SEMI: (f64, f64) = (100.0, 130.0);
const SUPERSAMPLE: u32 = 8;

#[derive(Debug, Clone)]
pub(crate) struct SyntheticEye {
    pub pupil_center: Point2<f64>,
    pub pupil_radius: f64,
    pub iris_radius: f64,
    pub pupil_level: u8,
    /// `(center, sigma px, peak)`; rendered into the lit frame only.
    pub glint: Option<(Point2<f64>, f64, f64)>,
}

impl SyntheticEye {
    pub fn at(pupil_center: Point2<f64>) -> Self {
        Self {
            pupil_center,
            pupil_radius: 3.0,
            iris_radius: 6.0,
            pupil_level: 220,
            glint: None,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SyntheticIr {
    pub size: (u32, u32),
    pub skin_level: u8,
    pub iris_level: u8,
    pub eyes: Vec<SyntheticEye>,
    pub ambient: Option<(Roi, u8)>,
    pub specular: Vec<(Point2<f64>, f64, u8)>,
    pub noise_sigma: f64,
    pub seed: u64,
}

impl SyntheticIr {
    pub fn default_scene() -> Self {
        Self {
            size: (640, 360),
            skin_level: 90,
            iris_level: 35,
            eyes: vec![
                SyntheticEye::at(Point2::new(290.3, 180.7)),
                SyntheticEye::at(Point2::new(350.6, 181.2)),
            ],
            ambient: None,
            specular: Vec::new(),
            noise_sigma: 0.0,
            seed: 7,
        }
    }

    pub fn render(&self) -> (GrayImage, GrayImage) {
        let (w, h) = self.size;
        let mut lit = vec![0.0f64; w as usize * h as usize];

        for y in 0..h {
            for x in 0..w {
                if in_face(x, y) {
                    lit[(y * w + x) as usize] = f64::from(self.skin_level);
                }
            }
        }
        for eye in &self.eyes {
            composite_disc(
                &mut lit,
                w,
                h,
                eye.pupil_center,
                eye.iris_radius,
                f64::from(self.iris_level),
            );
            composite_disc(
                &mut lit,
                w,
                h,
                eye.pupil_center,
                eye.pupil_radius,
                f64::from(eye.pupil_level),
            );
            if let Some((center, sigma, peak)) = eye.glint {
                composite_glint(&mut lit, w, h, center, sigma, peak);
            }
        }
        for &(center, radius, level) in &self.specular {
            composite_disc(&mut lit, w, h, center, radius, f64::from(level));
        }

        let mut dark = vec![0.0f64; w as usize * h as usize];

        if let Some((roi, level)) = self.ambient {
            add_roi(&mut lit, w, roi, f64::from(level));
            add_roi(&mut dark, w, roi, f64::from(level));
        }

        let mut rng = Xorshift64Star::new(self.seed);
        add_noise(&mut lit, self.noise_sigma, &mut rng);
        add_noise(&mut dark, self.noise_sigma, &mut rng);

        let lit_u8: Vec<u8> = lit.into_iter().map(round_clamp).collect();
        let dark_u8: Vec<u8> = dark.into_iter().map(round_clamp).collect();

        (
            GrayImage::new(w, h, lit_u8).expect("render produces a full buffer"),
            GrayImage::new(w, h, dark_u8).expect("render produces a full buffer"),
        )
    }
}

fn in_face(x: u32, y: u32) -> bool {
    let dx = (x as f64 + 0.5 - FACE_CENTER.0) / FACE_SEMI.0;
    let dy = (y as f64 + 0.5 - FACE_CENTER.1) / FACE_SEMI.1;
    dx * dx + dy * dy <= 1.0
}

fn disc_coverage(x: u32, y: u32, center: Point2<f64>, radius: f64) -> f64 {
    let mut covered = 0u32;
    for ky in 0..SUPERSAMPLE {
        for kx in 0..SUPERSAMPLE {
            let sx = x as f64 + (kx as f64 + 0.5) / f64::from(SUPERSAMPLE);
            let sy = y as f64 + (ky as f64 + 0.5) / f64::from(SUPERSAMPLE);
            let dx = sx - center.x;
            let dy = sy - center.y;
            if dx * dx + dy * dy <= radius * radius {
                covered += 1;
            }
        }
    }
    f64::from(covered) / f64::from(SUPERSAMPLE * SUPERSAMPLE)
}

fn composite_disc(buf: &mut [f64], w: u32, h: u32, center: Point2<f64>, radius: f64, level: f64) {
    let margin = radius.ceil() as i64 + 2;
    let cx = center.x.floor() as i64;
    let cy = center.y.floor() as i64;
    let x0 = (cx - margin).max(0);
    let x1 = (cx + margin).min(i64::from(w) - 1);
    let y0 = (cy - margin).max(0);
    let y1 = (cy + margin).min(i64::from(h) - 1);
    for y in y0..=y1 {
        for x in x0..=x1 {
            let cov = disc_coverage(x as u32, y as u32, center, radius);
            if cov > 0.0 {
                let idx = y as usize * w as usize + x as usize;
                buf[idx] = buf[idx] * (1.0 - cov) + level * cov;
            }
        }
    }
}

fn glint_coverage(x: u32, y: u32, center: Point2<f64>, sigma: f64) -> f64 {
    let mut sum = 0.0;
    for ky in 0..SUPERSAMPLE {
        for kx in 0..SUPERSAMPLE {
            let sx = x as f64 + (kx as f64 + 0.5) / f64::from(SUPERSAMPLE);
            let sy = y as f64 + (ky as f64 + 0.5) / f64::from(SUPERSAMPLE);
            let dx = sx - center.x;
            let dy = sy - center.y;
            sum += (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp();
        }
    }
    sum / f64::from(SUPERSAMPLE * SUPERSAMPLE)
}

fn composite_glint(buf: &mut [f64], w: u32, h: u32, center: Point2<f64>, sigma: f64, peak: f64) {
    let margin = (4.0 * sigma).ceil() as i64 + 1;
    let cx = center.x.floor() as i64;
    let cy = center.y.floor() as i64;
    let x0 = (cx - margin).max(0);
    let x1 = (cx + margin).min(i64::from(w) - 1);
    let y0 = (cy - margin).max(0);
    let y1 = (cy + margin).min(i64::from(h) - 1);
    for y in y0..=y1 {
        for x in x0..=x1 {
            let g = glint_coverage(x as u32, y as u32, center, sigma);
            let idx = y as usize * w as usize + x as usize;
            buf[idx] += (peak - buf[idx]) * g;
        }
    }
}

fn add_roi(buf: &mut [f64], w: u32, roi: Roi, level: f64) {
    for y in roi.y..roi.y + roi.height {
        for x in roi.x..roi.x + roi.width {
            buf[(y * w + x) as usize] += level;
        }
    }
}

fn round_clamp(v: f64) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}

pub(crate) struct Xorshift64Star {
    state: u64,
}

impl Xorshift64Star {
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn next_uniform(&mut self) -> f64 {
        let next = self.next_u64();
        ((next >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    pub(crate) fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_uniform();
        let u2 = self.next_uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn add_noise(buf: &mut [f64], sigma: f64, rng: &mut Xorshift64Star) {
    if sigma <= 0.0 {
        return;
    }
    for v in buf.iter_mut() {
        *v += sigma * rng.next_gaussian();
    }
}
