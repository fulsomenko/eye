use nalgebra::{Point2, Vector2};

use crate::DetectError;
use crate::image::RgbImage;
use crate::mediapipe::anchors::Anchor;

pub const DETECTOR_INPUT: usize = 128;
pub const NUM_ANCHORS: usize = 896;
pub const NUM_COORDS: usize = 16;
pub const NUM_KEYPOINTS: usize = 6;

#[derive(Debug, Clone, PartialEq)]
pub struct FaceDetectorOutput {
    pub regressors: Vec<f32>,
    pub logits: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FaceDetection {
    pub score: f32,
    pub center: Point2<f64>,
    pub size: Vector2<f64>,
    pub keypoints: [Point2<f64>; NUM_KEYPOINTS],
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn iou(a: &FaceDetection, b: &FaceDetection) -> f64 {
    let (a_half_w, a_half_h) = (a.size.x / 2.0, a.size.y / 2.0);
    let (b_half_w, b_half_h) = (b.size.x / 2.0, b.size.y / 2.0);
    let (a_x0, a_x1) = (a.center.x - a_half_w, a.center.x + a_half_w);
    let (a_y0, a_y1) = (a.center.y - a_half_h, a.center.y + a_half_h);
    let (b_x0, b_x1) = (b.center.x - b_half_w, b.center.x + b_half_w);
    let (b_y0, b_y1) = (b.center.y - b_half_h, b.center.y + b_half_h);

    let inter_w = (a_x1.min(b_x1) - a_x0.max(b_x0)).max(0.0);
    let inter_h = (a_y1.min(b_y1) - a_y0.max(b_y0)).max(0.0);
    let inter = inter_w * inter_h;
    let union = a.size.x * a.size.y + b.size.x * b.size.y - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

pub fn decode(
    raw: &FaceDetectorOutput,
    anchors: &[Anchor],
    min_score: f32,
) -> Result<Vec<FaceDetection>, DetectError> {
    if raw.regressors.len() != NUM_ANCHORS * NUM_COORDS
        || raw.logits.len() != NUM_ANCHORS
        || anchors.len() != NUM_ANCHORS
    {
        return Err(DetectError::Inference(format!(
            "expected regressors.len() == {}, logits.len() == {}, anchors.len() == {}; got {}, {}, {}",
            NUM_ANCHORS * NUM_COORDS,
            NUM_ANCHORS,
            NUM_ANCHORS,
            raw.regressors.len(),
            raw.logits.len(),
            anchors.len(),
        )));
    }

    let mut detections = Vec::new();
    for (i, anchor) in anchors.iter().enumerate() {
        let score = sigmoid(raw.logits[i].clamp(-100.0, 100.0));
        if score < min_score {
            continue;
        }
        let r = &raw.regressors[i * NUM_COORDS..i * NUM_COORDS + NUM_COORDS];
        let center = Point2::new(
            f64::from(r[0] / 128.0 + anchor.x),
            f64::from(r[1] / 128.0 + anchor.y),
        );
        let size = Vector2::new(f64::from(r[2] / 128.0), f64::from(r[3] / 128.0));
        let mut keypoints = [Point2::new(0.0, 0.0); NUM_KEYPOINTS];
        for (k, kp) in keypoints.iter_mut().enumerate() {
            *kp = Point2::new(
                f64::from(r[4 + 2 * k] / 128.0 + anchor.x),
                f64::from(r[5 + 2 * k] / 128.0 + anchor.y),
            );
        }
        detections.push(FaceDetection {
            score,
            center,
            size,
            keypoints,
        });
    }
    Ok(detections)
}

pub fn weighted_nms(mut detections: Vec<FaceDetection>, iou_threshold: f32) -> Vec<FaceDetection> {
    detections.sort_by(|a, b| b.score.total_cmp(&a.score));

    let mut merged = Vec::new();
    while let Some(top) = detections.first().cloned() {
        let mut cluster = vec![top.clone()];
        detections.remove(0);
        detections.retain(|d| {
            if iou(&top, d) as f32 > iou_threshold {
                cluster.push(d.clone());
                false
            } else {
                true
            }
        });

        let weight_sum: f64 = cluster.iter().map(|d| f64::from(d.score)).sum();
        let mut center = Vector2::new(0.0, 0.0);
        let mut size = Vector2::new(0.0, 0.0);
        let mut keypoints = [Vector2::new(0.0, 0.0); NUM_KEYPOINTS];
        for d in &cluster {
            let w = f64::from(d.score);
            center += d.center.coords * w;
            size += d.size * w;
            for (kp, dkp) in keypoints.iter_mut().zip(d.keypoints.iter()) {
                *kp += dkp.coords * w;
            }
        }
        center /= weight_sum;
        size /= weight_sum;
        let keypoints = keypoints.map(|kp| Point2::from(kp / weight_sum));

        merged.push(FaceDetection {
            score: top.score,
            center: Point2::from(center),
            size,
            keypoints,
        });
    }
    merged
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Letterbox {
    pub scale: f64,
    pub pad_x: f64,
    pub pad_y: f64,
}

impl Letterbox {
    pub fn new(image_w: u32, image_h: u32, n: usize) -> Self {
        let n = n as f64;
        let (w, h) = (f64::from(image_w), f64::from(image_h));
        let scale = n / w.max(h);
        let pad_x = (n - w * scale) / 2.0;
        let pad_y = (n - h * scale) / 2.0;
        Letterbox {
            scale,
            pad_x,
            pad_y,
        }
    }

    pub fn to_image(&self, normalized: &Point2<f64>, n: usize) -> Point2<f64> {
        let n = n as f64;
        Point2::new(
            (normalized.x * n - self.pad_x) / self.scale,
            (normalized.y * n - self.pad_y) / self.scale,
        )
    }
}

pub fn letterbox_to_tensor(
    image: &RgbImage,
    n: usize,
    range: (f32, f32),
    out: &mut [f32],
) -> Letterbox {
    let letterbox = Letterbox::new(image.width, image.height, n);
    let (lo, hi) = range;
    let (w, h) = (f64::from(image.width), f64::from(image.height));
    for v in 0..n {
        for u in 0..n {
            let x = (u as f64 + 0.5 - letterbox.pad_x) / letterbox.scale;
            let y = (v as f64 + 0.5 - letterbox.pad_y) / letterbox.scale;
            let in_bounds = x >= 0.0 && x < w && y >= 0.0 && y < h;
            for c in 0..3 {
                let idx = (v * n + u) * 3 + c;
                out[idx] = if in_bounds {
                    (image.sample(x, y, c) / 255.0) as f32 * (hi - lo) + lo
                } else {
                    lo
                };
            }
        }
    }
    letterbox
}

#[cfg(test)]
pub(crate) fn encode_detection(
    anchors: &[Anchor],
    i: usize,
    det: &FaceDetection,
) -> FaceDetectorOutput {
    let a = anchors[i];
    let mut regressors = vec![0.0f32; NUM_ANCHORS * NUM_COORDS];
    let base = i * NUM_COORDS;
    regressors[base] = (det.center.x - f64::from(a.x)) as f32 * 128.0;
    regressors[base + 1] = (det.center.y - f64::from(a.y)) as f32 * 128.0;
    regressors[base + 2] = det.size.x as f32 * 128.0;
    regressors[base + 3] = det.size.y as f32 * 128.0;
    for k in 0..NUM_KEYPOINTS {
        let kp = det.keypoints[k];
        regressors[base + 4 + 2 * k] = (kp.x - f64::from(a.x)) as f32 * 128.0;
        regressors[base + 5 + 2 * k] = (kp.y - f64::from(a.y)) as f32 * 128.0;
    }
    let mut logits = vec![-10.0f32; NUM_ANCHORS];
    logits[i] = 8.0;
    FaceDetectorOutput { regressors, logits }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;
    use crate::mediapipe::anchors::short_range_anchors;

    fn face_detection(center: (f64, f64), size: (f64, f64)) -> FaceDetection {
        let center = Point2::new(center.0, center.1);
        let keypoints = [
            Point2::new(center.x - 0.05, center.y - 0.02),
            Point2::new(center.x + 0.05, center.y - 0.02),
            Point2::new(center.x, center.y),
            Point2::new(center.x, center.y + 0.05),
            Point2::new(center.x - 0.08, center.y + 0.01),
            Point2::new(center.x + 0.08, center.y + 0.01),
        ];
        FaceDetection {
            score: 0.0,
            center,
            size: Vector2::new(size.0, size.1),
            keypoints,
        }
    }

    #[test]
    fn test_decode_recovers_encoded_detection() {
        let anchors = short_range_anchors();
        let det = face_detection((0.40, 0.55), (0.30, 0.32));
        let raw = encode_detection(&anchors, 600, &det);

        let decoded = decode(&raw, &anchors, 0.5).unwrap();
        assert_eq!(decoded.len(), 1);
        let got = &decoded[0];
        assert_abs_diff_eq!(got.center.x, det.center.x, epsilon = 1e-6);
        assert_abs_diff_eq!(got.center.y, det.center.y, epsilon = 1e-6);
        assert_abs_diff_eq!(got.size.x, det.size.x, epsilon = 1e-6);
        assert_abs_diff_eq!(got.size.y, det.size.y, epsilon = 1e-6);
        for k in 0..NUM_KEYPOINTS {
            assert_abs_diff_eq!(got.keypoints[k].x, det.keypoints[k].x, epsilon = 1e-6);
            assert_abs_diff_eq!(got.keypoints[k].y, det.keypoints[k].y, epsilon = 1e-6);
        }
        let expected_score = 1.0 / (1.0 + (-8.0f64).exp());
        assert_abs_diff_eq!(f64::from(got.score), expected_score, epsilon = 1e-6);
    }

    #[test]
    fn test_decode_drops_scores_below_threshold() {
        let anchors = short_range_anchors();
        let det = face_detection((0.40, 0.55), (0.30, 0.32));
        let raw = encode_detection(&anchors, 600, &det);

        let decoded = decode(&raw, &anchors, 0.9999).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_decode_wrong_length_is_inference_error() {
        let anchors = short_range_anchors();
        let raw = FaceDetectorOutput {
            regressors: vec![0.0; NUM_ANCHORS * NUM_COORDS - 1],
            logits: vec![0.0; NUM_ANCHORS],
        };
        let err = decode(&raw, &anchors, 0.5).unwrap_err();
        assert!(matches!(err, DetectError::Inference(_)));
    }

    #[test]
    fn test_weighted_nms_merges_overlapping_by_score() {
        let mut a = face_detection((0.50, 0.50), (0.18, 0.18));
        a.score = 0.9;
        let mut b = face_detection((0.52, 0.50), (0.18, 0.18));
        b.score = 0.6;

        let merged = weighted_nms(vec![a, b], 0.3);
        assert_eq!(merged.len(), 1);
        assert_abs_diff_eq!(merged[0].center.x, 0.508, epsilon = 1e-9);
        assert_abs_diff_eq!(f64::from(merged[0].score), 0.9, epsilon = 1e-6);
    }

    #[test]
    fn test_weighted_nms_keeps_disjoint_boxes() {
        let mut a = face_detection((0.25, 0.5), (0.2, 0.2));
        a.score = 0.9;
        let mut b = face_detection((0.75, 0.5), (0.2, 0.2));
        b.score = 0.8;

        let kept = weighted_nms(vec![a, b], 0.3);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn test_letterbox_1280x720_pads_vertically() {
        let letterbox = Letterbox::new(1280, 720, 128);
        assert_abs_diff_eq!(letterbox.scale, 0.1, epsilon = 1e-12);
        assert_abs_diff_eq!(letterbox.pad_x, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(letterbox.pad_y, 28.0, epsilon = 1e-12);

        let image = RgbImage {
            width: 1280,
            height: 720,
            data: vec![128u8; 1280 * 720 * 3],
        };
        let mut out = vec![0.0f32; 128 * 128 * 3];
        letterbox_to_tensor(&image, 128, (-1.0, 1.0), &mut out);

        for v in 0..28 {
            for u in 0..128 {
                for c in 0..3 {
                    let idx = (v * 128 + u) * 3 + c;
                    assert_abs_diff_eq!(out[idx] as f64, -1.0, epsilon = 1e-12);
                }
            }
        }
        for v in 100..128 {
            for u in 0..128 {
                for c in 0..3 {
                    let idx = (v * 128 + u) * 3 + c;
                    assert_abs_diff_eq!(out[idx] as f64, -1.0, epsilon = 1e-12);
                }
            }
        }
        let expected_content = 128.0 / 255.0 * 2.0 - 1.0;
        for v in 28..100 {
            for u in 0..128 {
                for c in 0..3 {
                    let idx = (v * 128 + u) * 3 + c;
                    assert_abs_diff_eq!(out[idx] as f64, expected_content, epsilon = 1e-6);
                }
            }
        }
    }

    #[test]
    fn test_letterbox_maps_normalized_centre_to_image_centre() {
        let letterbox = Letterbox::new(1280, 720, 128);
        let p = letterbox.to_image(&Point2::new(0.5, 0.5), 128);
        assert_abs_diff_eq!(p.x, 640.0, epsilon = 1e-9);
        assert_abs_diff_eq!(p.y, 360.0, epsilon = 1e-9);
    }
}
