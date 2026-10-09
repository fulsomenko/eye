use std::collections::HashMap;

use eye_core::log::field;
use eye_core::observation::SCHEME_MEDIAPIPE_478;
use eye_core::stage::{Detector, StageError};
use eye_core::{CameraId, FaceObservation, FrameSet, Illumination, Observations, PixelFormat};

use crate::DetectError;
use crate::image::RgbImage;
use crate::mediapipe::anchors::{Anchor, short_range_anchors};
use crate::mediapipe::blazeface::{DETECTOR_INPUT, decode, letterbox_to_tensor, weighted_nms};
use crate::mediapipe::landmarks::{eyes_from_landmarks, unproject_landmarks};
use crate::mediapipe::roi::{RotatedRect, roi_from_detection, roi_from_landmarks};
use crate::mediapipe::warp::warp_roi_to_tensor;
use crate::mediapipe::{LANDMARK_INPUT, MediaPipeOptions, MediaPipeRuntime};

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[derive(Debug)]
pub struct MediaPipeDetector<R: MediaPipeRuntime> {
    name: &'static str,
    runtime: R,
    options: MediaPipeOptions,
    anchors: Vec<Anchor>,
    tracked: HashMap<CameraId, RotatedRect>,
    detector_input: Vec<f32>,
    landmark_input: Vec<f32>,
}

impl<R: MediaPipeRuntime> MediaPipeDetector<R> {
    pub fn new(name: &'static str, runtime: R, options: MediaPipeOptions) -> Self {
        Self {
            name,
            runtime,
            options,
            anchors: short_range_anchors(),
            tracked: HashMap::new(),
            detector_input: vec![0.0; DETECTOR_INPUT * DETECTOR_INPUT * 3],
            landmark_input: vec![0.0; LANDMARK_INPUT * LANDMARK_INPUT * 3],
        }
    }

    pub fn detect_image(
        &mut self,
        camera: &CameraId,
        image: &RgbImage,
    ) -> Result<Option<FaceObservation>, DetectError> {
        self.detect_image_retrying(camera, image, true)
    }

    fn detect_image_retrying(
        &mut self,
        camera: &CameraId,
        image: &RgbImage,
        allow_retry: bool,
    ) -> Result<Option<FaceObservation>, DetectError> {
        let (roi, from_track) = match self.tracked.get(camera).copied() {
            Some(roi) if self.options.track => (roi, true),
            _ => match self.detect_roi(image)? {
                Some(roi) => (roi, false),
                None => {
                    self.tracked.remove(camera);
                    return Ok(None);
                }
            },
        };

        warp_roi_to_tensor(
            image,
            &roi,
            LANDMARK_INPUT,
            (0.0, 1.0),
            &mut self.landmark_input,
        );
        let output = self.runtime.run_landmarks(&self.landmark_input)?;
        let presence = sigmoid(output.presence_logit);

        if presence < self.options.min_presence {
            self.tracked.remove(camera);
            let retry = from_track && allow_retry;
            tracing::debug!(
                { field::REASON } = "low_presence",
                presence = f64::from(presence),
                min_presence = f64::from(self.options.min_presence),
                from_track,
                retry,
                "face rejected"
            );
            if retry {
                return self.detect_image_retrying(camera, image, false);
            }
            return Ok(None);
        }

        let landmarks = unproject_landmarks(&output.landmarks, &roi, LANDMARK_INPUT)?;
        let eyes = eyes_from_landmarks(&landmarks, &roi, &self.options)?;

        tracing::trace!(
            presence = f64::from(presence),
            landmarks = landmarks.len() as u64,
            from_track,
            roi_size = roi.size,
            "landmarks"
        );

        if self.options.track {
            self.tracked
                .insert(camera.clone(), roi_from_landmarks(&landmarks));
        }

        Ok(Some(FaceObservation {
            scheme: SCHEME_MEDIAPIPE_478,
            landmarks,
            eyes: eyes.into(),
        }))
    }

    fn detect_roi(&mut self, image: &RgbImage) -> Result<Option<RotatedRect>, DetectError> {
        let letterbox =
            letterbox_to_tensor(image, DETECTOR_INPUT, (-1.0, 1.0), &mut self.detector_input);
        let raw = self.runtime.run_face_detector(&self.detector_input)?;
        let detections = decode(&raw, &self.anchors, self.options.min_detection_score)?;
        let n_detections = detections.len() as u64;
        let merged = weighted_nms(detections, self.options.nms_iou);
        let n_merged = merged.len() as u64;
        match merged.into_iter().next() {
            Some(det) => {
                let roi = roi_from_detection(&det, &letterbox, DETECTOR_INPUT);
                tracing::trace!(
                    score = f64::from(det.score),
                    x = roi.center.x,
                    y = roi.center.y,
                    size = roi.size,
                    rotation_rad = roi.rotation,
                    merged = n_merged,
                    "face roi"
                );
                Ok(Some(roi))
            }
            None => {
                tracing::debug!(
                    { field::REASON } = "no_face",
                    detections = n_detections,
                    min_detection_score = f64::from(self.options.min_detection_score),
                    "no face"
                );
                Ok(None)
            }
        }
    }

    pub fn detect_frames(&mut self, frames: &FrameSet) -> Result<Vec<Observations>, DetectError> {
        let mut out = Vec::new();
        for frame in frames.frames() {
            let header = frame.header();
            if !self.accepts(header.format, header.illumination) {
                tracing::trace!(
                    { field::REASON } = "not_accepted",
                    format = ?header.format,
                    { field::ILLUMINATION } = header.illumination.as_str(),
                    "frame not accepted"
                );
                continue;
            }
            let image = RgbImage::from_frame(frame)?;
            let start = std::time::Instant::now();
            let face = self.detect_image(&header.camera, &image)?;
            tracing::trace!(
                { field::ELAPSED_US } = start.elapsed().as_micros() as u64,
                face = face.is_some(),
                "mediapipe frame"
            );
            out.push(Observations {
                camera: header.camera.clone(),
                timestamp: header.timestamp,
                face,
            });
        }
        Ok(out)
    }
}

impl<R: MediaPipeRuntime> Detector for MediaPipeDetector<R> {
    fn name(&self) -> &'static str {
        self.name
    }

    fn accepts(&self, format: PixelFormat, illumination: Illumination) -> bool {
        matches!(format, PixelFormat::Mjpeg | PixelFormat::Rgb8)
            && matches!(illumination, Illumination::Ambient | Illumination::Unknown)
    }

    fn detect(&mut self, frames: &FrameSet) -> Result<Vec<Observations>, StageError> {
        Ok(self.detect_frames(frames)?)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::Arc;

    use eye_core::{Frame, FrameHeader, Side, Timestamp};
    use nalgebra::{Point2, Vector2};

    use super::*;
    use crate::mediapipe::LandmarkOutput;
    use crate::mediapipe::anchors::short_range_anchors;
    use crate::mediapipe::blazeface::{
        FaceDetection, FaceDetectorOutput, Letterbox, NUM_ANCHORS, NUM_COORDS, decode,
        encode_detection,
    };

    #[derive(Debug)]
    struct FakeRuntime {
        detector: FaceDetectorOutput,
        landmarks: VecDeque<LandmarkOutput>,
        detector_calls: usize,
    }

    impl MediaPipeRuntime for FakeRuntime {
        fn run_face_detector(&mut self, _input: &[f32]) -> Result<FaceDetectorOutput, DetectError> {
            self.detector_calls += 1;
            Ok(self.detector.clone())
        }

        fn run_landmarks(&mut self, _input: &[f32]) -> Result<LandmarkOutput, DetectError> {
            Ok(self
                .landmarks
                .pop_front()
                .expect("no landmark output queued"))
        }
    }

    fn sample_detection() -> FaceDetection {
        FaceDetection {
            score: 0.0,
            center: Point2::new(0.5, 0.5),
            size: Vector2::new(0.3, 0.3),
            keypoints: [
                Point2::new(0.45, 0.48),
                Point2::new(0.55, 0.48),
                Point2::new(0.5, 0.5),
                Point2::new(0.5, 0.55),
                Point2::new(0.42, 0.51),
                Point2::new(0.58, 0.51),
            ],
        }
    }

    fn sample_raw_landmarks() -> Vec<f32> {
        let mut raw = vec![0.0f32; crate::mediapipe::NUM_LANDMARKS * 3];
        for i in 0..crate::mediapipe::NUM_LANDMARKS {
            raw[i * 3] = 40.0 + i as f32 * 0.3;
            raw[i * 3 + 1] = 50.0 + i as f32 * 0.2;
        }
        raw
    }

    fn sample_image() -> RgbImage {
        RgbImage {
            width: 1280,
            height: 720,
            data: vec![128u8; 1280 * 720 * 3],
        }
    }

    #[test]
    fn test_pipeline_fake_runtime_produces_face() {
        let anchors = short_range_anchors();
        let det = sample_detection();
        let detector_output = encode_detection(&anchors, 400, &det);
        let raw_landmarks = sample_raw_landmarks();
        let runtime = FakeRuntime {
            detector: detector_output,
            landmarks: VecDeque::from(vec![LandmarkOutput {
                landmarks: raw_landmarks.clone(),
                presence_logit: 5.0,
            }]),
            detector_calls: 0,
        };
        let mut pipeline =
            MediaPipeDetector::new("mediapipe-fake", runtime, MediaPipeOptions::default());
        let image = sample_image();
        let camera = CameraId::from("rgb");

        let face = pipeline
            .detect_image(&camera, &image)
            .unwrap()
            .expect("face expected");
        assert_eq!(face.scheme, SCHEME_MEDIAPIPE_478);
        assert_eq!(face.landmarks.len(), crate::mediapipe::NUM_LANDMARKS);
        assert_eq!(face.eyes.len(), 2);

        let letterbox = Letterbox::new(image.width, image.height, DETECTOR_INPUT);
        let raw_for_roi = encode_detection(&anchors, 400, &det);
        let decoded = decode(
            &raw_for_roi,
            &anchors,
            MediaPipeOptions::default().min_detection_score,
        )
        .unwrap();
        let roi = roi_from_detection(&decoded[0], &letterbox, DETECTOR_INPUT);
        let expected_iris_center = roi.crop_to_image(
            &Point2::new(
                f64::from(raw_landmarks[468 * 3]),
                f64::from(raw_landmarks[468 * 3 + 1]),
            ),
            LANDMARK_INPUT,
        );

        let right = face.eye(Side::Right).unwrap();
        let iris_center = right.iris.as_ref().unwrap().value().center();
        approx::assert_abs_diff_eq!(iris_center.x, expected_iris_center.x, epsilon = 1e-9);
        approx::assert_abs_diff_eq!(iris_center.y, expected_iris_center.y, epsilon = 1e-9);
    }

    #[test]
    fn test_pipeline_tracking_skips_detector_on_second_frame() {
        let anchors = short_range_anchors();
        let det = sample_detection();
        let detector_output = encode_detection(&anchors, 400, &det);
        let raw_landmarks = sample_raw_landmarks();
        let runtime = FakeRuntime {
            detector: detector_output,
            landmarks: VecDeque::from(vec![
                LandmarkOutput {
                    landmarks: raw_landmarks.clone(),
                    presence_logit: 5.0,
                },
                LandmarkOutput {
                    landmarks: raw_landmarks,
                    presence_logit: 5.0,
                },
            ]),
            detector_calls: 0,
        };
        let mut pipeline =
            MediaPipeDetector::new("mediapipe-fake", runtime, MediaPipeOptions::default());
        let image = sample_image();
        let camera = CameraId::from("rgb");

        pipeline.detect_image(&camera, &image).unwrap();
        pipeline.detect_image(&camera, &image).unwrap();

        assert_eq!(pipeline.runtime.detector_calls, 1);
    }

    #[test]
    fn test_pipeline_lost_track_retries_detector_same_frame() {
        let anchors = short_range_anchors();
        let det = sample_detection();
        let detector_output = encode_detection(&anchors, 400, &det);
        let raw_landmarks = sample_raw_landmarks();
        let runtime = FakeRuntime {
            detector: detector_output,
            landmarks: VecDeque::from(vec![
                LandmarkOutput {
                    landmarks: raw_landmarks.clone(),
                    presence_logit: 5.0,
                },
                LandmarkOutput {
                    landmarks: raw_landmarks.clone(),
                    presence_logit: -5.0,
                },
                LandmarkOutput {
                    landmarks: raw_landmarks,
                    presence_logit: 5.0,
                },
            ]),
            detector_calls: 0,
        };
        let mut pipeline =
            MediaPipeDetector::new("mediapipe-fake", runtime, MediaPipeOptions::default());
        let image = sample_image();
        let camera = CameraId::from("rgb");

        let first = pipeline.detect_image(&camera, &image).unwrap();
        assert!(first.is_some());
        let second = pipeline.detect_image(&camera, &image).unwrap();
        assert!(second.is_some());

        assert_eq!(pipeline.runtime.detector_calls, 2);
    }

    #[test]
    fn test_pipeline_no_detection_yields_none_face() {
        let runtime = FakeRuntime {
            detector: FaceDetectorOutput {
                regressors: vec![0.0; NUM_ANCHORS * NUM_COORDS],
                logits: vec![-10.0; NUM_ANCHORS],
            },
            landmarks: VecDeque::new(),
            detector_calls: 0,
        };
        let mut pipeline =
            MediaPipeDetector::new("mediapipe-fake", runtime, MediaPipeOptions::default());
        let image = sample_image();
        let camera = CameraId::from("rgb");

        let face = pipeline.detect_image(&camera, &image).unwrap();
        assert!(face.is_none());
        assert!(!pipeline.tracked.contains_key(&camera));
    }

    #[test]
    fn test_detect_through_trait_on_rgb_frame() {
        let anchors = short_range_anchors();
        let det = sample_detection();
        let detector_output = encode_detection(&anchors, 400, &det);
        let raw_landmarks = sample_raw_landmarks();
        let runtime = FakeRuntime {
            detector: detector_output,
            landmarks: VecDeque::from(vec![LandmarkOutput {
                landmarks: raw_landmarks,
                presence_logit: 5.0,
            }]),
            detector_calls: 0,
        };
        let mut detector: Box<dyn Detector> = Box::new(MediaPipeDetector::new(
            "mediapipe-fake",
            runtime,
            MediaPipeOptions::default(),
        ));

        let rgb_data: Arc<[u8]> = vec![128u8; 1280 * 720 * 3].into();
        let rgb_frame = Frame::new(
            FrameHeader {
                camera: CameraId::from("rgb"),
                seq: 0,
                timestamp: Timestamp::from_nanos(0),
                width: 1280,
                height: 720,
                format: PixelFormat::Rgb8,
                illumination: Illumination::Ambient,
            },
            rgb_data,
        )
        .unwrap();

        let ir_data: Arc<[u8]> = vec![0u8; 4].into();
        let ir_frame = Frame::new(
            FrameHeader {
                camera: CameraId::from("ir"),
                seq: 0,
                timestamp: Timestamp::from_nanos(0),
                width: 2,
                height: 2,
                format: PixelFormat::Gray8,
                illumination: Illumination::IrLit,
            },
            ir_data,
        )
        .unwrap();

        let frames = FrameSet::new(vec![rgb_frame, ir_frame]).unwrap();
        let obs = detector.detect(&frames).unwrap();
        assert_eq!(obs.len(), 1);
        assert_eq!(obs[0].camera, CameraId::from("rgb"));
        assert!(obs[0].face.is_some());
    }

    #[test]
    fn test_logs_no_face_at_debug() {
        use eye_log::testing::capture_logs;
        use eye_log::{Level as LogLevel, Value};

        let runtime = FakeRuntime {
            detector: FaceDetectorOutput {
                regressors: vec![0.0; NUM_ANCHORS * NUM_COORDS],
                logits: vec![-10.0; NUM_ANCHORS],
            },
            landmarks: VecDeque::new(),
            detector_calls: 0,
        };
        let mut pipeline =
            MediaPipeDetector::new("mediapipe-fake", runtime, MediaPipeOptions::default());
        let image = sample_image();
        let camera = CameraId::from("rgb");

        let (face, logs) = capture_logs(tracing::Level::TRACE, || {
            pipeline.detect_image(&camera, &image)
        });
        assert!(face.unwrap().is_none());

        let rec = logs
            .iter()
            .find(|r| r.message == "no face")
            .expect("event emitted");
        assert_eq!(rec.level, LogLevel::Debug);
        assert_eq!(rec.fields[field::REASON], Value::Str("no_face".into()));
        assert_eq!(rec.fields["detections"], Value::U64(0));
        assert_eq!(rec.fields["min_detection_score"], Value::F64(0.5));
    }

    #[test]
    fn test_logs_low_presence_at_debug() {
        use eye_log::testing::capture_logs;
        use eye_log::{Level as LogLevel, Value};

        let anchors = short_range_anchors();
        let det = sample_detection();
        let detector_output = encode_detection(&anchors, 400, &det);
        let raw_landmarks = sample_raw_landmarks();
        let runtime = FakeRuntime {
            detector: detector_output,
            landmarks: VecDeque::from(vec![
                LandmarkOutput {
                    landmarks: raw_landmarks.clone(),
                    presence_logit: 5.0,
                },
                LandmarkOutput {
                    landmarks: raw_landmarks.clone(),
                    presence_logit: -5.0,
                },
                LandmarkOutput {
                    landmarks: raw_landmarks,
                    presence_logit: 5.0,
                },
            ]),
            detector_calls: 0,
        };
        let mut pipeline =
            MediaPipeDetector::new("mediapipe-fake", runtime, MediaPipeOptions::default());
        let image = sample_image();
        let camera = CameraId::from("rgb");

        pipeline.detect_image(&camera, &image).unwrap();
        let (second, logs) = capture_logs(tracing::Level::TRACE, || {
            pipeline.detect_image(&camera, &image)
        });
        assert!(second.unwrap().is_some());

        let rejected = logs
            .iter()
            .find(|r| r.message == "face rejected")
            .expect("event emitted");
        assert_eq!(rejected.level, LogLevel::Debug);
        assert_eq!(
            rejected.fields[field::REASON],
            Value::Str("low_presence".into())
        );
        assert_eq!(rejected.fields["from_track"], Value::Bool(true));
        assert_eq!(rejected.fields["retry"], Value::Bool(true));

        let landmarks = logs
            .iter()
            .find(|r| r.message == "landmarks")
            .expect("event emitted");
        assert_eq!(landmarks.fields["from_track"], Value::Bool(false));
    }

    #[test]
    fn test_logs_landmarks_at_trace() {
        use eye_core::{Frame, FrameHeader, Timestamp};
        use eye_log::Value;
        use eye_log::testing::capture_logs;

        let anchors = short_range_anchors();
        let det = sample_detection();
        let detector_output = encode_detection(&anchors, 400, &det);
        let raw_landmarks = sample_raw_landmarks();
        let runtime = FakeRuntime {
            detector: detector_output,
            landmarks: VecDeque::from(vec![LandmarkOutput {
                landmarks: raw_landmarks,
                presence_logit: 5.0,
            }]),
            detector_calls: 0,
        };
        let mut pipeline =
            MediaPipeDetector::new("mediapipe-fake", runtime, MediaPipeOptions::default());
        let image = sample_image();

        let frame = Frame::new(
            FrameHeader {
                camera: CameraId::from("rgb"),
                seq: 0,
                timestamp: Timestamp::from_nanos(0),
                width: image.width,
                height: image.height,
                format: PixelFormat::Rgb8,
                illumination: Illumination::Ambient,
            },
            image.data.clone().into(),
        )
        .unwrap();
        let frames = FrameSet::single(frame);

        let (out, logs) = capture_logs(tracing::Level::TRACE, || {
            pipeline.detect_frames(&frames).unwrap()
        });
        assert_eq!(out.len(), 1);
        assert!(out[0].face.is_some());

        let face_roi = logs
            .iter()
            .find(|r| r.message == "face roi")
            .expect("event emitted");
        assert_eq!(face_roi.fields["merged"], Value::U64(1));

        let landmarks_rec = logs
            .iter()
            .find(|r| r.message == "landmarks")
            .expect("event emitted");
        assert_eq!(landmarks_rec.fields["landmarks"], Value::U64(478));
        match landmarks_rec.fields["presence"] {
            Value::F64(p) => assert!(p > 0.99, "presence {p}"),
            ref other => panic!("expected F64 presence, got {other:?}"),
        }

        let frame_rec = logs
            .iter()
            .find(|r| r.message == "mediapipe frame")
            .expect("event emitted");
        assert_eq!(frame_rec.fields["face"], Value::Bool(true));
    }

    #[test]
    fn test_accepts_mjpeg_and_rgb_ambient_only() {
        let runtime = FakeRuntime {
            detector: FaceDetectorOutput {
                regressors: vec![],
                logits: vec![],
            },
            landmarks: VecDeque::new(),
            detector_calls: 0,
        };
        let pipeline =
            MediaPipeDetector::new("mediapipe-fake", runtime, MediaPipeOptions::default());

        for format in [PixelFormat::Mjpeg, PixelFormat::Rgb8, PixelFormat::Gray8] {
            for illumination in [
                Illumination::Ambient,
                Illumination::Unknown,
                Illumination::IrLit,
                Illumination::IrDark,
            ] {
                let expected = matches!(format, PixelFormat::Mjpeg | PixelFormat::Rgb8)
                    && matches!(illumination, Illumination::Ambient | Illumination::Unknown);
                assert_eq!(
                    pipeline.accepts(format, illumination),
                    expected,
                    "{format:?}/{illumination:?}"
                );
            }
        }
    }
}
