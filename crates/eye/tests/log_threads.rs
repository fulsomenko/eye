use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use eye::pipeline::Pipeline;
use eye::registry::PassThroughFilter;
use eye::tracker::{Tracker, TrackerError, TrackerOptions};
use eye_capture::pairing::Pairer;
use eye_capture::{CaptureError, FrameSource};
use eye_core::log::{field, span};
use eye_core::{
    CameraId, CameraInfo, CameraModel, Frame, FrameHeader, FrameSet, GazeRay, Illumination,
    Observations, OutputId, PixelFormat, Rig, ScreenModel, Timestamp,
    stage::{Detector, GazeEstimator, StageError},
};
use eye_log::Value;
use tracing_subscriber::layer::SubscriberExt;

struct VecSource {
    info: CameraInfo,
    frames: VecDeque<Frame>,
}

impl FrameSource for VecSource {
    fn camera(&self) -> &CameraInfo {
        &self.info
    }

    fn next_frame(&mut self) -> Result<Frame, CaptureError> {
        self.frames.pop_front().ok_or(CaptureError::EndOfStream)
    }
}

#[derive(Debug)]
struct FixedDetector;

impl Detector for FixedDetector {
    fn name(&self) -> &'static str {
        "fixed"
    }

    fn accepts(&self, _format: PixelFormat, _illumination: Illumination) -> bool {
        true
    }

    fn detect(&mut self, frames: &FrameSet) -> Result<Vec<Observations>, StageError> {
        Ok(frames
            .frames()
            .iter()
            .map(|f| Observations::empty(f.header().camera.clone(), f.header().timestamp))
            .collect())
    }
}

#[derive(Debug)]
struct FixedEstimator;

impl GazeEstimator for FixedEstimator {
    fn name(&self) -> &'static str {
        "fixed"
    }

    fn estimate(&mut self, _obs: &[Observations], _rig: &Rig) -> Result<Vec<GazeRay>, StageError> {
        Ok(vec![GazeRay {
            side: None,
            origin: nalgebra::Point3::new(155.0, 85.0, -500.0),
            direction: nalgebra::Vector3::z_axis(),
            angular_cov: nalgebra::Matrix2::identity() * 1e-6,
            origin_cov: nalgebra::Matrix3::zeros(),
            head_rotation: nalgebra::UnitQuaternion::identity(),
        }])
    }
}

fn rig() -> Rig {
    let camera = CameraModel {
        id: CameraId::from("ir"),
        width: 640,
        height: 360,
        fx: 430.0,
        fy: 430.0,
        cx: 320.0,
        cy: 180.0,
        distortion: [0.0; 5],
        screen_from_camera: nalgebra::Isometry3::identity(),
    };
    let screen = ScreenModel {
        output: OutputId::from("eDP-1"),
        size_mm: nalgebra::Vector2::new(310.0, 170.0),
        size_px: (3840, 2160),
        scale: 2.0,
    };
    Rig::new(vec![camera], screen).expect("rig is valid")
}

fn info() -> CameraInfo {
    CameraInfo {
        id: CameraId::from("ir"),
        format: PixelFormat::Gray8,
        width: 4,
        height: 2,
        frame_interval: Duration::from_millis(66),
    }
}

fn frame(seq: u64, t_ms: u64) -> Frame {
    Frame::new(
        FrameHeader {
            camera: CameraId::from("ir"),
            seq,
            timestamp: Timestamp::from_nanos(t_ms * 1_000_000),
            width: 4,
            height: 2,
            format: PixelFormat::Gray8,
            illumination: Illumination::IrLit,
        },
        vec![0u8; 8].into(),
    )
    .expect("frame is valid")
}

#[test]
fn test_capture_thread_inherits_parent_span() {
    let buf = Arc::new(Mutex::new(Vec::new()));
    let sink = eye_log::testing::VecSink(Arc::clone(&buf));
    let (layer, handle) = eye_log::SinkLayer::spawn(Box::new(sink), 4096);
    let subscriber = tracing_subscriber::Registry::default().with(layer);
    tracing::subscriber::set_global_default(subscriber)
        .expect("this is the only global subscriber installed in this process");

    let run = tracing::info_span!(span::RUN, { field::RUN_ID } = "log-threads");
    let _run = run.entered();

    let source = VecSource {
        info: info(),
        frames: VecDeque::from(vec![frame(0, 0), frame(1, 66), frame(2, 132)]),
    };

    let pipeline = Pipeline::new(
        rig(),
        Pairer::new(&[info()]).expect("single-camera pairer"),
        vec![(
            CameraId::from("ir"),
            Box::new(FixedDetector) as Box<dyn Detector>,
        )],
        Box::new(FixedEstimator),
        Box::new(PassThroughFilter),
        None,
        3,
    );

    let mut tracker = Tracker::start(
        pipeline,
        vec![Box::new(source) as Box<dyn FrameSource>],
        TrackerOptions {
            capture_capacity: 8,
            output_capacity: 8,
            sinks: Vec::new(),
        },
    )
    .expect("tracker starts");

    loop {
        match tracker.next() {
            Ok(_) => {}
            Err(TrackerError::EndOfStream) => break,
            Err(e) => panic!("unexpected error: {e:?}"),
        }
    }
    tracker.shutdown().expect("tracker shuts down cleanly");
    drop(_run);

    handle.shutdown();
    let records = buf.lock().expect("eye-log VecSink mutex poisoned").clone();

    let mut checked = 0;
    for rec in &records {
        if matches!(
            rec.message.as_str(),
            "capture ended" | "frame set" | "all captures ended; pipeline stopping"
        ) {
            assert_eq!(
                rec.context.get(field::RUN_ID),
                Some(&Value::Str("log-threads".to_string())),
                "missing {} on {:?}",
                field::RUN_ID,
                rec.message
            );
            checked += 1;
            if rec.message == "frame set" {
                assert_eq!(rec.context.get(field::SET_CAMERAS), Some(&Value::U64(1)));
            }
        }
    }
    assert!(
        checked >= 3,
        "expected capture ended, frame set, and all captures ended records, found {checked} in {records:?}"
    );
}
