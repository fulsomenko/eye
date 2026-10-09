//! The overlay thread emits to the GLOBAL default subscriber, so this file installs one.
use std::sync::{Arc, Mutex};

use eye_log::testing::VecSink;
use eye_overlay::scene::{Scene, Schedule};
use eye_overlay::surface::{SurfaceOptions, spawn};
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::layer::{Layer, SubscriberExt};

struct BlankScene;
impl Scene for BlankScene {
    type Msg = ();
    fn on_msg(&mut self, _: (), _: std::time::Instant) {}
    fn render(
        &mut self,
        _: &mut eye_overlay::canvas::Canvas<'_>,
        _: std::time::Instant,
    ) -> Schedule {
        Schedule::Idle
    }
}

#[test]
#[ignore = "needs wayland"]
fn test_overlay_thread_inherits_run_span() {
    let records = Arc::new(Mutex::new(Vec::new()));
    let (layer, sink) = eye_log::SinkLayer::spawn(Box::new(VecSink(Arc::clone(&records))), 1024);
    let subscriber = tracing_subscriber::registry().with(layer.with_filter(LevelFilter::TRACE));
    tracing::subscriber::set_global_default(subscriber)
        .expect("first and only global subscriber in this binary");

    let span = tracing::info_span!("run", "run.id" = "t");
    let handle = {
        let _g = span.enter();
        spawn(
            SurfaceOptions {
                output: std::env::var("EYE_OUTPUT").ok(),
                namespace: "eye-overlay-test",
            },
            BlankScene,
        )
        .expect("spawn succeeds")
    };
    handle.close().expect("clean exit");
    sink.shutdown();

    let recs = records.lock().unwrap();
    let configured = recs
        .iter()
        .find(|r| r.message == "layer surface configured")
        .expect("configured record");
    assert_eq!(
        configured.context.get("run.id"),
        Some(&eye_log::Value::Str("t".into()))
    );
    assert_eq!(configured.target, "eye_overlay::surface");
}
