//! Names shared by every emitter and every sink.
pub mod span {
    pub const RUN: &str = "run";
    pub const SESSION: &str = "session";
    pub const FRAME: &str = "frame";
    pub const STAGE: &str = "stage";
}
pub mod field {
    pub const RUN_ID: &str = "run.id";
    pub const COMMAND: &str = "command";
    pub const SESSION_ID: &str = "session.id";
    pub const CAMERA: &str = "camera";
    pub const SEQ: &str = "seq";
    pub const TS_NS: &str = "ts_ns";
    pub const ILLUMINATION: &str = "illumination";
    pub const SET_CAMERAS: &str = "set.cameras";
    pub const STAGE_KIND: &str = "stage.kind";
    pub const STAGE_NAME: &str = "stage.name";
    pub const REASON: &str = "reason";
    pub const ELAPSED_US: &str = "elapsed_us";
    pub const SCHEMA: &str = "schema";
}
pub const LOG_SCHEMA: u32 = 1;

/// The per-frame span every stage event is nested in. Entered where a frame is produced or processed.
/// `set_cameras` is the number of frames in the FrameSet; 1 for a single frame.
pub fn frame_span(
    camera: &str,
    seq: u64,
    ts_ns: u64,
    illumination: &str,
    set_cameras: u64,
) -> tracing::Span {
    tracing::debug_span!(
        span::FRAME,
        { field::CAMERA } = camera,
        { field::SEQ } = seq,
        { field::TS_NS } = ts_ns,
        { field::ILLUMINATION } = illumination,
        { field::SET_CAMERAS } = set_cameras,
    )
}

/// Spawns a named thread that runs `f` inside the caller's current span, so every record the
/// thread emits carries the same `run`/`session` context as the spawner.
pub fn spawn_in_current_span<F, T>(
    name: impl Into<String>,
    f: F,
) -> std::io::Result<std::thread::JoinHandle<T>>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    let parent = tracing::Span::current();
    std::thread::Builder::new()
        .name(name.into())
        .spawn(move || {
            let _parent = parent.entered();
            f()
        })
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use tracing::field::{Field, Visit};
    use tracing::span::{Attributes, Record};
    use tracing::{Event, Id, Metadata, Subscriber};

    use super::*;

    #[derive(Default)]
    struct Recorded {
        span_name: Option<&'static str>,
        fields: Vec<&'static str>,
        meta: Option<&'static Metadata<'static>>,
        last_entered: Option<u64>,
        entered: Vec<(u64, Option<String>)>,
    }
    struct NameVisitor<'a>(&'a mut Vec<&'static str>);
    impl Visit for NameVisitor<'_> {
        fn record_debug(&mut self, field: &Field, _: &dyn std::fmt::Debug) {
            self.0.push(field.name());
        }
    }
    struct Recorder(Arc<Mutex<Recorded>>);
    impl Subscriber for Recorder {
        fn enabled(&self, _: &Metadata<'_>) -> bool {
            true
        }
        fn new_span(&self, attrs: &Attributes<'_>) -> Id {
            let mut r = self.0.lock().unwrap();
            r.span_name = Some(attrs.metadata().name());
            r.meta = Some(attrs.metadata());
            attrs.record(&mut NameVisitor(&mut r.fields));
            Id::from_u64(1)
        }
        fn record(&self, _: &Id, _: &Record<'_>) {}
        fn record_follows_from(&self, _: &Id, _: &Id) {}
        fn event(&self, _: &Event<'_>) {}
        fn enter(&self, id: &Id) {
            let mut r = self.0.lock().unwrap();
            r.last_entered = Some(id.into_u64());
            r.entered.push((
                id.into_u64(),
                std::thread::current().name().map(String::from),
            ));
        }
        fn exit(&self, _: &Id) {
            self.0.lock().unwrap().last_entered = None;
        }
        fn current_span(&self) -> tracing_core::span::Current {
            let r = self.0.lock().unwrap();
            match (r.last_entered, r.meta) {
                (Some(id), Some(meta)) => tracing_core::span::Current::new(Id::from_u64(id), meta),
                _ => tracing_core::span::Current::none(),
            }
        }
    }

    #[test]
    fn test_frame_span_records_the_vocabulary_names() {
        let rec = Arc::new(Mutex::new(Recorded::default()));
        tracing::subscriber::with_default(Recorder(rec.clone()), || {
            let _s = frame_span("ir", 7, 1, "ir_lit", 2).entered();
        });
        let r = rec.lock().unwrap();
        assert_eq!(r.span_name, Some(span::FRAME));
        assert_eq!(
            r.fields,
            vec![
                field::CAMERA,
                field::SEQ,
                field::TS_NS,
                field::ILLUMINATION,
                field::SET_CAMERAS,
            ]
        );
    }

    #[test]
    fn test_spawn_in_current_span_runs_inside_the_spawners_span() {
        let rec = Arc::new(Mutex::new(Recorded::default()));
        tracing::subscriber::with_default(Recorder(rec.clone()), || {
            let _parent = frame_span("ir", 1, 1, "ir_lit", 1).entered();
            let name = super::spawn_in_current_span("probe", || {
                std::thread::current().name().map(String::from)
            })
            .unwrap()
            .join()
            .unwrap();
            assert_eq!(name, Some("probe".to_string()));
        });
        assert!(
            rec.lock()
                .unwrap()
                .entered
                .contains(&(1, Some("probe".to_string())))
        );
    }

    #[test]
    fn test_constants_are_namespaced() {
        let dotted = [
            field::RUN_ID,
            field::SESSION_ID,
            field::STAGE_KIND,
            field::STAGE_NAME,
            field::SET_CAMERAS,
        ];
        for name in dotted {
            assert_eq!(
                name.matches('.').count(),
                1,
                "expected exactly one '.' in {name}"
            );
        }

        let undotted = [
            field::COMMAND,
            field::CAMERA,
            field::SEQ,
            field::TS_NS,
            field::ILLUMINATION,
            field::REASON,
            field::ELAPSED_US,
            field::SCHEMA,
            span::RUN,
            span::SESSION,
            span::FRAME,
            span::STAGE,
        ];
        for name in undotted {
            assert_eq!(name.matches('.').count(), 0, "expected no '.' in {name}");
        }

        for name in dotted.iter().chain(undotted.iter()) {
            assert!(
                !name.chars().any(char::is_whitespace),
                "{name} contains whitespace"
            );
            assert!(
                !name.chars().any(char::is_uppercase),
                "{name} contains an uppercase letter"
            );
        }
    }
}
