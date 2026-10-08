use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::JoinHandle;
use std::time::{SystemTime, UNIX_EPOCH};

use crossbeam_channel::Sender;
use tracing::Event;
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record as SpanRecord};
use tracing::subscriber::Subscriber;
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::registry::LookupSpan;

use crate::record::{Record, Value};
use crate::sink::Sink;

#[derive(Default)]
struct FieldVisitor(BTreeMap<String, Value>);

impl Visit for FieldVisitor {
    fn record_bool(&mut self, field: &Field, value: bool) {
        self.0.insert(field.name().to_string(), Value::Bool(value));
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        self.0.insert(field.name().to_string(), Value::I64(value));
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.0.insert(field.name().to_string(), Value::U64(value));
    }

    fn record_f64(&mut self, field: &Field, value: f64) {
        self.0.insert(field.name().to_string(), Value::F64(value));
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        self.0
            .insert(field.name().to_string(), Value::Str(value.to_string()));
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.0
            .insert(field.name().to_string(), Value::Str(format!("{value:?}")));
    }
}

struct SpanFields(BTreeMap<String, Value>);

enum Msg {
    Record(Record),
    Shutdown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShutdownReport {
    pub written: u64,
    pub dropped: u64,
}

pub struct SinkLayer {
    tx: Sender<Msg>,
    dropped: Arc<AtomicU64>,
}

impl std::fmt::Debug for SinkLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SinkLayer").finish_non_exhaustive()
    }
}

pub struct SinkHandle {
    tx: Sender<Msg>,
    handle: JoinHandle<ShutdownReport>,
}

impl std::fmt::Debug for SinkHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SinkHandle").finish_non_exhaustive()
    }
}

impl SinkHandle {
    pub fn shutdown(self) -> ShutdownReport {
        let _ = self.tx.send(Msg::Shutdown);
        self.handle.join().unwrap_or(ShutdownReport {
            written: 0,
            dropped: 0,
        })
    }
}

impl SinkLayer {
    pub fn spawn(mut sink: Box<dyn Sink>, capacity: usize) -> (Self, SinkHandle) {
        let (tx, rx) = crossbeam_channel::bounded::<Msg>(capacity);
        let dropped = Arc::new(AtomicU64::new(0));
        let dropped_for_thread = Arc::clone(&dropped);
        let span = tracing::Span::current();

        let handle = std::thread::Builder::new()
            .name("eye-log-writer".to_string())
            .spawn(move || {
                let _enter = span.enter();
                let mut written: u64 = 0;
                while let Ok(msg) = rx.recv() {
                    match msg {
                        Msg::Record(record) => {
                            if sink.write(&record).is_ok() {
                                written += 1;
                            }
                        }
                        Msg::Shutdown => break,
                    }
                }
                let _ = sink.flush();
                ShutdownReport {
                    written,
                    dropped: dropped_for_thread.load(Ordering::Relaxed),
                }
            })
            .expect("spawning eye-log-writer thread");

        (
            SinkLayer {
                tx: tx.clone(),
                dropped,
            },
            SinkHandle { tx, handle },
        )
    }
}

impl<S> Layer<S> for SinkLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let mut visitor = FieldVisitor::default();
        attrs.record(&mut visitor);
        if let Some(span) = ctx.span(id) {
            span.extensions_mut().insert(SpanFields(visitor.0));
        }
    }

    fn on_record(&self, id: &Id, values: &SpanRecord<'_>, ctx: Context<'_, S>) {
        let mut visitor = FieldVisitor::default();
        values.record(&mut visitor);
        if let Some(span) = ctx.span(id) {
            let mut ext = span.extensions_mut();
            if let Some(existing) = ext.get_mut::<SpanFields>() {
                existing.0.extend(visitor.0);
            } else {
                ext.insert(SpanFields(visitor.0));
            }
        }
    }

    fn on_event(&self, event: &Event<'_>, ctx: Context<'_, S>) {
        let mut visitor = FieldVisitor::default();
        event.record(&mut visitor);
        let mut fields = visitor.0;
        let message = match fields.remove("message") {
            Some(Value::Str(s)) => s,
            _ => String::new(),
        };

        let mut context = BTreeMap::new();
        if let Some(scope) = ctx.event_scope(event) {
            for span in scope.from_root() {
                let ext = span.extensions();
                if let Some(span_fields) = ext.get::<SpanFields>() {
                    for (k, v) in &span_fields.0 {
                        context.insert(k.clone(), v.clone());
                    }
                }
            }
        }

        let ts_unix_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);

        let record = Record {
            schema: crate::LOG_SCHEMA,
            ts_unix_ns,
            level: (*event.metadata().level()).into(),
            target: event.metadata().target().to_string(),
            message,
            context,
            fields,
        };

        if self.tx.try_send(Msg::Record(record)).is_err() {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use tracing_subscriber::Registry;
    use tracing_subscriber::layer::SubscriberExt;

    use super::*;
    use crate::record::Level as RecordLevel;

    #[derive(Debug, Default)]
    struct VecSink(Arc<Mutex<Vec<Record>>>);

    impl Sink for VecSink {
        fn write(&mut self, record: &Record) -> std::io::Result<()> {
            self.0.lock().unwrap().push(record.clone());
            Ok(())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn with_sink<T>(capacity: usize, f: impl FnOnce() -> T) -> (T, Vec<Record>, ShutdownReport) {
        let buf = Arc::new(Mutex::new(Vec::new()));
        let (layer, handle) = SinkLayer::spawn(Box::new(VecSink(Arc::clone(&buf))), capacity);
        let subscriber = Registry::default().with(layer);
        let result = tracing::subscriber::with_default(subscriber, f);
        let report = handle.shutdown();
        let records = buf.lock().unwrap().clone();
        (result, records, report)
    }

    #[test]
    fn test_layer_merges_span_context_leaf_wins() {
        let (_, records, _) = with_sink(64, || {
            let run = tracing::info_span!("run", run.id = "r");
            let _run = run.entered();
            let frame = tracing::debug_span!("frame", camera = "ir", seq = 7u64);
            let _frame = frame.entered();
            let stage = tracing::debug_span!("stage", stage.kind = "detect", camera = "rgb");
            let _stage = stage.entered();
            tracing::debug!(reason = "x", "decision");
        });

        assert_eq!(records.len(), 1);
        let rec = &records[0];
        assert_eq!(rec.context.len(), 4);
        assert_eq!(rec.context["run.id"], Value::Str("r".to_string()));
        assert_eq!(rec.context["camera"], Value::Str("rgb".to_string()));
        assert_eq!(rec.context["seq"], Value::U64(7));
        assert_eq!(rec.context["stage.kind"], Value::Str("detect".to_string()));

        let mut expected_fields = BTreeMap::new();
        expected_fields.insert("reason".to_string(), Value::Str("x".to_string()));
        assert_eq!(rec.fields, expected_fields);
    }

    #[test]
    fn test_full_queue_drops_and_counts() {
        use std::sync::Barrier;

        struct Blocking {
            barrier: Arc<Barrier>,
            waited: bool,
        }
        impl Sink for Blocking {
            fn write(&mut self, _record: &Record) -> std::io::Result<()> {
                if !self.waited {
                    self.waited = true;
                    self.barrier.wait();
                }
                Ok(())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let barrier = Arc::new(Barrier::new(2));
        let sink = Blocking {
            barrier: Arc::clone(&barrier),
            waited: false,
        };
        let (layer, handle) = SinkLayer::spawn(Box::new(sink), 1);
        let subscriber = Registry::default().with(layer);

        tracing::subscriber::with_default(subscriber, || {
            for _ in 0..10 {
                tracing::info!("event");
            }
            barrier.wait();
        });
        let report = handle.shutdown();
        assert!(report.dropped >= 8, "dropped = {}", report.dropped);
    }

    #[test]
    fn test_sink_handle_shutdown_drains_and_reports() {
        let (_, records, report) = with_sink(64, || {
            tracing::info!("one");
            tracing::info!("two");
        });
        assert_eq!(records.len(), 2);
        assert_eq!(report.written, 2);
        assert_eq!(report.dropped, 0);
    }

    #[test]
    fn test_shutdown_flushes_and_reports() {
        use std::fs::File;
        use std::io::{BufRead, BufReader};

        use crate::cli::{self, LogArgs, LogFormat};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x.jsonl");
        let args = LogArgs {
            log_level: Some("off".to_string()),
            log_file: Some(path.to_str().unwrap().to_string()),
            log_file_level: "trace".to_string(),
            log_format: LogFormat::Text,
        };

        let guard = cli::init(&args, "test").unwrap();
        tracing::info!("one");
        tracing::info!("two");
        drop(guard);

        let file = File::open(&path).unwrap();
        let lines: Vec<String> = BufReader::new(file).lines().map(|l| l.unwrap()).collect();
        assert!(!lines.is_empty());

        let records: Vec<Record> = lines
            .iter()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();

        let messages: Vec<&str> = records.iter().map(|r| r.message.as_str()).collect();
        for expected in ["run started", "logging to file", "one", "two"] {
            assert!(
                messages.contains(&expected),
                "missing {expected:?} in {messages:?}"
            );
        }

        for record in &records {
            match record.context.get("run.id") {
                Some(Value::Str(id)) => assert!(!id.is_empty()),
                other => panic!("expected run.id in context, got {other:?}"),
            }
            assert_eq!(
                record.context.get("command"),
                Some(&Value::Str("test".to_string()))
            );
        }
    }

    #[test]
    fn test_sink_filter_is_independent_of_terminal_filter() {
        use tracing_subscriber::filter::LevelFilter;
        use tracing_subscriber::fmt;

        let term_buf: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
        let sink_buf = Arc::new(Mutex::new(Vec::new()));
        let (layer, handle) = SinkLayer::spawn(Box::new(VecSink(Arc::clone(&sink_buf))), 64);

        let term_writer = Arc::clone(&term_buf);
        let subscriber = Registry::default()
            .with(
                fmt::layer()
                    .with_writer(move || -> Box<dyn std::io::Write> {
                        Box::new(SharedBuf(Arc::clone(&term_writer)))
                    })
                    .with_filter(LevelFilter::INFO),
            )
            .with(layer.with_filter(LevelFilter::TRACE));

        tracing::subscriber::with_default(subscriber, || {
            tracing::trace!("trace-only event");
        });
        handle.shutdown();

        assert_eq!(sink_buf.lock().unwrap().len(), 1);
        assert!(term_buf.lock().unwrap().is_empty());
    }

    struct SharedBuf(Arc<Mutex<Vec<u8>>>);
    impl std::io::Write for SharedBuf {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn test_level_conversion_matches_tracing() {
        assert_eq!(RecordLevel::from(tracing::Level::ERROR), RecordLevel::Error);
        assert_eq!(RecordLevel::from(tracing::Level::TRACE), RecordLevel::Trace);
    }
}
