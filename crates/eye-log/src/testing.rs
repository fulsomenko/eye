use std::sync::{Arc, Mutex};

use tracing::Level;
use tracing_subscriber::Registry;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::layer::{Layer, SubscriberExt};

use crate::layer::SinkLayer;
use crate::record::Record;
use crate::sink::Sink;

#[derive(Debug, Default)]
pub struct VecSink(pub Arc<Mutex<Vec<Record>>>);

impl Sink for VecSink {
    fn write(&mut self, record: &Record) -> std::io::Result<()> {
        self.0
            .lock()
            .expect("eye-log VecSink mutex poisoned")
            .push(record.clone());
        Ok(())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub fn capture_logs<T>(max_level: Level, f: impl FnOnce() -> T) -> (T, Vec<Record>) {
    let buf = Arc::new(Mutex::new(Vec::new()));
    let sink = VecSink(Arc::clone(&buf));
    let (layer, handle) = SinkLayer::spawn(Box::new(sink), 1 << 16);
    let subscriber =
        Registry::default().with(layer.with_filter(LevelFilter::from_level(max_level)));
    let result = tracing::subscriber::with_default(subscriber, f);
    handle.shutdown();
    let records = buf.lock().expect("eye-log VecSink mutex poisoned").clone();
    (result, records)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capture_logs_captures_up_to_max_level() {
        let (value, records) = capture_logs(Level::INFO, || {
            tracing::trace!("below threshold");
            tracing::info!("at threshold");
            42
        });
        assert_eq!(value, 42);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].message, "at threshold");
    }
}
