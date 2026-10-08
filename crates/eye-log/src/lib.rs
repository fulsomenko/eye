pub mod cli;
mod layer;
mod record;
mod sink;
mod stamp;
pub mod testing;

pub use eye_core::log::LOG_SCHEMA;
pub use layer::{ShutdownReport, SinkHandle, SinkLayer};
pub use record::{Level, Record, Value};
pub use sink::{JsonLinesSink, Sink};
pub use stamp::utc_stamp;
