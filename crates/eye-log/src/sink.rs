use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

use crate::record::Record;

pub trait Sink: Send {
    fn write(&mut self, record: &Record) -> io::Result<()>;
    fn flush(&mut self) -> io::Result<()>;
}

#[derive(Debug)]
pub struct JsonLinesSink {
    writer: BufWriter<File>,
}

impl JsonLinesSink {
    pub fn create(path: &Path) -> io::Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .mode(0o600)
            .open(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
        })
    }
}

impl Sink for JsonLinesSink {
    fn write(&mut self, record: &Record) -> io::Result<()> {
        serde_json::to_writer(&mut self.writer, record).map_err(io::Error::other)?;
        self.writer.write_all(b"\n")
    }

    fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }
}

impl Drop for JsonLinesSink {
    fn drop(&mut self) {
        let _ = self.writer.flush();
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::io::{BufRead, BufReader};

    use tempfile::tempdir;

    use super::*;
    use crate::record::{Level, Value};

    fn sample(seq: u64) -> Record {
        let mut fields = BTreeMap::new();
        fields.insert("seq".to_string(), Value::U64(seq));
        Record {
            schema: crate::LOG_SCHEMA,
            ts_unix_ns: 1000 + seq,
            level: Level::Info,
            target: "eye_log::tests".to_string(),
            message: format!("event {seq}"),
            context: BTreeMap::new(),
            fields,
        }
    }

    #[test]
    fn test_record_round_trips_through_json_lines() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nested").join("log.jsonl");
        let records: Vec<Record> = (0..3).map(sample).collect();

        {
            let mut sink = JsonLinesSink::create(&path).unwrap();
            for r in &records {
                sink.write(r).unwrap();
            }
            sink.flush().unwrap();
        }

        let file = File::open(&path).unwrap();
        let lines: Vec<String> = BufReader::new(file).lines().map(|l| l.unwrap()).collect();
        assert_eq!(lines.len(), 3);
        for (line, expected) in lines.iter().zip(records.iter()) {
            let parsed: Record = serde_json::from_str(line).unwrap();
            assert_eq!(&parsed, expected);
            let raw: serde_json::Value = serde_json::from_str(line).unwrap();
            assert_eq!(raw["schema"], 1);
        }
    }
}
