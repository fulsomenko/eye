use crate::GazePoint;

#[derive(Debug, thiserror::Error)]
pub enum SinkError {
    #[error("sink closed")]
    Closed,
    #[error("sink I/O")]
    Io(#[from] std::io::Error),
    #[error("{sink} sink failed")]
    Backend {
        sink: &'static str,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

/// Consumer of gaze points at the end of the pipeline (e.g. the overlay).
pub trait GazeSink: Send {
    fn name(&self) -> &'static str;
    fn push(&mut self, point: &GazePoint) -> Result<(), SinkError>;
}

#[cfg(test)]
mod tests {
    use std::error::Error;
    use std::io;

    use nalgebra::{Matrix2, Point2};

    use super::*;
    use crate::{OutputId, Timestamp};

    struct VecSink(Vec<GazePoint>);

    impl GazeSink for VecSink {
        fn name(&self) -> &'static str {
            "vec"
        }

        fn push(&mut self, point: &GazePoint) -> Result<(), SinkError> {
            self.0.push(point.clone());
            Ok(())
        }
    }

    struct ClosedSink;

    impl GazeSink for ClosedSink {
        fn name(&self) -> &'static str {
            "closed"
        }

        fn push(&mut self, _point: &GazePoint) -> Result<(), SinkError> {
            Err(SinkError::Closed)
        }
    }

    fn point_at(nanos: u64) -> GazePoint {
        GazePoint {
            timestamp: Timestamp::from_nanos(nanos),
            output: OutputId::from("eDP-1"),
            mm: Point2::new(0.0, 0.0),
            px_physical: Point2::new(0.0, 0.0),
            px_logical: Point2::new(0.0, 0.0),
            cov_mm: Matrix2::zeros(),
            confidence: 1.0,
        }
    }

    const fn assert_send<T: Send>() {}
    const _: () = assert_send::<Box<dyn GazeSink>>();

    #[test]
    fn test_sinks_work_as_boxed_trait_objects() {
        let mut sinks: Vec<Box<dyn GazeSink>> =
            vec![Box::new(VecSink(Vec::new())), Box::new(ClosedSink)];
        let results: Vec<bool> = sinks
            .iter_mut()
            .map(|sink| sink.push(&point_at(1)).is_ok())
            .collect();
        assert_eq!(results, [true, false]);
        assert_eq!(sinks[1].name(), "closed");
    }

    #[test]
    fn test_vec_sink_keeps_push_order() {
        let mut sink = VecSink(Vec::new());
        sink.push(&point_at(1)).unwrap();
        sink.push(&point_at(2)).unwrap();
        sink.push(&point_at(3)).unwrap();
        let timestamps: Vec<u64> = sink.0.iter().map(|p| p.timestamp.as_nanos()).collect();
        assert_eq!(timestamps, [1, 2, 3]);
    }

    #[test]
    fn test_backend_error_keeps_source() {
        let err = SinkError::Backend {
            sink: "layer-shell",
            source: "compositor gone".into(),
        };
        assert_eq!(err.to_string(), "layer-shell sink failed");
        assert_eq!(err.source().unwrap().to_string(), "compositor gone");
    }

    #[test]
    fn test_io_error_converts_into_sink_error() {
        fn produces() -> Result<(), SinkError> {
            Err(io::Error::from(io::ErrorKind::BrokenPipe))?;
            Ok(())
        }
        assert!(matches!(produces(), Err(SinkError::Io(_))));
    }
}
