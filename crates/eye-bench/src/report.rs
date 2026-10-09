use std::path::Path;

use serde::Serialize;

use crate::error::BenchError;
use crate::metrics::{MetricParams, RegionHit, SessionMetrics, Summary};
use crate::row::{BenchRow, RowKind, RowOutcome};

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BenchReport {
    pub schema_version: u32,
    pub eye_version: String,
    pub params: MetricParams,
    pub rows: Vec<BenchRow>,
}

impl BenchReport {
    /// Sorts rows by (pipeline, calibration, kind, session); session rows before the aggregate.
    pub fn new(params: MetricParams, mut rows: Vec<BenchRow>) -> Self {
        rows.sort_by(|a, b| {
            (&a.pipeline, a.calibration, a.kind, &a.session).cmp(&(
                &b.pipeline,
                b.calibration,
                b.kind,
                &b.session,
            ))
        });
        Self {
            schema_version: SCHEMA_VERSION,
            eye_version: env!("CARGO_PKG_VERSION").into(),
            params,
            rows,
        }
    }

    #[allow(clippy::result_large_err)]
    pub fn to_json(&self) -> Result<String, BenchError> {
        let mut s = serde_json::to_string_pretty(self)?;
        s.push('\n');
        Ok(s)
    }

    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str("# eye bench report\n\n");
        out.push_str(&format!(
            "eye {}. Grids {}; boundary margin {} px; dropout bin {} ms.\n",
            self.eye_version,
            grids_label(&self.params.grids),
            self.params.boundary_margin_px,
            self.params.dropout_bin.as_millis(),
        ));
        out.push_str(
            "`proc ms` is pipeline processing time per FrameSet, not end-to-end latency (see `eye run --stats`).\n\n",
        );
        out.push_str(&header_line(&self.params.grids));
        out.push('\n');
        out.push_str(&align_line(&self.params.grids));
        out.push('\n');
        for row in &self.rows {
            out.push_str(&self.row_line(row));
            out.push('\n');
        }

        let targets = targets_section(&self.rows);
        if let Some(targets) = targets {
            out.push('\n');
            out.push_str(&targets);
        }

        let errors = errors_section(&self.rows);
        if let Some(errors) = errors {
            out.push('\n');
            out.push_str(&errors);
        }

        let warnings = warnings_section(&self.rows);
        if let Some(warnings) = warnings {
            out.push('\n');
            out.push_str(&warnings);
        }

        out
    }

    #[allow(clippy::result_large_err)]
    pub fn write_to(&self, dir: &Path) -> Result<(), BenchError> {
        std::fs::create_dir_all(dir).map_err(|source| BenchError::Io {
            path: dir.to_path_buf(),
            source,
        })?;

        let json_path = dir.join("report.json");
        std::fs::write(&json_path, self.to_json()?).map_err(|source| BenchError::Io {
            path: json_path.clone(),
            source,
        })?;

        let md_path = dir.join("report.md");
        std::fs::write(&md_path, self.to_markdown()).map_err(|source| BenchError::Io {
            path: md_path.clone(),
            source,
        })?;

        tracing::info!(
            json = %json_path.display(),
            markdown = %md_path.display(),
            rows = self.rows.len() as u64,
            "report written"
        );
        Ok(())
    }

    fn row_line(&self, row: &BenchRow) -> String {
        let session = match row.kind {
            RowKind::Session => cell(&row.session),
            RowKind::Aggregate => "**all**".to_owned(),
        };
        let mut cols = vec![
            cell(&row.pipeline),
            row.calibration.as_str().to_owned(),
            session,
        ];
        match &row.outcome {
            RowOutcome::Ok { metrics: m } => {
                let s = |x: &Option<Summary>, f: fn(&Summary) -> f64| x.as_ref().map(f);
                cols.push("ok".into());
                cols.push(m.samples.to_string());
                cols.push(opt(s(&m.angular_error_deg, |s| s.mean), deg));
                cols.push(opt(s(&m.angular_error_deg, |s| s.p95), deg));
                cols.push(opt(m.accuracy_deg, deg));
                cols.push(opt(m.precision_rms_s2s_deg, deg));
                cols.push(opt(s(&m.px_error_logical, |s| s.mean), one));
                for g in &self.params.grids {
                    cols.push(opt(region(m, *g).and_then(|r| r.hit_rate), pct));
                }
                cols.push(opt(s(&m.processing_ms, |s| s.p50), one));
                cols.push(opt(s(&m.processing_ms, |s| s.p95), one));
                cols.push(opt(m.dropout_rate, pct));
            }
            RowOutcome::Error { .. } => {
                cols.push("error".into());
                cols.extend(std::iter::repeat_n(
                    "n/a".to_owned(),
                    9 + self.params.grids.len(),
                ));
            }
        }
        format!("| {} |", cols.join(" | "))
    }
}

fn grids_label(grids: &[[u32; 2]]) -> String {
    grids
        .iter()
        .map(|[c, r]| format!("{c}x{r}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn header_line(grids: &[[u32; 2]]) -> String {
    let mut cols = vec![
        "pipeline".to_owned(),
        "calib".to_owned(),
        "session".to_owned(),
        "status".to_owned(),
        "samples".to_owned(),
        "err mean deg".to_owned(),
        "err p95 deg".to_owned(),
        "acc deg".to_owned(),
        "prec deg".to_owned(),
        "err mean px".to_owned(),
    ];
    for [c, r] in grids {
        cols.push(format!("{c}x{r} hit"));
    }
    cols.push("proc p50 ms".to_owned());
    cols.push("proc p95 ms".to_owned());
    cols.push("dropout".to_owned());
    format!("| {} |", cols.join(" | "))
}

fn align_line(grids: &[[u32; 2]]) -> String {
    let mut cols = vec!["---"; 4];
    cols.extend(std::iter::repeat_n("---:", 9 + grids.len()));
    format!("|{}|", cols.join("|"))
}

fn targets_section(rows: &[BenchRow]) -> Option<String> {
    let aggregate_ok: Vec<&BenchRow> = rows
        .iter()
        .filter(|r| r.kind == RowKind::Aggregate && matches!(r.outcome, RowOutcome::Ok { .. }))
        .collect();
    if aggregate_ok.is_empty() {
        return None;
    }

    let mut out = String::new();
    out.push_str("## Accuracy targets (aggregate rows)\n\n");
    out.push_str("| pipeline | calib | 3x3 hit >= 90 % | 4x4 hit >= 90 % | err mean deg | err p95 deg | mean < 2 deg (stretch) |\n");
    out.push_str("|---|---|---|---|---:|---:|---|\n");
    for row in aggregate_ok {
        let RowOutcome::Ok { metrics: m } = &row.outcome else {
            unreachable!()
        };
        let hit = |grid: [u32; 2]| -> String {
            match region(m, grid).and_then(|r| r.hit_rate) {
                Some(rate) => if rate >= 0.9 { "yes" } else { "no" }.to_owned(),
                None => "n/a".to_owned(),
            }
        };
        let mean = m.angular_error_deg.as_ref().map(|s| s.mean);
        let p95 = m.angular_error_deg.as_ref().map(|s| s.p95);
        let stretch = match mean {
            Some(v) if v < 2.0 => "yes",
            Some(_) => "no",
            None => "n/a",
        };
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} |\n",
            cell(&row.pipeline),
            row.calibration.as_str(),
            hit([3, 3]),
            hit([4, 4]),
            opt(mean, deg),
            opt(p95, deg),
            stretch,
        ));
    }
    Some(out)
}

fn errors_section(rows: &[BenchRow]) -> Option<String> {
    let entries: Vec<String> = rows
        .iter()
        .filter_map(|row| match &row.outcome {
            RowOutcome::Error { message } => Some(format!(
                "- {} / {} / {}: {}\n",
                cell(&row.pipeline),
                row.calibration.as_str(),
                cell(&row.session),
                cell(message),
            )),
            RowOutcome::Ok { .. } => None,
        })
        .collect();
    if entries.is_empty() {
        return None;
    }
    let mut out = String::new();
    out.push_str("## Errors\n\n");
    for entry in entries {
        out.push_str(&entry);
    }
    Some(out)
}

fn warnings_section(rows: &[BenchRow]) -> Option<String> {
    let entries: Vec<String> = rows
        .iter()
        .flat_map(|row| {
            row.warnings.iter().map(move |w| {
                format!(
                    "- {} / {} / {}: {}\n",
                    cell(&row.pipeline),
                    row.calibration.as_str(),
                    cell(&row.session),
                    cell(w),
                )
            })
        })
        .collect();
    if entries.is_empty() {
        return None;
    }
    let mut out = String::new();
    out.push_str("## Warnings\n\n");
    for entry in entries {
        out.push_str(&entry);
    }
    Some(out)
}

fn cell(text: &str) -> String {
    text.replace('|', "\\|")
}

fn opt(v: Option<f64>, f: impl Fn(f64) -> String) -> String {
    v.map(f).unwrap_or_else(|| "n/a".to_owned())
}

fn deg(v: f64) -> String {
    format!("{v:.2}")
}

fn one(v: f64) -> String {
    format!("{v:.1}")
}

fn pct(v: f64) -> String {
    format!("{:.1} %", v * 100.0)
}

fn region(m: &SessionMetrics, grid: [u32; 2]) -> Option<&RegionHit> {
    m.regions.iter().find(|r| [r.cols, r.rows] == grid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::row::CalibrationMode;

    fn metrics() -> SessionMetrics {
        SessionMetrics {
            windows: 16,
            samples: 812,
            angular_error_deg: Some(Summary {
                mean: 3.41,
                p50: 3.10,
                p95: 7.90,
            }),
            px_error_logical: Some(Summary {
                mean: 184.2,
                p50: 170.0,
                p95: 350.0,
            }),
            accuracy_deg: Some(2.95),
            precision_rms_s2s_deg: Some(0.42),
            regions: vec![
                RegionHit {
                    cols: 3,
                    rows: 3,
                    windows: 9,
                    excluded: 0,
                    hits: 8,
                    hit_rate: Some(8.0 / 9.0),
                    sample_hit_rate: Some(0.85),
                },
                RegionHit {
                    cols: 4,
                    rows: 4,
                    windows: 15,
                    excluded: 1,
                    hits: 10,
                    hit_rate: Some(10.0 / 15.0),
                    sample_hit_rate: Some(0.6),
                },
            ],
            processing_ms: Some(Summary {
                mean: 4.3,
                p50: 4.0,
                p95: 6.1,
            }),
            dropout_rate: Some(0.042),
            output_rate_hz: Some(30.0),
        }
    }

    fn row(pipeline: &str, kind: RowKind, session: &str, outcome: RowOutcome) -> BenchRow {
        BenchRow {
            pipeline: pipeline.to_owned(),
            calibration: CalibrationMode::None,
            kind,
            session: session.to_owned(),
            step_errors: 0,
            warnings: vec![],
            outcome,
        }
    }

    #[test]
    fn test_rows_sorted_by_pipeline_calibration_kind_session() {
        let rows = vec![
            row(
                "b",
                RowKind::Aggregate,
                "all",
                RowOutcome::Ok { metrics: metrics() },
            ),
            row(
                "a",
                RowKind::Session,
                "s2",
                RowOutcome::Ok { metrics: metrics() },
            ),
            row(
                "b",
                RowKind::Session,
                "s1",
                RowOutcome::Ok { metrics: metrics() },
            ),
            row(
                "a",
                RowKind::Session,
                "s1",
                RowOutcome::Ok { metrics: metrics() },
            ),
        ];
        let report = BenchReport::new(MetricParams::default(), rows);
        let actual: Vec<(&str, &str)> = report
            .rows
            .iter()
            .map(|r| (r.pipeline.as_str(), r.session.as_str()))
            .collect();
        assert_eq!(
            actual,
            vec![("a", "s1"), ("a", "s2"), ("b", "s1"), ("b", "all")]
        );
    }

    #[test]
    fn test_json_is_deterministic_under_input_order() {
        let rows_a = vec![
            row(
                "a",
                RowKind::Session,
                "s1",
                RowOutcome::Ok { metrics: metrics() },
            ),
            row(
                "b",
                RowKind::Session,
                "s1",
                RowOutcome::Ok { metrics: metrics() },
            ),
        ];
        let rows_b = vec![
            row(
                "b",
                RowKind::Session,
                "s1",
                RowOutcome::Ok { metrics: metrics() },
            ),
            row(
                "a",
                RowKind::Session,
                "s1",
                RowOutcome::Ok { metrics: metrics() },
            ),
        ];
        let report_a = BenchReport::new(MetricParams::default(), rows_a);
        let report_b = BenchReport::new(MetricParams::default(), rows_b);
        assert_eq!(report_a.to_json().unwrap(), report_b.to_json().unwrap());
    }

    #[test]
    fn test_json_shape_has_status_tag_and_metrics() {
        let rows = vec![row(
            "ir-classic",
            RowKind::Session,
            "20261008T090000Z",
            RowOutcome::Ok { metrics: metrics() },
        )];
        let report = BenchReport::new(MetricParams::default(), rows);
        let json = report.to_json().unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["schema_version"], 1);
        assert_eq!(value["rows"][0]["status"], "ok");
        assert_eq!(
            value["rows"][0]["metrics"]["angular_error_deg"]["mean"],
            3.41
        );
        assert!(value["rows"][0].get("outcome").is_none());
        assert_eq!(
            value["params"]["grids"],
            serde_json::json!([[3, 3], [4, 4]])
        );
        assert_eq!(value["params"]["dropout_bin"], 200);
    }

    #[test]
    fn test_error_row_serializes_message() {
        let rows = vec![row(
            "p",
            RowKind::Session,
            "s",
            RowOutcome::Error {
                message: "boom".to_owned(),
            },
        )];
        let report = BenchReport::new(MetricParams::default(), rows);
        let json = report.to_json().unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["rows"][0]["status"], "error");
        assert_eq!(value["rows"][0]["message"], "boom");
    }

    #[test]
    fn test_markdown_golden_single_ok_row() {
        let rows = vec![row(
            "ir-classic",
            RowKind::Session,
            "20261008T090000Z",
            RowOutcome::Ok { metrics: metrics() },
        )];
        let report = BenchReport::new(MetricParams::default(), rows);
        let v = env!("CARGO_PKG_VERSION");
        let expected = format!(
            "# eye bench report\n\neye {v}. Grids 3x3, 4x4; boundary margin 20 px; dropout bin 200 ms.\n`proc ms` is pipeline processing time per FrameSet, not end-to-end latency (see `eye run --stats`).\n\n| pipeline | calib | session | status | samples | err mean deg | err p95 deg | acc deg | prec deg | err mean px | 3x3 hit | 4x4 hit | proc p50 ms | proc p95 ms | dropout |\n|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n| ir-classic | none | 20261008T090000Z | ok | 812 | 3.41 | 7.90 | 2.95 | 0.42 | 184.2 | 88.9 % | 66.7 % | 4.0 | 6.1 | 4.2 % |\n"
        );
        assert_eq!(report.to_markdown(), expected);
    }

    #[test]
    fn test_markdown_none_renders_na() {
        let mut m = metrics();
        m.precision_rms_s2s_deg = None;
        let rows = vec![row(
            "p",
            RowKind::Session,
            "s",
            RowOutcome::Ok { metrics: m },
        )];
        let report = BenchReport::new(MetricParams::default(), rows);
        let md = report.to_markdown();
        let line = md.lines().find(|l| l.starts_with("| p |")).unwrap();
        let cols: Vec<&str> = line.split('|').map(str::trim).collect();
        // header: pipeline, calib, session, status, samples, err mean deg, err p95 deg, acc deg, prec deg, ...
        assert_eq!(cols[9], "n/a");
    }

    #[test]
    fn test_markdown_error_row_listed_under_errors() {
        let rows = vec![row(
            "p",
            RowKind::Session,
            "s",
            RowOutcome::Error {
                message: "boom".to_owned(),
            },
        )];
        let report = BenchReport::new(MetricParams::default(), rows);
        let md = report.to_markdown();
        let line = md.lines().find(|l| l.starts_with("| p |")).unwrap();
        let na_count = line.matches("n/a").count();
        assert_eq!(na_count, 11);
        assert!(line.contains("| error |"));
        assert!(md.contains("## Errors\n\n- p / none / s: boom\n"));
    }

    #[test]
    fn test_markdown_no_errors_heading_when_all_ok() {
        let rows = vec![row(
            "p",
            RowKind::Session,
            "s",
            RowOutcome::Ok { metrics: metrics() },
        )];
        let report = BenchReport::new(MetricParams::default(), rows);
        assert!(!report.to_markdown().contains("## Errors"));
    }

    #[test]
    fn test_markdown_lists_warnings() {
        let mut r = row(
            "p",
            RowKind::Session,
            "s",
            RowOutcome::Ok { metrics: metrics() },
        );
        r.warnings = vec!["w1".to_owned()];
        let report = BenchReport::new(MetricParams::default(), vec![r]);
        let md = report.to_markdown();
        assert!(md.contains("## Warnings\n\n- p / none / s: w1\n"));
    }

    #[test]
    fn test_markdown_escapes_pipe() {
        let rows = vec![row(
            "a|b",
            RowKind::Session,
            "s",
            RowOutcome::Ok { metrics: metrics() },
        )];
        let report = BenchReport::new(MetricParams::default(), rows);
        assert!(report.to_markdown().contains("a\\|b"));
    }

    #[test]
    fn test_targets_table_marks_pass_fail() {
        let mut m = metrics();
        m.regions = vec![
            RegionHit {
                cols: 3,
                rows: 3,
                windows: 20,
                excluded: 0,
                hits: 19,
                hit_rate: Some(0.95),
                sample_hit_rate: Some(0.95),
            },
            RegionHit {
                cols: 4,
                rows: 4,
                windows: 20,
                excluded: 0,
                hits: 16,
                hit_rate: Some(0.80),
                sample_hit_rate: Some(0.80),
            },
        ];
        m.angular_error_deg = Some(Summary {
            mean: 1.5,
            p50: 1.2,
            p95: 3.0,
        });
        let rows = vec![row(
            "p",
            RowKind::Aggregate,
            "all",
            RowOutcome::Ok { metrics: m },
        )];
        let report = BenchReport::new(MetricParams::default(), rows);
        let md = report.to_markdown();
        assert!(md.contains("| p | none | yes | no | 1.50 | 3.00 | yes |"));
    }

    #[test]
    fn test_targets_table_na_when_grid_missing() {
        let mut m = metrics();
        m.regions[0].hit_rate = None;
        m.regions[0].sample_hit_rate = None;
        m.angular_error_deg = Some(Summary {
            mean: 1.5,
            p50: 1.2,
            p95: 3.0,
        });
        let rows = vec![row(
            "p",
            RowKind::Aggregate,
            "all",
            RowOutcome::Ok { metrics: m },
        )];
        let report = BenchReport::new(MetricParams::default(), rows);
        let md = report.to_markdown();
        assert!(md.contains("| p | none | n/a | no | 1.50 | 3.00 | yes |"));
    }

    #[test]
    fn test_write_to_creates_both_files() {
        let dir = tempfile::tempdir().unwrap();
        let rows = vec![row(
            "p",
            RowKind::Session,
            "s",
            RowOutcome::Ok { metrics: metrics() },
        )];
        let report = BenchReport::new(MetricParams::default(), rows);
        let out_dir = dir.path().join("r");
        report.write_to(&out_dir).unwrap();

        let json_path = out_dir.join("report.json");
        let md_path = out_dir.join("report.md");
        assert!(json_path.exists());
        assert!(md_path.exists());

        let json_content = std::fs::read_to_string(&json_path).unwrap();
        let _: serde_json::Value = serde_json::from_str(&json_content).unwrap();

        let md_content = std::fs::read_to_string(&md_path).unwrap();
        assert!(md_content.starts_with("# eye bench report"));
    }
}
