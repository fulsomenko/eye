use std::time::Duration;

use eye_capture::FrameSource;
use eye_core::Illumination;

use crate::{
    case::{
        Measurement, Needs, ParamError, TestCase, TestCtx, TestError, TestOutput, parse_params,
    },
    hw::{Tagging, tagged_ir},
    mode::EmitterSetting,
    regress::require_emitter,
    stats,
};

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TaggingParams {
    pub frames: usize,
    pub warmup: usize,
    pub lit_threshold: f64,
    pub min_agreement: f64,
    pub max_dark_p95: f64,
    pub min_lit_p5: f64,
    pub require_metadata: bool,
}

impl Default for TaggingParams {
    fn default() -> Self {
        Self {
            frames: 1200,
            warmup: 8,
            lit_threshold: 20.0,
            min_agreement: 0.999,
            max_dark_p95: 15.0,
            min_lit_p5: 25.0,
            require_metadata: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TaggingStats {
    pub frames: usize,
    pub agreement: f64,
    pub unexplained_breaks: usize,
    pub seq_gaps: usize,
    pub parity_breaks_at_gaps: usize,
    pub dark_p95: f64,
    pub lit_p5: f64,
}

/// `rows` = (seq, full-resolution mean, tag), in order, warmup already removed.
pub fn tagging_stats(rows: &[(u64, f64, Illumination)], lit_threshold: f64) -> TaggingStats {
    let truth_lit = |mean: f64| mean > lit_threshold;

    let agree = rows
        .iter()
        .filter(|&&(_, mean, tag)| match tag {
            Illumination::IrLit => truth_lit(mean),
            Illumination::IrDark => !truth_lit(mean),
            Illumination::Ambient | Illumination::Unknown => false,
        })
        .count();
    let agreement = if rows.is_empty() {
        f64::NAN
    } else {
        agree as f64 / rows.len() as f64
    };

    let mut unexplained_breaks = 0usize;
    let mut seq_gaps = 0usize;
    let mut parity_breaks_at_gaps = 0usize;
    for w in rows.windows(2) {
        let (seq0, mean0, _) = w[0];
        let (seq1, mean1, _) = w[1];
        let gap = seq1.saturating_sub(seq0);
        let lit0 = truth_lit(mean0);
        let lit1 = truth_lit(mean1);
        if gap == 1 {
            if lit0 == lit1 {
                unexplained_breaks += 1;
            }
        } else {
            seq_gaps += 1;
            let should_flip = gap % 2 == 1;
            let flipped = lit0 != lit1;
            if should_flip != flipped {
                parity_breaks_at_gaps += 1;
            }
        }
    }

    let dark_means: Vec<f64> = rows
        .iter()
        .filter(|&&(_, _, tag)| tag == Illumination::IrDark)
        .map(|&(_, mean, _)| mean)
        .collect();
    let lit_means: Vec<f64> = rows
        .iter()
        .filter(|&&(_, _, tag)| tag == Illumination::IrLit)
        .map(|&(_, mean, _)| mean)
        .collect();

    TaggingStats {
        frames: rows.len(),
        agreement,
        unexplained_breaks,
        seq_gaps,
        parity_breaks_at_gaps,
        dark_p95: stats::pct(&dark_means, 95.0),
        lit_p5: stats::pct(&lit_means, 5.0),
    }
}

#[derive(Debug)]
struct IrTaggingAgreement {
    params: TaggingParams,
}

impl TestCase for IrTaggingAgreement {
    fn name(&self) -> &'static str {
        "ir_tagging_agreement"
    }

    fn needs(&self) -> Needs {
        Needs {
            ir: true,
            ..Needs::default()
        }
    }

    fn default_timeout(&self) -> Duration {
        let p = &self.params;
        Duration::from_secs_f64(15.0 + (p.warmup + p.frames) as f64 / 15.0)
    }

    fn run(&self, ctx: &TestCtx) -> Result<TestOutput, TestError> {
        require_emitter(ctx, EmitterSetting::On, "ir_tagging_agreement", "on")?;
        let p = &self.params;
        ctx.instruct("keep a face or a hand 30-50 cm in front of the camera for the IR checks");

        let mut tagged = tagged_ir(ctx, Tagging::Auto)?;
        let mut rows = Vec::with_capacity(p.warmup + p.frames);
        for _ in 0..(p.warmup + p.frames) {
            ctx.check()?;
            let frame = tagged.next_frame()?;
            let h = frame.header();
            let mean = eye_capture::mean_brightness(frame.data(), h.width, h.height, 1);
            rows.push((h.seq, mean, h.illumination));
        }
        let metadata_active = tagged.metadata_active();
        rows.drain(0..p.warmup);

        let s = tagging_stats(&rows, p.lit_threshold);

        let mut out = TestOutput::default();
        out.push(if p.require_metadata {
            Measurement::at_least(
                "metadata_active",
                if metadata_active { 1.0 } else { 0.0 },
                "",
                1.0,
            )
        } else {
            Measurement::info(
                "metadata_active",
                if metadata_active { 1.0 } else { 0.0 },
                "",
            )
        });
        out.push(Measurement::at_least(
            "agreement",
            s.agreement,
            "",
            p.min_agreement,
        ));
        out.push(Measurement::at_most(
            "unexplained_breaks",
            s.unexplained_breaks as f64,
            "",
            0.0,
        ));
        out.push(Measurement::at_most(
            "parity_breaks_at_gaps",
            s.parity_breaks_at_gaps as f64,
            "",
            0.0,
        ));
        out.push(Measurement::at_most(
            "dark_p95",
            s.dark_p95,
            "",
            p.max_dark_p95,
        ));
        out.push(Measurement::at_least("lit_p5", s.lit_p5, "", p.min_lit_p5));
        out.push(Measurement::info("frames", s.frames as f64, ""));
        out.push(Measurement::info("seq_gaps", s.seq_gaps as f64, ""));
        Ok(out)
    }
}

pub fn build(params: &toml::Table) -> Result<Box<dyn TestCase>, ParamError> {
    let p: TaggingParams = parse_params(params)?;
    if p.frames == 0 {
        return Err(ParamError::Invalid("frames must be >= 1".into()));
    }
    for (name, v) in [
        ("lit_threshold", p.lit_threshold),
        ("min_agreement", p.min_agreement),
        ("max_dark_p95", p.max_dark_p95),
        ("min_lit_p5", p.min_lit_p5),
    ] {
        if !v.is_finite() {
            return Err(ParamError::Invalid(format!("{name} must be finite")));
        }
    }
    Ok(Box::new(IrTaggingAgreement { params: p }))
}

#[cfg(test)]
mod tests {
    use std::{collections::VecDeque, sync::Arc};

    use super::*;
    use crate::{
        mode::Role,
        testkit::{self, FakeMeta},
    };

    fn rows_perfect(n: usize) -> Vec<(u64, f64, Illumination)> {
        (0..n as u64)
            .map(|seq| {
                if seq % 2 == 1 {
                    (seq, 46.0, Illumination::IrLit)
                } else {
                    (seq, 0.0, Illumination::IrDark)
                }
            })
            .collect()
    }

    #[test]
    fn test_tagging_stats_perfect_alternation() {
        let rows = rows_perfect(100);
        let s = tagging_stats(&rows, 20.0);
        assert_eq!(s.agreement, 1.0);
        assert_eq!(s.unexplained_breaks, 0);
        assert_eq!(s.dark_p95, 0.0);
        assert_eq!(s.lit_p5, 46.0);
    }

    #[test]
    fn test_tagging_stats_counts_mis_tags_and_unknown() {
        let mut rows = rows_perfect(100);
        rows[10] = (10, 0.0, Illumination::IrLit);
        rows[20] = (20, 0.0, Illumination::Unknown);
        let s = tagging_stats(&rows, 20.0);
        approx::assert_abs_diff_eq!(s.agreement, 0.98, epsilon = 1e-12);
    }

    #[test]
    fn test_unexplained_break_is_counted() {
        let rows = vec![
            (0u64, 0.0, Illumination::IrDark),
            (1u64, 46.0, Illumination::IrLit),
            (2u64, 46.0, Illumination::IrLit),
            (3u64, 0.0, Illumination::IrDark),
        ];
        let s = tagging_stats(&rows, 20.0);
        assert_eq!(s.unexplained_breaks, 1);
    }

    #[test]
    fn test_gap_that_keeps_parity_is_not_a_break() {
        let rows = vec![
            (10u64, 46.0, Illumination::IrLit),
            (12u64, 46.0, Illumination::IrLit),
        ];
        let s = tagging_stats(&rows, 20.0);
        assert_eq!(s.seq_gaps, 1);
        assert_eq!(s.parity_breaks_at_gaps, 0);
    }

    #[test]
    fn test_gap_that_breaks_parity_is_counted() {
        let rows = vec![
            (10u64, 46.0, Illumination::IrLit),
            (12u64, 0.0, Illumination::IrDark),
        ];
        let s = tagging_stats(&rows, 20.0);
        assert_eq!(s.parity_breaks_at_gaps, 1);
    }

    fn ir_frames(n: usize) -> Vec<eye_core::Frame> {
        testkit::synth_gray(
            "ir",
            n,
            0,
            0,
            66_666_666,
            |seq| {
                if seq % 2 == 1 { 46 } else { 0 }
            },
        )
    }

    #[test]
    fn test_tagging_case_end_to_end_with_metadata() {
        let frames = ir_frames(1208);
        let records: VecDeque<_> = (0..1208u64)
            .map(|seq| eye_capture::MetaRecord {
                timestamp: eye_core::Timestamp::from_nanos(seq * 66_666_666),
                lit: Some(seq % 2 == 1),
            })
            .collect();
        let session = Arc::new(
            testkit::FakeSession::empty()
                .with_source(Role::Ir, testkit::boxed(frames))
                .with_meta(Box::new(FakeMeta { records })),
        );
        let ctx = testkit::ctx(session, testkit::mode_ir(EmitterSetting::On));
        let case = build(&toml::Table::new()).unwrap();
        let out = case.run(&ctx).unwrap();
        assert!(out.passed());
        assert_eq!(
            out.measurements
                .iter()
                .find(|m| m.name == "metadata_active")
                .unwrap()
                .value,
            1.0
        );
        assert_eq!(
            out.measurements
                .iter()
                .find(|m| m.name == "agreement")
                .unwrap()
                .value,
            1.0
        );
    }

    #[test]
    fn test_tagging_case_requires_metadata_by_default() {
        let frames = ir_frames(1208);
        let session =
            Arc::new(testkit::FakeSession::empty().with_source(Role::Ir, testkit::boxed(frames)));
        let ctx = testkit::ctx(session, testkit::mode_ir(EmitterSetting::On));
        let case = build(&toml::Table::new()).unwrap();
        let out = case.run(&ctx).unwrap();
        assert_eq!(
            out.measurements
                .iter()
                .find(|m| m.name == "metadata_active")
                .unwrap()
                .value,
            0.0
        );
        assert!(!out.passed());

        let frames = ir_frames(1208);
        let session =
            Arc::new(testkit::FakeSession::empty().with_source(Role::Ir, testkit::boxed(frames)));
        let ctx = testkit::ctx(session, testkit::mode_ir(EmitterSetting::On));
        let mut table = toml::Table::new();
        table.insert("require_metadata".into(), toml::Value::Boolean(false));
        let case = build(&table).unwrap();
        let out = case.run(&ctx).unwrap();
        assert!(out.passed());
        assert_eq!(
            out.measurements
                .iter()
                .find(|m| m.name == "agreement")
                .unwrap()
                .value,
            1.0
        );
    }

    #[test]
    fn test_tagging_requires_emitter_on_mode() {
        let session = Arc::new(testkit::FakeSession::empty());
        let ctx = testkit::ctx(session, testkit::mode_ir(EmitterSetting::Off));
        let case = build(&toml::Table::new()).unwrap();
        let err = case.run(&ctx).unwrap_err();
        assert!(matches!(err, TestError::Other(_)));
    }
}
