//! Small numeric helpers shared by the hardware test cases.

use eye_core::Timestamp;

pub fn ms(t: Timestamp) -> f64 {
    t.0.as_secs_f64() * 1e3
}

pub fn sorted(values: &[f64]) -> Vec<f64> {
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    v
}

/// Nearest-rank percentile shared with eye-bench; NaN when empty.
pub fn pct(values: &[f64], p: f64) -> f64 {
    eye_bench::metrics::percentile(&sorted(values), p).unwrap_or(f64::NAN)
}

pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

pub fn std_dev(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let m = mean(values);
    (values.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / values.len() as f64).sqrt()
}

pub fn intervals_ms(ts: &[Timestamp]) -> Vec<f64> {
    ts.windows(2).map(|w| ms(w[1]) - ms(w[0])).collect()
}

/// `(n - 1) / (last - first)` in 1/s; NaN for fewer than 2 frames or a non-positive span.
pub fn fps(ts: &[Timestamp]) -> f64 {
    match (ts.first(), ts.last()) {
        (Some(&first), Some(&last)) if ts.len() >= 2 => {
            let span_s = (ms(last) - ms(first)) / 1e3;
            if span_s > 0.0 {
                (ts.len() - 1) as f64 / span_s
            } else {
                f64::NAN
            }
        }
        _ => f64::NAN,
    }
}

pub fn seq_gaps(seqs: &[u64]) -> usize {
    seqs.windows(2).filter(|w| w[1] != w[0] + 1).count()
}

/// For each `a[k]`, the index of the nearest `b` (ties go to the later one). Both ascending; O(n + m). Empty when `b` is empty.
pub fn nearest_indices(a: &[Timestamp], b: &[Timestamp]) -> Vec<usize> {
    if b.is_empty() {
        return Vec::new();
    }
    let mut j = 0;
    a.iter()
        .map(|&ta| {
            while j + 1 < b.len() && (ms(b[j + 1]) - ms(ta)).abs() <= (ms(b[j]) - ms(ta)).abs() {
                j += 1;
            }
            j
        })
        .collect()
}

/// Least-squares slope of `y` over `x_s` (seconds), per minute; NaN when undefined.
pub fn slope_per_min(x_s: &[f64], y: &[f64]) -> f64 {
    let n = x_s.len().min(y.len());
    if n < 2 {
        return f64::NAN;
    }
    let (mx, my) = (mean(&x_s[..n]), mean(&y[..n]));
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for i in 0..n {
        let dx = x_s[i] - mx;
        sxy += dx * (y[i] - my);
        sxx += dx * dx;
    }
    if sxx <= 0.0 {
        f64::NAN
    } else {
        sxy / sxx * 60.0
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    const P: u64 = 33_333_333;

    #[test]
    fn test_fps_of_30hz_timestamps_is_30() {
        let ts: Vec<Timestamp> = (0..91).map(|k| Timestamp::from_nanos(k * P)).collect();
        assert_relative_eq!(fps(&ts), 30.0, epsilon = 1e-6);
        assert!(fps(&[Timestamp::from_nanos(0)]).is_nan());
    }

    #[test]
    fn test_seq_gaps_and_intervals() {
        assert_eq!(seq_gaps(&[1, 2, 4, 5]), 1);
        let ts = [
            Timestamp::from_nanos(0),
            Timestamp::from_nanos(33_000_000),
            Timestamp::from_nanos(66_000_000),
        ];
        let intervals = intervals_ms(&ts);
        assert_abs_diff(&intervals, &[33.0, 33.0]);
    }

    fn assert_abs_diff(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() <= 1e-9, "{x} vs {y}");
        }
    }

    #[test]
    fn test_nearest_indices_match_brute_force() {
        let a: Vec<Timestamp> = (0..50)
            .map(|k| Timestamp::from_nanos(k * P + (k % 3) * 2_000_000))
            .collect();
        let b: Vec<Timestamp> = (0..50)
            .map(|k| Timestamp::from_nanos(k * P + 5_000_000))
            .collect();
        let got = nearest_indices(&a, &b);
        for (k, &ta) in a.iter().enumerate() {
            let mut best = 0usize;
            let mut best_d = f64::INFINITY;
            for (j, &tb) in b.iter().enumerate() {
                let d = (ms(tb) - ms(ta)).abs();
                if d <= best_d {
                    best_d = d;
                    best = j;
                }
            }
            assert_eq!(got[k], best, "k={k}");
        }
    }

    #[test]
    fn test_slope_per_min_recovers_known_drift() {
        let x: Vec<f64> = (0..1800).map(|k| (k as f64 * P as f64) / 1e9).collect();
        let y: Vec<f64> = (0..1800).map(|k| -(k as f64) * 0.001).collect();
        assert_relative_eq!(slope_per_min(&x, &y), -1.8, epsilon = 1e-6);
    }

    #[test]
    fn test_pct_matches_bench_nearest_rank() {
        let values: Vec<f64> = (1..=20).map(f64::from).collect();
        assert_eq!(pct(&values, 50.0), 10.0);
        assert_eq!(pct(&values, 95.0), 19.0);
        assert!(pct(&[], 50.0).is_nan());
    }
}
