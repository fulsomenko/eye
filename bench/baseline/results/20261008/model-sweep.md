# Model/prior sweep (EYE-119, 2026-10-08)

RGB pipeline (`bench/baseline/rgb.toml`), sessions `20261008T201638Z` (A) and
`20261008T201709Z` (B), release binary (`cargo build --release -p eye-app
--features mediapipe-ort`), `eye-bench` library grid over `[evaluation.fit]`.
LOTO pools both sessions (matches `bench/baseline/rgb.bench.toml`);
cross-session trains on one session and evaluates on the other with a fixed
profile (no cross `CalibrationMode` exists, so this ran as two single-session
fit/evaluate passes instead of a bench.toml matrix). Timing axis dropped per
the settle/window refit already on this card.

`HeadFrame` is included as a full row at every `slope_prior_sigma`: it is
reachable today only via `model_override` (the ladder in
`DotSessionFit::fit_with` never selects it on its own), so every `HeadFrame`
cell below used `FitConfig { model_override: Some(CorrectionModel::HeadFrame),
.. }`.

| slope_prior_sigma | model | LOTO err mean | LOTO err p95 | LOTO acc | LOTO prec | LOTO 3x3 hit | LOTO 4x4 hit | cross A train, B eval mean | cross A train, B eval prec | cross B train, A eval mean | cross B train, A eval prec |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.15 (default) | affine | 3.95 | 8.31 | 3.87 | 0.54 | 68.8 % | 50.0 % | 3.88 | 0.64 | 3.63 | 0.38 |
| 0.15 (default) | quadratic | 2.95 | 5.72 | 2.85 | 0.49 | 62.5 % | 50.0 % | 3.04 | 0.64 | 2.76 | 0.34 |
| 0.15 (default) | headframe | 3.12 | 7.92 | 3.04 | 0.52 | 75.0 % | 84.4 % | 2.89 | 0.64 | 2.47 | 0.39 |
| 0.5 | affine | 3.95 | 8.29 | 3.86 | 0.54 | 71.9 % | 50.0 % | 3.96 | 0.64 | 3.66 | 0.38 |
| 0.5 | quadratic | 2.93 | 5.78 | 2.80 | 0.52 | 68.8 % | 53.1 % | 3.06 | 0.64 | 2.80 | 0.35 |
| 0.5 | headframe | 3.12 | 8.80 | 3.04 | 0.52 | 75.0 % | 84.4 % | 2.87 | 0.64 | 2.48 | 0.39 |
| 2.0 | affine | 3.99 | 8.30 | 3.90 | 0.54 | 71.9 % | 53.1 % | 3.96 | 0.64 | 3.66 | 0.38 |
| 2.0 | quadratic | 2.99 | 5.78 | 2.87 | 0.54 | 68.8 % | 56.2 % | 3.05 | 0.64 | 2.82 | 0.36 |
| 2.0 | headframe | 3.12 | 8.81 | 3.04 | 0.53 | 75.0 % | 84.4 % | 2.87 | 0.64 | 2.48 | 0.39 |

Cross-session mean (average of both directions), for the decision rule:

| slope_prior_sigma | model | cross-session mean deg |
|---|---|---:|
| 0.15 | affine | 3.755 |
| 0.15 | quadratic | 2.90 |
| 0.15 | headframe | **2.68** |
| 0.5 | affine | 3.81 |
| 0.5 | quadratic | 2.93 |
| 0.5 | headframe | 2.675 |
| 2.0 | affine | 3.81 |
| 2.0 | quadratic | 2.935 |
| 2.0 | headframe | 2.675 |

## Landmark head-pose spread on these two sessions

`eye_bench::runner::replay_session` re-runs the real `landmark` estimator, so
`GazeRay::head_rotation` (`screen_from_viewer`, eyeball.rs:80-85) is the actual
per-frame PnP rotation, not a stand-in. Magnitude of the rotation angle
(`UnitQuaternion::axis_angle`) across every ray of each session:

| session | n rays | mean | p5 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| A (`20261008T201638Z`) | 1548 | 3.49 deg | 1.21 deg | 5.80 deg | 0.56 deg | 6.64 deg |
| B (`20261008T201709Z`) | 1544 | 2.51 deg | 0.84 deg | 4.62 deg | 0.34 deg | 5.96 deg |

Both sessions carry a few degrees of incidental head rotation throughout (the
subject was not holding a frontal pose to a nominal camera, the one case
where `screen_from_viewer` is exactly identity, eyeball.rs:298). The affine
`theta` learned in the screen frame is therefore wrong by a rotation-dependent
amount on every frame of both sessions, not just on a hypothetical
deliberately-reposed recording. `HeadFrame`'s cross-session numbers above
(2.68-2.68 deg mean vs quadratic's 2.90-2.94 and affine's 3.76-3.81) are the
direct evidence that correcting in the head frame already helps on the
rotation these two sessions happen to contain, even without the dedicated
10-30 deg deliberate-rotation recording the DECIDED section still calls for.

## Precision guard, same harness and config as the grid above

The card's +20 % precision cap is against the **uncalibrated** baseline, and
that baseline must come from the same filter config as the grid
(`bench/baseline/rgb.toml`'s heavy one-euro, `min_cutoff = 0.3`,
`beta = 0.002`), not from `bench/baseline/results/20261008/rgb/report.md`'s
0.43 deg, which was measured under the crate-default filter
(`OneEuroConfig` 1.0 / 0.007) before that heavier config was adopted. Measured
here, pooled, same config as the grid:

| calibration | mean | precision |
|---|---:|---:|
| none | 21.25 deg | **0.35 deg** |

Recomputed guard (LOTO precision vs 0.35 deg, not 0.43):

| slope_prior_sigma | model | LOTO prec | vs 0.35 deg |
|---|---|---:|---:|
| 0.15 | affine | 0.54 | +54 % |
| 0.15 | quadratic | 0.49 | +40 % |
| 0.15 | headframe | 0.52 | +49 % |
| 0.5 | affine | 0.54 | +54 % |
| 0.5 | quadratic | 0.52 | +49 % |
| 0.5 | headframe | 0.52 | +49 % |
| 2.0 | affine | 0.54 | +54 % |
| 2.0 | quadratic | 0.54 | +54 % |
| 2.0 | headframe | 0.53 | +51 % |

**Every model and prior in the grid is over the +20 % cap** against the
correctly-matched baseline. The previous "+16 %, inside the cap" claim used
0.43 deg from a different filter config than the one this grid ran under
(affine's own LOTO error, 3.95, is identical between the two filter configs,
which is what made the mismatch easy to miss: the fit model's accuracy did not
move, only the precision denominator did). None of affine, quadratic or
headframe passes the guard as written.

## Conclusion

`slope_prior_sigma` barely moves the numbers at all (within 0.06 deg mean
across 0.15 to 2.0, for all three models) -- consistent with the DATA section
above: the fit model is the lever, not its priors.

The affine cross-session numbers (3.76-3.81 deg mean) are *lower* than the
affine LOTO number (3.95-3.99 deg) on these two sessions, not higher: LOTO is
not the flattering number here, cross-session is (a 16-target LOTO fold still
trains on 15 of this session's own targets, so it is not a strong test of
transfer to a different session). This does not, by itself, say anything
about head-pose sensitivity one way or the other; the real evidence for that
is the head-pose spread and the `HeadFrame` cross-session numbers above.

**Decision: adopt `HeadFrame` as the primary candidate (per the card's DECIDED
section); do not change `FitConfig::default()`'s automatic ladder yet.**

- `HeadFrame` has the best cross-session mean of the whole grid, 2.68 deg at
  `slope_prior_sigma = 0.15`, beating quadratic (2.90, -8 %) and affine (3.76,
  -29 %). Cross-session is the metric that measures posture change, and this
  is on real, if modest (2.5-3.5 deg mean), incidental head rotation already
  present in both recordings (see above), not synthetic data.
- `HeadFrame`'s LOTO mean (3.12 deg) is comfortably inside the "no more than
  10 % worse" bound against either reference: 5.8 % worse than quadratic's
  LOTO (2.95), and 21 % *better* than affine's LOTO (3.95-3.99).
- The precision guard is not met by `HeadFrame`, or by anything else in the
  grid (see above): +49 % against the honest 0.35 deg baseline. This is a
  real, unresolved gap, not a formality to wave through. The ADDENDUM already
  names the fix (a filter re-tune under the calibrated profile) as a
  follow-up to evaluate on these same recordings; that work is out of scope
  here and is tracked as a follow-up rather than silently skipped.
- `FitConfig::default()` is left unchanged: the ladder still has no
  `HeadFrame` branch, and `min_targets_quadratic: 12` keeps selecting
  `Quadratic` for 16-target sessions. Two things block flipping the default:
  (1) the precision guard above is unmet by every model, so there is no
  currently-passing candidate to promote; (2) the card's own DECIDED section
  requires a dedicated recording with deliberate large head rotation (HUMAN
  step, not yet collected) before `HeadFrame` becomes the unconditional
  default rather than an explicit `model_override`. Both are tracked as
  follow-up work on this card rather than left as an implicit "pick one later"
  in this file.

## Stale card claim

`eye_bench::testing::kappa_ray_config` (the pinning-test fixture named in the
card's acceptance criteria) only generates a constant per-axis `offset_deg`
bias (`KappaRayOptions` has no gain/curvature field); it cannot produce "a
known non-identity gain" as written. The pinning coverage for `Quadratic`
instead lives in `eye-calibration`'s own synthetic generator
(`test_quadratic_recovers_synthetic_curvature`, `user_fit.rs`), which injects
a known `yaw^2` curvature and asserts the fitted `quad[0]` recovers it.
