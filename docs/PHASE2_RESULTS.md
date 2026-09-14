# Phase 2 results — processing with oracle parity

Status: **done**. `crates/mstudio-processing` reproduces every filter, interpolation
method, the pattern-based gap filler and the analysis primitives of MStudio v0.1.5,
checked against `tests/golden/` (78 arrays).

## Parity (all green, `cargo test -p mstudio-processing`)

| Group | Cases | Max |diff| vs Python | Tolerance |
|---|---|---|---|
| Butterworth (3 parameter sets × clean/gapped) | 6 | 5.6e-15 | 1e-6 |
| Butterworth on speed (2 × 2) | 4 | 3.1e-15 | 1e-6 |
| Kalman + RTS (3 × 2) | 6 | 4.4e-15 | 1e-5 |
| Gaussian (2 × 2) | 4 | 1.8e-15 | 1e-6 |
| LOESS (2 × 2) | 4 | 6.0e-14 | 1e-5 |
| Median (2 × 2) | 4 | 0 (away from NaN windows, see below) | 1e-6 |
| Interpolation: linear, nearest, zero, slinear, quadratic, cubic, polynomial(3) × 3 gaps | 21 | < 1e-9 | 1e-9 |
| Interpolation: spline | 3 | intentional deviation (below) | — |
| Pattern-based (1 / 2 / 3 references) | 4 | < 1e-9 | 1e-9 |
| Analysis: distance, joint angle, velocity, acceleration, arc | 5 | < 1e-9 (velocity 1e-6 from float32 golden rounding) | — |

The filters agree to floating-point noise because each is a step-by-step port
of the reference implementation: scipy `butter` (bilinear + `zpk2tf`),
`lfilter_zi` / transposed direct-form II `lfilter`, `filtfilt` odd padding;
filterpy `predict` / Joseph-form `update` / `rts_smoother`; statsmodels'
`_smoothers_lowess.pyx`; `ndimage.gaussian_filter1d` reflect mode; scipy
`make_interp_spline` knot rules (not-a-knot / midpoint) with a banded solver;
scipy `Rotation.align_vectors` (Kabsch via nalgebra SVD, single-pair shortest arc).

## Performance (`cargo bench -p mstudio-processing`, Apple M5 Max, 300 markers × 50 000 frames, all axes, rayon)

| Filter | Time | Plan target |
|---|---|---|
| Butterworth (order 4) | **90 ms** | < 200 ms ✅ |
| Gaussian (σ = 3) | 115 ms | — |
| Median (k = 5) | 106 ms | — |
| LOESS (10 points) | 136 ms | — |

## Bugs found in the shipped Python app (fixed on both sides)

**Pattern-based interpolation rotated the offset the wrong way.**
`Rotation.align_vectors(a, b)` returns the rotation taking `b` onto `a`; the code
passed `(initial, current)` and then applied the result to the *initial* offset,
i.e. it rotated by the inverse. With a synthetic rigid body rotating 0.05 rad/frame
the reconstruction error was 0.059 m; after swapping the arguments it is 1e-16.
Fixed in `MStudio/utils/dataProcessor.py` (2- and 3+-reference modes), regression
test added to `tests/test.py`, goldens `pattern_RKnee_2ref`, `pattern_RKnee_3ref`,
`pattern_LWrist_3ref` regenerated. Pure-translation cases (1 reference) were
unaffected.

## Intentional deviations

| Where | Python v0.1.5 | Rust | Why |
|---|---|---|---|
| `spline` interpolation | `UnivariateSpline(k=order)` with pandas' default smoothing `s = len(x)` — a *smoothing* spline that does not pass through the data; gap values were 0.23 m off the trajectory in the oracle | interpolating B-spline of the given order (identical to `cubic` for order 3) | a gap filler must interpolate; the golden test asserts both that Rust equals `cubic` and that the oracle really differs |
| `median` with NaN in the window | `scipy.signal.medfilt` quickselect: unspecified values in and shortly after any window that contains NaN (e.g. frame 52 returned frame 51's value in the oracle) | NaN samples are excluded from the window; output is NaN only where the input is | parity is asserted on every sample farther than one kernel width from a NaN |
| Kalman on sequences shorter than 3 samples | `IndexError` (cannot form the derivative states) | returned unchanged | never crash on a 1–2 sample run |
| Edge NaNs for scipy interpolation kinds | left NaN (`fill_value=nan`) | same | documented so the UI can say so |

Not changed but worth a follow-up (behaviour kept for parity): the Gaussian
filter propagates NaN over ±(4σ) frames around every gap because the vendored
Pose2Sim code filters the whole column instead of the valid runs.

## API

```rust
mstudio_processing::filter_column(&col, &Filter::Butterworth { order: 4, cutoff_hz: 6.0 }, fps)?;
mstudio_processing::apply_filter(&mut frames, marker, first, last, &filter, fps)?   // writes back only [first, last]
mstudio_processing::filter_take(&mut frames, &markers, &filter, fps)?             // all markers, rayon
mstudio_processing::interpolate_in_range(&mut frames, marker, first, last, InterpMethod::Cubic)?;
mstudio_processing::pattern_interpolate(&mut frames, target, &refs, first, last)?;
mstudio_processing::analysis::{distance, joint_angle, segment_angle, arc_points, velocity, acceleration}
```

Every mutating call returns a `DirtyRange` — the frames the renderer must re-upload (plan rule R2).

## Next: Phase 3 — renderer (`mstudio-render`)
