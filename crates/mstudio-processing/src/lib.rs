//! MStudio signal processing.
//!
//! Ports of `MStudio/utils/filtering.py` (Pose2Sim, BSD-3, David Pagnon),
//! `utils/dataProcessor.py` and `utils/analysisMode.py`, pinned to the Python
//! oracle in `tests/golden/`. Every routine is a pure function on plain
//! arrays; the application decides what to run on a worker thread.
//!
//! Intentional deviations from the Python code are documented on the item
//! concerned and summarised in `docs/PHASE2_RESULTS.md`.

pub mod analysis;
mod bspline;
pub mod filters;
pub mod interp;
mod kalman;
mod lowess;
pub mod pattern;
mod rotation;
mod signal;

pub use filters::{apply_filter, filter_column, filter_take, Filter};
pub use interp::{interpolate_column, interpolate_in_range, InterpMethod};
pub use pattern::pattern_interpolate;

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum ProcessingError {
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
    #[error("not enough valid samples: need {need}, have {have}")]
    NotEnoughData { need: usize, have: usize },
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, ProcessingError>;

/// Runs of consecutive indices where `keep(v)` is true, in order — the
/// `np.split(np.where(~mask)[0], gaps)` idiom used by every Pose2Sim filter.
pub(crate) fn valid_runs(values: &[f64], keep: impl Fn(f64) -> bool) -> Vec<std::ops::Range<usize>> {
    let mut runs = Vec::new();
    let mut start: Option<usize> = None;
    for (i, &v) in values.iter().enumerate() {
        match (start, keep(v)) {
            (None, true) => start = Some(i),
            (Some(s), false) => {
                runs.push(s..i);
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        runs.push(s..values.len());
    }
    runs
}

#[cfg(test)]
mod tests {
    use super::valid_runs;

    #[test]
    fn runs_split_on_nan_and_zero() {
        let v = [1.0, 2.0, f64::NAN, 3.0, 0.0, 4.0, 5.0];
        let runs = valid_runs(&v, |x| !x.is_nan() && x != 0.0);
        assert_eq!(runs, vec![0..2, 3..4, 5..7]);
        assert!(valid_runs(&[f64::NAN], |x| !x.is_nan()).is_empty());
    }
}
