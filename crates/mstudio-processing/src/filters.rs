//! The six filters of `filtering.py` (Pose2Sim) and the way
//! `dataProcessor.filter_selected_data` applies them.

use mstudio_core::DirtyRange;
use ndarray::{s, Array3};
use rayon::prelude::*;

use crate::kalman::kalman_smooth;
use crate::lowess::lowess;
use crate::signal::{butter_lowpass, filtfilt, gaussian_filter1d, medfilt};
use crate::{valid_runs, ProcessingError, Result};

/// Filter selection with the parameters the UI exposes. Numeric coercions
/// match the Python code (`int()` on order, cutoff, trust ratio, sigma and
/// kernel size).
#[derive(Debug, Clone, PartialEq)]
pub enum Filter {
    /// Zero-phase Butterworth low-pass (`order` must be even; scipy is given `order/2`).
    Butterworth {
        order: u32,
        cutoff_hz: f64,
    },
    /// Butterworth applied to the derivative, then re-integrated.
    ButterworthOnSpeed {
        order: u32,
        cutoff_hz: f64,
    },
    /// Constant-acceleration Kalman filter, optionally RTS-smoothed.
    Kalman {
        trust_ratio: f64,
        smooth: bool,
    },
    Gaussian {
        sigma_kernel: f64,
    },
    /// LOWESS with `nb_values_used` points per local fit.
    Loess {
        nb_values_used: f64,
    },
    Median {
        kernel_size: f64,
    },
}

impl Filter {
    /// Name used in the Python config dict / UI.
    pub fn name(&self) -> &'static str {
        match self {
            Filter::Butterworth { .. } => "butterworth",
            Filter::ButterworthOnSpeed { .. } => "butterworth_on_speed",
            Filter::Kalman { .. } => "kalman",
            Filter::Gaussian { .. } => "gaussian",
            Filter::Loess { .. } => "LOESS",
            Filter::Median { .. } => "median",
        }
    }
}

fn butter_coeffs(order: u32, cutoff_hz: f64, fps: f64) -> Result<(Vec<f64>, Vec<f64>)> {
    if order == 0 || !order.is_multiple_of(2) {
        return Err(ProcessingError::InvalidParameter(format!(
            "Butterworth order must be a positive even number, got {order}"
        )));
    }
    let cutoff = cutoff_hz.trunc(); // Python: int(cut_off_frequency)
    if cutoff <= 0.0 || cutoff >= fps / 2.0 {
        return Err(ProcessingError::InvalidParameter(format!("cutoff {cutoff} Hz must be in (0, {})", fps / 2.0)));
    }
    Ok(butter_lowpass((order / 2) as usize, cutoff / (fps / 2.0)))
}

/// Filter one whole coordinate column (`filter1d` in Python).
pub fn filter_column(values: &[f64], filter: &Filter, fps: f64) -> Result<Vec<f64>> {
    let mut out = values.to_vec();
    match *filter {
        Filter::Butterworth { order, cutoff_hz } => {
            let (b, a) = butter_coeffs(order, cutoff_hz, fps)?;
            let padlen = 3 * a.len().max(b.len());
            for run in valid_runs(values, |v| !v.is_nan() && v != 0.0) {
                if run.len() > padlen {
                    let y = filtfilt(&b, &a, &values[run.clone()]);
                    out[run].copy_from_slice(&y);
                }
            }
        }
        Filter::ButterworthOnSpeed { order, cutoff_hz } => {
            let (b, a) = butter_coeffs(order, cutoff_hz, fps)?;
            let padlen = 3 * a.len().max(b.len());
            let n = values.len();
            if n < 2 {
                return Ok(out);
            }
            // col.diff() then fillna(diff.iloc[1] / 2) — every NaN, including gaps
            let mut diff: Vec<f64> =
                (0..n).map(|i| if i == 0 { f64::NAN } else { values[i] - values[i - 1] }).collect();
            let fill = diff[1] / 2.0;
            for d in diff.iter_mut() {
                if d.is_nan() {
                    *d = fill;
                }
            }
            for run in valid_runs(&diff, |v| !v.is_nan() && v != 0.0) {
                if run.len() > padlen {
                    let y = filtfilt(&b, &a, &diff[run.clone()]);
                    diff[run].copy_from_slice(&y);
                }
            }
            // cumsum (pandas skips NaN) + first sample
            let mut acc = 0.0;
            for i in 0..n {
                if diff[i].is_nan() {
                    out[i] = f64::NAN;
                } else {
                    acc += diff[i];
                    out[i] = acc + values[0];
                }
            }
        }
        Filter::Kalman { trust_ratio, smooth } => {
            let trust = trust_ratio.trunc() as i64;
            for run in valid_runs(values, |v| !v.is_nan() && v != 0.0) {
                let y = kalman_smooth(&values[run.clone()], fps, trust, smooth);
                out[run].copy_from_slice(&y);
            }
        }
        Filter::Gaussian { sigma_kernel } => {
            let sigma = sigma_kernel.trunc();
            if sigma <= 0.0 {
                return Err(ProcessingError::InvalidParameter(format!("sigma must be >= 1, got {sigma_kernel}")));
            }
            out = gaussian_filter1d(values, sigma);
        }
        Filter::Loess { nb_values_used } => {
            let kernel = nb_values_used;
            if kernel <= 0.0 {
                return Err(ProcessingError::InvalidParameter(format!(
                    "nb_values_used must be positive, got {kernel}"
                )));
            }
            for run in valid_runs(values, |v| !v.is_nan()) {
                if (run.len() as f64) > kernel {
                    let x: Vec<f64> = run.clone().map(|i| i as f64).collect();
                    let y = lowess(&values[run.clone()], &x, kernel / run.len() as f64);
                    out[run].copy_from_slice(&y);
                }
            }
        }
        Filter::Median { kernel_size } => {
            let mut k = kernel_size.trunc() as i64;
            if k <= 0 {
                k = 3;
            }
            if k % 2 == 0 {
                k += 1;
            }
            out = medfilt(values, k.max(1) as usize);
        }
    }
    Ok(out)
}

/// `filter_selected_data`: filter the whole column of every axis of one
/// marker, then write back only the inclusive frame range.
pub fn apply_filter(
    frames: &mut Array3<f64>,
    marker: usize,
    first: usize,
    last: usize,
    filter: &Filter,
    fps: f64,
) -> Result<DirtyRange> {
    let n = frames.dim().0;
    let range = DirtyRange::inclusive(first.min(n.saturating_sub(1)), last.min(n.saturating_sub(1)));
    for axis in 0..3 {
        let col: Vec<f64> = frames.slice(s![.., marker, axis]).to_vec();
        let filtered = filter_column(&col, filter, fps)?;
        let mut dst = frames.slice_mut(s![range.start..range.end, marker, axis]);
        for (d, v) in dst.iter_mut().zip(&filtered[range.start..range.end]) {
            *d = *v;
        }
    }
    Ok(range)
}

/// Filter every axis of the given markers over the whole take, in parallel.
pub fn filter_take(frames: &mut Array3<f64>, markers: &[usize], filter: &Filter, fps: f64) -> Result<DirtyRange> {
    let (n, _, _) = frames.dim();
    let jobs: Vec<(usize, usize)> = markers.iter().flat_map(|&m| (0..3).map(move |a| (m, a))).collect();
    type Job = ((usize, usize), Vec<f64>);
    let results: Vec<Result<Job>> = jobs
        .par_iter()
        .map(|&(m, a)| {
            let col: Vec<f64> = frames.slice(s![.., m, a]).to_vec();
            filter_column(&col, filter, fps).map(|v| ((m, a), v))
        })
        .collect();
    for r in results {
        let ((m, a), v) = r?;
        frames.slice_mut(s![.., m, a]).assign(&ndarray::Array1::from(v));
    }
    Ok(DirtyRange::all(n))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn butterworth_rejects_odd_order_and_bad_cutoff() {
        let v = vec![1.0; 50];
        assert!(filter_column(&v, &Filter::Butterworth { order: 3, cutoff_hz: 6.0 }, 100.0).is_err());
        assert!(filter_column(&v, &Filter::Butterworth { order: 4, cutoff_hz: 60.0 }, 100.0).is_err());
        assert!(filter_column(&v, &Filter::Butterworth { order: 4, cutoff_hz: 6.0 }, 100.0).is_ok());
    }

    #[test]
    fn short_runs_and_gaps_are_left_alone() {
        let mut v = vec![1.0; 30];
        v[10] = f64::NAN;
        v[20] = 0.0;
        let y = filter_column(&v, &Filter::Butterworth { order: 4, cutoff_hz: 6.0 }, 100.0).unwrap();
        assert!(y[10].is_nan());
        assert_eq!(y[20], 0.0);
        assert!((y[0] - 1.0).abs() < 1e-12); // run 0..10 has 10 samples > padlen 9
        assert_eq!(y[11..20], v[11..20]); // 9 samples: not filtered
    }

    #[test]
    fn apply_filter_only_touches_the_range() {
        let mut f = Array3::<f64>::zeros((40, 1, 3));
        for i in 0..40 {
            f[[i, 0, 0]] = if i % 2 == 0 { 1.0 } else { 1.5 };
        }
        let before = f.clone();
        let r = apply_filter(&mut f, 0, 10, 19, &Filter::Median { kernel_size: 3.0 }, 100.0).unwrap();
        assert_eq!(r, DirtyRange { start: 10, end: 20 });
        assert_eq!(f.slice(s![..10, .., ..]), before.slice(s![..10, .., ..]));
        assert_eq!(f.slice(s![20.., .., ..]), before.slice(s![20.., .., ..]));
        assert!(f.slice(s![10..20, 0, 0]).iter().all(|v| *v == 1.0 || *v == 1.5));
    }
}
