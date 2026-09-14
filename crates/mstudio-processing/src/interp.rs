//! Gap interpolation, mirroring `dataProcessor.interpolate_selected_data`,
//! i.e. `pandas.Series.interpolate(method, limit_direction='both', order)`
//! applied to the whole column with only the NaNs inside the selected range
//! written back.

use mstudio_core::DirtyRange;
use ndarray::{s, Array3};

use crate::bspline::BSpline;
use crate::{ProcessingError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpMethod {
    /// `np.interp`: linear inside, clamped to the nearest valid value outside.
    Linear,
    /// `interp1d(kind='nearest')`: ties go to the lower neighbour.
    Nearest,
    /// `interp1d(kind='zero')`: previous valid value.
    Zero,
    /// `interp1d(kind='slinear')`: degree-1 spline.
    Slinear,
    /// `interp1d(kind='quadratic')`.
    Quadratic,
    /// `interp1d(kind='cubic')`: not-a-knot cubic.
    Cubic,
    /// `interp1d(kind=order)`: interpolating spline of that degree.
    Polynomial(u32),
    /// Interpolating spline of the given degree.
    ///
    /// Deviation: pandas' `spline` is `UnivariateSpline(k=order)` with the
    /// default smoothing `s = len(x)`, which does not pass through the data
    /// and produced gap values 0.2 m off the trajectory in the oracle run.
    /// An interpolating spline is what users of a gap-filling tool expect.
    Spline(u32),
}

impl InterpMethod {
    pub fn from_name(name: &str, order: u32) -> Option<InterpMethod> {
        Some(match name {
            "linear" => InterpMethod::Linear,
            "nearest" => InterpMethod::Nearest,
            "zero" => InterpMethod::Zero,
            "slinear" => InterpMethod::Slinear,
            "quadratic" => InterpMethod::Quadratic,
            "cubic" => InterpMethod::Cubic,
            "polynomial" => InterpMethod::Polynomial(order),
            "spline" => InterpMethod::Spline(order),
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            InterpMethod::Linear => "linear",
            InterpMethod::Nearest => "nearest",
            InterpMethod::Zero => "zero",
            InterpMethod::Slinear => "slinear",
            InterpMethod::Quadratic => "quadratic",
            InterpMethod::Cubic => "cubic",
            InterpMethod::Polynomial(_) => "polynomial",
            InterpMethod::Spline(_) => "spline",
        }
    }
}

/// Fill every NaN of `values` the way pandas would, returning the full column.
/// Leading/trailing NaNs are clamped for `Linear` and left NaN for the scipy
/// kinds (`fill_value=nan`), as in pandas.
pub fn interpolate_column(values: &[f64], method: InterpMethod) -> Result<Vec<f64>> {
    let n = values.len();
    let valid: Vec<usize> = (0..n).filter(|&i| !values[i].is_nan()).collect();
    let mut out = values.to_vec();
    if valid.len() == n {
        return Ok(out);
    }
    let x: Vec<f64> = valid.iter().map(|&i| i as f64).collect();
    let y: Vec<f64> = valid.iter().map(|&i| values[i]).collect();
    let need = |k: usize| -> Result<()> {
        if valid.len() < k + 1 {
            Err(ProcessingError::NotEnoughData { need: k + 1, have: valid.len() })
        } else {
            Ok(())
        }
    };
    let inside = |i: usize| x.first().is_some_and(|&a| (i as f64) >= a) && x.last().is_some_and(|&b| (i as f64) <= b);

    match method {
        InterpMethod::Linear => {
            need(0)?;
            for i in 0..n {
                if values[i].is_nan() {
                    out[i] = np_interp(i as f64, &x, &y);
                }
            }
        }
        InterpMethod::Nearest => {
            need(0)?;
            let bds: Vec<f64> = x.windows(2).map(|w| w[0] / 2.0 + w[1] / 2.0).collect();
            for i in 0..n {
                if values[i].is_nan() && inside(i) {
                    // searchsorted(side='left'), clipped
                    let idx = bds.partition_point(|&b| b < i as f64).min(y.len() - 1);
                    out[i] = y[idx];
                }
            }
        }
        InterpMethod::Zero
        | InterpMethod::Slinear
        | InterpMethod::Quadratic
        | InterpMethod::Cubic
        | InterpMethod::Polynomial(_)
        | InterpMethod::Spline(_) => {
            let k = match method {
                InterpMethod::Zero => 0,
                InterpMethod::Slinear => 1,
                InterpMethod::Quadratic => 2,
                InterpMethod::Cubic => 3,
                InterpMethod::Polynomial(o) | InterpMethod::Spline(o) => o as usize,
                _ => unreachable!(),
            };
            if k > 5 {
                return Err(ProcessingError::InvalidParameter(format!("spline order {k} not supported (max 5)")));
            }
            need(k)?;
            let spline =
                BSpline::interpolate(&x, &y, k).ok_or(ProcessingError::Other("spline system is singular".into()))?;
            for i in 0..n {
                if values[i].is_nan() && inside(i) {
                    out[i] = spline.eval(i as f64);
                }
            }
        }
    }
    Ok(out)
}

/// `numpy.interp` for one point (`xp` increasing), constant outside.
fn np_interp(xq: f64, xp: &[f64], fp: &[f64]) -> f64 {
    if xq <= xp[0] {
        return fp[0];
    }
    if xq >= xp[xp.len() - 1] {
        return fp[fp.len() - 1];
    }
    let j = xp.partition_point(|&v| v <= xq) - 1;
    let t = (xq - xp[j]) / (xp[j + 1] - xp[j]);
    fp[j] + t * (fp[j + 1] - fp[j])
}

/// `interpolate_selected_data` for one marker: interpolate every axis over
/// the whole column and write back the NaNs inside the inclusive range.
/// Returns the range actually touched (empty if there was nothing to fill).
pub fn interpolate_in_range(
    frames: &mut Array3<f64>,
    marker: usize,
    first: usize,
    last: usize,
    method: InterpMethod,
) -> Result<DirtyRange> {
    let n = frames.dim().0;
    let range = DirtyRange::inclusive(first.min(n.saturating_sub(1)), last.min(n.saturating_sub(1)));
    let mut touched = DirtyRange { start: 0, end: 0 };
    for axis in 0..3 {
        let col: Vec<f64> = frames.slice(s![.., marker, axis]).to_vec();
        let targets: Vec<usize> = (range.start..range.end).filter(|&i| col[i].is_nan()).collect();
        if targets.is_empty() {
            continue;
        }
        let filled = interpolate_column(&col, method)?;
        for &i in &targets {
            frames[[i, marker, axis]] = filled[i];
            touched = touched.union(DirtyRange::single(i));
        }
    }
    Ok(touched)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gapped() -> Vec<f64> {
        let mut v: Vec<f64> = (0..20).map(|i| (i as f64) * 0.5).collect();
        for x in &mut v[8..12] {
            *x = f64::NAN;
        }
        v[0] = f64::NAN;
        v
    }

    #[test]
    fn linear_clamps_edges_and_fills_inside() {
        let y = interpolate_column(&gapped(), InterpMethod::Linear).unwrap();
        assert_eq!(y[0], 0.5); // clamped to first valid
        assert!((y[9] - 4.5).abs() < 1e-12);
    }

    #[test]
    fn scipy_kinds_leave_edges_nan() {
        for m in [InterpMethod::Nearest, InterpMethod::Zero, InterpMethod::Cubic, InterpMethod::Quadratic] {
            let y = interpolate_column(&gapped(), m).unwrap();
            assert!(y[0].is_nan(), "{m:?}");
            assert!(!y[9].is_nan(), "{m:?}");
        }
        assert!((interpolate_column(&gapped(), InterpMethod::Cubic).unwrap()[10] - 5.0).abs() < 1e-9);
        assert_eq!(interpolate_column(&gapped(), InterpMethod::Zero).unwrap()[11], 3.5);
        // nearest: valid neighbours are 7 and 12 (midpoint 9.5): 8, 9 → 3.5; 10, 11 → 6.0
        let y = interpolate_column(&gapped(), InterpMethod::Nearest).unwrap();
        assert_eq!((y[8], y[9], y[10], y[11]), (3.5, 3.5, 6.0, 6.0));
        // a true tie goes to the lower neighbour (searchsorted side='left')
        let v = [1.0, f64::NAN, f64::NAN, f64::NAN, 5.0];
        assert_eq!(interpolate_column(&v, InterpMethod::Nearest).unwrap()[2], 1.0);
    }

    #[test]
    fn range_write_back_only_fills_selected_nans() {
        let mut f = Array3::<f64>::zeros((20, 1, 3));
        for i in 0..20 {
            for a in 0..3 {
                f[[i, 0, a]] = i as f64;
            }
        }
        f[[5, 0, 0]] = f64::NAN;
        f[[15, 0, 0]] = f64::NAN;
        let r = interpolate_in_range(&mut f, 0, 3, 8, InterpMethod::Linear).unwrap();
        assert_eq!(r, DirtyRange { start: 5, end: 6 });
        assert_eq!(f[[5, 0, 0]], 5.0);
        assert!(f[[15, 0, 0]].is_nan());
    }

    #[test]
    fn not_enough_points_is_an_error() {
        let v = [f64::NAN, 1.0, f64::NAN, 2.0, f64::NAN];
        assert_eq!(
            interpolate_column(&v, InterpMethod::Cubic),
            Err(ProcessingError::NotEnoughData { need: 4, have: 2 })
        );
    }
}
