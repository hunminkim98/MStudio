//! B-spline interpolation equivalent to `scipy.interpolate.make_interp_spline`
//! with `bc_type=None` (not-a-knot for odd degree, Greville-style midpoint
//! knots for even degree), as used by `interp1d` for the `zero`, `slinear`,
//! `quadratic`, `cubic` and integer kinds.

/// Knot vector for degree `k` through strictly increasing `x`.
pub fn knots(x: &[f64], k: usize) -> Vec<f64> {
    let n = x.len();
    match k {
        0 => {
            let mut t = x.to_vec();
            t.push(x[n - 1]);
            t
        }
        1 => {
            let mut t = vec![x[0]];
            t.extend_from_slice(x);
            t.push(x[n - 1]);
            t
        }
        _ => {
            let (k2, inner): (usize, Vec<f64>) = if k % 2 == 1 {
                (k.div_ceil(2), x.to_vec())
            } else {
                (k / 2, x.windows(2).map(|w| (w[0] + w[1]) / 2.0).collect())
            };
            let core = if inner.len() > 2 * k2 { inner[k2..inner.len() - k2].to_vec() } else { Vec::new() };
            let mut t = vec![x[0]; k + 1];
            t.extend(core);
            t.extend(std::iter::repeat_n(x[n - 1], k + 1));
            t
        }
    }
}

/// Index `i` with `t[i] <= x < t[i+1]` inside the valid span `[k, len-k-1]`;
/// `x == t.last()` maps to the last interval.
fn find_interval(t: &[f64], k: usize, x: f64) -> usize {
    let hi = t.len() - k - 2;
    if x >= t[hi + 1] {
        return hi;
    }
    let mut lo = k;
    let mut hi_i = hi;
    while lo < hi_i {
        let mid = (lo + hi_i).div_ceil(2);
        if t[mid] <= x {
            lo = mid;
        } else {
            hi_i = mid - 1;
        }
    }
    lo
}

/// Cox–de Boor: the `k+1` non-zero basis functions `B_{i-k..=i,k}(x)`.
fn basis(t: &[f64], k: usize, i: usize, x: f64, out: &mut [f64]) {
    out[0] = 1.0;
    for j in 1..=k {
        let mut saved = 0.0;
        for r in 0..j {
            let left = x - t[i + 1 + r - j];
            let right = t[i + 1 + r] - x;
            let denom = right + left;
            let term = if denom != 0.0 { out[r] / denom } else { 0.0 };
            out[r] = saved + right * term;
            saved = left * term;
        }
        out[j] = saved;
    }
}

pub struct BSpline {
    t: Vec<f64>,
    c: Vec<f64>,
    k: usize,
}

impl BSpline {
    /// Interpolating spline of degree `k` through `(x, y)`; `x` strictly
    /// increasing with at least `k + 1` points.
    pub fn interpolate(x: &[f64], y: &[f64], k: usize) -> Option<BSpline> {
        let n = x.len();
        if n < k + 1 || n != y.len() {
            return None;
        }
        let t = knots(x, k);
        if k == 0 {
            return Some(BSpline { t, c: y.to_vec(), k });
        }
        // Banded collocation matrix: row j has k+1 entries starting at column i-k.
        let mut rows: Vec<(usize, Vec<f64>)> = Vec::with_capacity(n);
        let mut b = vec![0.0; k + 1];
        for &xj in x {
            let i = find_interval(&t, k, xj);
            basis(&t, k, i, xj, &mut b);
            rows.push((i - k, b.clone()));
        }
        let c = solve_banded(rows, y, k)?;
        Some(BSpline { t, c, k })
    }

    pub fn eval(&self, x: f64) -> f64 {
        let k = self.k;
        let i = find_interval(&self.t, k, x);
        if k == 0 {
            return self.c[i];
        }
        let mut b = vec![0.0; k + 1];
        basis(&self.t, k, i, x, &mut b);
        (0..=k).map(|r| b[r] * self.c[i - k + r]).sum()
    }
}

/// Banded Gaussian elimination without pivoting. The collocation matrix of
/// an interpolating B-spline is totally positive, for which elimination
/// without pivoting is stable (de Boor), so no fill-in occurs outside the
/// band: row `r` holds columns `start[r] .. start[r] + 2k`.
fn solve_banded(rows: Vec<(usize, Vec<f64>)>, y: &[f64], k: usize) -> Option<Vec<f64>> {
    let n = y.len();
    let width = 2 * k + 1;
    let mut a = vec![vec![0.0; width]; n];
    let mut start = vec![0usize; n];
    let mut rhs = y.to_vec();
    for (r, (s, vals)) in rows.into_iter().enumerate() {
        start[r] = s;
        a[r][..=k].copy_from_slice(&vals);
    }
    let idx = |start: &[usize], r: usize, c: usize| -> Option<usize> {
        (c >= start[r] && c < start[r] + width).then(|| c - start[r])
    };
    for col in 0..n {
        let pc = idx(&start, col, col)?;
        let pv = a[col][pc];
        if pv == 0.0 {
            return None;
        }
        for row in col + 1..(col + k + 1).min(n) {
            let Some(rc) = idx(&start, row, col) else { continue };
            let f = a[row][rc] / pv;
            if f == 0.0 {
                continue;
            }
            for c in col..(start[col] + width).min(n) {
                if let (Some(rj), Some(pj)) = (idx(&start, row, c), idx(&start, col, c)) {
                    a[row][rj] -= f * a[col][pj];
                }
            }
            rhs[row] -= f * rhs[col];
        }
    }
    let mut c = vec![0.0; n];
    for row in (0..n).rev() {
        let mut s = rhs[row];
        for (col, cv) in c.iter().enumerate().take((start[row] + width).min(n)).skip(row + 1) {
            if let Some(j) = idx(&start, row, col) {
                s -= a[row][j] * cv;
            }
        }
        c[row] = s / a[row][idx(&start, row, row)?];
    }
    Some(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knots_follow_scipy_rules() {
        let x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(knots(&x, 0), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0]);
        assert_eq!(knots(&x, 1), vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0]);
        assert_eq!(knots(&x, 3), vec![0.0, 0.0, 0.0, 0.0, 2.0, 3.0, 5.0, 5.0, 5.0, 5.0]);
        assert_eq!(knots(&x, 2), vec![0.0, 0.0, 0.0, 1.5, 2.5, 3.5, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn reproduces_polynomials_of_its_degree() {
        let x: Vec<f64> = (0..12).map(|i| i as f64 * 0.7).collect();
        for k in 1..=3usize {
            let y: Vec<f64> = x.iter().map(|v| v.powi(k as i32) - 2.0 * v + 1.0).collect();
            let s = BSpline::interpolate(&x, &y, k).unwrap();
            for q in [0.35f64, 1.9, 4.44, 7.7] {
                let want = q.powi(k as i32) - 2.0 * q + 1.0;
                assert!((s.eval(q) - want).abs() < 1e-9, "k={k} x={q}: {} vs {want}", s.eval(q));
            }
            for (xi, yi) in x.iter().zip(&y) {
                assert!((s.eval(*xi) - yi).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn zero_degree_is_previous_value() {
        let x = [0.0, 1.0, 2.0, 3.0];
        let y = [10.0, 20.0, 30.0, 40.0];
        let s = BSpline::interpolate(&x, &y, 0).unwrap();
        assert_eq!(s.eval(0.5), 10.0);
        assert_eq!(s.eval(2.9), 30.0);
        assert_eq!(s.eval(3.0), 40.0);
    }
}
