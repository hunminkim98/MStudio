//! `scipy.signal` / `scipy.ndimage` equivalents: Butterworth design,
//! zero-phase filtering, Gaussian and median filters. Each follows the scipy
//! implementation step by step so results agree to floating-point noise.

use num_complex::Complex64;

// ----------------------------------------------------------- butterworth --

/// `scipy.signal.butter(n, wn, 'low', analog=False)` → `(b, a)`.
/// `wn` is the cutoff normalised to the Nyquist frequency.
pub fn butter_lowpass(n: usize, wn: f64) -> (Vec<f64>, Vec<f64>) {
    // buttap: poles on the unit circle
    let poles: Vec<Complex64> = (0..n)
        .map(|i| {
            let m = -(n as f64) + 1.0 + 2.0 * i as f64;
            -(Complex64::i() * std::f64::consts::PI * m / (2.0 * n as f64)).exp()
        })
        .collect();
    // lp2lp_zpk with the bilinear pre-warp (fs = 2)
    let fs = 2.0;
    let warped = 2.0 * fs * (std::f64::consts::PI * wn / fs).tan();
    let poles: Vec<Complex64> = poles.iter().map(|p| p * warped).collect();
    let k = warped.powi(n as i32);
    // bilinear_zpk
    let fs2 = 2.0 * fs;
    let p_z: Vec<Complex64> = poles.iter().map(|p| (fs2 + p) / (fs2 - p)).collect();
    let z_z: Vec<Complex64> = vec![Complex64::new(-1.0, 0.0); n];
    let denom: Complex64 = poles.iter().fold(Complex64::new(1.0, 0.0), |acc, p| acc * (fs2 - p));
    let k_z = k * (Complex64::new(1.0, 0.0) / denom).re;
    // zpk2tf
    let b: Vec<f64> = poly(&z_z).iter().map(|c| (k_z * c).re).collect();
    let a: Vec<f64> = poly(&p_z).iter().map(|c| c.re).collect();
    (b, a)
}

/// `numpy.poly`: coefficients of ∏ (x − rᵢ), highest power first, built by
/// sequential convolution exactly like numpy.
fn poly(roots: &[Complex64]) -> Vec<Complex64> {
    let mut a = vec![Complex64::new(1.0, 0.0)];
    for r in roots {
        let mut next = vec![Complex64::new(0.0, 0.0); a.len() + 1];
        for (i, &c) in a.iter().enumerate() {
            next[i] += c;
            next[i + 1] -= c * r;
        }
        a = next;
    }
    a
}

/// `scipy.signal.lfilter_zi`: initial state giving a steady-state step
/// response, for the transposed direct-form II structure.
#[allow(clippy::needless_range_loop)] // index arithmetic mirrors the matrix definition
pub fn lfilter_zi(b: &[f64], a: &[f64]) -> Vec<f64> {
    let n = a.len().max(b.len());
    let mut a: Vec<f64> = a.iter().copied().chain(std::iter::repeat(0.0)).take(n).collect();
    let mut b: Vec<f64> = b.iter().copied().chain(std::iter::repeat(0.0)).take(n).collect();
    if a[0] != 1.0 {
        let a0 = a[0];
        for v in a.iter_mut() {
            *v /= a0;
        }
        for v in b.iter_mut() {
            *v /= a0;
        }
    }
    let m = n - 1;
    // IminusA = I - companion(a).T ; companion first row = -a[1:], subdiagonal ones
    let mut mat = vec![vec![0.0; m]; m];
    for i in 0..m {
        for j in 0..m {
            let comp_t = if j == 0 {
                -a[i + 1]
            } else if j == i + 1 {
                1.0
            } else {
                0.0
            };
            mat[i][j] = (if i == j { 1.0 } else { 0.0 }) - comp_t;
        }
    }
    let rhs: Vec<f64> = (0..m).map(|i| b[i + 1] - a[i + 1] * b[0]).collect();
    solve_dense(mat, rhs)
}

/// Gaussian elimination with partial pivoting (tiny systems only).
#[allow(clippy::needless_range_loop)]
fn solve_dense(mut m: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Vec<f64> {
    let n = rhs.len();
    for col in 0..n {
        let piv = (col..n).max_by(|&i, &j| m[i][col].abs().total_cmp(&m[j][col].abs())).unwrap();
        m.swap(col, piv);
        rhs.swap(col, piv);
        for row in col + 1..n {
            let f = m[row][col] / m[col][col];
            if f != 0.0 {
                for k in col..n {
                    m[row][k] -= f * m[col][k];
                }
                rhs[row] -= f * rhs[col];
            }
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let mut s = rhs[row];
        for k in row + 1..n {
            s -= m[row][k] * x[k];
        }
        x[row] = s / m[row][row];
    }
    x
}

/// `scipy.signal.lfilter(b, a, x, zi=zi)` (transposed direct form II).
pub fn lfilter(b: &[f64], a: &[f64], x: &[f64], zi: &[f64]) -> Vec<f64> {
    let n = a.len().max(b.len());
    let a0 = a[0];
    let a: Vec<f64> = a.iter().map(|v| v / a0).chain(std::iter::repeat(0.0)).take(n).collect();
    let b: Vec<f64> = b.iter().map(|v| v / a0).chain(std::iter::repeat(0.0)).take(n).collect();
    let mut z: Vec<f64> = zi.to_vec();
    z.resize(n - 1, 0.0);
    let mut y = Vec::with_capacity(x.len());
    for &xk in x {
        let yk = b[0] * xk + z.first().copied().unwrap_or(0.0);
        for i in 0..n.saturating_sub(2) {
            z[i] = b[i + 1] * xk + z[i + 1] - a[i + 1] * yk;
        }
        if n >= 2 {
            z[n - 2] = b[n - 1] * xk - a[n - 1] * yk;
        }
        y.push(yk);
    }
    y
}

/// `scipy.signal.filtfilt(b, a, x)` with the default odd padding and
/// `padlen = 3 * max(len(a), len(b))`. Requires `x.len() > padlen`.
pub fn filtfilt(b: &[f64], a: &[f64], x: &[f64]) -> Vec<f64> {
    let edge = 3 * a.len().max(b.len());
    let n = x.len();
    assert!(n > edge, "filtfilt: signal shorter than padlen");
    let mut ext = Vec::with_capacity(n + 2 * edge);
    for i in (1..=edge).rev() {
        ext.push(2.0 * x[0] - x[i]);
    }
    ext.extend_from_slice(x);
    for i in 1..=edge {
        ext.push(2.0 * x[n - 1] - x[n - 1 - i]);
    }
    let zi = lfilter_zi(b, a);
    let zi0: Vec<f64> = zi.iter().map(|v| v * ext[0]).collect();
    let y = lfilter(b, a, &ext, &zi0);
    let y0 = *y.last().unwrap();
    let zi1: Vec<f64> = zi.iter().map(|v| v * y0).collect();
    let rev: Vec<f64> = y.iter().rev().copied().collect();
    let y = lfilter(b, a, &rev, &zi1);
    let mut out: Vec<f64> = y.iter().rev().copied().collect();
    out.drain(..edge);
    out.truncate(n);
    out
}

// -------------------------------------------------------------- gaussian --

/// `scipy.ndimage.gaussian_filter1d(x, sigma)` with `mode='reflect'`,
/// `truncate=4.0`. NaN propagates through the kernel, as in scipy.
pub fn gaussian_filter1d(x: &[f64], sigma: f64) -> Vec<f64> {
    let lw = (4.0 * sigma + 0.5) as i64;
    let sigma2 = sigma * sigma;
    let mut w: Vec<f64> = (-lw..=lw).map(|i| (-0.5 / sigma2 * (i * i) as f64).exp()).collect();
    let sum: f64 = w.iter().sum();
    for v in w.iter_mut() {
        *v /= sum;
    }
    let n = x.len() as i64;
    let reflect = |mut i: i64| -> usize {
        // scipy 'reflect': (d c b a | a b c d | d c b a)
        loop {
            if i < 0 {
                i = -i - 1;
            } else if i >= n {
                i = 2 * n - i - 1;
            } else {
                return i as usize;
            }
        }
    };
    (0..n)
        .map(|i| {
            let mut acc = 0.0;
            for (j, wj) in w.iter().enumerate() {
                acc += wj * x[reflect(i + j as i64 - lw)];
            }
            acc
        })
        .collect()
}

// ---------------------------------------------------------------- median --

/// Median filter with zero padding at the edges like `scipy.signal.medfilt`.
///
/// Deviation from scipy: samples that are NaN are left out of the window
/// (scipy's quickselect gives unspecified results when a window contains
/// NaN). An output is NaN only where its own input sample is NaN.
pub fn medfilt(x: &[f64], kernel: usize) -> Vec<f64> {
    let k = kernel.max(1) | 1; // odd
    let half = (k / 2) as i64;
    let n = x.len() as i64;
    let mut window = Vec::with_capacity(k);
    (0..n)
        .map(|i| {
            if x[i as usize].is_nan() {
                return f64::NAN;
            }
            window.clear();
            for j in i - half..=i + half {
                let v = if j < 0 || j >= n { 0.0 } else { x[j as usize] };
                if !v.is_nan() {
                    window.push(v);
                }
            }
            window.sort_by(|a, b| a.total_cmp(b));
            window[window.len() / 2]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn butter_matches_scipy_coefficients() {
        // scipy.signal.butter(2, 10/60, 'low'):
        let (b, a) = butter_lowpass(2, 10.0 / 60.0);
        let want_b = [0.04948995626867706, 0.09897991253735412, 0.04948995626867706];
        let want_a = [1.0, -1.2796324249978088, 0.47759225007251704];
        for i in 0..3 {
            assert!((b[i] - want_b[i]).abs() < 1e-12, "b[{i}] {} vs {}", b[i], want_b[i]);
            assert!((a[i] - want_a[i]).abs() < 1e-12, "a[{i}] {} vs {}", a[i], want_a[i]);
        }
    }

    #[test]
    fn lfilter_zi_is_steady_state_for_a_step() {
        let (b, a) = butter_lowpass(2, 0.2);
        let zi = lfilter_zi(&b, &a);
        let y = lfilter(&b, &a, &[1.0; 5], &zi);
        for v in y {
            assert!((v - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn filtfilt_matches_scipy_on_a_parabola() {
        let (b, a) = butter_lowpass(2, 0.2);
        let zi = lfilter_zi(&b, &a);
        assert!((zi[0] - 0.9325447261109284).abs() < 1e-12 && (zi[1] + 0.3453463242071169).abs() < 1e-12);
        let x = vec![3.0; 40];
        let y = filtfilt(&b, &a, &x);
        assert!(y.iter().all(|v| (v - 3.0).abs() < 1e-12));
        // scipy.signal.filtfilt(b, a, ((i-20)/5)**2)
        let x: Vec<f64> = (0..41).map(|i| ((i as f64 - 20.0) / 5.0).powi(2)).collect();
        let y = filtfilt(&b, &a, &x);
        let want = [15.997334791518753, 14.490827385024394, 13.01919490898537, 11.606676995105401];
        for i in 0..4 {
            assert!((y[i] - want[i]).abs() < 1e-10, "y[{i}] {} vs {}", y[i], want[i]);
        }
        assert!((y[20] + 5.686829307143599e-06).abs() < 1e-10);
        assert!((y[40] - 16.01998512512694).abs() < 1e-10);
    }

    #[test]
    fn gaussian_reflect_and_nan_propagation() {
        let x = vec![1.0; 20];
        assert!(gaussian_filter1d(&x, 2.0).iter().all(|v| (v - 1.0).abs() < 1e-12));
        let mut x = vec![1.0; 20];
        x[10] = f64::NAN;
        let y = gaussian_filter1d(&x, 1.0); // lw = 4
        assert!(y[6..=14].iter().all(|v| v.is_nan()));
        assert!(!y[5].is_nan() && !y[15].is_nan());
    }

    #[test]
    fn medfilt_zero_pads_and_skips_nan() {
        assert_eq!(medfilt(&[5.0, 6.0, 7.0], 3), vec![5.0, 6.0, 6.0]);
        let y = medfilt(&[1.0, f64::NAN, 3.0, 4.0], 3);
        assert_eq!(y[0], 1.0);
        assert!(y[1].is_nan());
        assert_eq!(y[2], 4.0); // window {3, 4} → upper median
        assert_eq!(y[3], 3.0); // window {3, 4, 0} → 3
    }
}
