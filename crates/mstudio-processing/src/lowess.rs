//! LOWESS, a line-by-line port of statsmodels'
//! `_smoothers_lowess.pyx` for the case Pose2Sim uses: `it=0`, `delta=0`,
//! fitted at the data points, unit residual weights.

/// Locally weighted linear regression of `y` on `x` (`x` increasing),
/// using `frac` of the points for each local fit.
pub fn lowess(y: &[f64], x: &[f64], frac: f64) -> Vec<f64> {
    let n = x.len();
    let mut k = (frac * n as f64 + 1e-10) as usize;
    k = k.clamp(2, n.max(2)).min(n);
    let mut fit = vec![0.0; n];
    let mut w = vec![0.0; n];
    let mut left = 0usize;
    let mut right = k;
    for i in 0..n {
        let xv = x[i];
        // update_neighborhood
        while right < n && xv > (x[left] + x[right]) / 2.0 {
            left += 1;
            right += 1;
        }
        let radius = (xv - x[left]).max(x[right - 1] - xv);
        // calculate_weights: tricube of the normalised distance
        let mut sum = 0.0;
        let mut nonzero = 0usize;
        for j in left..right {
            let d = (x[j] - xv).abs() / radius;
            let d3 = d * d * d;
            let t = 1.0 - d3;
            w[j] = t * t * t;
            sum += w[j];
            if w[j] > 1e-12 {
                nonzero += 1;
            }
        }
        if nonzero < 2 {
            fit[i] = y[i];
            continue;
        }
        for wj in &mut w[left..right] {
            *wj /= sum;
        }
        // calculate_y_fit
        let mut swx = 0.0;
        for j in left..right {
            swx += w[j] * x[j];
        }
        let mut sqdev = 0.0;
        for j in left..right {
            sqdev += w[j] * (x[j] - swx) * (x[j] - swx);
        }
        let sqdev = sqdev.max(1e-12);
        let mut acc = 0.0;
        for j in left..right {
            let p = w[j] * (1.0 + (xv - swx) * (x[j] - swx) / sqdev);
            acc += p * y[j];
        }
        fit[i] = acc;
    }
    fit
}

#[cfg(test)]
mod tests {
    use super::lowess;

    #[test]
    fn reproduces_a_line_exactly() {
        let x: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|v| 2.0 * v + 1.0).collect();
        let f = lowess(&y, &x, 0.3);
        for (a, b) in f.iter().zip(&y) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn smooths_noise_towards_the_trend() {
        let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|v| v + if (*v as i64) % 2 == 0 { 0.5 } else { -0.5 }).collect();
        let f = lowess(&y, &x, 0.4);
        let err: f64 = f.iter().zip(&x).skip(5).take(40).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(err < 0.2, "{err}");
    }
}
