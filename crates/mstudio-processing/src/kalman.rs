//! Constant-acceleration Kalman filter + RTS smoother for one coordinate,
//! reproducing `filtering.kalman_filter` (Pose2Sim) on top of filterpy:
//! `dim_x = 3` (position, velocity, acceleration), `dim_z = 1`, Joseph-form
//! covariance update, `Q_discrete_white_noise(3, dt, var)`.
#![allow(clippy::needless_range_loop)] // 3×3 matrix code reads best with explicit indices

type M3 = [[f64; 3]; 3];
type V3 = [f64; 3];

fn matmul(a: &M3, b: &M3) -> M3 {
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    r
}

fn transpose(a: &M3) -> M3 {
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = a[j][i];
        }
    }
    r
}

fn matvec(a: &M3, v: &V3) -> V3 {
    [
        a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
        a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
        a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
    ]
}

fn add(a: &M3, b: &M3) -> M3 {
    let mut r = *a;
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] += b[i][j];
        }
    }
    r
}

fn sub(a: &M3, b: &M3) -> M3 {
    let mut r = *a;
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] -= b[i][j];
        }
    }
    r
}

fn inverse(m: &M3) -> M3 {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let inv_det = 1.0 / det;
    let mut r = [[0.0; 3]; 3];
    r[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det;
    r[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det;
    r[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det;
    r[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det;
    r[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det;
    r[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det;
    r[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det;
    r[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det;
    r[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det;
    r
}

/// Filter one gap-free sequence. `trust_ratio` and `smooth` follow the
/// Pose2Sim parameters (`process_noise = 20 * trust_ratio`, measurement
/// noise 20). Sequences shorter than 3 samples are returned unchanged (the
/// Python code cannot initialise its derivative states for them).
pub fn kalman_smooth(coords: &[f64], frame_rate: f64, trust_ratio: i64, smooth: bool) -> Vec<f64> {
    let n = coords.len();
    if n < 3 {
        return coords.to_vec();
    }
    let measurement_noise = 20.0;
    let process_noise = measurement_noise * trust_ratio as f64;
    let dt = 1.0 / frame_rate;

    // initial state from finite differences with unit step (derivate_array dt=1)
    let d1: Vec<f64> = coords.windows(2).map(|w| w[1] - w[0]).collect();
    let d2: Vec<f64> = d1.windows(2).map(|w| w[1] - w[0]).collect();
    let mut x: V3 = [coords[0], d1[0], d2[0]];

    let f: M3 = [[1.0, dt, dt * dt / 2.0], [0.0, 1.0, dt], [0.0, 0.0, 1.0]];
    let ft = transpose(&f);
    let h: V3 = [1.0, 0.0, 0.0];
    let mut p: M3 = [[measurement_noise, 0.0, 0.0], [0.0, measurement_noise, 0.0], [0.0, 0.0, measurement_noise]];
    let r = measurement_noise * measurement_noise;
    let var = process_noise * process_noise;
    let q: M3 = [
        [0.25 * dt.powi(4) * var, 0.5 * dt.powi(3) * var, 0.5 * dt * dt * var],
        [0.5 * dt.powi(3) * var, dt * dt * var, dt * var],
        [0.5 * dt * dt * var, dt * var, var],
    ];

    let mut means: Vec<V3> = Vec::with_capacity(n);
    let mut covs: Vec<M3> = Vec::with_capacity(n);
    for &z in coords {
        // predict
        x = matvec(&f, &x);
        p = add(&matmul(&matmul(&f, &p), &ft), &q);
        // update (Joseph form)
        let y = z - (h[0] * x[0] + h[1] * x[1] + h[2] * x[2]);
        let pht: V3 = [p[0][0], p[1][0], p[2][0]];
        let s = pht[0] + r;
        let k: V3 = [pht[0] / s, pht[1] / s, pht[2] / s];
        for i in 0..3 {
            x[i] += k[i] * y;
        }
        let mut i_kh = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                i_kh[i][j] = (if i == j { 1.0 } else { 0.0 }) - k[i] * h[j];
            }
        }
        let mut krk = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                krk[i][j] = k[i] * r * k[j];
            }
        }
        p = add(&matmul(&matmul(&i_kh, &p), &transpose(&i_kh)), &krk);
        means.push(x);
        covs.push(p);
    }

    if smooth {
        // Rauch–Tung–Striebel backward pass
        for k in (0..n - 1).rev() {
            let pp = add(&matmul(&matmul(&f, &covs[k]), &ft), &q);
            let g = matmul(&matmul(&covs[k], &ft), &inverse(&pp));
            let fx = matvec(&f, &means[k]);
            let innov = [means[k + 1][0] - fx[0], means[k + 1][1] - fx[1], means[k + 1][2] - fx[2]];
            let dx = matvec(&g, &innov);
            for i in 0..3 {
                means[k][i] += dx[i];
            }
            let dp = matmul(&matmul(&g, &sub(&covs[k + 1], &pp)), &transpose(&g));
            covs[k] = add(&covs[k], &dp);
        }
    }
    means.iter().map(|m| m[0]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_signal_is_preserved() {
        let x = vec![2.0; 50];
        for smooth in [false, true] {
            let y = kalman_smooth(&x, 100.0, 20, smooth);
            assert!(y.iter().all(|v| (v - 2.0).abs() < 1e-9), "{smooth}");
        }
    }

    #[test]
    fn short_sequences_pass_through() {
        assert_eq!(kalman_smooth(&[1.0, 2.0], 100.0, 20, true), vec![1.0, 2.0]);
    }

    #[test]
    fn inverse_is_correct() {
        let m: M3 = [[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]];
        let id = matmul(&m, &inverse(&m));
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((id[i][j] - want).abs() < 1e-12);
            }
        }
    }
}
