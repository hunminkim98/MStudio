//! `scipy.spatial.transform.Rotation.align_vectors(a, b)`: the rotation `R`
//! minimising Σ‖aᵢ − R bᵢ‖² (Kabsch), plus scipy's special case for a single
//! vector pair (shortest-arc rotation).

use nalgebra::{Matrix3, Vector3};

pub type Mat3 = [[f64; 3]; 3];

pub fn apply(r: &Mat3, v: [f64; 3]) -> [f64; 3] {
    [
        r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
        r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
        r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2],
    ]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

/// Rodrigues: rotation of `angle` radians about the unit `axis`.
fn from_axis_angle(axis: [f64; 3], angle: f64) -> Mat3 {
    let (s, c) = angle.sin_cos();
    let t = 1.0 - c;
    let [x, y, z] = axis;
    [
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ]
}

/// `None` if a set is empty, sizes differ, or (single pair) a vector is zero.
pub fn align_vectors(a: &[[f64; 3]], b: &[[f64; 3]]) -> Option<Mat3> {
    if a.is_empty() || a.len() != b.len() {
        return None;
    }
    if a.len() == 1 {
        return align_single(a[0], b[0]);
    }
    // Kabsch: B = Σ aᵢ bᵢᵀ ; R = U diag(1, 1, det(U Vᵀ)) Vᵀ
    let mut m = Matrix3::<f64>::zeros();
    for (ai, bi) in a.iter().zip(b) {
        let av = Vector3::new(ai[0], ai[1], ai[2]);
        let bv = Vector3::new(bi[0], bi[1], bi[2]);
        m += av * bv.transpose();
    }
    let svd = m.svd(true, true);
    let u = svd.u?;
    let vt = svd.v_t?;
    let mut c = u * vt;
    if c.determinant() < 0.0 {
        let mut u2 = u;
        for i in 0..3 {
            u2[(i, 2)] = -u2[(i, 2)];
        }
        c = u2 * vt;
    }
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = c[(i, j)];
        }
    }
    Some(r)
}

/// scipy's single-pair path: rotate unit `b` onto unit `a` about `b × a`.
fn align_single(a: [f64; 3], b: [f64; 3]) -> Option<Mat3> {
    let (na, nb) = (norm(a), norm(b));
    if na == 0.0 || nb == 0.0 {
        return None;
    }
    let a = [a[0] / na, a[1] / na, a[2] / na];
    let b = [b[0] / nb, b[1] / nb, b[2] / nb];
    let cr = cross(b, a);
    let cn = norm(cr);
    let theta = cn.atan2(dot(a, b));
    if cn == 0.0 {
        if theta < 1e-3 {
            return Some([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        }
        // antiparallel: 180° about an axis perpendicular to a (scipy picks the
        // component of smallest magnitude to build it)
        let i = (0..3).min_by(|&p, &q| a[p].abs().total_cmp(&a[q].abs())).unwrap();
        let r = match i {
            0 => [0.0, -a[2], a[1]],
            1 => [a[2], 0.0, -a[0]],
            _ => [-a[1], a[0], 0.0],
        };
        let rn = norm(r);
        return Some(from_axis_angle([r[0] / rn, r[1] / rn, r[2] / rn], std::f64::consts::PI));
    }
    Some(from_axis_angle([cr[0] / cn, cr[1] / cn, cr[2] / cn], theta))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f64; 3], b: [f64; 3]) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() < 1e-9)
    }

    #[test]
    fn single_pair_rotates_b_onto_a() {
        let a = [0.0, 2.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let r = align_vectors(&[a], &[b]).unwrap();
        assert!(close(apply(&r, [1.0, 0.0, 0.0]), [0.0, 1.0, 0.0]));
        // antiparallel
        let r = align_vectors(&[[1.0, 0.0, 0.0]], &[[-1.0, 0.0, 0.0]]).unwrap();
        assert!(close(apply(&r, [-1.0, 0.0, 0.0]), [1.0, 0.0, 0.0]));
    }

    #[test]
    fn kabsch_recovers_a_known_rotation() {
        let rot = from_axis_angle([0.0, 0.0, 1.0], 0.7);
        let b = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.5], [0.3, -0.2, 0.9]];
        let a: Vec<[f64; 3]> = b.iter().map(|v| apply(&rot, *v)).collect();
        let r = align_vectors(&a, &b).unwrap();
        for (ai, bi) in a.iter().zip(&b) {
            assert!(close(apply(&r, *bi), *ai));
        }
    }
}
