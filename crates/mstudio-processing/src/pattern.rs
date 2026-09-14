//! Pattern-based gap filling, a port of `dataProcessor.interpolate_with_pattern`.
//!
//! The missing target marker is reconstructed from reference markers using
//! their spatial relationship in the nearest frame where target and all
//! references are valid:
//! * 1 reference: constant offset,
//! * 2 references: offset rotated with the reference segment, scaled by its
//!   length change,
//! * 3+ references: offset in the frame of the reference cloud (Kabsch).
//!
//! The rotation is estimated from the initial to the current reference
//! configuration and applied to the initial offset. (MStudio v0.1.5 had the
//! arguments of `Rotation.align_vectors` reversed, which rotated the offset
//! the wrong way; fixed in both implementations and the goldens regenerated.)

use mstudio_core::DirtyRange;
use ndarray::Array3;

use crate::rotation::{align_vectors, apply};
use crate::{ProcessingError, Result};

fn pos(frames: &Array3<f64>, f: usize, m: usize) -> Option<[f64; 3]> {
    let p = [frames[[f, m, 0]], frames[[f, m, 1]], frames[[f, m, 2]]];
    (!p.iter().any(|v| v.is_nan())).then_some(p)
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn norm(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

fn centroid(pts: &[[f64; 3]]) -> [f64; 3] {
    let n = pts.len() as f64;
    let s = pts.iter().fold([0.0; 3], |acc, p| add(acc, *p));
    [s[0] / n, s[1] / n, s[2] / n]
}

/// `np.isclose(x, 0)`.
fn is_zero(x: f64) -> bool {
    x.abs() <= 1e-8
}

/// Fill the target marker's missing frames inside the inclusive range.
/// Returns the range of frames written (empty if none).
pub fn pattern_interpolate(
    frames: &mut Array3<f64>,
    target: usize,
    references: &[usize],
    first: usize,
    last: usize,
) -> Result<DirtyRange> {
    if references.is_empty() {
        return Err(ProcessingError::InvalidParameter("select at least 1 reference marker".into()));
    }
    let n = frames.dim().0;
    let range = DirtyRange::inclusive(first.min(n.saturating_sub(1)), last.min(n.saturating_sub(1)));
    let refs_at = |frames: &Array3<f64>, f: usize| -> Option<Vec<[f64; 3]>> {
        references.iter().map(|&r| pos(frames, f, r)).collect()
    };

    // nearest frame (to either end of the range) where target and all refs are valid
    let (start, end) = (range.start as i64, (range.end - 1) as i64);
    let closest = (0..n)
        .filter(|&f| pos(frames, f, target).is_some() && refs_at(frames, f).is_some())
        .min_by_key(|&f| {
            let f = f as i64;
            (f - start).abs().min((f - end).abs())
        })
        .ok_or_else(|| {
            ProcessingError::Other(
                "No frame found where target and ALL selected reference markers have valid data simultaneously.".into(),
            )
        })?;

    let target_init = pos(frames, closest, target).unwrap();
    let refs_init = refs_at(frames, closest).unwrap();

    enum Mode {
        One { offset: [f64; 3] },
        Two { v_ref_init: [f64; 3], norm_init: f64, v_target_rel: [f64; 3] },
        Many { centered: Vec<[f64; 3]>, target_rel: [f64; 3] },
    }
    let mode = match references.len() {
        1 => Mode::One { offset: sub(target_init, refs_init[0]) },
        2 => {
            let v_ref_init = sub(refs_init[1], refs_init[0]);
            let norm_init = norm(v_ref_init);
            if is_zero(norm_init) {
                Mode::One { offset: sub(target_init, refs_init[0]) }
            } else {
                Mode::Two { v_ref_init, norm_init, v_target_rel: sub(target_init, refs_init[0]) }
            }
        }
        _ => {
            let c = centroid(&refs_init);
            Mode::Many { centered: refs_init.iter().map(|p| sub(*p, c)).collect(), target_rel: sub(target_init, c) }
        }
    };

    let mut touched = DirtyRange { start: 0, end: 0 };
    for f in range.start..range.end {
        if pos(frames, f, target).is_some() {
            continue;
        }
        let Some(refs) = refs_at(frames, f) else { continue };
        let est = match &mode {
            Mode::One { offset } => Some(add(refs[0], *offset)),
            Mode::Two { v_ref_init, norm_init, v_target_rel } => {
                let v_cur = sub(refs[1], refs[0]);
                let n_cur = norm(v_cur);
                if is_zero(n_cur) {
                    None
                } else {
                    let s = n_cur / norm_init;
                    align_vectors(&[v_cur], &[*v_ref_init]).map(|r| add(refs[0], apply(&r, scale(*v_target_rel, s))))
                }
            }
            Mode::Many { centered, target_rel } => {
                let c = centroid(&refs);
                let q: Vec<[f64; 3]> = refs.iter().map(|p| sub(*p, c)).collect();
                align_vectors(&q, centered).map(|r| add(apply(&r, *target_rel), c))
            }
        };
        if let Some(p) = est {
            for k in 0..3 {
                frames[[f, target, k]] = p[k];
            }
            touched = touched.union(DirtyRange::single(f));
        }
    }
    Ok(touched)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rigid body: reference triangle + target, translated and rotated per frame.
    fn rigid_take() -> Array3<f64> {
        let n = 30;
        let mut f = Array3::<f64>::zeros((n, 4, 3));
        let base = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.3, 0.4, 0.5]];
        for i in 0..n {
            let ang = i as f64 * 0.05;
            let (s, c) = ang.sin_cos();
            let t = [i as f64 * 0.01, 0.0, 0.02 * i as f64];
            for (m, p) in base.iter().enumerate() {
                let r = [c * p[0] - s * p[1] + t[0], s * p[0] + c * p[1] + t[1], p[2] + t[2]];
                for k in 0..3 {
                    f[[i, m, k]] = r[k];
                }
            }
        }
        f
    }

    #[test]
    fn three_references_recover_a_rigidly_moving_target() {
        let truth = rigid_take();
        let mut f = truth.clone();
        for i in 10..20 {
            for k in 0..3 {
                f[[i, 3, k]] = f64::NAN;
            }
        }
        let r = pattern_interpolate(&mut f, 3, &[0, 1, 2], 8, 22).unwrap();
        assert_eq!(r, DirtyRange { start: 10, end: 20 });
        for i in 10..20 {
            for k in 0..3 {
                assert!((f[[i, 3, k]] - truth[[i, 3, k]]).abs() < 1e-9, "frame {i}");
            }
        }
    }

    #[test]
    fn one_reference_uses_a_constant_offset() {
        let mut f = Array3::<f64>::zeros((5, 2, 3));
        for i in 0..5 {
            f[[i, 0, 0]] = i as f64;
            f[[i, 1, 0]] = i as f64 + 1.0;
        }
        f[[2, 1, 0]] = f64::NAN;
        pattern_interpolate(&mut f, 1, &[0], 0, 4).unwrap();
        assert_eq!(f[[2, 1, 0]], 3.0);
    }

    #[test]
    fn errors_without_a_common_valid_frame() {
        let mut f = Array3::<f64>::from_elem((3, 2, 3), f64::NAN);
        assert!(pattern_interpolate(&mut f, 1, &[0], 0, 2).is_err());
        assert!(pattern_interpolate(&mut f, 1, &[], 0, 2).is_err());
    }
}
