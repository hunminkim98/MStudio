//! Bone-length outlier detection. Port of `core/outlier_detector.py`.
//!
//! For every skeleton pair the segment length is computed per frame; a frame
//! whose length changed by more than `threshold` (relative to the previous
//! frame) flags **both** markers of the pair at that frame. Frames with a
//! missing sample never compare (`NaN > t` is false), exactly as in numpy.

use ndarray::{Array2, Array3, Axis};
use rayon::prelude::*;

pub const DEFAULT_OUTLIER_THRESHOLD: f64 = 0.3;

/// Returns `[n_markers, n_frames]` flags.
pub fn detect_outliers(frames: &Array3<f64>, pairs: &[(usize, usize)], threshold: f64) -> Array2<bool> {
    let n_frames = frames.len_of(Axis(0));
    let n_markers = frames.len_of(Axis(1));
    let mut out = Array2::from_elem((n_markers, n_frames), false);
    if pairs.is_empty() || n_frames < 2 {
        return out;
    }

    // Each pair is independent; collect flagged frames per pair, then merge in
    // order so the result is deterministic regardless of thread scheduling.
    let flagged: Vec<Vec<usize>> = pairs
        .par_iter()
        .map(|&(parent, child)| {
            let len = |f: usize| {
                let mut s = 0.0;
                for k in 0..3 {
                    let d = frames[[f, child, k]] - frames[[f, parent, k]];
                    s += d * d;
                }
                s.sqrt()
            };
            let mut prev = len(0);
            let mut hits = Vec::new();
            for f in 1..n_frames {
                let cur = len(f);
                let change = (cur - prev).abs() / (prev + 1e-8);
                if change > threshold {
                    hits.push(f);
                }
                prev = cur;
            }
            hits
        })
        .collect();

    for (&(parent, child), hits) in pairs.iter().zip(flagged) {
        for f in hits {
            out[[parent, f]] = true;
            out[[child, f]] = true;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_both_markers_at_the_frame_after_a_jump() {
        // marker 0 fixed at origin; marker 1 at x = 1, 1, 2 (jump between frame 1 and 2)
        let mut f = Array3::<f64>::zeros((3, 2, 3));
        f[[0, 1, 0]] = 1.0;
        f[[1, 1, 0]] = 1.0;
        f[[2, 1, 0]] = 2.0;
        let o = detect_outliers(&f, &[(0, 1)], 0.3);
        assert_eq!(o.row(0).to_vec(), vec![false, false, true]);
        assert_eq!(o.row(1).to_vec(), vec![false, false, true]);
    }

    #[test]
    fn nan_never_flags() {
        let mut f = Array3::<f64>::zeros((3, 2, 3));
        f[[0, 1, 0]] = 1.0;
        f[[1, 1, 0]] = f64::NAN;
        f[[2, 1, 0]] = 5.0;
        let o = detect_outliers(&f, &[(0, 1)], 0.3);
        assert!(!o.iter().any(|&b| b));
    }

    #[test]
    fn empty_pairs_gives_all_false() {
        let f = Array3::<f64>::zeros((5, 3, 3));
        let o = detect_outliers(&f, &[], 0.3);
        assert_eq!(o.dim(), (3, 5));
        assert!(!o.iter().any(|&b| b));
    }
}
