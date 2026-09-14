//! Trajectory of one marker around the current frame, as thick segments.

use mstudio_core::Take;

use crate::segments::Segment;

/// Segments joining consecutive valid samples in `[frame − window, frame + window]`;
/// the past is drawn at full colour, the future dimmed.
pub fn trajectory_segments(
    take: &Take,
    marker: usize,
    frame: usize,
    window: usize,
    color: [f32; 3],
    width_px: f32,
) -> Vec<Segment> {
    let n = take.n_frames();
    if n == 0 || marker >= take.n_markers() {
        return Vec::new();
    }
    let lo = frame.saturating_sub(window);
    let hi = (frame + window).min(n - 1);
    let mut out = Vec::with_capacity(hi - lo);
    let mut prev: Option<(usize, [f32; 3])> = None;
    for f in lo..=hi {
        if let Some(p) = take.position(f, marker) {
            let p = [p[0] as f32, p[1] as f32, p[2] as f32];
            if let Some((pf, pp)) = prev {
                if pf + 1 == f {
                    let t = if f <= frame { 1.0 } else { 0.45 };
                    out.push(Segment::new(pp, p, width_px, [color[0] * t, color[1] * t, color[2] * t, 1.0]));
                }
            }
            prev = Some((f, p));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn gaps_break_the_line_and_future_is_dimmed() {
        let mut f = Array3::<f64>::zeros((10, 1, 3));
        for i in 0..10 {
            f[[i, 0, 0]] = i as f64;
        }
        f[[7, 0, 0]] = f64::NAN;
        let take = Take::new(vec!["m".into()], 10.0, f);
        let s = trajectory_segments(&take, 0, 5, 3, [1.0, 1.0, 1.0], 2.0);
        // frames 2..=8 valid except 7: pairs (2,3)(3,4)(4,5)(5,6) → 4 segments; (6,7),(7,8) dropped
        assert_eq!(s.len(), 4);
        assert_eq!(s[2].color[0], 1.0); // (4,5) past
        assert_eq!(s[3].color[0], 0.45); // (5,6) future
        assert!(trajectory_segments(&take, 5, 5, 3, [1.0; 3], 2.0).is_empty());
    }
}
