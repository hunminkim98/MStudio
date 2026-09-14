//! Ground grid and axis triad, in view-up space (Y up).

use crate::segments::Segment;

pub const GRID_COLOR: [f32; 4] = [0.24, 0.24, 0.26, 1.0];
pub const GRID_MAJOR_COLOR: [f32; 4] = [0.34, 0.34, 0.37, 1.0];

/// `half_extent` metres each side, one line per `step`, every fifth line
/// brighter; plus X (red) / Y (green) / Z (blue) axes of `axis_len` at the
/// origin.
pub fn grid_segments(ground_y: f32, half_extent: f32, step: f32, axis_len: f32) -> Vec<Segment> {
    let mut v = Vec::new();
    let n = (half_extent / step).round() as i32;
    for i in -n..=n {
        let a = i as f32 * step;
        let (w, c) = if i % 5 == 0 { (1.5, GRID_MAJOR_COLOR) } else { (1.0, GRID_COLOR) };
        v.push(Segment::new([a, ground_y, -half_extent], [a, ground_y, half_extent], w, c));
        v.push(Segment::new([-half_extent, ground_y, a], [half_extent, ground_y, a], w, c));
    }
    let o = [0.0, ground_y, 0.0];
    v.push(Segment::new(o, [axis_len, ground_y, 0.0], 2.5, [0.9, 0.25, 0.25, 1.0]));
    v.push(Segment::new(o, [0.0, ground_y + axis_len, 0.0], 2.5, [0.3, 0.9, 0.3, 1.0]));
    v.push(Segment::new(o, [0.0, ground_y, axis_len], 2.5, [0.35, 0.45, 1.0, 1.0]));
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_and_axes() {
        let g = grid_segments(0.0, 10.0, 1.0, 0.5);
        assert_eq!(g.len(), 21 * 2 + 3);
        let x_axis = g[g.len() - 3];
        assert_eq!(x_axis.b[0], 0.5);
        assert_eq!(x_axis.a[3], 2.5);
    }
}
