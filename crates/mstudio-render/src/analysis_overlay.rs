//! Geometry for analysis mode: the segment between two markers with the
//! reference axis, or the two arms of a three-marker joint with its angle
//! arc. Labels are left to the host (anchor points are returned).

use mstudio_core::{ReferenceAxis, Take};
use mstudio_processing::analysis;

use crate::segments::Segment;

pub const SEGMENT_COLOR: [f32; 4] = [1.0, 0.85, 0.2, 1.0];
pub const REFERENCE_COLOR: [f32; 4] = [0.4, 0.8, 1.0, 1.0];
pub const ARC_COLOR: [f32; 4] = [1.0, 0.5, 0.2, 1.0];

#[derive(Debug, Clone, PartialEq)]
pub struct AnalysisOverlay {
    pub segments: Vec<Segment>,
    /// Text to draw and where (data space).
    pub labels: Vec<(String, [f32; 3])>,
    /// Endpoints of the reference line, for hit-testing the axis-cycle click.
    pub reference_line: Option<([f32; 3], [f32; 3])>,
}

fn f32x3(p: [f64; 3]) -> [f32; 3] {
    [p[0] as f32, p[1] as f32, p[2] as f32]
}

/// Build the overlay for the selected analysis markers at `frame`.
pub fn analysis_overlay(
    take: &Take,
    frame: usize,
    markers: &[usize],
    axis: ReferenceAxis,
    line_width: f32,
) -> AnalysisOverlay {
    let mut out = AnalysisOverlay { segments: Vec::new(), labels: Vec::new(), reference_line: None };
    let pos = |m: usize| take.position(frame, m);
    match markers {
        [a, b] => {
            let (Some(pa), Some(pb)) = (pos(*a), pos(*b)) else { return out };
            let dist = analysis::distance(pa, pb);
            out.segments.push(Segment::new(f32x3(pa), f32x3(pb), line_width * 1.5, SEGMENT_COLOR));
            // reference axis from the first marker, as long as the segment
            let dir = axis.vector();
            let end = [pa[0] + dir[0] * dist, pa[1] + dir[1] * dist, pa[2] + dir[2] * dist];
            out.segments.push(Segment::new(f32x3(pa), f32x3(end), line_width, REFERENCE_COLOR));
            out.reference_line = Some((f32x3(pa), f32x3(end)));
            let mid = [(pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0, (pa[2] + pb[2]) / 2.0];
            out.labels.push((format!("{:.3} m", dist), f32x3(mid)));
            if let Some(angle) = analysis::segment_angle(pa, pb, dir) {
                out.labels.push((format!("{angle:.1}° vs {}", axis.label()), f32x3(end)));
            }
        }
        [a, v, c] => {
            let (Some(pa), Some(pv), Some(pc)) = (pos(*a), pos(*v), pos(*c)) else { return out };
            out.segments.push(Segment::new(f32x3(pv), f32x3(pa), line_width * 1.5, SEGMENT_COLOR));
            out.segments.push(Segment::new(f32x3(pv), f32x3(pc), line_width * 1.5, SEGMENT_COLOR));
            if let Some(arc) = analysis::arc_points(pv, pa, pc, analysis::ARC_RADIUS, analysis::ARC_SEGMENTS) {
                for w in arc.windows(2) {
                    out.segments.push(Segment::new(f32x3(w[0]), f32x3(w[1]), line_width, ARC_COLOR));
                }
                let mid = arc[arc.len() / 2];
                if let Some(angle) = analysis::joint_angle(pa, pv, pc) {
                    out.labels.push((format!("{angle:.1}°"), f32x3(mid)));
                }
            }
        }
        _ => {}
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    fn take() -> Take {
        let mut f = Array3::<f64>::zeros((1, 3, 3));
        f[[0, 1, 0]] = 1.0; // b at (1,0,0)
        f[[0, 2, 1]] = 1.0; // c at (0,1,0)
        Take::new(vec!["a".into(), "v".into(), "c".into()], 10.0, f)
    }

    #[test]
    fn two_markers_give_segment_reference_and_labels() {
        let o = analysis_overlay(&take(), 0, &[0, 1], ReferenceAxis::Y, 2.0);
        assert_eq!(o.segments.len(), 2);
        assert_eq!(o.reference_line.unwrap().1, [0.0, 1.0, 0.0]);
        assert_eq!(o.labels[0].0, "1.000 m");
        assert!(o.labels[1].0.starts_with("90.0° vs Y"));
    }

    #[test]
    fn three_markers_give_arms_and_arc() {
        let o = analysis_overlay(&take(), 0, &[1, 0, 2], ReferenceAxis::X, 2.0);
        assert_eq!(o.segments.len(), 2 + analysis::ARC_SEGMENTS);
        assert_eq!(o.labels[0].0, "90.0°");
        assert!(analysis_overlay(&take(), 0, &[0], ReferenceAxis::X, 2.0).segments.is_empty());
    }
}
