//! Marker picking by CPU projection (Phase 0 finding: exact, < 20 µs for a
//! thousand markers, no GPU readback latency).

use glam::Mat4;
use mstudio_core::Take;

use crate::camera::project;

/// The marker whose projected centre is within `radius_px` of `pointer`,
/// nearest to the camera first, then nearest to the pointer.
pub fn pick_marker(
    take: &Take,
    frame: usize,
    view_proj: Mat4,
    viewport: [f32; 2],
    pointer: [f32; 2],
    radius_px: f32,
) -> Option<usize> {
    if frame >= take.n_frames() {
        return None;
    }
    (0..take.n_markers())
        .filter_map(|m| {
            let p = take.position(frame, m)?;
            let (s, depth) = project(view_proj, viewport, [p[0] as f32, p[1] as f32, p[2] as f32])?;
            let d = ((s[0] - pointer[0]).powi(2) + (s[1] - pointer[1]).powi(2)).sqrt();
            (d <= radius_px).then_some((m, depth, d))
        })
        .min_by(|a, b| a.1.total_cmp(&b.1).then(a.2.total_cmp(&b.2)))
        .map(|(m, _, _)| m)
}

/// Distance in pixels from `pointer` to the projected segment `(a, b)`,
/// `None` if either end is off screen.
pub fn distance_to_segment_px(
    view_proj: Mat4,
    viewport: [f32; 2],
    a: [f32; 3],
    b: [f32; 3],
    pointer: [f32; 2],
) -> Option<f32> {
    let (sa, _) = project(view_proj, viewport, a)?;
    let (sb, _) = project(view_proj, viewport, b)?;
    let (dx, dy) = (sb[0] - sa[0], sb[1] - sa[1]);
    let len2 = dx * dx + dy * dy;
    let t = if len2 == 0.0 {
        0.0
    } else {
        (((pointer[0] - sa[0]) * dx + (pointer[1] - sa[1]) * dy) / len2).clamp(0.0, 1.0)
    };
    let (px, py) = (sa[0] + t * dx, sa[1] + t * dy);
    Some(((pointer[0] - px).powi(2) + (pointer[1] - py).powi(2)).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::Camera;
    use mstudio_core::CoordinateSystem;
    use ndarray::Array3;

    #[test]
    fn picks_the_nearest_marker_under_the_pointer() {
        let mut f = Array3::<f64>::zeros((1, 2, 3));
        f[[0, 1, 0]] = 0.5;
        let take = Take::new(vec!["a".into(), "b".into()], 10.0, f);
        let cam = Camera { target: glam::Vec3::ZERO, yaw: 0.0, pitch: 0.0, dist: 3.0, ..Camera::default() };
        let vp = cam.view_proj(1.0, CoordinateSystem::YUp);
        let (sb, _) = project(vp, [400.0, 400.0], [0.5, 0.0, 0.0]).unwrap();
        assert_eq!(pick_marker(&take, 0, vp, [400.0, 400.0], sb, 6.0), Some(1));
        assert_eq!(pick_marker(&take, 0, vp, [400.0, 400.0], [200.0, 200.0], 6.0), Some(0));
        assert_eq!(pick_marker(&take, 0, vp, [400.0, 400.0], [5.0, 5.0], 6.0), None);
        assert_eq!(pick_marker(&take, 3, vp, [400.0, 400.0], sb, 6.0), None);
        let d = distance_to_segment_px(vp, [400.0, 400.0], [0.0; 3], [0.5, 0.0, 0.0], [200.0, 210.0]).unwrap();
        assert!((d - 10.0).abs() < 1e-3);
    }
}
