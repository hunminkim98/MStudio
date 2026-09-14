//! Analysis-mode primitives, a port of `utils/analysisMode.py`.

use std::f64::consts::PI;

pub type Point = [f64; 3];

fn sub(a: Point, b: Point) -> Point {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: Point, b: Point) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: Point) -> f64 {
    dot(a, a).sqrt()
}

fn cross(a: Point, b: Point) -> Point {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// `np.isclose(x, target)` with numpy's default tolerances.
fn isclose(x: f64, target: f64) -> bool {
    (x - target).abs() <= 1e-8 + 1e-5 * target.abs()
}

/// Euclidean distance between two markers (meters).
pub fn distance(a: Point, b: Point) -> f64 {
    norm(sub(a, b))
}

/// Angle at `vertex` between `p1` and `p3`, in degrees; `None` for a
/// zero-length arm.
pub fn joint_angle(p1: Point, vertex: Point, p3: Point) -> Option<f64> {
    let a = sub(p1, vertex);
    let b = sub(p3, vertex);
    let (na, nb) = (norm(a), norm(b));
    if na == 0.0 || nb == 0.0 {
        return None;
    }
    let cos = (dot(a, b) / (na * nb)).clamp(-1.0, 1.0);
    Some(cos.acos().to_degrees())
}

/// Angle in degrees between a segment and a reference axis direction.
pub fn segment_angle(from: Point, to: Point, axis: Point) -> Option<f64> {
    let v = sub(to, from);
    let (nv, na) = (norm(v), norm(axis));
    if nv == 0.0 || na == 0.0 {
        return None;
    }
    Some((dot(v, axis) / (nv * na)).clamp(-1.0, 1.0).acos().to_degrees())
}

/// Points along the arc that marks the joint angle in the 3D view
/// (`radius` from the vertex, `segments + 1` points). `None` when the arms
/// are collinear.
pub fn arc_points(vertex: Point, p1: Point, p3: Point, radius: f64, segments: usize) -> Option<Vec<Point>> {
    let v1 = sub(p1, vertex);
    let v3 = sub(p3, vertex);
    let (n1, n3) = (norm(v1), norm(v3));
    if n1 == 0.0 || n3 == 0.0 {
        return None;
    }
    let u1 = [v1[0] / n1, v1[1] / n1, v1[2] / n1];
    let u3 = [v3[0] / n3, v3[1] / n3, v3[2] / n3];
    let angle = dot(u1, u3).clamp(-1.0, 1.0).acos();
    if isclose(angle, 0.0) || isclose(angle, PI) {
        return None;
    }
    let c = cross(u1, u3);
    let nc = norm(c);
    if isclose(nc, 0.0) {
        return None;
    }
    let normal = [c[0] / nc, c[1] / nc, c[2] / nc];
    let a2 = cross(normal, u1);
    let na2 = norm(a2);
    if isclose(na2, 0.0) {
        return None;
    }
    let a2 = [a2[0] / na2, a2[1] / na2, a2[2] / na2];
    Some(
        (0..=segments)
            .map(|i| {
                let phi = angle * (i as f64 / segments as f64);
                let (s, c) = phi.sin_cos();
                [
                    vertex[0] + radius * (u1[0] * c + a2[0] * s),
                    vertex[1] + radius * (u1[1] * c + a2[1] * s),
                    vertex[2] + radius * (u1[2] * c + a2[2] * s),
                ]
            })
            .collect(),
    )
}

pub const ARC_RADIUS: f64 = 0.05;
pub const ARC_SEGMENTS: usize = 20;

/// Central-difference velocity at frame `i` from frames `i−1` and `i+1`.
pub fn velocity(prev: Point, next: Point, frame_rate: f64) -> Option<Point> {
    if frame_rate <= 0.0 || prev.iter().chain(next.iter()).any(|v| v.is_nan()) {
        return None;
    }
    let dt = 1.0 / frame_rate;
    let d = sub(next, prev);
    Some([d[0] / (2.0 * dt), d[1] / (2.0 * dt), d[2] / (2.0 * dt)])
}

/// Central-difference acceleration from the velocities at `i−1` and `i+1`.
pub fn acceleration(vel_prev: Point, vel_next: Point, frame_rate: f64) -> Option<Point> {
    velocity(vel_prev, vel_next, frame_rate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn right_angle_and_distance() {
        let o = [0.0, 0.0, 0.0];
        assert_eq!(distance(o, [3.0, 4.0, 0.0]), 5.0);
        let a = joint_angle([1.0, 0.0, 0.0], o, [0.0, 1.0, 0.0]).unwrap();
        assert!((a - 90.0).abs() < 1e-12);
        assert!(joint_angle(o, o, [1.0, 0.0, 0.0]).is_none());
        assert!((segment_angle(o, [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]).unwrap() - 45.0).abs() < 1e-12);
    }

    #[test]
    fn arc_starts_on_first_arm_and_ends_on_second() {
        let o = [0.0, 0.0, 0.0];
        let arc = arc_points(o, [2.0, 0.0, 0.0], [0.0, 3.0, 0.0], 0.05, 20).unwrap();
        assert_eq!(arc.len(), 21);
        assert!((arc[0][0] - 0.05).abs() < 1e-12 && arc[0][1].abs() < 1e-12);
        assert!(arc[20][0].abs() < 1e-12 && (arc[20][1] - 0.05).abs() < 1e-12);
        assert!(arc_points(o, [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], 0.05, 20).is_none());
    }

    #[test]
    fn central_differences() {
        let v = velocity([0.0, 0.0, 0.0], [2.0, 0.0, 0.0], 100.0).unwrap();
        assert_eq!(v, [100.0, 0.0, 0.0]);
        assert!(velocity([f64::NAN, 0.0, 0.0], [1.0, 0.0, 0.0], 100.0).is_none());
        assert_eq!(acceleration([0.0; 3], [1.0, 0.0, 0.0], 50.0).unwrap(), [25.0, 0.0, 0.0]);
    }
}
