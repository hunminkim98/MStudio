//! Orbit camera and the coordinate-system model transform.
//!
//! The data is never transformed (plan): Y-up is the identity, Z-up is a
//! rotation applied in the view chain so that the data's Z axis points up on
//! screen. The grid lives in view-up space.
#![allow(deprecated)] // glam 0.33 deprecates the Mat4 camera helpers; the replacements land with the app crate

use glam::{Mat4, Vec3, Vec4};
use mstudio_core::CoordinateSystem;

/// Model matrix taking data space to view-up space.
pub fn coordinate_model(cs: CoordinateSystem) -> Mat4 {
    match cs {
        CoordinateSystem::YUp => Mat4::IDENTITY,
        CoordinateSystem::ZUp => Mat4::from_rotation_x(-std::f32::consts::FRAC_PI_2),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Camera {
    /// Orbit centre, in view-up space.
    pub target: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub dist: f32,
    pub fov_y: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Camera {
            target: Vec3::ZERO,
            yaw: 0.6,
            pitch: 0.35,
            dist: 4.0,
            fov_y: 45f32.to_radians(),
            near: 0.01,
            far: 1000.0,
        }
    }
}

impl Camera {
    /// Frame an axis-aligned box (already in view-up space).
    pub fn fit_bounds(&mut self, lo: [f32; 3], hi: [f32; 3]) {
        let lo = Vec3::from(lo);
        let hi = Vec3::from(hi);
        self.target = (lo + hi) * 0.5;
        let radius = ((hi - lo).length() * 0.5).max(0.25);
        self.dist = radius / (self.fov_y * 0.5).tan() * 1.15;
    }

    pub fn eye(&self) -> Vec3 {
        self.target
            + self.dist
                * Vec3::new(self.pitch.cos() * self.yaw.sin(), self.pitch.sin(), self.pitch.cos() * self.yaw.cos())
    }

    pub fn view(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye(), self.target, Vec3::Y)
    }

    pub fn projection(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, aspect.max(1e-3), self.near, self.far)
    }

    /// `projection * view * coordinate_model` — what the shaders and the
    /// picker use on raw data positions.
    pub fn view_proj(&self, aspect: f32, cs: CoordinateSystem) -> Mat4 {
        self.projection(aspect) * self.view() * coordinate_model(cs)
    }

    /// Screen-space drag in pixels.
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * 0.008;
        self.pitch = (self.pitch + dy * 0.008).clamp(-1.55, 1.55);
    }

    /// Screen-space drag in pixels; moves the target in the view plane.
    pub fn pan(&mut self, dx: f32, dy: f32) {
        let forward = (self.target - self.eye()).normalize();
        let right = forward.cross(Vec3::Y).normalize();
        let up = right.cross(forward);
        let s = self.dist * 0.0015;
        self.target += (-right * dx + up * dy) * s;
    }

    /// Wheel delta (positive = zoom in); scale is exponential so it feels
    /// the same at every distance.
    pub fn zoom(&mut self, delta: f32) {
        self.dist = (self.dist * (-delta * 0.002).exp()).clamp(0.02, 5000.0);
    }
}

/// Screen position (pixels, origin top-left) and NDC depth of a data-space
/// point, `None` when behind the camera or outside the depth range.
pub fn project(view_proj: Mat4, viewport: [f32; 2], p: [f32; 3]) -> Option<([f32; 2], f32)> {
    let c = view_proj * Vec4::new(p[0], p[1], p[2], 1.0);
    if c.w <= 0.0 {
        return None;
    }
    let ndc = c.truncate() / c.w;
    if !(0.0..=1.0).contains(&ndc.z) {
        return None;
    }
    Some(([(ndc.x + 1.0) * 0.5 * viewport[0], (1.0 - ndc.y) * 0.5 * viewport[1]], ndc.z))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn z_up_maps_data_z_to_screen_up() {
        let m = coordinate_model(CoordinateSystem::ZUp);
        let v = m * Vec4::new(0.0, 0.0, 1.0, 1.0);
        assert!((v.y - 1.0).abs() < 1e-6 && v.z.abs() < 1e-6);
        assert_eq!(coordinate_model(CoordinateSystem::YUp), Mat4::IDENTITY);
    }

    #[test]
    fn fit_puts_target_at_centre_and_target_projects_to_screen_centre() {
        let mut c = Camera::default();
        c.fit_bounds([-1.0, 0.0, -1.0], [1.0, 2.0, 1.0]);
        assert_eq!(c.target, Vec3::new(0.0, 1.0, 0.0));
        let vp = c.view_proj(2.0, CoordinateSystem::YUp);
        let (s, depth) = project(vp, [800.0, 400.0], [0.0, 1.0, 0.0]).unwrap();
        assert!((s[0] - 400.0).abs() < 1e-3 && (s[1] - 200.0).abs() < 1e-3);
        assert!(depth > 0.0 && depth < 1.0);
        assert!(project(vp, [800.0, 400.0], c.eye().into()).is_none());
    }

    #[test]
    fn zoom_is_exponential_and_clamped() {
        let mut c = Camera::default();
        let d0 = c.dist;
        c.zoom(100.0);
        assert!(c.dist < d0);
        c.zoom(-100.0);
        assert!((c.dist - d0).abs() < 1e-5);
        c.zoom(-1e9);
        assert_eq!(c.dist, 5000.0);
    }
}
