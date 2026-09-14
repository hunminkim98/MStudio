//! Marker and skeleton appearance. Port of `core/marker_visual_settings.py`
//! (clamping ranges and the four color schemes are the same).

pub type Rgb = [f32; 3];

fn clamp_color(c: Rgb) -> Rgb {
    [c[0].clamp(0.0, 1.0), c[1].clamp(0.0, 1.0), c[2].clamp(0.0, 1.0)]
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MarkerVisualConfig {
    /// 1.0 … 20.0
    pub size: f32,
    pub color_normal: Rgb,
    pub color_selected: Rgb,
    pub color_pattern: Rgb,
    /// 0.1 … 1.0
    pub opacity: f32,
}

impl Default for MarkerVisualConfig {
    fn default() -> Self {
        Self {
            size: 5.0,
            color_normal: [1.0, 1.0, 1.0],
            color_selected: [1.0, 0.9, 0.4],
            color_pattern: [1.0, 0.0, 0.0],
            opacity: 1.0,
        }
    }
}

impl MarkerVisualConfig {
    pub fn clamped(mut self) -> Self {
        self.size = self.size.clamp(1.0, 20.0);
        self.opacity = self.opacity.clamp(0.1, 1.0);
        self.color_normal = clamp_color(self.color_normal);
        self.color_selected = clamp_color(self.color_selected);
        self.color_pattern = clamp_color(self.color_pattern);
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkeletonVisualConfig {
    /// 0.5 … 5.0
    pub line_width: f32,
    pub color_normal: Rgb,
    pub color_outlier: Rgb,
    /// 0.1 … 1.0
    pub opacity: f32,
}

impl Default for SkeletonVisualConfig {
    fn default() -> Self {
        Self { line_width: 2.0, color_normal: [0.7, 0.7, 0.7], color_outlier: [1.0, 0.0, 0.0], opacity: 0.8 }
    }
}

impl SkeletonVisualConfig {
    pub fn clamped(mut self) -> Self {
        self.line_width = self.line_width.clamp(0.5, 5.0);
        self.opacity = self.opacity.clamp(0.1, 1.0);
        self.color_normal = clamp_color(self.color_normal);
        self.color_outlier = clamp_color(self.color_outlier);
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorScheme {
    pub name: &'static str,
    pub marker_normal: Rgb,
    pub marker_selected: Rgb,
    pub marker_pattern: Rgb,
    pub skeleton_normal: Rgb,
    pub skeleton_outlier: Rgb,
}

pub const COLOR_SCHEMES: &[ColorScheme] = &[
    ColorScheme {
        name: "Default",
        marker_normal: [1.0, 1.0, 1.0],
        marker_selected: [1.0, 0.9, 0.4],
        marker_pattern: [1.0, 0.0, 0.0],
        skeleton_normal: [0.7, 0.7, 0.7],
        skeleton_outlier: [1.0, 0.0, 0.0],
    },
    ColorScheme {
        name: "Blue Theme",
        marker_normal: [0.7, 0.8, 1.0],
        marker_selected: [0.0, 0.5, 1.0],
        marker_pattern: [1.0, 0.0, 0.5],
        skeleton_normal: [0.5, 0.7, 1.0],
        skeleton_outlier: [1.0, 0.3, 0.3],
    },
    ColorScheme {
        name: "Green Theme",
        marker_normal: [0.7, 1.0, 0.7],
        marker_selected: [0.0, 1.0, 0.0],
        marker_pattern: [1.0, 0.5, 0.0],
        skeleton_normal: [0.5, 0.9, 0.5],
        skeleton_outlier: [1.0, 0.4, 0.0],
    },
    ColorScheme {
        name: "Warm Theme",
        marker_normal: [1.0, 0.9, 0.7],
        marker_selected: [1.0, 0.6, 0.0],
        marker_pattern: [1.0, 0.0, 0.0],
        skeleton_normal: [0.9, 0.7, 0.5],
        skeleton_outlier: [1.0, 0.2, 0.2],
    },
];

/// The settings object the UI edits and the renderer reads. Every setter
/// clamps and returns whether the value actually changed, so callers can
/// decide whether a redraw is needed (the Python version fired callbacks).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct VisualSettings {
    pub marker: MarkerVisualConfig,
    pub skeleton: SkeletonVisualConfig,
}

impl VisualSettings {
    pub fn new() -> Self {
        Self::default()
    }

    fn update<T: PartialEq + Copy>(slot: &mut T, value: T) -> bool {
        let changed = *slot != value;
        *slot = value;
        changed
    }

    pub fn set_marker_size(&mut self, size: f32) -> bool {
        Self::update(&mut self.marker.size, size.clamp(1.0, 20.0))
    }

    pub fn set_marker_opacity(&mut self, opacity: f32) -> bool {
        Self::update(&mut self.marker.opacity, opacity.clamp(0.1, 1.0))
    }

    pub fn set_marker_normal_color(&mut self, c: Rgb) -> bool {
        Self::update(&mut self.marker.color_normal, clamp_color(c))
    }

    pub fn set_marker_selected_color(&mut self, c: Rgb) -> bool {
        Self::update(&mut self.marker.color_selected, clamp_color(c))
    }

    pub fn set_marker_pattern_color(&mut self, c: Rgb) -> bool {
        Self::update(&mut self.marker.color_pattern, clamp_color(c))
    }

    pub fn set_skeleton_line_width(&mut self, w: f32) -> bool {
        Self::update(&mut self.skeleton.line_width, w.clamp(0.5, 5.0))
    }

    pub fn set_skeleton_opacity(&mut self, opacity: f32) -> bool {
        Self::update(&mut self.skeleton.opacity, opacity.clamp(0.1, 1.0))
    }

    pub fn set_skeleton_normal_color(&mut self, c: Rgb) -> bool {
        Self::update(&mut self.skeleton.color_normal, clamp_color(c))
    }

    pub fn set_skeleton_outlier_color(&mut self, c: Rgb) -> bool {
        Self::update(&mut self.skeleton.color_outlier, clamp_color(c))
    }

    pub fn scheme_names() -> impl Iterator<Item = &'static str> {
        COLOR_SCHEMES.iter().map(|s| s.name)
    }

    /// Apply a named scheme; `false` if the name is unknown.
    pub fn apply_scheme(&mut self, name: &str) -> bool {
        let Some(s) = COLOR_SCHEMES.iter().find(|s| s.name == name) else { return false };
        self.marker.color_normal = s.marker_normal;
        self.marker.color_selected = s.marker_selected;
        self.marker.color_pattern = s.marker_pattern;
        self.skeleton.color_normal = s.skeleton_normal;
        self.skeleton.color_outlier = s.skeleton_outlier;
        true
    }

    pub fn reset(&mut self) -> bool {
        Self::update(self, VisualSettings::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn setters_clamp_and_report_change() {
        let mut v = VisualSettings::new();
        assert!(v.set_marker_size(50.0));
        assert_eq!(v.marker.size, 20.0);
        assert!(!v.set_marker_size(25.0));
        assert!(v.set_skeleton_line_width(0.1));
        assert_eq!(v.skeleton.line_width, 0.5);
        assert!(v.set_marker_normal_color([2.0, -1.0, 0.5]));
        assert_eq!(v.marker.color_normal, [1.0, 0.0, 0.5]);
    }

    #[test]
    fn schemes_apply_and_reset() {
        let mut v = VisualSettings::new();
        assert!(v.apply_scheme("Blue Theme"));
        assert_eq!(v.marker.color_selected, [0.0, 0.5, 1.0]);
        assert_eq!(v.skeleton.color_outlier, [1.0, 0.3, 0.3]);
        assert!(!v.apply_scheme("Nope"));
        assert!(v.reset());
        assert_eq!(v, VisualSettings::default());
        assert_eq!(VisualSettings::scheme_names().count(), 4);
    }
}
