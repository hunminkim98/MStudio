//! Dark / light themes with MStudio's accent colour.

use eframe::egui::{self, Color32, Visuals};

pub const ACCENT: Color32 = Color32::from_rgb(255, 196, 61);

pub fn apply(ctx: &egui::Context, dark: bool) {
    let mut v = if dark { Visuals::dark() } else { Visuals::light() };
    v.selection.bg_fill = if dark { Color32::from_rgb(86, 70, 20) } else { Color32::from_rgb(255, 232, 160) };
    v.selection.stroke.color = ACCENT;
    v.hyperlink_color = ACCENT;
    v.widgets.hovered.bg_stroke.color = ACCENT;
    v.widgets.active.bg_stroke.color = ACCENT;
    if dark {
        v.panel_fill = Color32::from_rgb(28, 28, 31);
        v.window_fill = Color32::from_rgb(34, 34, 38);
        v.extreme_bg_color = Color32::from_rgb(18, 18, 20);
    }
    ctx.set_visuals(v);
}

/// Viewport clear colour (behind the 3D scene).
pub fn viewport_background(dark: bool) -> Color32 {
    if dark {
        Color32::from_rgb(14, 14, 16)
    } else {
        Color32::from_rgb(58, 60, 66)
    }
}
