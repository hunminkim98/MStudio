//! The 3D viewport: an egui paint callback around `mstudio_render::Renderer`,
//! camera input, picking, labels.

use std::sync::{Arc, Mutex};

use eframe::egui::{self, Align2, Color32, FontId};
use eframe::egui_wgpu::{self, wgpu};
use mstudio_core::{ReferenceAxis, Take, VisualSettings};
use mstudio_render::{
    analysis_overlay, distance_to_segment_px, pick_marker, project, trajectory_segments, Camera, FrameParams,
    MarkerState, Renderer, Segment,
};

pub const TRAJECTORY_WINDOW: usize = 10;
const PICK_RADIUS_PX: f32 = 8.0;

/// Shared with the render callback.
pub struct SharedRenderer {
    pub renderer: Arc<Mutex<Renderer>>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl SharedRenderer {
    pub fn new(rs: &egui_wgpu::RenderState, msaa: u32) -> Self {
        let renderer = Renderer::new(
            &rs.device,
            mstudio_render::RenderConfig {
                color_format: rs.target_format,
                depth_format: Some(wgpu::TextureFormat::Depth24Plus),
                msaa_samples: msaa,
            },
        );
        Self { renderer: Arc::new(Mutex::new(renderer)), device: rs.device.clone(), queue: rs.queue.clone() }
    }
}

struct PaintCallback {
    renderer: Arc<Mutex<Renderer>>,
}

impl egui_wgpu::CallbackTrait for PaintCallback {
    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        pass: &mut wgpu::RenderPass<'static>,
        _res: &egui_wgpu::CallbackResources,
    ) {
        if let Ok(r) = self.renderer.lock() {
            r.draw(pass);
        }
    }
}

/// What the app tells the viewport each frame.
pub struct ViewportInput<'a> {
    pub take: Option<&'a Take>,
    pub frame: usize,
    pub selected: Option<usize>,
    pub pattern_markers: &'a [usize],
    pub analysis_markers: &'a [usize],
    pub analysis_mode: bool,
    pub reference_axis: ReferenceAxis,
    pub show_names: bool,
    pub show_trajectory: bool,
    pub show_skeleton: bool,
    pub show_grid: bool,
    pub coordinate_system: mstudio_core::CoordinateSystem,
    pub visual: &'a VisualSettings,
    pub dark: bool,
}

#[derive(Debug, Default)]
pub struct ViewportOutput {
    pub picked: Option<usize>,
    pub clicked_empty: bool,
    pub cycle_reference_axis: bool,
    /// Analysis readouts to show in the panel.
    pub analysis_labels: Vec<String>,
}

pub struct Viewport {
    pub camera: Camera,
    pub shared: SharedRenderer,
    last_states: Vec<u32>,
}

impl Viewport {
    pub fn new(shared: SharedRenderer) -> Self {
        Viewport { camera: Camera::default(), shared, last_states: Vec::new() }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, input: ViewportInput<'_>) -> ViewportOutput {
        let mut out = ViewportOutput::default();
        let rect = ui.available_rect_before_wrap();
        let response = ui.allocate_rect(rect, egui::Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, crate::theme::viewport_background(input.dark));

        // camera input
        let d = response.drag_delta();
        if response.dragged_by(egui::PointerButton::Primary) {
            self.camera.orbit(d.x, d.y);
        } else if response.dragged_by(egui::PointerButton::Secondary)
            || response.dragged_by(egui::PointerButton::Middle)
        {
            self.camera.pan(d.x, d.y);
        }
        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                self.camera.zoom(scroll);
            }
        }

        let ppp = ui.ctx().pixels_per_point();
        let viewport_px = [rect.width() * ppp, rect.height() * ppp];
        let aspect = rect.width().max(1.0) / rect.height().max(1.0);
        let vp = self.camera.view_proj(aspect, input.coordinate_system);
        let to_screen = |p: [f32; 3]| {
            project(vp, viewport_px, p).map(|(s, z)| (egui::pos2(rect.min.x + s[0] / ppp, rect.min.y + s[1] / ppp), z))
        };

        let Some(take) = input.take else {
            painter.text(
                rect.center(),
                Align2::CENTER_CENTER,
                "Open a TRC / C3D file or a JSON folder",
                FontId::proportional(16.0),
                Color32::from_gray(150),
            );
            return out;
        };
        let frame = input.frame.min(take.n_frames().saturating_sub(1));

        // overlay geometry (small; rebuilt every frame)
        let mut segments: Vec<Segment> = Vec::new();
        if input.show_trajectory {
            if let Some(m) = input.selected {
                let c = input.visual.marker.color_selected;
                segments.extend(trajectory_segments(take, m, frame, TRAJECTORY_WINDOW, c, 2.0));
            }
        }
        let mut overlay_labels: Vec<(String, [f32; 3])> = Vec::new();
        let mut reference_line = None;
        if input.analysis_mode && !input.analysis_markers.is_empty() {
            let ov = analysis_overlay(
                take,
                frame,
                input.analysis_markers,
                input.reference_axis,
                input.visual.skeleton.line_width.max(1.5),
            );
            segments.extend(ov.segments);
            overlay_labels = ov.labels;
            reference_line = ov.reference_line;
            out.analysis_labels = overlay_labels.iter().map(|(t, _)| t.clone()).collect();
        }

        // click: reference line first, then marker picking
        if response.clicked_by(egui::PointerButton::Primary) {
            if let Some(pos) = response.interact_pointer_pos() {
                let px = [(pos.x - rect.min.x) * ppp, (pos.y - rect.min.y) * ppp];
                let on_reference = reference_line
                    .and_then(|(a, b)| distance_to_segment_px(vp, viewport_px, a, b, px))
                    .is_some_and(|dist| dist <= 6.0 * ppp);
                if on_reference {
                    out.cycle_reference_axis = true;
                } else {
                    match pick_marker(take, frame, vp, viewport_px, px, PICK_RADIUS_PX * ppp) {
                        Some(m) => out.picked = Some(m),
                        None => out.clicked_empty = true,
                    }
                }
            }
        }

        // per-marker states (only re-upload when changed)
        let mut states = vec![MarkerState::Normal; take.n_markers()];
        for &m in input.pattern_markers {
            if m < states.len() {
                states[m] = MarkerState::Pattern;
            }
        }
        for &m in input.analysis_markers {
            if m < states.len() {
                states[m] = MarkerState::Analysis;
            }
        }
        if let Some(m) = input.selected {
            if m < states.len() {
                states[m] = MarkerState::Selected;
            }
        }
        let words: Vec<u32> = states.iter().map(|s| *s as u32).collect();

        {
            let mut r = self.shared.renderer.lock().unwrap();
            if words != self.last_states {
                r.set_marker_states(&self.shared.queue, &states);
                self.last_states = words;
            }
            r.set_overlay(&self.shared.device, &self.shared.queue, &segments);
            r.prepare(
                &self.shared.device,
                &self.shared.queue,
                &FrameParams {
                    view_proj: vp,
                    viewport: viewport_px,
                    frame,
                    selected: input.selected,
                    visual: input.visual,
                    show_skeleton: input.show_skeleton,
                    show_grid: input.show_grid,
                },
            );
        }
        painter.add(egui_wgpu::Callback::new_paint_callback(
            rect,
            PaintCallback { renderer: self.shared.renderer.clone() },
        ));

        // labels
        if input.show_names {
            let font = FontId::proportional(11.0);
            for m in 0..take.n_markers() {
                let Some(p) = take.position(frame, m) else { continue };
                let Some((s, _)) = to_screen([p[0] as f32, p[1] as f32, p[2] as f32]) else { continue };
                if !rect.contains(s) {
                    continue;
                }
                let color = if input.selected == Some(m) { crate::theme::ACCENT } else { Color32::from_gray(215) };
                painter.text(s + egui::vec2(6.0, -4.0), Align2::LEFT_BOTTOM, &take.markers[m], font.clone(), color);
            }
        }
        for (text, p) in &overlay_labels {
            if let Some((s, _)) = to_screen(*p) {
                let galley = painter.layout_no_wrap(text.clone(), FontId::proportional(13.0), Color32::WHITE);
                let r = egui::Rect::from_min_size(s + egui::vec2(8.0, -8.0), galley.size() + egui::vec2(8.0, 4.0));
                painter.rect_filled(r, 3.0, Color32::from_black_alpha(160));
                painter.galley(r.min + egui::vec2(4.0, 2.0), galley, Color32::WHITE);
            }
        }
        out
    }

    /// Frame the markers of `frame` (falls back to the whole take when that
    /// frame has no valid sample), bounds transformed into view-up space.
    pub fn fit(&mut self, take: &Take, cs: mstudio_core::CoordinateSystem, frame: usize) {
        let frame_bounds = || {
            let mut lo = [f64::INFINITY; 3];
            let mut hi = [f64::NEG_INFINITY; 3];
            for m in 0..take.n_markers() {
                if let Some(p) = take.position(frame, m) {
                    for k in 0..3 {
                        lo[k] = lo[k].min(p[k]);
                        hi[k] = hi[k].max(p[k]);
                    }
                }
            }
            lo[0].is_finite().then_some((lo, hi))
        };
        if let Some((lo, hi)) = frame_bounds().or_else(|| take.bounds()) {
            let m = mstudio_render::coordinate_model(cs);
            let corners = [
                [lo[0], lo[1], lo[2]],
                [hi[0], lo[1], lo[2]],
                [lo[0], hi[1], lo[2]],
                [lo[0], lo[1], hi[2]],
                [hi[0], hi[1], lo[2]],
                [hi[0], lo[1], hi[2]],
                [lo[0], hi[1], hi[2]],
                [hi[0], hi[1], hi[2]],
            ];
            let mut vlo = [f32::MAX; 3];
            let mut vhi = [f32::MIN; 3];
            for c in corners {
                let v = m * glam::Vec4::new(c[0] as f32, c[1] as f32, c[2] as f32, 1.0);
                for k in 0..3 {
                    vlo[k] = vlo[k].min(v[k]);
                    vhi[k] = vhi[k].max(v[k]);
                }
            }
            self.camera.fit_bounds(vlo, vhi);
        }
    }
}
