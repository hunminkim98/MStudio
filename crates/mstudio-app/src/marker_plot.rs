//! X / Y / Z coordinate plots of the selected marker (port of
//! `gui/markerPlot.py`): current-frame line, drag to select a frame range,
//! click to seek, double-click to clear the selection.

use eframe::egui::{self, Color32};
use egui_plot::{Line, Plot, PlotPoints, Polygon, VLine};
use mstudio_core::Take;

#[derive(Debug, Default)]
pub struct MarkerPlotState {
    drag_anchor: Option<usize>,
    /// Marker the cached series belong to (rebuilt when it changes or data is edited).
    cached_for: Option<(usize, u64)>,
    series: [Vec<[f64; 2]>; 3],
}

#[derive(Debug, Default)]
pub struct MarkerPlotOutput {
    pub seek: Option<usize>,
    pub selection: Option<Option<(usize, usize)>>,
}

pub struct MarkerPlotInput<'a> {
    pub take: &'a Take,
    pub marker: Option<usize>,
    pub frame: usize,
    pub selection: Option<(usize, usize)>,
    /// Bumped by the app whenever the take's data changes.
    pub data_version: u64,
    pub state: &'a mut MarkerPlotState,
}

const AXES: [(&str, Color32); 3] = [
    ("X", Color32::from_rgb(240, 90, 90)),
    ("Y", Color32::from_rgb(90, 220, 110)),
    ("Z", Color32::from_rgb(100, 150, 255)),
];

pub fn marker_plot(ui: &mut egui::Ui, input: MarkerPlotInput<'_>) -> MarkerPlotOutput {
    let mut out = MarkerPlotOutput::default();
    let Some(marker) = input.marker else {
        ui.centered_and_justified(|ui| ui.weak("Click a marker in the 3D view to plot its coordinates"));
        return out;
    };
    if input.state.cached_for != Some((marker, input.data_version)) {
        for (axis, series) in input.state.series.iter_mut().enumerate() {
            series.clear();
            series.extend((0..input.take.n_frames()).filter_map(|f| {
                let v = input.take.frames[[f, marker, axis]];
                (!v.is_nan()).then_some([f as f64, v])
            }));
        }
        input.state.cached_for = Some((marker, input.data_version));
    }

    let n = input.take.n_frames();
    ui.horizontal(|ui| {
        ui.strong(&input.take.markers[marker]);
        ui.weak("drag = select range · click = seek · double-click = clear · ctrl-scroll = zoom");
    });
    // three equal plots; the last one carries the x axis labels
    let height = ((ui.available_height() - 3.0 * ui.spacing().item_spacing.y - 34.0) / 3.0).max(40.0);
    for (axis, (name, color)) in AXES.iter().enumerate() {
        let series = &input.state.series[axis];
        let (mut lo, mut hi) =
            series.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), p| (lo.min(p[1]), hi.max(p[1])));
        if !lo.is_finite() {
            lo = -1.0;
            hi = 1.0;
        }
        let pad = ((hi - lo) * 0.08).max(1e-3);
        let (lo, hi) = (lo - pad, hi + pad);
        let last = axis == 2;
        let plot = Plot::new(("marker_plot", axis))
            .height(if last { height + 20.0 } else { height })
            .allow_drag(false)
            .allow_boxed_zoom(false)
            .allow_scroll(false)
            .allow_zoom([true, true])
            .allow_double_click_reset(false)
            .show_grid(true)
            .show_axes([last, true])
            .y_axis_min_width(52.0)
            .include_x(0.0)
            .include_x(n.saturating_sub(1) as f64)
            .include_y(lo)
            .include_y(hi);
        let response = plot.show(ui, |plot_ui| {
            if let Some((a, b)) = input.selection {
                let poly = Polygon::new(
                    "selection",
                    PlotPoints::from(vec![[a as f64, lo], [b as f64, lo], [b as f64, hi], [a as f64, hi]]),
                )
                .fill_color(crate::theme::ACCENT.gamma_multiply(0.18))
                .stroke(egui::Stroke::NONE);
                plot_ui.polygon(poly);
            }
            plot_ui.line(Line::new(*name, PlotPoints::from(series.clone())).color(*color).width(1.5));
            plot_ui.vline(VLine::new("frame", input.frame as f64).color(crate::theme::ACCENT).width(1.5));

            // interaction (plot coordinates → frame)
            let r = plot_ui.response().clone();
            if let Some(p) = plot_ui.pointer_coordinate() {
                let f = (p.x.round().max(0.0) as usize).min(n.saturating_sub(1));
                if r.drag_started_by(egui::PointerButton::Primary) {
                    // store anchor in a temporary; applied below through the state
                    plot_ui.ctx().data_mut(|d| d.insert_temp(egui::Id::new("mplot_anchor"), f));
                }
                if r.dragged_by(egui::PointerButton::Primary) {
                    if let Some(a) = plot_ui.ctx().data(|d| d.get_temp::<usize>(egui::Id::new("mplot_anchor"))) {
                        out.selection = Some(Some((a.min(f), a.max(f))));
                    }
                } else if r.clicked_by(egui::PointerButton::Primary) {
                    out.seek = Some(f);
                }
            }
            if r.double_clicked() {
                out.selection = Some(None);
            }
        });
        let _ = response;
    }
    let _ = &mut input.state.drag_anchor;
    out
}
