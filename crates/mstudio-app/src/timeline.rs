//! Timeline strip: frame or time ticks, current-frame cursor, scrubbing and
//! range selection (port of `TRCViewer.update_timeline` and friends).

use eframe::egui::{self, Align2, Color32, FontId, Pos2, Rect, Sense, Stroke};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TimelineMode {
    #[default]
    Frames,
    Time,
}

#[derive(Debug, Default)]
pub struct TimelineState {
    pub mode: TimelineMode,
    drag_anchor: Option<usize>,
}

#[derive(Debug, Default)]
pub struct TimelineOutput {
    pub seek: Option<usize>,
    pub selection: Option<Option<(usize, usize)>>,
}

pub struct TimelineInput<'a> {
    pub n_frames: usize,
    pub fps: f64,
    pub frame: usize,
    pub selection: Option<(usize, usize)>,
    pub state: &'a mut TimelineState,
}

const HEIGHT: f32 = 58.0;
const PAD_X: f32 = 12.0;

pub fn timeline(ui: &mut egui::Ui, input: TimelineInput<'_>) -> TimelineOutput {
    let mut out = TimelineOutput::default();
    let (rect, response) = ui.allocate_exact_size(egui::vec2(ui.available_width(), HEIGHT), Sense::click_and_drag());
    let painter = ui.painter_at(rect);
    let dark = ui.visuals().dark_mode;
    painter.rect_filled(rect, 4.0, if dark { Color32::from_rgb(22, 22, 25) } else { Color32::from_rgb(235, 235, 238) });
    if input.n_frames == 0 {
        painter.text(
            rect.center(),
            Align2::CENTER_CENTER,
            "no data",
            FontId::proportional(12.0),
            ui.visuals().weak_text_color(),
        );
        return out;
    }

    let n = input.n_frames;
    let track = Rect::from_min_max(
        Pos2::new(rect.left() + PAD_X, rect.top() + 22.0),
        Pos2::new(rect.right() - PAD_X, rect.bottom() - 8.0),
    );
    let x_of = |frame: usize| track.left() + track.width() * (frame as f32 / (n.max(2) - 1) as f32);
    let frame_of = |x: f32| (((x - track.left()) / track.width()).clamp(0.0, 1.0) * (n - 1) as f32).round() as usize;

    // selection band
    if let Some((a, b)) = input.selection {
        let sel = Rect::from_min_max(Pos2::new(x_of(a), track.top()), Pos2::new(x_of(b), track.bottom()));
        painter.rect_filled(sel, 0.0, crate::theme::ACCENT.gamma_multiply(0.25));
    }

    // ticks
    let px_per_frame = track.width() / (n - 1).max(1) as f32;
    let tick_color = ui.visuals().weak_text_color();
    let label_color = ui.visuals().text_color();
    painter.line_segment([track.left_bottom(), track.right_bottom()], Stroke::new(1.0, tick_color));
    let font = FontId::proportional(10.0);
    match input.state.mode {
        TimelineMode::Frames => {
            let step = nice_step(60.0 / px_per_frame.max(1e-3));
            let mut f = 0usize;
            while f < n {
                let x = x_of(f);
                painter.line_segment(
                    [Pos2::new(x, track.bottom()), Pos2::new(x, track.bottom() - 8.0)],
                    Stroke::new(1.0, tick_color),
                );
                painter.text(
                    Pos2::new(x, track.top() - 2.0),
                    Align2::CENTER_BOTTOM,
                    f.to_string(),
                    font.clone(),
                    label_color,
                );
                f += step.max(1) as usize;
            }
        }
        TimelineMode::Time => {
            let total = (n - 1) as f64 / input.fps;
            let px_per_sec = track.width() as f64 / total.max(1e-6);
            let step = nice_step_f64(70.0 / px_per_sec);
            let mut t = 0.0;
            while t <= total + 1e-9 {
                let x = track.left() + (t / total.max(1e-9)) as f32 * track.width();
                painter.line_segment(
                    [Pos2::new(x, track.bottom()), Pos2::new(x, track.bottom() - 8.0)],
                    Stroke::new(1.0, tick_color),
                );
                painter.text(
                    Pos2::new(x, track.top() - 2.0),
                    Align2::CENTER_BOTTOM,
                    format!("{t:.2}s"),
                    font.clone(),
                    label_color,
                );
                t += step;
            }
        }
    }

    // cursor
    let cx = x_of(input.frame.min(n - 1));
    painter.line_segment(
        [Pos2::new(cx, track.top() - 14.0), Pos2::new(cx, track.bottom())],
        Stroke::new(2.0, crate::theme::ACCENT),
    );
    painter.text(
        Pos2::new(cx, track.bottom() + 1.0),
        Align2::CENTER_TOP,
        match input.state.mode {
            TimelineMode::Frames => format!("{}", input.frame),
            TimelineMode::Time => format!("{:.3}s", input.frame as f64 / input.fps),
        },
        FontId::proportional(10.0),
        crate::theme::ACCENT,
    );

    // interaction: drag = scrub, shift-drag / right-drag = range select
    let shift = ui.input(|i| i.modifiers.shift);
    if let Some(pos) = response.interact_pointer_pos() {
        let f = frame_of(pos.x);
        let selecting = shift
            || response.dragged_by(egui::PointerButton::Secondary)
            || response.drag_started_by(egui::PointerButton::Secondary);
        if selecting {
            if response.drag_started() {
                input.state.drag_anchor = Some(f);
            }
            if let Some(a) = input.state.drag_anchor {
                out.selection = Some(Some((a.min(f), a.max(f))));
            }
        } else if response.dragged_by(egui::PointerButton::Primary) || response.clicked_by(egui::PointerButton::Primary)
        {
            out.seek = Some(f);
        }
    }
    if response.drag_stopped() {
        input.state.drag_anchor = None;
    }
    if response.double_clicked() {
        out.selection = Some(None);
    }
    if let Some(pos) = response.hover_pos() {
        response.clone().on_hover_text(format!("frame {}", frame_of(pos.x)));
    }
    out
}

fn nice_step(min_step: f32) -> u32 {
    let candidates = [1u32, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000, 10000, 20000, 50000];
    candidates.into_iter().find(|&c| c as f32 >= min_step).unwrap_or(100000)
}

fn nice_step_f64(min_step: f64) -> f64 {
    let candidates = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 300.0];
    candidates.into_iter().find(|&c| c >= min_step).unwrap_or(600.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steps_are_nice_numbers_at_least_min() {
        assert_eq!(nice_step(0.5), 1);
        assert_eq!(nice_step(3.0), 5);
        assert_eq!(nice_step(120.0), 200);
        assert_eq!(nice_step_f64(0.07), 0.1);
        assert_eq!(nice_step_f64(7.0), 10.0);
    }
}
