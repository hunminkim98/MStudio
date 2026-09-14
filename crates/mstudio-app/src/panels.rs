//! Right-hand tabs: Controls (file, playback, view, skeleton, appearance,
//! analysis, selection), Edit (edit mode, filters, interpolation, pattern
//! references, report) and Markers (list).

use eframe::egui::{self, RichText};
use mstudio_core::{skeleton, CoordinateSystem, VisualSettings};
use mstudio_processing::Filter;

use crate::app::{App, Command, FilterKind, INTERP_METHODS};
use crate::timeline::TimelineMode;

pub fn controls(ui: &mut egui::Ui, app: &mut App, cmds: &mut Vec<Command>) {
    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        ui.add_space(4.0);
        section(ui, "File", |ui| {
            ui.horizontal_wrapped(|ui| {
                if ui.button("Open…").clicked() {
                    cmds.push(Command::OpenFileDialog);
                }
                if ui.button("Open JSON folder…").clicked() {
                    cmds.push(Command::OpenFolderDialog);
                }
                if ui.add_enabled(app.doc.is_some(), egui::Button::new("Save As…")).clicked() {
                    cmds.push(Command::SaveAsDialog);
                }
            });
            if let Some(doc) = &app.doc {
                ui.label(
                    RichText::new(doc.path.as_ref().map_or("(unsaved)".to_string(), |p| p.display().to_string()))
                        .small()
                        .weak(),
                );
                ui.label(
                    RichText::new(format!(
                        "{} markers · {} frames · {:.1} Hz",
                        doc.take.n_markers(),
                        doc.take.n_frames(),
                        doc.take.fps
                    ))
                    .small(),
                );
            }
        });

        section(ui, "Playback", |ui| {
            let playing = app.playback.is_playing();
            ui.horizontal(|ui| {
                if ui.button(if playing { "⏸ Pause" } else { "▶ Play" }).clicked() {
                    cmds.push(Command::TogglePlay);
                }
                if ui.button("⏹ Stop").clicked() {
                    cmds.push(Command::Stop);
                }
                if ui.button("⏮").on_hover_text("previous frame").clicked() {
                    cmds.push(Command::Step(-1));
                }
                if ui.button("⏭").on_hover_text("next frame").clicked() {
                    cmds.push(Command::Step(1));
                }
            });
            ui.horizontal(|ui| {
                ui.checkbox(&mut app.playback.looping, "Loop");
                ui.label("speed");
                ui.add(egui::DragValue::new(&mut app.playback.speed).range(0.05..=8.0).speed(0.05).fixed_decimals(2));
                ui.label("fps");
                let mut fps = app.playback.fps();
                if ui.add(egui::DragValue::new(&mut fps).range(1.0..=1000.0).speed(1.0)).changed() {
                    cmds.push(Command::SetFps(fps));
                }
            });
            ui.horizontal(|ui| {
                ui.label("timeline");
                ui.selectable_value(&mut app.timeline_state.mode, TimelineMode::Frames, "frames");
                ui.selectable_value(&mut app.timeline_state.mode, TimelineMode::Time, "time");
            });
        });

        section(ui, "View", |ui| {
            if let Some(doc) = &mut app.doc {
                ui.horizontal_wrapped(|ui| {
                    ui.checkbox(&mut doc.state.view.show_names, "Names");
                    ui.checkbox(&mut doc.state.view.show_trajectory, "Trajectory");
                    ui.checkbox(&mut doc.state.view.show_skeleton, "Skeleton");
                    ui.checkbox(&mut app.show_grid, "Grid");
                });
                ui.horizontal(|ui| {
                    ui.label("up axis");
                    let mut cs = doc.state.view.coordinate_system;
                    ui.selectable_value(&mut cs, CoordinateSystem::YUp, "Y-up");
                    ui.selectable_value(&mut cs, CoordinateSystem::ZUp, "Z-up");
                    if cs != doc.state.view.coordinate_system {
                        cmds.push(Command::SetCoordinateSystem(cs));
                    }
                    if ui.button("Fit view").clicked() {
                        cmds.push(Command::FitView);
                    }
                });
            } else {
                ui.checkbox(&mut app.show_grid, "Grid");
            }
            ui.horizontal(|ui| {
                ui.label("theme");
                if ui.selectable_label(app.dark, "dark").clicked() {
                    cmds.push(Command::SetTheme(true));
                }
                if ui.selectable_label(!app.dark, "light").clicked() {
                    cmds.push(Command::SetTheme(false));
                }
            });
        });

        section(ui, "Skeleton", |ui| {
            let current = app.doc.as_ref().and_then(|d| d.state.skeleton_model).unwrap_or("No skeleton");
            egui::ComboBox::from_id_salt("skeleton_model").selected_text(current).show_ui(ui, |ui| {
                if ui.selectable_label(current == "No skeleton", "No skeleton").clicked() {
                    cmds.push(Command::SetSkeleton(None));
                }
                for m in skeleton::APP_MODELS {
                    if ui.selectable_label(current == m.name, m.name).clicked() {
                        cmds.push(Command::SetSkeleton(Some(m)));
                    }
                }
            });
            if let Some(doc) = &app.doc {
                let n_out = doc.outliers.iter().filter(|&&b| b).count();
                ui.label(
                    RichText::new(format!("{} pairs · {} outlier flags", doc.state.skeleton_pairs.len(), n_out))
                        .small()
                        .weak(),
                );
            }
        });

        section(ui, "Appearance", |ui| {
            let v: &mut VisualSettings = &mut app.visual;
            egui::Grid::new("appearance").num_columns(2).spacing([8.0, 4.0]).show(ui, |ui| {
                ui.label("marker size");
                ui.add(egui::Slider::new(&mut v.marker.size, 1.0..=20.0));
                ui.end_row();
                ui.label("marker opacity");
                ui.add(egui::Slider::new(&mut v.marker.opacity, 0.1..=1.0));
                ui.end_row();
                ui.label("line width");
                ui.add(egui::Slider::new(&mut v.skeleton.line_width, 0.5..=5.0));
                ui.end_row();
                ui.label("skeleton opacity");
                ui.add(egui::Slider::new(&mut v.skeleton.opacity, 0.1..=1.0));
                ui.end_row();
                ui.label("colours");
                ui.horizontal(|ui| {
                    ui.color_edit_button_rgb(&mut v.marker.color_normal).on_hover_text("marker");
                    ui.color_edit_button_rgb(&mut v.marker.color_selected).on_hover_text("selected marker");
                    ui.color_edit_button_rgb(&mut v.marker.color_pattern).on_hover_text("pattern reference");
                    ui.color_edit_button_rgb(&mut v.skeleton.color_normal).on_hover_text("skeleton");
                    ui.color_edit_button_rgb(&mut v.skeleton.color_outlier).on_hover_text("outlier");
                });
                ui.end_row();
                ui.label("scheme");
                ui.horizontal_wrapped(|ui| {
                    for name in VisualSettings::scheme_names() {
                        if ui.small_button(name).clicked() {
                            v.apply_scheme(name);
                        }
                    }
                    if ui.small_button("reset").clicked() {
                        v.reset();
                    }
                });
                ui.end_row();
            });
        });

        section(ui, "Analysis", |ui| {
            let Some(doc) = &mut app.doc else {
                ui.weak("open a file");
                return;
            };
            let mut on = doc.state.editing.is_analysis_mode;
            if ui.checkbox(&mut on, "Analysis mode (click 2 or 3 markers)").changed() {
                cmds.push(Command::SetAnalysisMode(on));
            }
            if doc.state.editing.is_analysis_mode {
                ui.horizontal_wrapped(|ui| {
                    for &m in &doc.state.selection.analysis_markers {
                        ui.label(RichText::new(&doc.take.markers[m]).color(crate::theme::ACCENT));
                    }
                    if ui.small_button("clear").clicked() {
                        cmds.push(Command::ClearAnalysis);
                    }
                });
                ui.horizontal(|ui| {
                    ui.label(format!("reference axis: {}", doc.state.editing.reference_axis.label()));
                    if ui.small_button("cycle").clicked() {
                        cmds.push(Command::CycleReferenceAxis);
                    }
                });
                for l in &app.analysis_labels {
                    ui.label(RichText::new(l).strong());
                }
            }
        });

        section(ui, "Selection", |ui| {
            let Some(doc) = &app.doc else {
                return;
            };
            match doc.state.selection.current_marker {
                Some(m) => {
                    ui.label(RichText::new(&doc.take.markers[m]).strong());
                    let f = app.playback.frame().min(doc.take.n_frames().saturating_sub(1));
                    match doc.take.position(f, m) {
                        Some(p) => ui.monospace(format!("x {:+.4}\ny {:+.4}\nz {:+.4}", p[0], p[1], p[2])),
                        None => ui.weak("missing in this frame"),
                    };
                    if let Some((a, b)) = doc.state.selection.selected_frames {
                        ui.label(format!("frames {a}–{b} selected"));
                    }
                }
                None => {
                    ui.weak("no marker selected");
                }
            }
        });
    });
}

pub fn edit(ui: &mut egui::Ui, app: &mut App, cmds: &mut Vec<Command>) {
    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        ui.add_space(4.0);
        let Some(doc) = &app.doc else {
            ui.weak("open a file");
            return;
        };
        let busy = app.worker.is_some();
        let has_marker = doc.state.selection.current_marker.is_some();
        let range_text = match doc.state.selection.selected_frames {
            Some((a, b)) => format!("frames {a}–{b}"),
            None => "whole take (select a range on the timeline or plot)".to_string(),
        };
        let marker_name = doc.state.selection.current_marker.map(|m| doc.take.markers[m].clone());

        if let Some(w) = &app.worker {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(format!("{} … {:.1} s", w.label, w.started.elapsed().as_secs_f32()));
            });
            ui.separator();
        }

        section(ui, "Edit", |ui| {
            let mut editing = doc.state.editing.is_editing;
            if ui.checkbox(&mut editing, "Edit mode").changed() {
                cmds.push(Command::SetEditMode(editing));
            }
            ui.label(
                RichText::new(format!("target: {}", marker_name.as_deref().unwrap_or("— select a marker"))).small(),
            );
            ui.label(RichText::new(format!("range: {range_text}")).small());
            ui.horizontal_wrapped(|ui| {
                if ui
                    .add_enabled(editing && has_marker && !busy, egui::Button::new("Delete range"))
                    .on_hover_text("set the marker to missing in the selected frames")
                    .clicked()
                {
                    cmds.push(Command::DeleteRange);
                }
                if ui.add_enabled(!busy && !app.undo_empty(), egui::Button::new("Undo    ⌘Z")).clicked() {
                    cmds.push(Command::Undo);
                }
                if ui
                    .add_enabled(!busy, egui::Button::new("Restore original"))
                    .on_hover_text("discard every edit since the file was opened")
                    .clicked()
                {
                    cmds.push(Command::RestoreOriginal);
                }
            });
            ui.label(RichText::new(app.last_edit.as_deref().unwrap_or("")).small().weak());
        });

        section(ui, "Filter", |ui| {
            let f = &mut app.edit_settings;
            egui::ComboBox::from_id_salt("filter_kind").selected_text(f.filter.label()).show_ui(ui, |ui| {
                for k in FilterKind::ALL {
                    ui.selectable_value(&mut f.filter, k, k.label());
                }
            });
            egui::Grid::new("filter_params").num_columns(2).spacing([8.0, 4.0]).show(ui, |ui| match f.filter {
                FilterKind::Butterworth | FilterKind::ButterworthOnSpeed => {
                    ui.label("order (even)");
                    ui.add(egui::DragValue::new(&mut f.butter_order).range(2..=12).speed(1.0));
                    ui.end_row();
                    ui.label("cutoff (Hz)");
                    ui.add(egui::DragValue::new(&mut f.butter_cutoff).range(1.0..=500.0).speed(0.5));
                    ui.end_row();
                }
                FilterKind::Kalman => {
                    ui.label("trust ratio");
                    ui.add(egui::DragValue::new(&mut f.kalman_trust).range(1.0..=1000.0).speed(1.0));
                    ui.end_row();
                    ui.label("RTS smoothing");
                    ui.checkbox(&mut f.kalman_smooth, "");
                    ui.end_row();
                }
                FilterKind::Gaussian => {
                    ui.label("sigma (frames)");
                    ui.add(egui::DragValue::new(&mut f.gaussian_sigma).range(1.0..=50.0).speed(0.5));
                    ui.end_row();
                }
                FilterKind::Loess => {
                    ui.label("points per fit");
                    ui.add(egui::DragValue::new(&mut f.loess_points).range(3.0..=500.0).speed(1.0));
                    ui.end_row();
                }
                FilterKind::Median => {
                    ui.label("kernel (odd)");
                    ui.add(egui::DragValue::new(&mut f.median_kernel).range(3.0..=99.0).speed(2.0));
                    ui.end_row();
                }
            });
            if ui.add_enabled(has_marker && !busy, egui::Button::new("Apply filter to range")).clicked() {
                cmds.push(Command::ApplyFilter);
            }
        });

        section(ui, "Interpolation", |ui| {
            let f = &mut app.edit_settings;
            egui::ComboBox::from_id_salt("interp_method").selected_text(INTERP_METHODS[f.interp_method]).show_ui(
                ui,
                |ui| {
                    for (i, name) in INTERP_METHODS.iter().enumerate() {
                        ui.selectable_value(&mut f.interp_method, i, *name);
                    }
                },
            );
            let name = INTERP_METHODS[f.interp_method];
            if name == "polynomial" || name == "spline" {
                ui.horizontal(|ui| {
                    ui.label("order");
                    ui.add(egui::DragValue::new(&mut f.interp_order).range(1..=5).speed(1.0));
                });
            }
            if name == "pattern-based" {
                let mode = doc.state.editing.pattern_selection_mode;
                let mut on = mode;
                if ui.checkbox(&mut on, "Select reference markers (click in the 3D view)").changed() {
                    cmds.push(Command::SetPatternMode(on));
                }
                ui.horizontal_wrapped(|ui| {
                    for &m in &doc.state.selection.pattern_markers {
                        ui.label(RichText::new(&doc.take.markers[m]).color(egui::Color32::from_rgb(255, 80, 80)));
                    }
                    if !doc.state.selection.pattern_markers.is_empty() && ui.small_button("clear").clicked() {
                        cmds.push(Command::ClearPattern);
                    }
                });
                let ready = has_marker && !doc.state.selection.pattern_markers.is_empty();
                if ui.add_enabled(ready && !busy, egui::Button::new("Run pattern-based interpolation")).clicked() {
                    cmds.push(Command::RunPattern);
                }
            } else if ui.add_enabled(has_marker && !busy, egui::Button::new("Interpolate gaps in range")).clicked() {
                cmds.push(Command::ApplyInterp);
            }
        });

        section(ui, "Report", |ui| {
            ui.label(RichText::new("Interactive HTML (opens in your browser; print for PDF)").small().weak());
            if ui.add_enabled(!busy, egui::Button::new("Generate report…")).clicked() {
                cmds.push(Command::GenerateReport);
            }
        });
    });
}

pub fn marker_list(ui: &mut egui::Ui, app: &mut App, cmds: &mut Vec<Command>) {
    let Some(doc) = &app.doc else {
        ui.weak("open a file");
        return;
    };
    let frame = app.playback.frame().min(doc.take.n_frames().saturating_sub(1));
    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        for (m, name) in doc.take.markers.iter().enumerate() {
            let selected = doc.state.selection.current_marker == Some(m);
            let missing = doc.take.position(frame, m).is_none();
            let text = if missing { RichText::new(name).weak() } else { RichText::new(name) };
            if ui.selectable_label(selected, text).clicked() {
                cmds.push(Command::SelectMarker(Some(m)));
            }
        }
    });
}

fn section(ui: &mut egui::Ui, title: &str, add: impl FnOnce(&mut egui::Ui)) {
    egui::CollapsingHeader::new(RichText::new(title).strong()).default_open(true).show(ui, add);
    ui.add_space(2.0);
}

impl FilterKind {
    pub const ALL: [FilterKind; 6] = [
        FilterKind::Butterworth,
        FilterKind::ButterworthOnSpeed,
        FilterKind::Kalman,
        FilterKind::Gaussian,
        FilterKind::Loess,
        FilterKind::Median,
    ];

    pub fn label(self) -> &'static str {
        match self {
            FilterKind::Butterworth => "Butterworth",
            FilterKind::ButterworthOnSpeed => "Butterworth on speed",
            FilterKind::Kalman => "Kalman",
            FilterKind::Gaussian => "Gaussian",
            FilterKind::Loess => "LOESS",
            FilterKind::Median => "Median",
        }
    }
}

/// Build the processing filter from the panel values.
pub fn filter_from_settings(s: &crate::app::EditSettings) -> Filter {
    match s.filter {
        FilterKind::Butterworth => Filter::Butterworth { order: s.butter_order, cutoff_hz: s.butter_cutoff },
        FilterKind::ButterworthOnSpeed => {
            Filter::ButterworthOnSpeed { order: s.butter_order, cutoff_hz: s.butter_cutoff }
        }
        FilterKind::Kalman => Filter::Kalman { trust_ratio: s.kalman_trust, smooth: s.kalman_smooth },
        FilterKind::Gaussian => Filter::Gaussian { sigma_kernel: s.gaussian_sigma },
        FilterKind::Loess => Filter::Loess { nb_values_used: s.loess_points },
        FilterKind::Median => Filter::Median { kernel_size: s.median_kernel },
    }
}
