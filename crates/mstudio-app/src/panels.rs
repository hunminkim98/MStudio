//! Right-hand tabs: Controls (file, playback, view, skeleton, visual,
//! analysis, info) and Markers (list).

use eframe::egui::{self, RichText};
use mstudio_core::{skeleton, CoordinateSystem, VisualSettings};

use crate::app::{App, Command};
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
            let mut changed = false;
            ui.horizontal(|ui| {
                ui.label("marker size");
                changed |= ui.add(egui::Slider::new(&mut v.marker.size, 1.0..=20.0)).changed();
            });
            ui.horizontal(|ui| {
                ui.label("marker opacity");
                changed |= ui.add(egui::Slider::new(&mut v.marker.opacity, 0.1..=1.0)).changed();
            });
            ui.horizontal(|ui| {
                ui.label("line width");
                changed |= ui.add(egui::Slider::new(&mut v.skeleton.line_width, 0.5..=5.0)).changed();
            });
            ui.horizontal(|ui| {
                ui.label("skeleton opacity");
                changed |= ui.add(egui::Slider::new(&mut v.skeleton.opacity, 0.1..=1.0)).changed();
            });
            ui.horizontal(|ui| {
                ui.label("scheme");
                for name in VisualSettings::scheme_names() {
                    if ui.small_button(name).clicked() {
                        v.apply_scheme(name);
                        changed = true;
                    }
                }
                if ui.small_button("reset").clicked() {
                    v.reset();
                    changed = true;
                }
            });
            let _ = changed; // the renderer reads the settings every frame
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
