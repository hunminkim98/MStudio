//! Application state, docking layout, commands, edits, background jobs and
//! shortcuts.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use eframe::egui::{self, RichText};
use egui_dock::{DockArea, DockState, NodeIndex, Style, TabViewer};
use mstudio_core::skeleton::{self, SkeletonModel};
use mstudio_core::{
    detect_outliers, CoordinateSystem, DirtyRange, Playback, StateManager, Take, VisualSettings,
    DEFAULT_OUTLIER_THRESHOLD,
};
use mstudio_processing::InterpMethod;
use ndarray::{s, Array2, Array3};

use crate::jobs::{Job, Outcome, Worker};
use crate::marker_plot::{marker_plot, MarkerPlotInput, MarkerPlotState};
use crate::panels::filter_from_settings;
use crate::timeline::{timeline, TimelineInput, TimelineState};
use crate::viewport::{SharedRenderer, Viewport, ViewportInput};
use crate::LaunchOptions;

pub const MSAA: u32 = 4;
pub const INTERP_METHODS: [&str; 9] =
    ["linear", "polynomial", "spline", "nearest", "zero", "slinear", "quadratic", "cubic", "pattern-based"];
const UNDO_LIMIT: usize = 50;

pub struct Document {
    pub take: Take,
    pub path: Option<PathBuf>,
    pub state: StateManager,
    pub outliers: Array2<bool>,
    /// Bumped on every data edit; caches (marker plot) key on it.
    pub data_version: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Viewport,
    Controls,
    Edit,
    Markers,
    Plot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterKind {
    Butterworth,
    ButterworthOnSpeed,
    Kalman,
    Gaussian,
    Loess,
    Median,
}

/// Panel values for filters and interpolation (defaults = the Python app's).
#[derive(Debug, Clone)]
pub struct EditSettings {
    pub filter: FilterKind,
    pub butter_order: u32,
    pub butter_cutoff: f64,
    pub kalman_trust: f64,
    pub kalman_smooth: bool,
    pub gaussian_sigma: f64,
    pub loess_points: f64,
    pub median_kernel: f64,
    pub interp_method: usize,
    pub interp_order: u32,
}

impl Default for EditSettings {
    fn default() -> Self {
        Self {
            filter: FilterKind::Butterworth,
            butter_order: 4,
            butter_cutoff: 10.0,
            kalman_trust: 20.0,
            kalman_smooth: true,
            gaussian_sigma: 3.0,
            loess_points: 10.0,
            median_kernel: 3.0,
            interp_method: 0,
            interp_order: 3,
        }
    }
}

/// One reversible edit: the frames of `range` for `marker` (or all markers) as they were.
struct EditRecord {
    label: String,
    marker: Option<usize>,
    range: DirtyRange,
    before: Array3<f64>,
}

/// UI events applied after the frame is drawn (keeps borrows simple).
#[derive(Debug, Clone)]
pub enum Command {
    OpenFileDialog,
    OpenFolderDialog,
    SaveAsDialog,
    Open(PathBuf),
    TogglePlay,
    Stop,
    Step(isize),
    Seek(usize),
    SetFps(f64),
    SetSelection(Option<(usize, usize)>),
    SelectMarker(Option<usize>),
    SetSkeleton(Option<&'static SkeletonModel>),
    SetCoordinateSystem(CoordinateSystem),
    FitView,
    SetTheme(bool),
    SetAnalysisMode(bool),
    ClearAnalysis,
    CycleReferenceAxis,
    SetEditMode(bool),
    DeleteRange,
    RestoreOriginal,
    Undo,
    ApplyFilter,
    ApplyInterp,
    SetPatternMode(bool),
    ClearPattern,
    RunPattern,
    GenerateReport,
    Quit,
}

pub struct App {
    pub doc: Option<Document>,
    pub playback: Playback,
    pub visual: VisualSettings,
    pub show_grid: bool,
    pub dark: bool,
    pub timeline_state: TimelineState,
    pub analysis_labels: Vec<String>,
    pub edit_settings: EditSettings,
    pub worker: Option<Worker>,
    pub last_edit: Option<String>,
    undo: Vec<EditRecord>,
    plot_state: MarkerPlotState,
    viewport: Viewport,
    dock: Option<DockState<Tab>>,
    status: String,
    error: Option<String>,
    frame_times: std::collections::VecDeque<Instant>,
    screenshot: Option<PathBuf>,
    shot_requested: bool,
    exit_at: Option<Instant>,
    started: Instant,
    /// 0 = off; 1 = filter running; 2 = report running.
    selftest_step: u8,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>, opts: LaunchOptions) -> App {
        let rs = cc.wgpu_render_state.as_ref().expect("wgpu renderer required");
        let shared = SharedRenderer::new(rs, MSAA);
        crate::theme::apply(&cc.egui_ctx, true);

        let mut dock = DockState::new(vec![Tab::Viewport]);
        let surface = dock.main_surface_mut();
        let [center, _right] =
            surface.split_right(NodeIndex::root(), 0.76, vec![Tab::Controls, Tab::Edit, Tab::Markers]);
        let _ = surface.split_below(center, 0.68, vec![Tab::Plot]);

        let mut app = App {
            doc: None,
            playback: Playback::new(),
            visual: VisualSettings::new(),
            show_grid: true,
            dark: true,
            timeline_state: TimelineState::default(),
            analysis_labels: Vec::new(),
            edit_settings: EditSettings::default(),
            worker: None,
            last_edit: None,
            undo: Vec::new(),
            plot_state: MarkerPlotState::default(),
            viewport: Viewport::new(shared),
            dock: Some(dock),
            status: "ready".into(),
            error: None,
            frame_times: Default::default(),
            screenshot: opts.screenshot,
            shot_requested: false,
            exit_at: opts.exit_after.map(|s| Instant::now() + Duration::from_secs_f32(s)),
            started: Instant::now(),
            selftest_step: 0,
        };
        if let Some(p) = opts.path {
            app.open(&p);
        }
        if opts.demo || opts.selftest {
            if let Some(doc) = &mut app.doc {
                doc.state.view.show_names = true;
                doc.state.view.show_trajectory = true;
                doc.state.set_current_marker(Some(2.min(doc.take.n_markers().saturating_sub(1))));
                doc.state.set_selected_frames(Some((30, 60)));
                doc.state.set_editing_mode(true);
            }
            if let Some(dock) = &mut app.dock {
                if let Some(path) = dock.find_tab(&Tab::Edit) {
                    let _ = dock.set_active_tab(path);
                }
            }
        }
        if opts.selftest && app.doc.is_some() {
            app.selftest_step = 1;
            app.apply_filter();
        }
        if opts.play {
            app.playback.play(Instant::now());
        }
        app
    }

    pub fn undo_empty(&self) -> bool {
        self.undo.is_empty()
    }

    // ------------------------------------------------------------ document --

    fn open(&mut self, path: &Path) {
        match mstudio_io::load(path) {
            Ok(take) => {
                let n = take.n_frames();
                let fps = take.fps;
                self.playback = Playback::new();
                self.playback.set_data_info(n, fps);
                self.playback.looping = true;
                let mut doc = Document {
                    take,
                    path: Some(path.to_path_buf()),
                    state: StateManager::new(),
                    outliers: Array2::from_elem((0, 0), false),
                    data_version: 0,
                };
                doc.state.view.show_skeleton = false;
                doc.outliers = Array2::from_elem((doc.take.n_markers(), n), false);
                self.plot_state = MarkerPlotState::default();
                self.undo.clear();
                self.last_edit = None;
                {
                    let mut r = self.viewport.shared.renderer.lock().unwrap();
                    r.load_take(&self.viewport.shared.device, &doc.take);
                    r.set_skeleton_pairs(&self.viewport.shared.device, &[]);
                    r.set_outliers(&self.viewport.shared.device, &doc.outliers);
                }
                self.doc = Some(doc);
                self.rebuild_grid();
                self.fit_view();
                // pick the skeleton model that explains the most marker pairs
                let markers = &self.doc.as_ref().unwrap().take.markers;
                let best = skeleton::APP_MODELS
                    .iter()
                    .map(|m| (m.resolve_pairs(markers).len(), *m))
                    .max_by_key(|(n, _)| *n)
                    .filter(|(n, _)| *n >= 3)
                    .map(|(_, m)| m);
                self.set_skeleton(best);
                self.status = format!("opened {}", path.display());
            }
            Err(e) => self.error = Some(format!("Could not open {}:\n{e}", path.display())),
        }
    }

    fn save_as(&mut self, path: &Path) {
        let Some(doc) = &mut self.doc else { return };
        match mstudio_io::save(path, &doc.take) {
            Ok(()) => {
                doc.path = Some(path.to_path_buf());
                self.status = format!("saved {}", path.display());
            }
            Err(e) => self.error = Some(format!("Could not save {}:\n{e}", path.display())),
        }
    }

    fn rebuild_grid(&mut self) {
        let ground = self.doc.as_ref().and_then(|d| {
            let cs = d.state.view.coordinate_system;
            d.take.bounds().map(|(lo, _)| if cs.is_z_up() { lo[2] as f32 } else { lo[1] as f32 })
        });
        let segments = mstudio_render::grid_segments(ground.unwrap_or(0.0), 10.0, 1.0, 0.5);
        let mut r = self.viewport.shared.renderer.lock().unwrap();
        r.set_grid(&self.viewport.shared.device, &self.viewport.shared.queue, &segments);
    }

    fn fit_view(&mut self) {
        if let Some(doc) = &self.doc {
            let frame = self.playback.frame().min(doc.take.n_frames().saturating_sub(1));
            self.viewport.fit(&doc.take, doc.state.view.coordinate_system, frame);
        }
    }

    fn set_skeleton(&mut self, model: Option<&'static SkeletonModel>) {
        let Some(doc) = &mut self.doc else { return };
        if let Some(m) = model {
            if doc.take.rename_markers(m) {
                doc.data_version += 1;
            }
        }
        doc.state.set_skeleton_model(model, &doc.take.markers);
        let mut r = self.viewport.shared.renderer.lock().unwrap();
        r.set_skeleton_pairs(&self.viewport.shared.device, &doc.state.skeleton_pairs);
        drop(r);
        self.recompute_outliers();
    }

    fn recompute_outliers(&mut self) {
        let Some(doc) = &mut self.doc else { return };
        doc.outliers = if doc.state.skeleton_pairs.is_empty() {
            Array2::from_elem((doc.take.n_markers(), doc.take.n_frames()), false)
        } else {
            detect_outliers(&doc.take.frames, &doc.state.skeleton_pairs, DEFAULT_OUTLIER_THRESHOLD)
        };
        let mut r = self.viewport.shared.renderer.lock().unwrap();
        r.set_outliers(&self.viewport.shared.device, &doc.outliers);
    }

    // --------------------------------------------------------------- edits --

    /// Selected frame range as inclusive bounds, or the whole take.
    fn edit_range(&self) -> Option<(usize, usize)> {
        let doc = self.doc.as_ref()?;
        let n = doc.take.n_frames();
        Some(doc.state.selection.selected_frames.unwrap_or((0, n.saturating_sub(1))))
    }

    fn snapshot(&self, marker: Option<usize>, range: DirtyRange) -> Array3<f64> {
        let take = &self.doc.as_ref().unwrap().take;
        match marker {
            Some(m) => take.frames.slice(s![range.start..range.end, m..m + 1, ..]).to_owned(),
            None => take.frames.slice(s![range.start..range.end, .., ..]).to_owned(),
        }
    }

    fn push_undo(&mut self, label: &str, marker: Option<usize>, range: DirtyRange) {
        if range.is_empty() {
            return;
        }
        let before = self.snapshot(marker, range);
        self.undo.push(EditRecord { label: label.to_string(), marker, range, before });
        if self.undo.len() > UNDO_LIMIT {
            self.undo.remove(0);
        }
    }

    /// After the take changed in `range`: GPU patch (R2), caches, outliers.
    fn after_edit(&mut self, range: DirtyRange, label: String) {
        if let Some(doc) = &mut self.doc {
            doc.data_version += 1;
            self.viewport.shared.renderer.lock().unwrap().update_frames(&self.viewport.shared.queue, &doc.take, range);
        }
        self.recompute_outliers();
        self.status = label.clone();
        self.last_edit = Some(label);
    }

    #[allow(clippy::needless_range_loop)]
    fn apply_columns(&mut self, marker: usize, range: DirtyRange, cols: [Vec<f64>; 3], label: String) {
        if range.is_empty() {
            self.status = format!("{label}: nothing to change");
            self.last_edit = Some(self.status.clone());
            return;
        }
        self.push_undo(&label, Some(marker), range);
        let doc = self.doc.as_mut().unwrap();
        for f in range.start..range.end.min(doc.take.n_frames()) {
            for k in 0..3 {
                doc.take.frames[[f, marker, k]] = cols[k][f];
            }
        }
        self.after_edit(range, format!("{label} applied to frames {}–{}", range.start, range.end - 1));
    }

    fn delete_range(&mut self) {
        let Some((first, last)) = self.edit_range() else { return };
        let Some(marker) = self.doc.as_ref().and_then(|d| d.state.selection.current_marker) else { return };
        let range = DirtyRange::inclusive(first, last);
        self.push_undo("delete", Some(marker), range);
        let doc = self.doc.as_mut().unwrap();
        doc.take.clear_range(marker, first, last);
        let name = doc.take.markers[marker].clone();
        self.after_edit(range, format!("deleted {name} frames {first}–{last}"));
    }

    fn restore_original(&mut self) {
        let Some(doc) = &self.doc else { return };
        let all = DirtyRange::all(doc.take.n_frames());
        self.push_undo("restore original", None, all);
        self.doc.as_mut().unwrap().take.restore_original();
        self.after_edit(all, "restored the original data".into());
    }

    fn undo(&mut self) {
        let Some(rec) = self.undo.pop() else { return };
        let doc = self.doc.as_mut().unwrap();
        match rec.marker {
            Some(m) => doc.take.frames.slice_mut(s![rec.range.start..rec.range.end, m..m + 1, ..]).assign(&rec.before),
            None => doc.take.frames.slice_mut(s![rec.range.start..rec.range.end, .., ..]).assign(&rec.before),
        }
        self.after_edit(rec.range, format!("undid {}", rec.label));
    }

    fn start_job(&mut self, job: Job) {
        if self.worker.is_some() {
            self.error = Some("A processing job is still running.".into());
            return;
        }
        self.worker = Some(Worker::spawn(job));
    }

    fn marker_columns(&self, marker: usize) -> Array3<f64> {
        let take = &self.doc.as_ref().unwrap().take;
        take.frames.slice(s![.., marker..marker + 1, ..]).to_owned()
    }

    fn apply_filter(&mut self) {
        let Some((first, last)) = self.edit_range() else { return };
        let Some(marker) = self.doc.as_ref().and_then(|d| d.state.selection.current_marker) else { return };
        let fps = self.doc.as_ref().unwrap().take.fps;
        let filter = filter_from_settings(&self.edit_settings);
        let cols = self.marker_columns(marker);
        self.start_job(Job::Filter { marker, cols, first, last, filter, fps });
    }

    fn apply_interp(&mut self) {
        let Some((first, last)) = self.edit_range() else { return };
        let Some(marker) = self.doc.as_ref().and_then(|d| d.state.selection.current_marker) else { return };
        let name = INTERP_METHODS[self.edit_settings.interp_method];
        let Some(method) = InterpMethod::from_name(name, self.edit_settings.interp_order) else { return };
        let cols = self.marker_columns(marker);
        self.start_job(Job::Interp { marker, cols, first, last, method });
    }

    fn run_pattern(&mut self) {
        let Some((first, last)) = self.edit_range() else { return };
        let Some(doc) = &self.doc else { return };
        let Some(marker) = doc.state.selection.current_marker else { return };
        let refs: Vec<usize> = doc.state.selection.pattern_markers.iter().copied().filter(|&r| r != marker).collect();
        if refs.is_empty() {
            self.error = Some("Select at least one reference marker (other than the target).".into());
            return;
        }
        let n = doc.take.n_frames();
        let mut sub = Array3::<f64>::zeros((n, 1 + refs.len(), 3));
        sub.slice_mut(s![.., 0, ..]).assign(&doc.take.frames.slice(s![.., marker, ..]));
        for (i, &r) in refs.iter().enumerate() {
            sub.slice_mut(s![.., i + 1, ..]).assign(&doc.take.frames.slice(s![.., r, ..]));
        }
        self.start_job(Job::Pattern { marker, sub, first, last });
    }

    fn generate_report(&mut self) {
        let Some(doc) = &self.doc else { return };
        let stem = doc
            .path
            .as_ref()
            .and_then(|p| p.file_stem())
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "take".into());
        let Some(path) = rfd::FileDialog::new()
            .add_filter("HTML", &["html"])
            .set_file_name(format!("{stem}_report.html"))
            .save_file()
        else {
            return;
        };
        let options = mstudio_report::ReportOptions {
            title: format!("{stem} — MStudio analysis report"),
            source: doc.path.as_ref().map(|p| p.display().to_string()).unwrap_or_default(),
            skeleton_model: doc.state.skeleton_model.map(String::from),
            skeleton_pairs: doc.state.skeleton_pairs.clone(),
            ..Default::default()
        };
        self.start_job(Job::Report { take: doc.take.clone(), options, path });
    }

    fn poll_worker(&mut self) {
        let Some(outcome) = self.worker.as_ref().and_then(|w| w.poll()) else { return };
        self.worker = None;
        match outcome {
            Outcome::Columns { marker, range, cols, label } => {
                self.apply_columns(marker, range, cols, label);
                if self.selftest_step == 1 {
                    let before = self.doc.as_ref().unwrap().take.frames.clone();
                    self.delete_range();
                    self.undo();
                    let same = self
                        .doc
                        .as_ref()
                        .unwrap()
                        .take
                        .frames
                        .iter()
                        .zip(before.iter())
                        .all(|(a, b)| a.to_bits() == b.to_bits());
                    eprintln!(
                        "selftest: filter applied ({}), delete+undo round trip {}",
                        self.last_edit.as_deref().unwrap_or(""),
                        if same { "ok" } else { "MISMATCH" }
                    );
                    let path = PathBuf::from("target/selftest_report.html");
                    let doc = self.doc.as_ref().unwrap();
                    let options = mstudio_report::ReportOptions {
                        title: "selftest".into(),
                        skeleton_model: doc.state.skeleton_model.map(String::from),
                        skeleton_pairs: doc.state.skeleton_pairs.clone(),
                        ..Default::default()
                    };
                    self.selftest_step = 2;
                    self.start_job(Job::Report { take: doc.take.clone(), options, path });
                }
            }
            Outcome::Report { path } => {
                self.status = format!("report written to {}", path.display());
                if self.selftest_step == 2 {
                    eprintln!(
                        "selftest: report written to {} ({} bytes)",
                        path.display(),
                        std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0)
                    );
                    self.selftest_step = 0;
                    self.exit_at = Some(Instant::now() + Duration::from_millis(300));
                } else if let Err(e) = mstudio_report::open_in_browser(&path) {
                    self.error = Some(format!("Report written to {} but could not be opened:\n{e}", path.display()));
                }
            }
            Outcome::Error(e) => {
                if self.selftest_step != 0 {
                    eprintln!("selftest: FAILED: {e}");
                    self.exit_at = Some(Instant::now());
                }
                self.error = Some(e)
            }
        }
    }

    fn apply(&mut self, cmd: Command, ctx: &egui::Context) {
        let now = Instant::now();
        match cmd {
            Command::OpenFileDialog => {
                if let Some(p) =
                    rfd::FileDialog::new().add_filter("Motion files", &["trc", "c3d", "TRC", "C3D"]).pick_file()
                {
                    self.open(&p);
                }
            }
            Command::OpenFolderDialog => {
                if let Some(p) = rfd::FileDialog::new().pick_folder() {
                    self.open(&p);
                }
            }
            Command::SaveAsDialog => {
                let name = self
                    .doc
                    .as_ref()
                    .and_then(|d| d.path.as_ref())
                    .and_then(|p| p.file_stem())
                    .map(|s| format!("{}_edited.trc", s.to_string_lossy()));
                let mut dlg = rfd::FileDialog::new().add_filter("TRC", &["trc"]).add_filter("C3D", &["c3d"]);
                if let Some(n) = name {
                    dlg = dlg.set_file_name(n);
                }
                if let Some(p) = dlg.save_file() {
                    self.save_as(&p);
                }
            }
            Command::Open(p) => self.open(&p),
            Command::TogglePlay => self.playback.toggle(now),
            Command::Stop => self.playback.stop(),
            Command::Step(d) => {
                if d > 0 {
                    self.playback.next_frame(now)
                } else {
                    self.playback.prev_frame(now)
                }
            }
            Command::Seek(f) => self.playback.set_frame(f, now),
            Command::SetFps(fps) => self.playback.set_fps(fps, now),
            Command::SetSelection(sel) => {
                if let Some(doc) = &mut self.doc {
                    doc.state.set_selected_frames(sel);
                }
            }
            Command::SelectMarker(m) => {
                if let Some(doc) = &mut self.doc {
                    if doc.state.editing.is_analysis_mode {
                        if let Some(m) = m {
                            if !doc.state.remove_analysis_marker(m) {
                                doc.state.add_analysis_marker(m);
                            }
                        }
                    } else if doc.state.editing.pattern_selection_mode {
                        if let Some(m) = m {
                            if doc.state.selection.current_marker != Some(m) {
                                doc.state.toggle_pattern_marker(m);
                            }
                        }
                    } else {
                        doc.state.set_current_marker(m);
                    }
                }
            }
            Command::SetSkeleton(model) => self.set_skeleton(model),
            Command::SetCoordinateSystem(cs) => {
                if let Some(doc) = &mut self.doc {
                    doc.state.view.coordinate_system = cs;
                }
                self.rebuild_grid();
                self.fit_view();
            }
            Command::FitView => self.fit_view(),
            Command::SetTheme(dark) => {
                self.dark = dark;
                crate::theme::apply(ctx, dark);
            }
            Command::SetAnalysisMode(on) => {
                if let Some(doc) = &mut self.doc {
                    doc.state.set_analysis_mode(on);
                }
            }
            Command::ClearAnalysis => {
                if let Some(doc) = &mut self.doc {
                    doc.state.clear_analysis_markers();
                }
            }
            Command::CycleReferenceAxis => {
                if let Some(doc) = &mut self.doc {
                    doc.state.cycle_reference_axis();
                }
            }
            Command::SetEditMode(on) => {
                if let Some(doc) = &mut self.doc {
                    doc.state.set_editing_mode(on);
                }
            }
            Command::DeleteRange => self.delete_range(),
            Command::RestoreOriginal => self.restore_original(),
            Command::Undo => self.undo(),
            Command::ApplyFilter => self.apply_filter(),
            Command::ApplyInterp => self.apply_interp(),
            Command::SetPatternMode(on) => {
                if let Some(doc) = &mut self.doc {
                    doc.state.set_pattern_selection_mode(on);
                }
            }
            Command::ClearPattern => {
                if let Some(doc) = &mut self.doc {
                    doc.state.clear_pattern_markers();
                }
            }
            Command::RunPattern => self.run_pattern(),
            Command::GenerateReport => self.generate_report(),
            Command::Quit => ctx.send_viewport_cmd(egui::ViewportCommand::Close),
        }
    }

    fn shortcuts(&self, ctx: &egui::Context, cmds: &mut Vec<Command>) {
        if ctx.egui_wants_keyboard_input() {
            return;
        }
        ctx.input(|i| {
            if i.key_pressed(egui::Key::Space) || i.key_pressed(egui::Key::Enter) {
                cmds.push(Command::TogglePlay);
            }
            if i.key_pressed(egui::Key::Escape) {
                cmds.push(Command::Stop);
            }
            if i.key_pressed(egui::Key::ArrowRight) {
                cmds.push(Command::Step(1));
            }
            if i.key_pressed(egui::Key::ArrowLeft) {
                cmds.push(Command::Step(-1));
            }
            if i.key_pressed(egui::Key::F) {
                cmds.push(Command::FitView);
            }
            if i.modifiers.command && i.key_pressed(egui::Key::O) {
                cmds.push(Command::OpenFileDialog);
            }
            if i.modifiers.command && i.modifiers.shift && i.key_pressed(egui::Key::S) {
                cmds.push(Command::SaveAsDialog);
            }
            if i.modifiers.command && i.key_pressed(egui::Key::Z) {
                cmds.push(Command::Undo);
            }
            if i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace) {
                cmds.push(Command::DeleteRange);
            }
        });
    }

    fn menu(&mut self, ui: &mut egui::Ui, cmds: &mut Vec<Command>) {
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.button("Open…    ⌘O").clicked() {
                    cmds.push(Command::OpenFileDialog);
                    ui.close();
                }
                if ui.button("Open JSON folder…").clicked() {
                    cmds.push(Command::OpenFolderDialog);
                    ui.close();
                }
                if ui.add_enabled(self.doc.is_some(), egui::Button::new("Save As…    ⇧⌘S")).clicked() {
                    cmds.push(Command::SaveAsDialog);
                    ui.close();
                }
                if ui
                    .add_enabled(self.doc.is_some() && self.worker.is_none(), egui::Button::new("Generate report…"))
                    .clicked()
                {
                    cmds.push(Command::GenerateReport);
                    ui.close();
                }
                ui.separator();
                if ui.button("Quit").clicked() {
                    cmds.push(Command::Quit);
                }
            });
            ui.menu_button("Edit", |ui| {
                if ui.add_enabled(!self.undo.is_empty(), egui::Button::new("Undo    ⌘Z")).clicked() {
                    cmds.push(Command::Undo);
                    ui.close();
                }
                if ui.add_enabled(self.doc.is_some(), egui::Button::new("Delete selected range    ⌫")).clicked() {
                    cmds.push(Command::DeleteRange);
                    ui.close();
                }
                if ui.add_enabled(self.doc.is_some(), egui::Button::new("Restore original")).clicked() {
                    cmds.push(Command::RestoreOriginal);
                    ui.close();
                }
            });
            ui.menu_button("View", |ui| {
                if ui.button("Fit view    F").clicked() {
                    cmds.push(Command::FitView);
                    ui.close();
                }
                ui.checkbox(&mut self.show_grid, "Grid");
                if let Some(doc) = &mut self.doc {
                    ui.checkbox(&mut doc.state.view.show_names, "Marker names");
                    ui.checkbox(&mut doc.state.view.show_trajectory, "Trajectory");
                    ui.checkbox(&mut doc.state.view.show_skeleton, "Skeleton");
                }
                ui.separator();
                if ui.button(if self.dark { "Light theme" } else { "Dark theme" }).clicked() {
                    cmds.push(Command::SetTheme(!self.dark));
                    ui.close();
                }
            });
            ui.separator();
            ui.label(RichText::new(&self.status).weak());
        });
    }

    fn status_bar(&self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let fps = self.frame_times.len();
            ui.small(format!("{fps} fps"));
            ui.separator();
            if let Some(doc) = &self.doc {
                ui.small(format!("frame {} / {}", self.playback.frame(), doc.take.n_frames()));
                ui.separator();
                ui.small(doc.state.view.coordinate_system.label());
                ui.separator();
                if doc.state.editing.is_editing {
                    ui.small(RichText::new("EDIT").color(crate::theme::ACCENT));
                    ui.separator();
                }
                if doc.state.editing.pattern_selection_mode {
                    ui.small(RichText::new("click reference markers").color(egui::Color32::from_rgb(255, 80, 80)));
                    ui.separator();
                }
            }
            if let Some(w) = &self.worker {
                ui.spinner();
                ui.small(&w.label);
                ui.separator();
            }
            ui.small("Space play · Esc stop · ←/→ step · F fit · ⌘Z undo · ⌫ delete range · LMB orbit · RMB/MMB pan · wheel zoom · click marker to select");
        });
    }
}

struct Tabs<'a> {
    app: &'a mut App,
    cmds: &'a mut Vec<Command>,
    frame: usize,
}

impl TabViewer for Tabs<'_> {
    type Tab = Tab;

    fn title(&mut self, tab: &mut Tab) -> egui::WidgetText {
        match tab {
            Tab::Viewport => "3D View",
            Tab::Controls => "Controls",
            Tab::Edit => "Edit",
            Tab::Markers => "Markers",
            Tab::Plot => "Marker plot",
        }
        .into()
    }

    fn is_closeable(&self, _tab: &Tab) -> bool {
        false
    }

    fn id(&mut self, tab: &mut Tab) -> egui::Id {
        egui::Id::new(("mstudio_tab", *tab as u8))
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Tab) {
        let app = &mut *self.app;
        match tab {
            Tab::Viewport => {
                let input = match &app.doc {
                    Some(doc) => ViewportInput {
                        take: Some(&doc.take),
                        frame: self.frame,
                        selected: doc.state.selection.current_marker,
                        pattern_markers: &doc.state.selection.pattern_markers,
                        analysis_markers: &doc.state.selection.analysis_markers,
                        analysis_mode: doc.state.editing.is_analysis_mode,
                        reference_axis: doc.state.editing.reference_axis,
                        show_names: doc.state.view.show_names,
                        show_trajectory: doc.state.view.show_trajectory,
                        show_skeleton: doc.state.view.show_skeleton,
                        show_grid: app.show_grid,
                        coordinate_system: doc.state.view.coordinate_system,
                        visual: &app.visual,
                        dark: app.dark,
                    },
                    None => ViewportInput {
                        take: None,
                        frame: 0,
                        selected: None,
                        pattern_markers: &[],
                        analysis_markers: &[],
                        analysis_mode: false,
                        reference_axis: Default::default(),
                        show_names: false,
                        show_trajectory: false,
                        show_skeleton: false,
                        show_grid: app.show_grid,
                        coordinate_system: Default::default(),
                        visual: &app.visual,
                        dark: app.dark,
                    },
                };
                // timeline strip under the 3D view
                let (n, fps, sel) = match &app.doc {
                    Some(d) => (d.take.n_frames(), d.take.fps, d.state.selection.selected_frames),
                    None => (0, 30.0, None),
                };
                let tl = egui::Panel::bottom("timeline_strip")
                    .show_separator_line(false)
                    .show(ui, |ui| {
                        timeline(
                            ui,
                            TimelineInput {
                                n_frames: n,
                                fps,
                                frame: self.frame,
                                selection: sel,
                                state: &mut app.timeline_state,
                            },
                        )
                    })
                    .inner;
                if let Some(f) = tl.seek {
                    self.cmds.push(Command::Seek(f));
                }
                if let Some(sel) = tl.selection {
                    self.cmds.push(Command::SetSelection(sel));
                }
                let out = app.viewport.ui(ui, input);
                app.analysis_labels = out.analysis_labels;
                if let Some(m) = out.picked {
                    self.cmds.push(Command::SelectMarker(Some(m)));
                } else if out.clicked_empty {
                    if let Some(doc) = &app.doc {
                        if !doc.state.editing.is_analysis_mode && !doc.state.editing.pattern_selection_mode {
                            self.cmds.push(Command::SelectMarker(None));
                        }
                    }
                }
                if out.cycle_reference_axis {
                    self.cmds.push(Command::CycleReferenceAxis);
                }
            }
            Tab::Controls => crate::panels::controls(ui, app, self.cmds),
            Tab::Edit => crate::panels::edit(ui, app, self.cmds),
            Tab::Markers => crate::panels::marker_list(ui, app, self.cmds),
            Tab::Plot => {
                if let Some(doc) = &app.doc {
                    let out = marker_plot(
                        ui,
                        MarkerPlotInput {
                            take: &doc.take,
                            marker: doc.state.selection.current_marker,
                            frame: self.frame,
                            selection: doc.state.selection.selected_frames,
                            data_version: doc.data_version,
                            state: &mut app.plot_state,
                        },
                    );
                    if let Some(f) = out.seek {
                        self.cmds.push(Command::Seek(f));
                    }
                    if let Some(sel) = out.selection {
                        self.cmds.push(Command::SetSelection(sel));
                    }
                } else {
                    ui.centered_and_justified(|ui| ui.weak("no data"));
                }
            }
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let now = Instant::now();
        self.frame_times.push_back(now);
        while self.frame_times.front().is_some_and(|t| now.duration_since(*t) > Duration::from_secs(1)) {
            self.frame_times.pop_front();
        }

        // screenshots for automated checks
        if let Some(img) = ctx.input(|i| {
            i.events.iter().find_map(|e| match e {
                egui::Event::Screenshot { image, .. } => Some(image.clone()),
                _ => None,
            })
        }) {
            if let Some(path) = self.screenshot.take() {
                let [w, h] = img.size;
                let bytes: Vec<u8> = img.pixels.iter().flat_map(|c| c.to_array()).collect();
                match image::RgbaImage::from_raw(w as u32, h as u32, bytes).map(|i| i.save(&path)) {
                    Some(Ok(())) => eprintln!("screenshot saved: {}", path.display()),
                    other => eprintln!("screenshot failed: {other:?}"),
                }
            }
        }
        if self.screenshot.is_some()
            && !self.shot_requested
            && now.duration_since(self.started) > Duration::from_millis(1500)
        {
            self.shot_requested = true;
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
        }
        if let Some(t) = self.exit_at {
            if now >= t {
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }
        if self.screenshot.is_some() || self.exit_at.is_some() {
            // timers only advance when frames are produced; screenshots need a continuous stream
            ctx.request_repaint();
        }

        self.poll_worker();
        if self.worker.is_some() {
            ctx.request_repaint_after(Duration::from_millis(50));
        }

        // drag & drop a motion file or JSON folder
        let dropped: Vec<PathBuf> = ctx.input(|i| i.raw.dropped_files.iter().map(|f| f.path().to_path_buf()).collect());

        let frame = self.playback.tick(now);
        let mut cmds: Vec<Command> = dropped.into_iter().map(Command::Open).collect();
        self.shortcuts(&ctx, &mut cmds);

        egui::Panel::top("menu").show(ui, |ui| self.menu(ui, &mut cmds));
        egui::Panel::bottom("status").show(ui, |ui| self.status_bar(ui));

        let mut dock = self.dock.take().expect("dock state");
        DockArea::new(&mut dock)
            .style(Style::from_egui(ui.style().as_ref()))
            .show_close_buttons(false)
            .show_inside(ui, &mut Tabs { app: self, cmds: &mut cmds, frame });
        self.dock = Some(dock);

        if let Some(msg) = self.error.clone() {
            egui::Window::new("Error")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(&ctx, |ui| {
                    ui.label(msg);
                    if ui.button("OK").clicked() {
                        self.error = None;
                    }
                });
        }

        for cmd in cmds {
            self.apply(cmd, &ctx);
        }
        if self.playback.is_playing() {
            ctx.request_repaint();
        }
    }
}
