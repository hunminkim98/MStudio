//! Application state. Port of `core/state_manager.py` without the callback
//! lists — an immediate-mode UI reads the state every frame instead.
//!
//! Markers are referenced by index into `Take::markers`, not by name.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CoordinateSystem {
    #[default]
    YUp,
    ZUp,
}

impl CoordinateSystem {
    pub fn is_z_up(self) -> bool {
        self == CoordinateSystem::ZUp
    }

    pub fn toggled(self) -> Self {
        match self {
            CoordinateSystem::YUp => CoordinateSystem::ZUp,
            CoordinateSystem::ZUp => CoordinateSystem::YUp,
        }
    }

    /// The string the Python app used (`"y-up"` / `"z-up"`).
    pub fn label(self) -> &'static str {
        match self {
            CoordinateSystem::YUp => "y-up",
            CoordinateSystem::ZUp => "z-up",
        }
    }
}

/// Reference axis for segment-angle analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReferenceAxis {
    #[default]
    X,
    Y,
    Z,
}

impl ReferenceAxis {
    pub fn next(self) -> Self {
        match self {
            ReferenceAxis::X => ReferenceAxis::Y,
            ReferenceAxis::Y => ReferenceAxis::Z,
            ReferenceAxis::Z => ReferenceAxis::X,
        }
    }

    pub fn vector(self) -> [f64; 3] {
        match self {
            ReferenceAxis::X => [1.0, 0.0, 0.0],
            ReferenceAxis::Y => [0.0, 1.0, 0.0],
            ReferenceAxis::Z => [0.0, 0.0, 1.0],
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            ReferenceAxis::X => "X",
            ReferenceAxis::Y => "Y",
            ReferenceAxis::Z => "Z",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ViewState {
    pub show_names: bool,
    pub show_trajectory: bool,
    pub show_skeleton: bool,
    pub coordinate_system: CoordinateSystem,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct SelectionState {
    pub current_marker: Option<usize>,
    /// Reference markers for pattern-based interpolation, in selection order.
    pub pattern_markers: Vec<usize>,
    /// Up to `MAX_ANALYSIS_MARKERS`, in selection order.
    pub analysis_markers: Vec<usize>,
    /// Inclusive `(first, last)` frame selection on the marker plot.
    pub selected_frames: Option<(usize, usize)>,
}

pub const MAX_ANALYSIS_MARKERS: usize = 3;

#[derive(Debug, Clone, PartialEq)]
pub struct EditingState {
    pub is_editing: bool,
    pub is_analysis_mode: bool,
    pub pattern_selection_mode: bool,
    pub filter_type: String,
    pub interp_method: String,
    pub interp_order: u32,
    pub reference_axis: ReferenceAxis,
}

impl Default for EditingState {
    fn default() -> Self {
        Self {
            is_editing: false,
            is_analysis_mode: false,
            pattern_selection_mode: false,
            filter_type: "butterworth".into(),
            interp_method: "linear".into(),
            interp_order: 3,
            reference_axis: ReferenceAxis::X,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct StateManager {
    pub view: ViewState,
    pub selection: SelectionState,
    pub editing: EditingState,
    /// Name of the active skeleton model (`None` = "No skeleton").
    pub skeleton_model: Option<&'static str>,
    /// Resolved `(parent, child)` marker-index pairs for the active model.
    pub skeleton_pairs: Vec<(usize, usize)>,
}

impl StateManager {
    pub fn new() -> Self {
        Self::default()
    }

    // ---- view

    pub fn toggle_marker_names(&mut self) -> bool {
        self.view.show_names = !self.view.show_names;
        self.view.show_names
    }

    pub fn toggle_trajectory(&mut self) -> bool {
        self.view.show_trajectory = !self.view.show_trajectory;
        self.view.show_trajectory
    }

    pub fn toggle_skeleton(&mut self) -> bool {
        self.view.show_skeleton = !self.view.show_skeleton;
        self.view.show_skeleton
    }

    pub fn toggle_coordinate_system(&mut self) -> CoordinateSystem {
        self.view.coordinate_system = self.view.coordinate_system.toggled();
        self.view.coordinate_system
    }

    // ---- selection

    pub fn set_current_marker(&mut self, marker: Option<usize>) {
        self.selection.current_marker = marker;
    }

    pub fn add_pattern_marker(&mut self, marker: usize) -> bool {
        if self.selection.pattern_markers.contains(&marker) {
            return false;
        }
        self.selection.pattern_markers.push(marker);
        true
    }

    pub fn remove_pattern_marker(&mut self, marker: usize) -> bool {
        let before = self.selection.pattern_markers.len();
        self.selection.pattern_markers.retain(|&m| m != marker);
        self.selection.pattern_markers.len() != before
    }

    /// Returns whether the marker is now selected.
    pub fn toggle_pattern_marker(&mut self, marker: usize) -> bool {
        if self.remove_pattern_marker(marker) {
            false
        } else {
            self.add_pattern_marker(marker)
        }
    }

    pub fn clear_pattern_markers(&mut self) {
        self.selection.pattern_markers.clear();
    }

    /// Adds an analysis marker unless it is already selected or the limit is
    /// reached. Returns whether it was added.
    pub fn add_analysis_marker(&mut self, marker: usize) -> bool {
        let list = &mut self.selection.analysis_markers;
        if list.contains(&marker) || list.len() >= MAX_ANALYSIS_MARKERS {
            return false;
        }
        list.push(marker);
        true
    }

    pub fn remove_analysis_marker(&mut self, marker: usize) -> bool {
        let before = self.selection.analysis_markers.len();
        self.selection.analysis_markers.retain(|&m| m != marker);
        self.selection.analysis_markers.len() != before
    }

    pub fn clear_analysis_markers(&mut self) {
        self.selection.analysis_markers.clear();
    }

    pub fn set_selected_frames(&mut self, range: Option<(usize, usize)>) {
        self.selection.selected_frames = range.map(|(a, b)| (a.min(b), a.max(b)));
    }

    // ---- editing

    pub fn set_editing_mode(&mut self, enabled: bool) {
        self.editing.is_editing = enabled;
    }

    pub fn set_analysis_mode(&mut self, enabled: bool) {
        if self.editing.is_analysis_mode != enabled {
            self.editing.is_analysis_mode = enabled;
            if !enabled {
                self.clear_analysis_markers();
            }
        }
    }

    pub fn toggle_analysis_mode(&mut self) -> bool {
        let next = !self.editing.is_analysis_mode;
        self.set_analysis_mode(next);
        next
    }

    pub fn cycle_reference_axis(&mut self) -> ReferenceAxis {
        self.editing.reference_axis = self.editing.reference_axis.next();
        self.editing.reference_axis
    }

    pub fn set_pattern_selection_mode(&mut self, enabled: bool) {
        if self.editing.pattern_selection_mode != enabled {
            self.editing.pattern_selection_mode = enabled;
            if !enabled {
                self.clear_pattern_markers();
            }
        }
    }

    // ---- skeleton

    /// `StateManager.set_skeleton_model` + `update_skeleton_pairs` combined:
    /// resolves pairs against the take's markers and turns skeleton display on
    /// or off accordingly.
    pub fn set_skeleton_model(&mut self, model: Option<&'static crate::skeleton::SkeletonModel>, markers: &[String]) {
        match model {
            None => {
                self.skeleton_model = None;
                self.skeleton_pairs.clear();
                self.view.show_skeleton = false;
            }
            Some(m) => {
                self.skeleton_model = Some(m.name);
                self.skeleton_pairs = m.resolve_pairs(markers);
                self.view.show_skeleton = true;
            }
        }
    }

    /// Everything back to defaults (new file loaded).
    pub fn reset(&mut self) {
        *self = StateManager::default();
    }

    /// Drop any marker reference that no longer exists (marker list changed).
    pub fn clamp_to_markers(&mut self, n_markers: usize) {
        let ok = |m: &usize| *m < n_markers;
        if !self.selection.current_marker.as_ref().is_some_and(ok) {
            self.selection.current_marker = None;
        }
        self.selection.pattern_markers.retain(ok);
        self.selection.analysis_markers.retain(ok);
        self.skeleton_pairs.retain(|(a, b)| ok(a) && ok(b));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analysis_markers_are_capped_at_three_and_unique() {
        let mut s = StateManager::new();
        s.set_analysis_mode(true);
        assert!(s.add_analysis_marker(1));
        assert!(!s.add_analysis_marker(1));
        assert!(s.add_analysis_marker(2));
        assert!(s.add_analysis_marker(3));
        assert!(!s.add_analysis_marker(4));
        assert_eq!(s.selection.analysis_markers, vec![1, 2, 3]);
        s.set_analysis_mode(false);
        assert!(s.selection.analysis_markers.is_empty());
    }

    #[test]
    fn reference_axis_cycles_x_y_z() {
        let mut s = StateManager::new();
        assert_eq!(s.cycle_reference_axis(), ReferenceAxis::Y);
        assert_eq!(s.cycle_reference_axis(), ReferenceAxis::Z);
        assert_eq!(s.cycle_reference_axis(), ReferenceAxis::X);
        assert_eq!(ReferenceAxis::Z.vector(), [0.0, 0.0, 1.0]);
    }

    #[test]
    fn skeleton_model_resolves_pairs_and_toggles_display() {
        let markers: Vec<String> = ["Hip", "RHip", "RKnee"].iter().map(|s| s.to_string()).collect();
        let mut s = StateManager::new();
        s.set_skeleton_model(Some(&crate::skeleton::HALPE_26), &markers);
        assert!(s.view.show_skeleton);
        assert_eq!(s.skeleton_pairs, vec![(0, 1), (1, 2)]);
        s.set_skeleton_model(None, &markers);
        assert!(!s.view.show_skeleton);
        assert!(s.skeleton_pairs.is_empty());
    }

    #[test]
    fn selected_frames_are_ordered() {
        let mut s = StateManager::new();
        s.set_selected_frames(Some((9, 4)));
        assert_eq!(s.selection.selected_frames, Some((4, 9)));
    }
}
