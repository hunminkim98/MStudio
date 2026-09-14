//! MStudio core data model.
//!
//! Port of `MStudio/core/` from the Python application (v0.1.5). Pure data and
//! algorithms: no I/O, no GPU, no UI. Behaviour is pinned to the Python oracle
//! by the golden files in `tests/golden/` (see `docs/CROSS_PLATFORM_PLAN.md`).

pub mod outliers;
pub mod playback;
pub mod skeleton;
mod skeleton_tables;
pub mod state;
pub mod take;
pub mod visual;

pub use outliers::{detect_outliers, DEFAULT_OUTLIER_THRESHOLD};
pub use playback::Playback;
pub use skeleton::{Node, SkeletonModel, ALL_MODELS, APP_MODELS};
pub use state::{CoordinateSystem, EditingState, ReferenceAxis, SelectionState, StateManager, ViewState};
pub use take::{DirtyRange, Limits, Take};
pub use visual::{ColorScheme, MarkerVisualConfig, Rgb, SkeletonVisualConfig, VisualSettings};
