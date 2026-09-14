//! MStudio file I/O. Port of `MStudio/utils/dataLoader.py` and `dataSaver.py`
//! without any UI: dialogs and message boxes belong to the application crate.
//!
//! Every reader produces a [`mstudio_core::Take`] in meters.

pub mod c3d;
pub mod json;
mod pyfmt;
pub mod trc;

use std::path::Path;

pub use c3d::{read_c3d, write_c3d};
pub use json::read_json_folder;
pub use pyfmt::py_repr;
pub use trc::{parse_trc, read_trc, trc_to_string, write_trc};

use mstudio_core::Take;

#[derive(Debug, thiserror::Error)]
pub enum IoError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("parse error: {0}")]
    Parse(String),
    #[error("C3D error: {0}")]
    C3d(String),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unsupported file type: {0}")]
    Unsupported(String),
}

pub type Result<T> = std::result::Result<T, IoError>;

/// Load a motion file by extension (`.trc` or `.c3d`); directories are
/// treated as Pose2Sim / Sports2D JSON folders (Y-up).
pub fn load(path: impl AsRef<Path>) -> Result<Take> {
    let path = path.as_ref();
    if path.is_dir() {
        return read_json_folder(path, mstudio_core::CoordinateSystem::YUp);
    }
    match path.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref() {
        Some("trc") => read_trc(path),
        Some("c3d") => read_c3d(path),
        other => Err(IoError::Unsupported(other.unwrap_or("<none>").to_string())),
    }
}

/// Save by extension (`.trc` or `.c3d`).
pub fn save(path: impl AsRef<Path>, take: &Take) -> Result<()> {
    let path = path.as_ref();
    match path.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref() {
        Some("trc") => write_trc(path, take),
        Some("c3d") => write_c3d(path, take),
        other => Err(IoError::Unsupported(other.unwrap_or("<none>").to_string())),
    }
}
