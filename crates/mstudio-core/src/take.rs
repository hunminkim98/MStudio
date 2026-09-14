//! `Take`: one motion-capture recording. Port of `core/data_manager.py`.
//!
//! The Python app keeps a wide `DataFrame` (`<Marker>_X/_Y/_Z` columns). Here
//! the same data is a dense `[frame, marker, xyz]` array in **f64** so that
//! filter outputs can be compared to the Python oracle at 1e-6; the renderer
//! converts to f32 when it uploads (`frames_f32`).

use ndarray::{s, Array1, Array3, Axis};

/// Half-open frame range `[start, end)` whose data changed and must be
/// re-uploaded to the GPU / re-analysed (plan rule R2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirtyRange {
    pub start: usize,
    pub end: usize,
}

impl DirtyRange {
    pub fn all(n_frames: usize) -> Self {
        Self { start: 0, end: n_frames }
    }

    pub fn single(frame: usize) -> Self {
        Self { start: frame, end: frame + 1 }
    }

    /// Inclusive frame bounds, as the UI selects them.
    pub fn inclusive(first: usize, last: usize) -> Self {
        let (a, b) = if first <= last { (first, last) } else { (last, first) };
        Self { start: a, end: b + 1 }
    }

    pub fn is_empty(&self) -> bool {
        self.end <= self.start
    }

    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    pub fn union(self, other: DirtyRange) -> DirtyRange {
        if self.is_empty() {
            return other;
        }
        if other.is_empty() {
            return self;
        }
        DirtyRange { start: self.start.min(other.start), end: self.end.max(other.end) }
    }
}

/// Axis-aligned bounds with the 10 % margin the Python app adds for the view
/// (`DataManager.calculate_data_limits`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Limits {
    pub x: (f64, f64),
    pub y: (f64, f64),
    pub z: (f64, f64),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Take {
    /// Marker names in column order. This order is also the GPU order.
    pub markers: Vec<String>,
    /// Data rate in Hz.
    pub fps: f64,
    /// `[n_frames, n_markers, 3]`, meters. `NaN` = missing sample.
    pub frames: Array3<f64>,
    /// Deep copy taken at load; `restore_original` copies it back.
    pub original: Array3<f64>,
    /// The `Frame#` column of the source file (TRC keeps the file's numbering).
    pub frame_numbers: Array1<i64>,
    /// The `Time` column of the source file, seconds.
    pub time: Array1<f64>,
}

impl Take {
    /// Build a take whose frame numbers are `1..=n` and time is `i / fps`.
    pub fn new(markers: Vec<String>, fps: f64, frames: Array3<f64>) -> Take {
        let n = frames.len_of(Axis(0));
        let frame_numbers = Array1::from_iter(1..=n as i64);
        let time = Array1::from_iter((0..n).map(|i| i as f64 / fps));
        Take::with_columns(markers, fps, frames, frame_numbers, time)
    }

    pub fn with_columns(
        markers: Vec<String>,
        fps: f64,
        frames: Array3<f64>,
        frame_numbers: Array1<i64>,
        time: Array1<f64>,
    ) -> Take {
        assert_eq!(frames.len_of(Axis(1)), markers.len(), "marker count must match frames axis 1");
        assert_eq!(frames.len_of(Axis(2)), 3, "frames axis 2 must be xyz");
        assert_eq!(frame_numbers.len(), frames.len_of(Axis(0)));
        assert_eq!(time.len(), frames.len_of(Axis(0)));
        let original = frames.clone();
        Take { markers, fps, frames, original, frame_numbers, time }
    }

    pub fn n_frames(&self) -> usize {
        self.frames.len_of(Axis(0))
    }

    pub fn n_markers(&self) -> usize {
        self.markers.len()
    }

    pub fn marker_index(&self, name: &str) -> Option<usize> {
        self.markers.iter().position(|m| m == name)
    }

    /// Position of a marker in a frame, `None` if any component is missing
    /// (`DataManager.get_marker_coordinates`).
    pub fn position(&self, frame: usize, marker: usize) -> Option<[f64; 3]> {
        let p = self.frames.slice(s![frame, marker, ..]);
        let v = [p[0], p[1], p[2]];
        if v.iter().any(|c| c.is_nan()) {
            None
        } else {
            Some(v)
        }
    }

    pub fn set_position(&mut self, frame: usize, marker: usize, value: Option<[f64; 3]>) -> DirtyRange {
        let v = value.unwrap_or([f64::NAN; 3]);
        let mut p = self.frames.slice_mut(s![frame, marker, ..]);
        p[0] = v[0];
        p[1] = v[1];
        p[2] = v[2];
        DirtyRange::single(frame)
    }

    /// Mark a marker missing over an inclusive frame range (edit mode delete).
    pub fn clear_range(&mut self, marker: usize, first: usize, last: usize) -> DirtyRange {
        let r = DirtyRange::inclusive(first, last.min(self.n_frames().saturating_sub(1)));
        self.frames.slice_mut(s![r.start..r.end, marker, ..]).fill(f64::NAN);
        r
    }

    /// Copy the load-time data back (`DataManager.restore_original_data`).
    pub fn restore_original(&mut self) -> DirtyRange {
        self.frames.assign(&self.original);
        DirtyRange::all(self.n_frames())
    }

    /// True when every marker is named `Keypoint_<i>` — 2D JSON data whose
    /// names are only known once a skeleton model is chosen.
    pub fn has_generic_names(&self) -> bool {
        !self.markers.is_empty() && self.markers.iter().all(|m| m.starts_with("Keypoint_"))
    }

    /// `DataManager.update_keypoint_names`: rename `Keypoint_<i>` markers to
    /// the skeleton node whose `id == i`. Returns whether anything changed.
    pub fn rename_markers(&mut self, model: &crate::skeleton::SkeletonModel) -> bool {
        if !self.has_generic_names() {
            return false;
        }
        let id_to_name = model.id_to_name();
        let mut changed = false;
        for (i, name) in self.markers.iter_mut().enumerate() {
            if let Some(new) = id_to_name.get(&(i as u32)) {
                if name != new {
                    *name = (*new).to_string();
                    changed = true;
                }
            }
        }
        changed
    }

    /// Raw min/max over all non-NaN samples, `None` when the take has no data.
    pub fn bounds(&self) -> Option<([f64; 3], [f64; 3])> {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for p in self.frames.rows() {
            for k in 0..3 {
                let v = p[k];
                if !v.is_nan() {
                    lo[k] = lo[k].min(v);
                    hi[k] = hi[k].max(v);
                }
            }
        }
        (lo[0].is_finite() && lo[1].is_finite() && lo[2].is_finite()).then_some((lo, hi))
    }

    /// Bounds with a 10 % margin per axis (`DataManager.calculate_data_limits`).
    pub fn data_limits(&self) -> Option<Limits> {
        let (lo, hi) = self.bounds()?;
        let pad = |k: usize| {
            let r = hi[k] - lo[k];
            (lo[k] - r * 0.1, hi[k] + r * 0.1)
        };
        Some(Limits { x: pad(0), y: pad(1), z: pad(2) })
    }

    /// Frame-major `[x, y, z, 0]` f32 stream for the GPU. Missing samples
    /// become `missing` on every component (a sentinel the shaders cull on).
    pub fn frames_f32(&self, missing: f32) -> Vec<[f32; 4]> {
        self.frames
            .rows()
            .into_iter()
            .map(|p| {
                if p.iter().any(|c| c.is_nan()) {
                    [missing, missing, missing, 0.0]
                } else {
                    [p[0] as f32, p[1] as f32, p[2] as f32, 0.0]
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn take() -> Take {
        let mut f = Array3::<f64>::zeros((4, 2, 3));
        for i in 0..4 {
            f[[i, 0, 0]] = i as f64;
            f[[i, 1, 1]] = -(i as f64);
        }
        f[[2, 1, 2]] = f64::NAN;
        Take::new(vec!["A".into(), "B".into()], 100.0, f)
    }

    #[test]
    fn position_is_none_when_any_component_missing() {
        let t = take();
        assert_eq!(t.position(1, 0), Some([1.0, 0.0, 0.0]));
        assert_eq!(t.position(2, 1), None);
    }

    #[test]
    fn restore_original_undoes_edits() {
        let mut t = take();
        let r = t.clear_range(0, 1, 2);
        assert_eq!(r, DirtyRange { start: 1, end: 3 });
        assert!(t.position(1, 0).is_none());
        assert_eq!(t.restore_original(), DirtyRange::all(4));
        assert_eq!(t.position(1, 0), Some([1.0, 0.0, 0.0]));
    }

    #[test]
    fn limits_add_ten_percent_margin_and_skip_nan() {
        let t = take();
        let l = t.data_limits().unwrap();
        let close = |a: (f64, f64), b: (f64, f64)| (a.0 - b.0).abs() < 1e-12 && (a.1 - b.1).abs() < 1e-12;
        assert!(close(l.x, (-0.3, 3.3)), "{:?}", l.x);
        assert!(close(l.y, (-3.3, 0.3)), "{:?}", l.y);
        assert!(close(l.z, (0.0, 0.0)), "{:?}", l.z);
    }

    #[test]
    fn frame_numbers_and_time_default_to_python_loader_convention() {
        let t = take();
        assert_eq!(t.frame_numbers.to_vec(), vec![1, 2, 3, 4]);
        assert_eq!(t.time[3], 0.03);
    }

    #[test]
    fn dirty_range_union_and_inclusive() {
        let a = DirtyRange::inclusive(5, 2);
        assert_eq!(a, DirtyRange { start: 2, end: 6 });
        assert_eq!(a.union(DirtyRange::single(10)), DirtyRange { start: 2, end: 11 });
        assert_eq!(DirtyRange { start: 3, end: 3 }.union(a), a);
    }
}
