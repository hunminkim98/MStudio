//! Python bindings for MStudio (`import mstudio`).
//!
//! `Take.frames`, `Take.original`, `Take.time` and `Take.frame_numbers` are
//! zero-copy NumPy views into the Rust arrays: the NumPy object keeps the
//! `Take` alive, and the `Take` never reallocates those buffers (every edit
//! is in place), which is the contract `borrow_from_array` needs.
//! Long-running routines release the GIL (`Python::detach`).

use std::path::PathBuf;

use mstudio_core::skeleton::SkeletonModel;
use mstudio_core::{DirtyRange, Take as CoreTake, APP_MODELS, DEFAULT_OUTLIER_THRESHOLD};
use mstudio_processing::{Filter as CoreFilter, InterpMethod};
use ndarray::{Array2, Array3};
use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;

fn io_err(e: mstudio_io::IoError) -> PyErr {
    PyIOError::new_err(e.to_string())
}

fn proc_err(e: mstudio_processing::ProcessingError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn report_err(e: mstudio_report::ReportError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

type Bounds = ((f64, f64, f64), (f64, f64, f64));

fn range_tuple(r: DirtyRange) -> (usize, usize) {
    (r.start, r.end)
}

fn model_by_name(name: &str) -> PyResult<&'static SkeletonModel> {
    SkeletonModel::by_name(name)
        .ok_or_else(|| PyValueError::new_err(format!("unknown skeleton model {name:?}; see mstudio.skeleton_models()")))
}

// ----------------------------------------------------------------- Filter --

/// A filter and its parameters. Build one with the static constructors
/// (`Filter.butterworth(order=4, cutoff_hz=10)` …); the defaults are the
/// desktop application's.
#[pyclass(module = "mstudio", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct Filter {
    inner: CoreFilter,
}

#[pymethods]
impl Filter {
    /// Zero-phase Butterworth low-pass; `order` must be even.
    #[staticmethod]
    #[pyo3(signature = (order = 4, cutoff_hz = 10.0))]
    fn butterworth(order: u32, cutoff_hz: f64) -> Filter {
        Filter { inner: CoreFilter::Butterworth { order, cutoff_hz } }
    }

    /// Butterworth applied to the derivative, then re-integrated.
    #[staticmethod]
    #[pyo3(signature = (order = 4, cutoff_hz = 10.0))]
    fn butterworth_on_speed(order: u32, cutoff_hz: f64) -> Filter {
        Filter { inner: CoreFilter::ButterworthOnSpeed { order, cutoff_hz } }
    }

    /// Constant-acceleration Kalman filter, optionally RTS-smoothed.
    #[staticmethod]
    #[pyo3(signature = (trust_ratio = 20.0, smooth = true))]
    fn kalman(trust_ratio: f64, smooth: bool) -> Filter {
        Filter { inner: CoreFilter::Kalman { trust_ratio, smooth } }
    }

    #[staticmethod]
    #[pyo3(signature = (sigma_kernel = 3.0))]
    fn gaussian(sigma_kernel: f64) -> Filter {
        Filter { inner: CoreFilter::Gaussian { sigma_kernel } }
    }

    /// LOWESS with `nb_values_used` points per local fit.
    #[staticmethod]
    #[pyo3(signature = (nb_values_used = 10.0))]
    fn loess(nb_values_used: f64) -> Filter {
        Filter { inner: CoreFilter::Loess { nb_values_used } }
    }

    #[staticmethod]
    #[pyo3(signature = (kernel_size = 3.0))]
    fn median(kernel_size: f64) -> Filter {
        Filter { inner: CoreFilter::Median { kernel_size } }
    }

    /// Name used by the Pose2Sim config dict (`"butterworth"`, `"kalman"`, `"LOESS"` …).
    #[getter]
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn __repr__(&self) -> String {
        format!("mstudio.Filter({:?})", self.inner)
    }
}

// ------------------------------------------------------------------- Take --

/// One motion-capture take: marker names, frame rate and an
/// `[n_frames, n_markers, 3]` float64 array in meters (`NaN` = missing).
#[pyclass(module = "mstudio")]
pub struct Take {
    inner: CoreTake,
}

/// Resolve a marker given as an index or a name.
fn marker_arg(take: &CoreTake, marker: &Bound<'_, PyAny>) -> PyResult<usize> {
    if let Ok(i) = marker.extract::<usize>() {
        return if i < take.n_markers() {
            Ok(i)
        } else {
            Err(PyValueError::new_err(format!("marker index {i} out of range (0..{})", take.n_markers())))
        };
    }
    let name: String = marker.extract()?;
    take.marker_index(&name).ok_or_else(|| PyValueError::new_err(format!("unknown marker {name:?}")))
}

fn last_or_end(take: &CoreTake, last: Option<usize>) -> usize {
    last.unwrap_or(take.n_frames().saturating_sub(1))
}

#[pymethods]
impl Take {
    /// `Take(markers, fps, frames)` — `frames` is `[n_frames, len(markers), 3]`
    /// and is copied; frame numbers are `1..=n` and time is `i / fps`.
    #[new]
    fn py_new(markers: Vec<String>, fps: f64, frames: PyReadonlyArray3<'_, f64>) -> PyResult<Take> {
        let shape = frames.shape();
        if shape[1] != markers.len() || shape[2] != 3 {
            return Err(PyValueError::new_err(format!(
                "frames must be [n_frames, {}, 3], got {:?}",
                markers.len(),
                shape
            )));
        }
        if fps <= 0.0 || !fps.is_finite() {
            return Err(PyValueError::new_err("fps must be positive"));
        }
        Ok(Take { inner: CoreTake::new(markers, fps, frames.as_array().to_owned()) })
    }

    /// Marker names in column order.
    #[getter]
    fn markers(&self) -> Vec<String> {
        self.inner.markers.clone()
    }

    #[setter]
    fn set_markers(&mut self, names: Vec<String>) -> PyResult<()> {
        if names.len() != self.inner.n_markers() {
            return Err(PyValueError::new_err(format!(
                "expected {} names, got {}",
                self.inner.n_markers(),
                names.len()
            )));
        }
        self.inner.markers = names;
        Ok(())
    }

    #[getter]
    fn fps(&self) -> f64 {
        self.inner.fps
    }

    #[setter]
    fn set_fps(&mut self, fps: f64) -> PyResult<()> {
        if fps <= 0.0 || !fps.is_finite() {
            return Err(PyValueError::new_err("fps must be positive"));
        }
        self.inner.fps = fps;
        Ok(())
    }

    #[getter]
    fn n_frames(&self) -> usize {
        self.inner.n_frames()
    }

    #[getter]
    fn n_markers(&self) -> usize {
        self.inner.n_markers()
    }

    /// The working data as a writable `[n_frames, n_markers, 3]` float64 view.
    /// Edits made through NumPy are seen by every other method immediately.
    #[getter]
    fn frames<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray3<f64>> {
        let take = this.borrow();
        // SAFETY: `inner.frames` is only ever modified in place (see module docs);
        // the returned array holds a reference to `this`, keeping the buffer alive.
        unsafe { PyArray3::borrow_from_array(&take.inner.frames, this.clone().into_any()) }
    }

    /// Copy new values into `frames` (same shape).
    #[setter]
    fn set_frames(&mut self, frames: PyReadonlyArray3<'_, f64>) -> PyResult<()> {
        if frames.shape() != self.inner.frames.shape() {
            return Err(PyValueError::new_err(format!(
                "shape {:?} does not match {:?}",
                frames.shape(),
                self.inner.frames.shape()
            )));
        }
        self.inner.frames.assign(&frames.as_array());
        Ok(())
    }

    /// The data as loaded (read-only view); `restore_original()` copies it back.
    #[getter]
    fn original<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray3<f64>> {
        let take = this.borrow();
        // SAFETY: as for `frames`.
        let arr = unsafe { PyArray3::borrow_from_array(&take.inner.original, this.clone().into_any()) };
        arr.readwrite().make_nonwriteable();
        arr
    }

    /// Time of each frame in seconds (read-only view).
    #[getter]
    fn time<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray1<f64>> {
        let take = this.borrow();
        // SAFETY: as for `frames`.
        let arr = unsafe { PyArray1::borrow_from_array(&take.inner.time, this.clone().into_any()) };
        arr.readwrite().make_nonwriteable();
        arr
    }

    /// The source file's `Frame#` column (read-only view).
    #[getter]
    fn frame_numbers<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray1<i64>> {
        let take = this.borrow();
        // SAFETY: as for `frames`.
        let arr = unsafe { PyArray1::borrow_from_array(&take.inner.frame_numbers, this.clone().into_any()) };
        arr.readwrite().make_nonwriteable();
        arr
    }

    /// Column index of a marker name.
    fn index(&self, name: &str) -> PyResult<usize> {
        self.inner.marker_index(name).ok_or_else(|| PyValueError::new_err(format!("unknown marker {name:?}")))
    }

    /// `(x, y, z)` of a marker in a frame, or `None` if any component is missing.
    fn position(&self, frame: usize, marker: &Bound<'_, PyAny>) -> PyResult<Option<(f64, f64, f64)>> {
        let m = marker_arg(&self.inner, marker)?;
        if frame >= self.inner.n_frames() {
            return Err(PyValueError::new_err(format!("frame {frame} out of range")));
        }
        Ok(self.inner.position(frame, m).map(|p| (p[0], p[1], p[2])))
    }

    /// Mark a marker missing over the inclusive frame range (edit-mode delete).
    /// Returns the half-open `(start, end)` range of frames changed.
    #[pyo3(signature = (marker, first = 0, last = None))]
    fn clear(&mut self, marker: &Bound<'_, PyAny>, first: usize, last: Option<usize>) -> PyResult<(usize, usize)> {
        let m = marker_arg(&self.inner, marker)?;
        let last = last_or_end(&self.inner, last);
        Ok(range_tuple(self.inner.clear_range(m, first, last)))
    }

    /// Copy the loaded data back over every edit.
    fn restore_original(&mut self) -> (usize, usize) {
        range_tuple(self.inner.restore_original())
    }

    /// True when the markers are numeric keypoint ids that a skeleton model can rename.
    fn has_generic_names(&self) -> bool {
        self.inner.has_generic_names()
    }

    /// Rename numeric keypoint ids to the model's joint names; returns whether anything changed.
    fn rename_markers(&mut self, model: &str) -> PyResult<bool> {
        Ok(self.inner.rename_markers(model_by_name(model)?))
    }

    /// Filter every axis of one marker over the whole column and write back
    /// the inclusive frame range (the application's *Filter* button).
    /// Returns the half-open `(start, end)` range written.
    #[pyo3(signature = (marker, filter, first = 0, last = None))]
    fn filter(
        &mut self,
        py: Python<'_>,
        marker: &Bound<'_, PyAny>,
        filter: &Filter,
        first: usize,
        last: Option<usize>,
    ) -> PyResult<(usize, usize)> {
        let m = marker_arg(&self.inner, marker)?;
        let last = last_or_end(&self.inner, last);
        let fps = self.inner.fps;
        let frames = &mut self.inner.frames;
        let f = &filter.inner;
        py.detach(|| mstudio_processing::apply_filter(frames, m, first, last, f, fps))
            .map(range_tuple)
            .map_err(proc_err)
    }

    /// Filter the whole take (all markers, or the given ones) in parallel.
    #[pyo3(signature = (filter, markers = None))]
    fn filter_all(
        &mut self,
        py: Python<'_>,
        filter: &Filter,
        markers: Option<Vec<Bound<'_, PyAny>>>,
    ) -> PyResult<(usize, usize)> {
        let idx: Vec<usize> = match markers {
            Some(list) => list.iter().map(|m| marker_arg(&self.inner, m)).collect::<PyResult<_>>()?,
            None => (0..self.inner.n_markers()).collect(),
        };
        let fps = self.inner.fps;
        let frames = &mut self.inner.frames;
        let f = &filter.inner;
        py.detach(|| mstudio_processing::filter_take(frames, &idx, f, fps)).map(range_tuple).map_err(proc_err)
    }

    /// Fill the missing samples of one marker inside the inclusive range.
    /// `method` is one of `linear, nearest, zero, slinear, quadratic, cubic,
    /// polynomial, spline`; `order` applies to the last two.
    /// Returns the half-open `(start, end)` range of frames written (empty if none).
    #[pyo3(signature = (marker, method = "linear", first = 0, last = None, order = 3))]
    fn interpolate(
        &mut self,
        py: Python<'_>,
        marker: &Bound<'_, PyAny>,
        method: &str,
        first: usize,
        last: Option<usize>,
        order: u32,
    ) -> PyResult<(usize, usize)> {
        let m = marker_arg(&self.inner, marker)?;
        let method = interp_method(method, order)?;
        let last = last_or_end(&self.inner, last);
        let frames = &mut self.inner.frames;
        py.detach(|| mstudio_processing::interpolate_in_range(frames, m, first, last, method))
            .map(range_tuple)
            .map_err(proc_err)
    }

    /// Pattern-based gap filling: reconstruct the target from the rigid
    /// offset to one or more reference markers.
    #[pyo3(signature = (marker, references, first = 0, last = None))]
    fn pattern_interpolate(
        &mut self,
        py: Python<'_>,
        marker: &Bound<'_, PyAny>,
        references: Vec<Bound<'_, PyAny>>,
        first: usize,
        last: Option<usize>,
    ) -> PyResult<(usize, usize)> {
        let m = marker_arg(&self.inner, marker)?;
        let refs: Vec<usize> = references.iter().map(|r| marker_arg(&self.inner, r)).collect::<PyResult<_>>()?;
        if refs.contains(&m) {
            return Err(PyValueError::new_err("the target marker cannot be one of its references"));
        }
        let last = last_or_end(&self.inner, last);
        let frames = &mut self.inner.frames;
        py.detach(|| mstudio_processing::pattern_interpolate(frames, m, &refs, first, last))
            .map(range_tuple)
            .map_err(proc_err)
    }

    /// Bone-length outlier flags `[n_markers, n_frames]` for the given
    /// `(parent, child)` marker pairs (see `skeleton_pairs`).
    #[pyo3(signature = (pairs, threshold = DEFAULT_OUTLIER_THRESHOLD))]
    fn outliers<'py>(
        &self,
        py: Python<'py>,
        pairs: Vec<Vec<Bound<'py, PyAny>>>,
        threshold: f64,
    ) -> PyResult<Bound<'py, PyArray2<bool>>> {
        let pairs: Vec<(usize, usize)> = pairs
            .iter()
            .map(|p| match p.as_slice() {
                [a, b] => Ok((marker_arg(&self.inner, a)?, marker_arg(&self.inner, b)?)),
                _ => Err(PyValueError::new_err("each pair must have exactly two markers")),
            })
            .collect::<PyResult<_>>()?;
        let frames = &self.inner.frames;
        let flags: Array2<bool> = py.detach(|| mstudio_core::detect_outliers(frames, &pairs, threshold));
        Ok(PyArray2::from_owned_array(py, flags))
    }

    /// `((min_x, min_y, min_z), (max_x, max_y, max_z))` over all valid samples.
    fn bounds(&self) -> Option<Bounds> {
        self.inner.bounds().map(|(lo, hi)| ((lo[0], lo[1], lo[2]), (hi[0], hi[1], hi[2])))
    }

    /// Write `.trc` or `.c3d` (by extension).
    fn save(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        let take = &self.inner;
        py.detach(|| mstudio_io::save(&path, take)).map_err(io_err)
    }

    /// Deep copy (frames, original and time).
    fn copy(&self) -> Take {
        Take { inner: self.inner.clone() }
    }

    fn __copy__(&self) -> Take {
        self.copy()
    }

    fn __len__(&self) -> usize {
        self.inner.n_frames()
    }

    fn __repr__(&self) -> String {
        format!(
            "mstudio.Take({} markers, {} frames, {} Hz)",
            self.inner.n_markers(),
            self.inner.n_frames(),
            self.inner.fps
        )
    }
}

fn interp_method(name: &str, order: u32) -> PyResult<InterpMethod> {
    InterpMethod::from_name(name, order).ok_or_else(|| {
        PyValueError::new_err(format!(
            "unknown interpolation method {name:?}; expected one of linear, nearest, zero, slinear, quadratic, cubic, polynomial, spline"
        ))
    })
}

// -------------------------------------------------------------- functions --

/// Load `.trc`, `.c3d`, or a Pose2Sim / Sports2D JSON folder (meters).
#[pyfunction]
fn load(py: Python<'_>, path: PathBuf) -> PyResult<Take> {
    py.detach(|| mstudio_io::load(&path)).map(|inner| Take { inner }).map_err(io_err)
}

/// Write a take as `.trc` or `.c3d` (by extension).
#[pyfunction]
fn save(py: Python<'_>, path: PathBuf, take: &Take) -> PyResult<()> {
    take.save(py, path)
}

/// Filter one 1-D column (`filter1d` in Pose2Sim). Returns a new array.
#[pyfunction]
fn filter1d<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
    filter: &Filter,
    fps: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let v: Vec<f64> = values.as_array().iter().copied().collect();
    let f = &filter.inner;
    let out = py.detach(|| mstudio_processing::filter_column(&v, f, fps)).map_err(proc_err)?;
    Ok(PyArray1::from_vec(py, out))
}

/// Fill the NaNs of one 1-D column the way `pandas.Series.interpolate` would.
#[pyfunction]
#[pyo3(signature = (values, method = "linear", order = 3))]
fn interpolate1d<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
    method: &str,
    order: u32,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let v: Vec<f64> = values.as_array().iter().copied().collect();
    let method = interp_method(method, order)?;
    let out = py.detach(|| mstudio_processing::interpolate_column(&v, method)).map_err(proc_err)?;
    Ok(PyArray1::from_vec(py, out))
}

/// Outlier flags `[n_markers, n_frames]` for a raw `[n_frames, n_markers, 3]` array.
#[pyfunction]
#[pyo3(signature = (frames, pairs, threshold = DEFAULT_OUTLIER_THRESHOLD))]
fn detect_outliers<'py>(
    py: Python<'py>,
    frames: PyReadonlyArray3<'py, f64>,
    pairs: Vec<Vec<usize>>,
    threshold: f64,
) -> PyResult<Bound<'py, PyArray2<bool>>> {
    let pairs: Vec<(usize, usize)> = pairs
        .iter()
        .map(|p| match p.as_slice() {
            [a, b] => Ok((*a, *b)),
            _ => Err(PyValueError::new_err("each pair must have exactly two marker indices")),
        })
        .collect::<PyResult<_>>()?;
    let n_markers = frames.shape()[1];
    if let Some(&(a, b)) = pairs.iter().find(|(a, b)| *a >= n_markers || *b >= n_markers) {
        return Err(PyValueError::new_err(format!("pair ({a}, {b}) out of range for {n_markers} markers")));
    }
    let arr: Array3<f64> = frames.as_array().to_owned();
    let flags = py.detach(|| mstudio_core::detect_outliers(&arr, &pairs, threshold));
    Ok(PyArray2::from_owned_array(py, flags))
}

/// Names of the skeleton models the application offers.
#[pyfunction]
fn skeleton_models() -> Vec<&'static str> {
    APP_MODELS.iter().map(|m| m.name).collect()
}

/// `(parent, child)` joint-name pairs of a skeleton model.
#[pyfunction]
fn skeleton_pair_names(model: &str) -> PyResult<Vec<(&'static str, &'static str)>> {
    Ok(model_by_name(model)?.pairs().collect())
}

/// `(parent, child)` marker-index pairs of a model that exist in `markers`
/// (`TRCViewer.update_skeleton_pairs`).
#[pyfunction]
fn skeleton_pairs(model: &str, markers: Vec<String>) -> PyResult<Vec<(usize, usize)>> {
    Ok(model_by_name(model)?.resolve_pairs(&markers))
}

/// Standard body segments found in `markers`: `(name, a, b)` index triples.
#[pyfunction]
fn auto_segments(markers: Vec<String>) -> Vec<(String, usize, usize)> {
    mstudio_processing::auto_segments(&markers).into_iter().map(|s| (s.name, s.a, s.b)).collect()
}

/// Standard joints found in `markers`: `(name, a, vertex, c)` index tuples.
#[pyfunction]
fn auto_joints(markers: Vec<String>) -> Vec<(String, usize, usize, usize)> {
    mstudio_processing::auto_joints(&markers).into_iter().map(|j| (j.name, j.a, j.vertex, j.c)).collect()
}

fn report_options(
    take: &CoreTake,
    title: &str,
    source: &str,
    skeleton: Option<&str>,
    max_points: usize,
) -> PyResult<mstudio_report::ReportOptions> {
    let (skeleton_model, skeleton_pairs) = match skeleton {
        Some(name) => {
            let model = model_by_name(name)?;
            (Some(model.name.to_string()), model.resolve_pairs(&take.markers))
        }
        None => (None, Vec::new()),
    };
    Ok(mstudio_report::ReportOptions {
        title: title.to_string(),
        source: source.to_string(),
        skeleton_model,
        skeleton_pairs,
        max_points,
    })
}

/// Render the interactive HTML analysis report and return it as a string.
#[pyfunction]
#[pyo3(signature = (take, title = "MStudio analysis report", source = "", skeleton = None, max_points = 3_000_000))]
fn build_report(
    py: Python<'_>,
    take: &Take,
    title: &str,
    source: &str,
    skeleton: Option<&str>,
    max_points: usize,
) -> PyResult<String> {
    let opts = report_options(&take.inner, title, source, skeleton, max_points)?;
    let inner = &take.inner;
    py.detach(|| mstudio_report::build_report(inner, &opts)).map_err(report_err)
}

/// Write the HTML analysis report to `path`.
#[pyfunction]
#[pyo3(signature = (take, path, title = "MStudio analysis report", source = None, skeleton = None, max_points = 3_000_000))]
fn write_report(
    py: Python<'_>,
    take: &Take,
    path: PathBuf,
    title: &str,
    source: Option<&str>,
    skeleton: Option<&str>,
    max_points: usize,
) -> PyResult<()> {
    let source = source
        .map(str::to_string)
        .unwrap_or_else(|| path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default());
    let opts = report_options(&take.inner, title, &source, skeleton, max_points)?;
    let inner = &take.inner;
    py.detach(|| mstudio_report::write_report(inner, &opts, &path)).map_err(report_err)
}

/// Open a file in the default browser.
#[pyfunction]
fn open_in_browser(path: PathBuf) -> PyResult<()> {
    mstudio_report::open_in_browser(path).map_err(report_err)
}

/// Open the MStudio desktop application and block until its window closes.
/// Must be called from the main thread, at most once per process.
///
/// `screenshot` (a PNG path) and `exit_after` (seconds) exist for automated
/// checks: the window saves an image once drawn and closes itself.
#[pyfunction]
#[pyo3(signature = (path = None, play = false, *, screenshot = None, exit_after = None))]
fn run(
    py: Python<'_>,
    path: Option<PathBuf>,
    play: bool,
    screenshot: Option<PathBuf>,
    exit_after: Option<f32>,
) -> PyResult<()> {
    let opts = mstudio_app::LaunchOptions { path, play, screenshot, exit_after, ..Default::default() };
    py.detach(|| mstudio_app::run(opts)).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("DEFAULT_OUTLIER_THRESHOLD", DEFAULT_OUTLIER_THRESHOLD)?;
    m.add(
        "INTERP_METHODS",
        PyList::new(m.py(), ["linear", "nearest", "zero", "slinear", "quadratic", "cubic", "polynomial", "spline"])?,
    )?;
    m.add_class::<Take>()?;
    m.add_class::<Filter>()?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(save, m)?)?;
    m.add_function(wrap_pyfunction!(filter1d, m)?)?;
    m.add_function(wrap_pyfunction!(interpolate1d, m)?)?;
    m.add_function(wrap_pyfunction!(detect_outliers, m)?)?;
    m.add_function(wrap_pyfunction!(skeleton_models, m)?)?;
    m.add_function(wrap_pyfunction!(skeleton_pair_names, m)?)?;
    m.add_function(wrap_pyfunction!(skeleton_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(auto_segments, m)?)?;
    m.add_function(wrap_pyfunction!(auto_joints, m)?)?;
    m.add_function(wrap_pyfunction!(build_report, m)?)?;
    m.add_function(wrap_pyfunction!(write_report, m)?)?;
    m.add_function(wrap_pyfunction!(open_in_browser, m)?)?;
    m.add_function(wrap_pyfunction!(run, m)?)?;
    Ok(())
}
