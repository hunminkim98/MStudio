//! Analysis report as one self-contained HTML file.
//!
//! Replaces `utils/reportGenerator.py` (matplotlib PDF). Sections are the
//! same — dataset overview and quality, marker coordinates with statistics,
//! velocity and acceleration, segment lengths and axis angles, joint angles —
//! but the charts are interactive (Plotly, inlined so the file works offline)
//! and a marker selector / time range at the top re-filters every chart.
//! Print the page from the browser for a PDF.

use std::path::Path;

use mstudio_core::Take;
use mstudio_processing::analysis;
use mstudio_processing::{auto_joints, auto_segments};
use serde::Serialize;

const PLOTLY_JS: &str = include_str!("../assets/plotly-basic.min.js");
const TEMPLATE: &str = include_str!("template.html");

#[derive(Debug, thiserror::Error)]
pub enum ReportError {
    #[error("template error: {0}")]
    Template(#[from] minijinja::Error),
    #[error("serialisation error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("empty take")]
    Empty,
}

#[derive(Debug, Clone)]
pub struct ReportOptions {
    pub title: String,
    /// File name shown on the cover.
    pub source: String,
    pub skeleton_model: Option<String>,
    /// Resolved `(parent, child)` pairs; each becomes a `Parent-Child` segment
    /// in addition to the standard ones (as the Python report did).
    pub skeleton_pairs: Vec<(usize, usize)>,
    /// Upper bound on numbers embedded for the charts; longer takes are
    /// strided down (tables always use every frame).
    pub max_points: usize,
}

impl Default for ReportOptions {
    fn default() -> Self {
        Self {
            title: "MStudio analysis report".into(),
            source: String::new(),
            skeleton_model: None,
            skeleton_pairs: Vec::new(),
            max_points: 3_000_000,
        }
    }
}

// ------------------------------------------------------------ statistics --

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Stats {
    pub n: usize,
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
}

pub fn stats(values: impl Iterator<Item = f64>) -> Stats {
    let v: Vec<f64> = values.filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return Stats { n: 0, mean: f64::NAN, std: f64::NAN, min: f64::NAN, max: f64::NAN };
    }
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n; // population, like np.std
    Stats {
        n: v.len(),
        mean,
        std: var.sqrt(),
        min: v.iter().cloned().fold(f64::INFINITY, f64::min),
        max: v.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    }
}

#[derive(Debug, Serialize)]
struct MarkerRow {
    name: String,
    valid: usize,
    missing: usize,
    completeness: f64,
    x: Stats,
    y: Stats,
    z: Stats,
    speed: Stats,
    accel: Stats,
}

#[derive(Debug, Serialize)]
struct SeriesRow {
    name: String,
    length: Stats,
    angle_x: Stats,
    angle_y: Stats,
    angle_z: Stats,
}

#[derive(Debug, Serialize)]
struct JointRow {
    name: String,
    angle: Stats,
    rom: f64,
}

#[derive(Debug, Serialize)]
struct ChartData {
    time: Vec<f64>,
    stride: usize,
    markers: Vec<String>,
    /// `[marker][axis][sample]`
    coords: Vec<[Vec<f64>; 3]>,
    speed: Vec<Vec<f64>>,
    accel: Vec<Vec<f64>>,
    segment_names: Vec<String>,
    segment_length: Vec<Vec<f64>>,
    segment_angle: Vec<[Vec<f64>; 3]>,
    joint_names: Vec<String>,
    joint_angle: Vec<Vec<f64>>,
}

/// Everything the template needs; also useful for tests.
#[derive(Debug, Serialize)]
pub struct ReportModel {
    title: String,
    source: String,
    generated: String,
    skeleton_model: String,
    n_markers: usize,
    n_frames: usize,
    fps: f64,
    duration: f64,
    completeness: f64,
    missing_points: usize,
    markers: Vec<MarkerRow>,
    segments: Vec<SeriesRow>,
    joints: Vec<JointRow>,
    chart_json: String,
}

fn p(take: &Take, f: usize, m: usize) -> Option<[f64; 3]> {
    take.position(f, m)
}

fn nan3() -> [f64; 3] {
    [f64::NAN; 3]
}

#[allow(clippy::needless_range_loop)] // frame indices are the natural notation for central differences
pub fn build_model(take: &Take, opts: &ReportOptions) -> Result<ReportModel, ReportError> {
    let n = take.n_frames();
    let nm = take.n_markers();
    if n == 0 || nm == 0 {
        return Err(ReportError::Empty);
    }
    let fps = take.fps;

    // kinematics (full resolution)
    let mut vel: Vec<Vec<[f64; 3]>> = vec![vec![nan3(); n]; nm];
    let mut acc: Vec<Vec<[f64; 3]>> = vec![vec![nan3(); n]; nm];
    for m in 0..nm {
        for f in 1..n.saturating_sub(1) {
            if let (Some(a), Some(b)) = (p(take, f - 1, m), p(take, f + 1, m)) {
                if let Some(v) = analysis::velocity(a, b, fps) {
                    vel[m][f] = v;
                }
            }
        }
        for f in 2..n.saturating_sub(2) {
            let (vp, vn) = (vel[m][f - 1], vel[m][f + 1]);
            if let Some(a) = analysis::acceleration(vp, vn, fps) {
                acc[m][f] = a;
            }
        }
    }
    let mag = |v: [f64; 3]| {
        if v.iter().any(|c| c.is_nan()) {
            f64::NAN
        } else {
            (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
        }
    };
    let speed: Vec<Vec<f64>> = vel.iter().map(|s| s.iter().map(|v| mag(*v)).collect()).collect();
    let accel: Vec<Vec<f64>> = acc.iter().map(|s| s.iter().map(|v| mag(*v)).collect()).collect();

    // marker table
    let mut missing_points = 0usize;
    let markers: Vec<MarkerRow> = (0..nm)
        .map(|m| {
            let valid = (0..n).filter(|&f| p(take, f, m).is_some()).count();
            let missing = n - valid;
            missing_points += missing;
            let col = |k: usize| (0..n).map(move |f| take.frames[[f, m, k]]);
            MarkerRow {
                name: take.markers[m].clone(),
                valid,
                missing,
                completeness: 100.0 * valid as f64 / n as f64,
                x: stats(col(0)),
                y: stats(col(1)),
                z: stats(col(2)),
                speed: stats(speed[m].iter().copied()),
                accel: stats(accel[m].iter().copied()),
            }
        })
        .collect();
    let completeness = 100.0 * (n * nm - missing_points) as f64 / (n * nm) as f64;

    // segments: standard patterns, then every skeleton pair not already listed
    let mut seg_defs: Vec<(String, usize, usize)> =
        auto_segments(&take.markers).into_iter().map(|s| (s.name, s.a, s.b)).collect();
    for &(a, b) in &opts.skeleton_pairs {
        // (Python deduplicated by name only, so "Thigh_R" and "RHip-RKnee" both appeared.)
        if a < nm && b < nm && !seg_defs.iter().any(|(_, x, y)| (*x == a && *y == b) || (*x == b && *y == a)) {
            seg_defs.push((format!("{}-{}", take.markers[a], take.markers[b]), a, b));
        }
    }
    let axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut seg_len: Vec<Vec<f64>> = Vec::new();
    let mut seg_ang: Vec<[Vec<f64>; 3]> = Vec::new();
    let mut segments = Vec::new();
    for (name, a, b) in &seg_defs {
        let mut len = vec![f64::NAN; n];
        let mut ang = [vec![f64::NAN; n], vec![f64::NAN; n], vec![f64::NAN; n]];
        for f in 0..n {
            if let (Some(pa), Some(pb)) = (p(take, f, *a), p(take, f, *b)) {
                len[f] = analysis::distance(pa, pb);
                for (k, axis) in axes.iter().enumerate() {
                    ang[k][f] = analysis::segment_angle(pa, pb, *axis).unwrap_or(f64::NAN);
                }
            }
        }
        segments.push(SeriesRow {
            name: name.clone(),
            length: stats(len.iter().copied()),
            angle_x: stats(ang[0].iter().copied()),
            angle_y: stats(ang[1].iter().copied()),
            angle_z: stats(ang[2].iter().copied()),
        });
        seg_len.push(len);
        seg_ang.push(ang);
    }

    // joints
    let joint_defs = auto_joints(&take.markers);
    let mut joint_angle: Vec<Vec<f64>> = Vec::new();
    let mut joints = Vec::new();
    for j in &joint_defs {
        let angles: Vec<f64> = (0..n)
            .map(|f| match (p(take, f, j.a), p(take, f, j.vertex), p(take, f, j.c)) {
                (Some(a), Some(v), Some(c)) => analysis::joint_angle(a, v, c).unwrap_or(f64::NAN),
                _ => f64::NAN,
            })
            .collect();
        let st = stats(angles.iter().copied());
        joints.push(JointRow { name: j.name.clone(), rom: st.max - st.min, angle: st });
        joint_angle.push(angles);
    }

    // chart data (strided)
    let total = n * (nm * 3 + nm * 2 + seg_defs.len() * 4 + joint_defs.len());
    let stride = total.div_ceil(opts.max_points.max(1)).max(1);
    let idx: Vec<usize> = (0..n).step_by(stride).collect();
    let pick = |v: &[f64]| idx.iter().map(|&i| v[i]).collect::<Vec<f64>>();
    let chart = ChartData {
        time: idx.iter().map(|&i| take.time[i]).collect(),
        stride,
        markers: take.markers.clone(),
        coords: (0..nm)
            .map(|m| {
                let col = |k: usize| idx.iter().map(|&f| take.frames[[f, m, k]]).collect::<Vec<f64>>();
                [col(0), col(1), col(2)]
            })
            .collect(),
        speed: speed.iter().map(|s| pick(s)).collect(),
        accel: accel.iter().map(|s| pick(s)).collect(),
        segment_names: seg_defs.iter().map(|(nme, _, _)| nme.clone()).collect(),
        segment_length: seg_len.iter().map(|s| pick(s)).collect(),
        segment_angle: seg_ang.iter().map(|a| [pick(&a[0]), pick(&a[1]), pick(&a[2])]).collect(),
        joint_names: joint_defs.iter().map(|j| j.name.clone()).collect(),
        joint_angle: joint_angle.iter().map(|s| pick(s)).collect(),
    };

    Ok(ReportModel {
        title: opts.title.clone(),
        source: opts.source.clone(),
        generated: now_string(),
        skeleton_model: opts.skeleton_model.clone().unwrap_or_else(|| "none".into()),
        n_markers: nm,
        n_frames: n,
        fps,
        duration: n as f64 / fps,
        completeness,
        missing_points,
        markers,
        segments,
        joints,
        chart_json: serde_json::to_string(&chart)?,
    })
}

fn now_string() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    // civil date from epoch days (Howard Hinnant's algorithm), UTC
    let days = (secs / 86_400) as i64;
    let (h, mi) = ((secs % 86_400) / 3600, (secs % 3600) / 60);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02} {h:02}:{mi:02} UTC")
}

/// Render the complete HTML document.
pub fn build_report(take: &Take, opts: &ReportOptions) -> Result<String, ReportError> {
    let model = build_model(take, opts)?;
    let mut env = minijinja::Environment::new();
    env.add_template("report.html", TEMPLATE)?;
    let tmpl = env.get_template("report.html")?;
    let html = tmpl.render(minijinja::context! {
        r => minijinja::Value::from_serialize(&model),
        chart_json => minijinja::Value::from_safe_string(model.chart_json.clone()),
        plotly_js => minijinja::Value::from_safe_string(PLOTLY_JS.to_string()),
    })?;
    Ok(html)
}

pub fn write_report(take: &Take, opts: &ReportOptions, path: impl AsRef<Path>) -> Result<(), ReportError> {
    std::fs::write(path, build_report(take, opts)?)?;
    Ok(())
}

/// Open a written report with the system's default browser.
pub fn open_in_browser(path: impl AsRef<Path>) -> Result<(), ReportError> {
    open::that(path.as_ref())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    fn take() -> Take {
        let n = 50;
        let names = ["RHip", "RKnee", "RAnkle", "Neck"];
        let mut f = Array3::<f64>::zeros((n, 4, 3));
        for i in 0..n {
            let t = i as f64 / 50.0;
            f[[i, 0, 1]] = 1.0;
            f[[i, 1, 0]] = 0.1 * t.sin();
            f[[i, 1, 1]] = 0.5;
            f[[i, 2, 1]] = 0.0;
            f[[i, 2, 2]] = 0.2 * t;
            f[[i, 3, 1]] = 1.5;
        }
        f[[10, 2, 0]] = f64::NAN;
        Take::new(names.iter().map(|s| s.to_string()).collect(), 50.0, f)
    }

    #[test]
    fn model_has_expected_sections() {
        let m = build_model(&take(), &ReportOptions { skeleton_pairs: vec![(0, 1), (1, 2)], ..Default::default() })
            .unwrap();
        assert_eq!(m.n_frames, 50);
        assert_eq!(m.missing_points, 1);
        assert!((m.completeness - 99.5).abs() < 1e-9);
        assert!(m.segments.iter().any(|s| s.name == "Thigh_R"));
        assert!(m.segments.iter().any(|s| s.name == "Trunk"));
        assert!(!m.segments.iter().any(|s| s.name == "RHip-RKnee"), "pair duplicates a standard segment");
        assert!(m.joints.iter().any(|j| j.name == "Knee_R"));
        assert!(m.markers[2].speed.n > 0);
        assert!(!m.chart_json.contains("NaN"));
        assert!(m.chart_json.contains("\"markers\":[\"RHip\",\"RKnee\",\"RAnkle\",\"Neck\"]"));
    }

    #[test]
    fn stride_limits_embedded_points() {
        let m = build_model(&take(), &ReportOptions { max_points: 100, ..Default::default() }).unwrap();
        let v: serde_json::Value = serde_json::from_str(&m.chart_json).unwrap();
        assert!(v["stride"].as_u64().unwrap() > 1);
        assert!(v["time"].as_array().unwrap().len() < 50);
    }

    #[test]
    fn html_is_self_contained_and_writes() {
        let html = build_report(&take(), &ReportOptions::default()).unwrap();
        assert!(
            html.contains("<html")
                && html.contains("plotly.js")
                && html.contains("Knee_R")
                && html.contains("@media print")
        );
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("r.html");
        write_report(&take(), &ReportOptions::default(), &path).unwrap();
        assert!(std::fs::metadata(&path).unwrap().len() > 1_000_000);
    }

    #[test]
    fn stats_are_nan_aware() {
        let s = stats([1.0, f64::NAN, 3.0].into_iter());
        assert_eq!((s.n, s.mean, s.min, s.max), (2, 2.0, 1.0, 3.0));
        assert!(stats(std::iter::empty()).mean.is_nan());
    }
}
