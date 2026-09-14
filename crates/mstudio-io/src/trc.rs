//! TRC (OpenSim / Vicon tab-separated) reader and writer.
//!
//! Reader mirrors `dataLoader.read_data_from_trc`: data rate from header
//! line 3, marker names from line 4, samples after the 5 header lines (the
//! blank separator line is optional), empty cells → NaN. Writer mirrors `dataSaver.save_to_trc` byte for
//! byte (see `pyfmt`).

use std::path::Path;

use mstudio_core::Take;
use ndarray::{Array1, Array3};

use crate::pyfmt::py_repr;
use crate::{IoError, Result};

pub fn read_trc(path: impl AsRef<Path>) -> Result<Take> {
    let text = std::fs::read_to_string(path)?;
    parse_trc(&text)
}

pub fn parse_trc(text: &str) -> Result<Take> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() < 4 {
        return Err(IoError::Parse("TRC file has fewer than 4 header lines".into()));
    }
    let fps: f64 = lines[2].split('\t').next().and_then(|s| s.trim().parse().ok()).unwrap_or(30.0);

    let mut markers: Vec<String> = Vec::new();
    for name in lines[3].split('\t').skip(2) {
        let name = name.trim();
        if !name.is_empty() && !markers.iter().any(|m| m == name) {
            markers.push(name.to_string());
        }
    }
    if markers.is_empty() {
        return Err(IoError::Parse("TRC file declares no markers".into()));
    }

    let mut rows: Vec<(i64, f64, Vec<f64>)> = Vec::new();
    // Standard TRC has a blank line after the 5 header lines; files written by
    // MStudio v0.1.5 (pandas) do not. Skipping blanks handles both.
    for line in lines.iter().skip(5).filter(|l| !l.trim().is_empty()) {
        let fields: Vec<&str> = line.split('\t').collect();
        let cell = |i: usize| -> f64 {
            fields
                .get(i)
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(f64::NAN)
        };
        let frame = cell(0);
        let frame = if frame.is_nan() { rows.len() as i64 + 1 } else { frame as i64 };
        let time = cell(1);
        let values: Vec<f64> = (0..markers.len() * 3).map(|k| cell(2 + k)).collect();
        rows.push((frame, time, values));
    }
    if rows.is_empty() {
        return Err(IoError::Parse("TRC file has no data rows".into()));
    }

    let n = rows.len();
    let m = markers.len();
    let mut frames = Array3::<f64>::from_elem((n, m, 3), f64::NAN);
    let mut frame_numbers = Array1::<i64>::zeros(n);
    let mut time = Array1::<f64>::zeros(n);
    for (i, (fnum, t, values)) in rows.into_iter().enumerate() {
        frame_numbers[i] = fnum;
        time[i] = t;
        for (k, v) in values.into_iter().enumerate() {
            frames[[i, k / 3, k % 3]] = v;
        }
    }
    Ok(Take::with_columns(markers, fps, frames, frame_numbers, time))
}

pub fn write_trc(path: impl AsRef<Path>, take: &Take) -> Result<()> {
    let path = path.as_ref();
    let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
    std::fs::write(path, trc_to_string(take, name))?;
    Ok(())
}

/// The exact text `save_to_trc` produces; `file_name` goes into the first
/// header line (Python writes `os.path.basename(file_path)` there).
pub fn trc_to_string(take: &Take, file_name: &str) -> String {
    let n = take.n_frames();
    let fps = py_repr(take.fps);
    let mut s = String::with_capacity(n * take.n_markers() * 36 + 512);

    s.push_str(&format!("PathFileType\t4\t(X/Y/Z)\t{file_name}\n"));
    s.push_str("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n");
    s.push_str(&format!("{fps}\t{fps}\t{n}\t{}\tm\t{fps}\t1\t{n}\n", take.n_markers()));
    s.push_str("Frame#\tTime");
    for m in &take.markers {
        s.push('\t');
        s.push_str(m);
        s.push_str("\t\t");
    }
    s.push('\n');
    s.push_str("\t\t");
    s.push_str(&vec!["X\tY\tZ"; take.n_markers()].join("\t"));
    s.push('\n');

    for f in 0..n {
        s.push_str(&take.frame_numbers[f].to_string());
        s.push('\t');
        s.push_str(&py_repr(take.time[f]));
        for m in 0..take.n_markers() {
            for k in 0..3 {
                s.push('\t');
                let v = take.frames[[f, m, k]];
                if !v.is_nan() {
                    s.push_str(&py_repr(v));
                }
            }
        }
        s.push('\n');
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "PathFileType\t4\t(X/Y/Z)\tx.trc\n\
DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n\
50.0\t50.0\t2\t2\tm\t50.0\t1\t2\n\
Frame#\tTime\tA\t\t\tB\t\t\n\
\t\tX1\tY1\tZ1\tX2\tY2\tZ2\n\
\n\
1\t0.0\t1.5\t2.5\t3.5\t\t\t\n\
2\t0.02\t1.6\t2.6\t3.6\t4.0\t5.0\t6.0\n";

    #[test]
    fn parses_header_names_and_missing_cells() {
        let t = parse_trc(SAMPLE).unwrap();
        assert_eq!(t.fps, 50.0);
        assert_eq!(t.markers, vec!["A", "B"]);
        assert_eq!(t.n_frames(), 2);
        assert_eq!(t.position(0, 0), Some([1.5, 2.5, 3.5]));
        assert_eq!(t.position(0, 1), None);
        assert_eq!(t.position(1, 1), Some([4.0, 5.0, 6.0]));
        assert_eq!(t.frame_numbers.to_vec(), vec![1, 2]);
        assert_eq!(t.time[1], 0.02);
    }

    #[test]
    fn round_trips_through_text_including_nan() {
        let t = parse_trc(SAMPLE).unwrap();
        let text = trc_to_string(&t, "x.trc");
        assert!(text.starts_with("PathFileType\t4\t(X/Y/Z)\tx.trc\n"));
        assert!(text.contains("\n1\t0.0\t1.5\t2.5\t3.5\t\t\t\n"));
        let back = parse_trc(&text).unwrap();
        assert_eq!(back.markers, t.markers);
        assert_eq!(
            back.frames.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            t.frames.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn rejects_files_without_markers_or_rows() {
        assert!(parse_trc("a\nb\nc\nFrame#\tTime\n\n\n").is_err());
        assert!(parse_trc("a\nb\n50\nFrame#\tTime\tA\n\n\n").is_err());
    }
}
