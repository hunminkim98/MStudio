//! Pose2Sim / Sports2D / OpenPose JSON folder reader. Port of
//! `dataLoader.read_data_from_json_folder`.
//!
//! One JSON file per frame, sorted by the trailing number in the file name.
//! The first person's `pose_keypoints_2d` (x, y, confidence triplets, pixels)
//! becomes a marker set named `Keypoint_<i>` (renamed once a skeleton model
//! is chosen). Pixels → meters by /1000, image Y is flipped, and the first
//! usable frame is translated so its lowest point sits at the origin.

use std::path::Path;

use mstudio_core::{CoordinateSystem, Take};
use ndarray::Array3;
use serde_json::Value;

use crate::{IoError, Result};

/// Frame rate assumed for JSON folders (the files carry none).
pub const JSON_FRAME_RATE: f64 = 30.0;

pub fn read_json_folder(dir: impl AsRef<Path>, coords: CoordinateSystem) -> Result<Take> {
    let dir = dir.as_ref();
    let mut files: Vec<String> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.to_ascii_lowercase().ends_with(".json"))
        .collect();
    if files.is_empty() {
        return Err(IoError::Parse("No JSON files found in the folder".into()));
    }
    files.sort(); // deterministic tie-break, then Python's stable sort by number
    files.sort_by_key(|n| trailing_number(n));

    let first: Value = serde_json::from_str(&std::fs::read_to_string(dir.join(&files[0]))?)?;
    let people = first.get("people").and_then(Value::as_array).filter(|p| !p.is_empty());
    let Some(people) = people else {
        return Err(IoError::Parse("Invalid JSON format: 'people' array not found or empty".into()));
    };
    let n_kp = people[0].get("pose_keypoints_2d").and_then(Value::as_array).map_or(0, |k| k.len() / 3);
    if n_kp == 0 {
        return Err(IoError::Parse("First JSON file has no pose_keypoints_2d".into()));
    }
    let markers: Vec<String> = (0..n_kp).map(|i| format!("Keypoint_{i}")).collect();

    let n = files.len();
    let mut frames = Array3::<f64>::from_elem((n, n_kp, 3), f64::NAN);
    for (f, name) in files.iter().enumerate() {
        let v: Value = serde_json::from_str(&std::fs::read_to_string(dir.join(name))?)?;
        let Some(person) = v.get("people").and_then(Value::as_array).and_then(|p| p.first()) else { continue };
        let Some(kp) = person.get("pose_keypoints_2d").and_then(Value::as_array) else { continue };
        for j in 0..n_kp {
            if j * 3 + 2 >= kp.len() {
                continue;
            }
            let (x, y, c) = (num(&kp[j * 3]), num(&kp[j * 3 + 1]), num(&kp[j * 3 + 2]));
            if c > 0.0 {
                let px = x / 1000.0;
                let py = -y / 1000.0; // image Y points down
                let (yy, zz) = match coords {
                    CoordinateSystem::YUp => (py, 0.0),
                    CoordinateSystem::ZUp => (0.0, py),
                };
                frames[[f, j, 0]] = px;
                frames[[f, j, 1]] = yy;
                frames[[f, j, 2]] = zz;
            }
        }
    }

    center_at_origin(&mut frames);
    Ok(Take::with_columns(
        markers,
        JSON_FRAME_RATE,
        frames,
        (0..n as i64).collect(),
        (0..n).map(|i| i as f64 / JSON_FRAME_RATE).collect(),
    ))
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

fn trailing_number(name: &str) -> u64 {
    let stem = &name[..name.len() - ".json".len()];
    let digits: String = stem.chars().rev().take_while(|c| c.is_ascii_digit()).collect();
    digits.chars().rev().collect::<String>().parse().unwrap_or(0)
}

/// Python: find the first frame with ≥ 3 valid X values, take the centroid of
/// its fully valid markers, and translate every frame so that X/Z are centred
/// and the lowest Y of that frame is at 0.
fn center_at_origin(frames: &mut Array3<f64>) {
    let (n, m, _) = frames.dim();
    let first_valid = (0..n).find(|&f| (0..m).filter(|&j| !frames[[f, j, 0]].is_nan()).count() >= 3);
    let Some(f0) = first_valid else { return };

    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut cz = 0.0;
    let mut min_y = f64::INFINITY;
    let mut count = 0usize;
    for j in 0..m {
        let (x, y, z) = (frames[[f0, j, 0]], frames[[f0, j, 1]], frames[[f0, j, 2]]);
        if !x.is_nan() && !y.is_nan() && !z.is_nan() {
            cx += x;
            cy += y;
            cz += z;
            min_y = min_y.min(y);
            count += 1;
        }
    }
    if count == 0 {
        return;
    }
    cx /= count as f64;
    cy /= count as f64;
    cz /= count as f64;
    let y_shift = cy - (cy - min_y); // == min_y, written as the Python does it

    for f in 0..n {
        for j in 0..m {
            if !frames[[f, j, 0]].is_nan() {
                frames[[f, j, 0]] -= cx;
            }
            if !frames[[f, j, 1]].is_nan() {
                frames[[f, j, 1]] -= y_shift;
            }
            if !frames[[f, j, 2]].is_nan() {
                frames[[f, j, 2]] -= cz;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_frame(dir: &Path, name: &str, kp: &[f64]) {
        let v = serde_json::json!({ "version": 1.3, "people": [{ "person_id": [-1], "pose_keypoints_2d": kp }] });
        std::fs::write(dir.join(name), v.to_string()).unwrap();
    }

    #[test]
    fn reads_sorts_scales_and_centers() {
        let dir = tempfile::tempdir().unwrap();
        // 3 keypoints; pixel coordinates; last keypoint of frame 2 has zero confidence
        write_frame(
            dir.path(),
            "cam_000000000010.json",
            &[1000.0, 2000.0, 0.9, 3000.0, 4000.0, 0.9, 2000.0, 1000.0, 0.9],
        );
        write_frame(
            dir.path(),
            "cam_000000000002.json",
            &[1000.0, 2000.0, 0.9, 3000.0, 4000.0, 0.9, 2000.0, 1000.0, 0.0],
        );
        std::fs::write(dir.path().join("notes.txt"), "ignored").unwrap();

        let t = read_json_folder(dir.path(), CoordinateSystem::YUp).unwrap();
        assert_eq!(t.markers, vec!["Keypoint_0", "Keypoint_1", "Keypoint_2"]);
        assert!(t.has_generic_names());
        assert_eq!(t.fps, 30.0);
        assert_eq!(t.n_frames(), 2);
        // file ..002 sorts first: its third keypoint is missing
        assert_eq!(t.position(0, 2), None);
        // frame 1 (file ..010) is the first with 3 valid markers → it is centred:
        // raw meters x = 1,3,2 (cx = 2); y = -2,-4,-1 (min -4) → y - (-4)
        assert_eq!(t.position(1, 0), Some([-1.0, 2.0, 0.0]));
        assert_eq!(t.position(1, 1), Some([1.0, 0.0, 0.0]));
        assert_eq!(t.position(1, 2), Some([0.0, 3.0, 0.0]));
        // frame 0 shares the translation
        assert_eq!(t.position(0, 0), Some([-1.0, 2.0, 0.0]));
    }

    #[test]
    fn z_up_puts_height_on_z() {
        let dir = tempfile::tempdir().unwrap();
        write_frame(dir.path(), "f1.json", &[0.0, 0.0, 1.0, 1000.0, 1000.0, 1.0, 2000.0, 0.0, 1.0]);
        let t = read_json_folder(dir.path(), CoordinateSystem::ZUp).unwrap();
        // y column is all zero → no vertical shift; z carries -y/1000, centred on its mean
        assert_eq!(t.position(0, 1), Some([0.0, 0.0, -1.0 + 1.0 / 3.0]));
    }

    #[test]
    fn rejects_folder_without_people() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.json"), r#"{"people": []}"#).unwrap();
        assert!(read_json_folder(dir.path(), CoordinateSystem::YUp).is_err());
    }
}
