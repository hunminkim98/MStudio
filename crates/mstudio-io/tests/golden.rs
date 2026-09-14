//! Parity with the Python oracle (`tests/golden/`).

use std::path::PathBuf;

use mstudio_core::Take;
use mstudio_io::{read_c3d, read_trc, write_c3d, write_trc};
use ndarray::{Array1, Array3};

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn golden(name: &str) -> PathBuf {
    repo().join("tests/golden").join(name)
}

fn manifest() -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(golden("manifest.json")).unwrap()).unwrap()
}

fn names(v: &serde_json::Value) -> Vec<String> {
    v.as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect()
}

/// NaN-aware element comparison with an absolute tolerance.
fn assert_close(got: &Array3<f64>, want: &Array3<f64>, atol: f64, what: &str) {
    assert_eq!(got.dim(), want.dim(), "{what}: shape");
    let mut worst = 0.0f64;
    for (g, w) in got.iter().zip(want.iter()) {
        match (g.is_nan(), w.is_nan()) {
            (true, true) => {}
            (false, false) => worst = worst.max((g - w).abs()),
            _ => panic!("{what}: NaN mismatch got {g} want {w}"),
        }
    }
    assert!(worst <= atol, "{what}: max abs diff {worst} > {atol}");
}

#[test]
fn trc_reader_matches_oracle() {
    let m = manifest();
    let take = read_trc(repo().join("tests/test.trc")).unwrap();
    assert_eq!(take.markers, names(&m["trc"]["markers"]));
    assert_eq!(take.fps, m["trc"]["fps"].as_f64().unwrap());
    let want: Array3<f64> = ndarray_npy::read_npy(golden("trc_frames.npy")).unwrap();
    assert_close(&take.frames, &want, 0.0, "trc frames");
    let time: Array1<f64> = ndarray_npy::read_npy(golden("trc_time.npy")).unwrap();
    assert_eq!(take.time, time);
}

#[test]
fn trc_writer_is_byte_exact_with_pandas() {
    let take = read_trc(repo().join("tests/test.trc")).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("trc_roundtrip.trc"); // basename appears in line 1
    write_trc(&out, &take).unwrap();
    let got = std::fs::read(&out).unwrap();
    let want = std::fs::read(golden("trc_roundtrip.trc")).unwrap();
    if got != want {
        let g = String::from_utf8_lossy(&got);
        let w = String::from_utf8_lossy(&want);
        // `str::lines` strips a trailing '\r', so this finds content differences only.
        for (i, (a, b)) in g.lines().zip(w.lines()).enumerate() {
            assert_eq!(a, b, "first differing line: {}", i + 1);
        }
        let crs = want.iter().filter(|&&b| b == b'\r').count();
        panic!(
            "every line matches but the byte count differs: got {} want {} ({} CR bytes in the golden file). \
             A golden file checked out with CRLF means .gitattributes is not marking tests/golden/** as -text.",
            got.len(),
            want.len(),
            crs
        );
    }
}

#[test]
fn c3d_reader_matches_oracle() {
    let m = manifest();
    let take = read_c3d(repo().join("tests/test.c3d")).unwrap();
    assert_eq!(take.markers, names(&m["c3d"]["markers"]));
    assert_eq!(take.fps, m["c3d"]["fps"].as_f64().unwrap());
    let want: Array3<f64> = ndarray_npy::read_npy(golden("c3d_frames.npy")).unwrap();
    assert_close(&take.frames, &want, 0.0, "c3d frames");
    let time: Array1<f64> = ndarray_npy::read_npy(golden("c3d_time.npy")).unwrap();
    let dt = (&take.time - &time).mapv(f64::abs).fold(0.0f64, |a, &b| a.max(b));
    assert!(dt < 1e-9, "c3d time column differs by {dt}");
}

#[test]
fn c3d_written_by_python_reads_back_the_trc_data() {
    let take = read_c3d(golden("c3d_roundtrip.c3d")).unwrap();
    let want: Array3<f64> = ndarray_npy::read_npy(golden("trc_frames.npy")).unwrap();
    assert_eq!(take.markers, names(&manifest()["trc"]["markers"]));
    // stored as float32 millimetres → ~1e-7 m for coordinates up to a few metres
    assert_close(&take.frames, &want, 1e-6, "python-written c3d");
}

#[test]
fn c3d_round_trip_preserves_values_and_missing_samples() {
    let mut take = read_trc(repo().join("tests/test.trc")).unwrap();
    take.clear_range(3, 10, 20);
    take.set_position(0, 0, None);
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("rt.c3d");
    write_c3d(&out, &take).unwrap();
    let back = read_c3d(&out).unwrap();
    assert_eq!(back.markers, take.markers);
    assert_eq!(back.fps, take.fps);
    assert_close(&back.frames, &take.frames, 1e-6, "c3d round trip");
    assert!(back.position(15, 3).is_none());
    assert!(back.position(0, 0).is_none());
}

#[test]
fn load_dispatches_on_extension() {
    assert!(mstudio_io::load(repo().join("tests/test.trc")).is_ok());
    assert!(mstudio_io::load(repo().join("tests/test.c3d")).is_ok());
    assert!(matches!(mstudio_io::load(repo().join("README.md")), Err(mstudio_io::IoError::Unsupported(_))));
    let _: &Take = &mstudio_io::load(repo().join("tests/test.trc")).unwrap();
}
