//! Parity with the Python oracle (`tests/golden/`, `scripts/gen_golden.py`).

use std::path::PathBuf;

use mstudio_processing::analysis;
use mstudio_processing::{filter_column, interpolate_in_range, pattern_interpolate, Filter, InterpMethod};
use ndarray::{s, Array1, Array2, Array3};

fn golden(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden").join(name)
}

fn manifest() -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(golden("manifest.json")).unwrap()).unwrap()
}

fn load3(name: &str) -> Array3<f64> {
    ndarray_npy::read_npy(golden(name)).unwrap()
}

fn marker_index(m: &serde_json::Value, name: &str) -> usize {
    m["trc"]["markers"].as_array().unwrap().iter().position(|v| v == name).unwrap()
}

/// Max abs difference, NaN positions must agree. `skip(i)` excludes samples.
fn max_diff<'a>(
    got: impl Iterator<Item = &'a f64>,
    want: impl Iterator<Item = &'a f64>,
    skip: impl Fn(usize) -> bool,
) -> f64 {
    let mut worst = 0.0f64;
    for (i, (g, w)) in got.zip(want).enumerate() {
        if skip(i) {
            continue;
        }
        match (g.is_nan(), w.is_nan()) {
            (true, true) => {}
            (false, false) => worst = worst.max((g - w).abs()),
            _ => panic!("NaN mismatch at {i}: got {g} want {w}"),
        }
    }
    worst
}

fn filter_from_case(case: &serde_json::Value) -> Filter {
    let p = &case["params"];
    let f = |k: &str| p[k].as_f64().unwrap();
    match case["type"].as_str().unwrap() {
        "butterworth" => Filter::Butterworth { order: f("order") as u32, cutoff_hz: f("cut_off_frequency") },
        "butterworth_on_speed" => {
            Filter::ButterworthOnSpeed { order: f("order") as u32, cutoff_hz: f("cut_off_frequency") }
        }
        "kalman" => Filter::Kalman { trust_ratio: f("trust_ratio"), smooth: f("smooth") != 0.0 },
        "gaussian" => Filter::Gaussian { sigma_kernel: f("sigma_kernel") },
        "LOESS" => Filter::Loess { nb_values_used: f("nb_values_used") },
        "median" => Filter::Median { kernel_size: f("kernel_size") },
        other => panic!("unknown filter {other}"),
    }
}

#[test]
fn every_filter_case_matches_the_oracle() {
    let m = manifest();
    let inputs = [("trc_frames.npy", ""), ("trc_frames_gapped.npy", "_gapped")];
    let mut report = Vec::new();
    for case in m["filter_cases"].as_array().unwrap() {
        let filter = filter_from_case(case);
        let fps = case["fps"].as_f64().unwrap();
        let idx = case["index"].as_u64().unwrap();
        for (input_file, suffix) in inputs {
            let input = load3(input_file);
            let want = load3(&format!("filter_{idx:02}_{}{suffix}.npy", filter.name()));
            let (n, nm, _) = input.dim();
            let mut worst = 0.0f64;
            for marker in 0..nm {
                for axis in 0..3 {
                    let col: Vec<f64> = input.slice(s![.., marker, axis]).to_vec();
                    let got = filter_column(&col, &filter, fps).unwrap();
                    let want_col: Vec<f64> = want.slice(s![.., marker, axis]).to_vec();
                    // Median: scipy's medfilt gives unspecified results in windows that contain a NaN
                    // *and* in the windows right after them (its selection state carries over), so the
                    // comparison skips samples within one full kernel width of a NaN.
                    let skip = |i: usize| -> bool {
                        if let Filter::Median { kernel_size } = filter {
                            let w = kernel_size as i64 | 1;
                            (i as i64 - w..=i as i64 + w)
                                .any(|j| j >= 0 && (j as usize) < n && col[j as usize].is_nan())
                        } else {
                            false
                        }
                    };
                    worst = worst.max(max_diff(got.iter(), want_col.iter(), skip));
                }
            }
            let tol = match filter {
                Filter::Kalman { .. } | Filter::Loess { .. } => 1e-5,
                _ => 1e-6,
            };
            let line = format!("{:>2} {:<22}{:<8} max|diff| = {:.2e}", idx, filter.name(), suffix, worst);
            eprintln!("{line}");
            report.push(line);
            assert!(worst <= tol, "case {idx} {} {suffix}: max diff {worst:.3e} > {tol:e}", filter.name());
        }
    }
    println!("{}", report.join("\n"));
}

#[test]
fn interpolation_cases_match_the_oracle_except_documented_spline() {
    let m = manifest();
    let gapped = load3("trc_frames_gapped.npy");
    for case in m["interp_cases"].as_array().unwrap() {
        let marker = marker_index(&m, case["marker"].as_str().unwrap());
        let (first, last) = (case["range"][0].as_u64().unwrap() as usize, case["range"][1].as_u64().unwrap() as usize);
        let name = case["method"].as_str().unwrap();
        let method = InterpMethod::from_name(name, case["order"].as_u64().unwrap() as u32).unwrap();
        let want: Array2<f64> =
            ndarray_npy::read_npy(golden(&format!("{}.npy", case["file"].as_str().unwrap()))).unwrap();

        let mut frames = gapped.clone();
        interpolate_in_range(&mut frames, marker, first, last, method).unwrap();
        let got = frames.slice(s![.., marker, ..]);

        if name == "spline" {
            // pandas 'spline' = UnivariateSpline with smoothing s=len(x): not an interpolant.
            // We interpolate instead (== cubic for order 3); prove the oracle really differs.
            let cubic_file = case["file"].as_str().unwrap().replace("spline", "cubic");
            let cubic: Array2<f64> = ndarray_npy::read_npy(golden(&format!("{cubic_file}.npy"))).unwrap();
            let d_cubic = max_diff(got.iter(), cubic.iter(), |_| false);
            let d_oracle = max_diff(got.iter(), want.iter(), |_| false);
            assert!(d_cubic < 1e-9, "spline should equal cubic interpolation: {d_cubic:.3e}");
            assert!(d_oracle > 1e-3, "oracle smoothing spline unexpectedly close to interpolation: {d_oracle:.3e}");
            continue;
        }
        let worst = max_diff(got.iter(), want.iter(), |_| false);
        assert!(worst < 1e-9, "{name} {}: max diff {worst:.3e}", case["marker"]);
    }
}

#[test]
fn pattern_interpolation_matches_the_oracle() {
    let m = manifest();
    let gapped = load3("trc_frames_gapped.npy");
    for case in m["pattern_cases"].as_array().unwrap() {
        let marker = marker_index(&m, case["marker"].as_str().unwrap());
        let refs: Vec<usize> =
            case["references"].as_array().unwrap().iter().map(|r| marker_index(&m, r.as_str().unwrap())).collect();
        let (first, last) = (case["range"][0].as_u64().unwrap() as usize, case["range"][1].as_u64().unwrap() as usize);
        let want: Array2<f64> =
            ndarray_npy::read_npy(golden(&format!("{}.npy", case["file"].as_str().unwrap()))).unwrap();
        let mut frames = gapped.clone();
        pattern_interpolate(&mut frames, marker, &refs, first, last).unwrap();
        let got = frames.slice(s![.., marker, ..]);
        let worst = max_diff(got.iter(), want.iter(), |_| false);
        assert!(worst < 1e-9, "{} with {} refs: max diff {worst:.3e}", case["marker"], refs.len());
    }
}

#[test]
fn analysis_primitives_match_the_oracle() {
    let m = manifest();
    let frames = load3("trc_frames.npy");
    let fps = m["trc"]["fps"].as_f64().unwrap();
    let p = |f: usize, name: &str| -> [f64; 3] {
        let i = marker_index(&m, name);
        [frames[[f, i, 0]], frames[[f, i, 1]], frames[[f, i, 2]]]
    };
    let n = frames.dim().0;

    let dist: Array1<f64> = ndarray_npy::read_npy(golden("analysis_distance_RHip_RKnee.npy")).unwrap();
    let ang: Array1<f64> = ndarray_npy::read_npy(golden("analysis_angle_RHip_RKnee_RAnkle.npy")).unwrap();
    for f in 0..n {
        assert!((analysis::distance(p(f, "RHip"), p(f, "RKnee")) - dist[f]).abs() < 1e-9);
        assert!((analysis::joint_angle(p(f, "RHip"), p(f, "RKnee"), p(f, "RAnkle")).unwrap() - ang[f]).abs() < 1e-9);
    }

    let vel: Array2<f64> = ndarray_npy::read_npy(golden("analysis_velocity_RKnee.npy")).unwrap();
    let acc: Array2<f64> = ndarray_npy::read_npy(golden("analysis_acceleration_RKnee.npy")).unwrap();
    let mut v = vec![[f64::NAN; 3]; n];
    for f in 1..n - 1 {
        v[f] = analysis::velocity(p(f - 1, "RKnee"), p(f + 1, "RKnee"), fps).unwrap();
        for k in 0..3 {
            assert!((v[f][k] - vel[[f, k]]).abs() < 1e-6, "vel frame {f}");
        }
    }
    for f in 2..n - 2 {
        let a = analysis::acceleration(v[f - 1], v[f + 1], fps).unwrap();
        for k in 0..3 {
            assert!((a[k] - acc[[f, k]]).abs() < 1e-4, "acc frame {f}: {} vs {}", a[k], acc[[f, k]]);
        }
    }
    assert!(vel[[0, 0]].is_nan() && acc[[1, 0]].is_nan());

    let arc: Array2<f64> = ndarray_npy::read_npy(golden("analysis_arc_RKnee_frame0.npy")).unwrap();
    let got =
        analysis::arc_points(p(0, "RKnee"), p(0, "RHip"), p(0, "RAnkle"), analysis::ARC_RADIUS, analysis::ARC_SEGMENTS)
            .unwrap();
    assert_eq!(got.len(), arc.dim().0);
    for (i, pt) in got.iter().enumerate() {
        for k in 0..3 {
            assert!((pt[k] - arc[[i, k]]).abs() < 1e-12);
        }
    }
}
