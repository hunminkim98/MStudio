"""Tests for the `mstudio` Python bindings.

They pin the bindings to the same golden files as the Rust crates
(`tests/golden/`, generated from the Python oracle) and check the NumPy
interop contract: zero-copy views, lifetimes, error mapping.

Run from the repository root after `maturin develop`:
    .venv/bin/pytest crates/mstudio-py/tests -q
"""
from __future__ import annotations

import gc
import json
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

import mstudio

ROOT = Path(__file__).resolve().parents[3]
GOLDEN = ROOT / "tests" / "golden"
TRC = ROOT / "tests" / "test.trc"
C3D = ROOT / "tests" / "test.c3d"
MANIFEST = json.loads((GOLDEN / "manifest.json").read_text())

FILTER_TOL = {"kalman": 1e-5, "LOESS": 1e-5}


def golden(name: str) -> np.ndarray:
    return np.load(GOLDEN / f"{name}.npy")


@pytest.fixture
def take() -> mstudio.Take:
    return mstudio.load(str(TRC))


@pytest.fixture
def gapped() -> mstudio.Take:
    return mstudio.Take(MANIFEST["trc"]["markers"], MANIFEST["trc"]["fps"], golden("trc_frames_gapped"))


def make_filter(ftype: str, p: dict) -> mstudio.Filter:
    if ftype == "butterworth":
        return mstudio.Filter.butterworth(int(p["order"]), p["cut_off_frequency"])
    if ftype == "butterworth_on_speed":
        return mstudio.Filter.butterworth_on_speed(int(p["order"]), p["cut_off_frequency"])
    if ftype == "kalman":
        return mstudio.Filter.kalman(p["trust_ratio"], bool(p["smooth"]))
    if ftype == "gaussian":
        return mstudio.Filter.gaussian(p["sigma_kernel"])
    if ftype == "LOESS":
        return mstudio.Filter.loess(p["nb_values_used"])
    if ftype == "median":
        return mstudio.Filter.median(p["kernel_size"])
    raise ValueError(ftype)


# ------------------------------------------------------------------ loaders --


def test_load_trc_matches_golden(take):
    assert take.markers == MANIFEST["trc"]["markers"]
    assert take.fps == MANIFEST["trc"]["fps"]
    assert take.n_frames == MANIFEST["trc"]["n_frames"] == len(take)
    np.testing.assert_array_equal(take.frames, golden("trc_frames"))
    np.testing.assert_array_equal(take.time, golden("trc_time"))
    assert take.frame_numbers[0] == 1 and take.frame_numbers.dtype == np.int64


def test_load_c3d_matches_golden():
    t = mstudio.load(str(C3D))
    assert t.markers == MANIFEST["c3d"]["markers"]
    np.testing.assert_array_equal(t.frames, golden("c3d_frames"))
    np.testing.assert_allclose(t.time, golden("c3d_time"), atol=1e-12)


def test_unsupported_extension_is_an_os_error(tmp_path):
    with pytest.raises(OSError):
        mstudio.load(str(tmp_path / "x.bvh"))
    with pytest.raises(OSError):
        mstudio.load(str(tmp_path / "missing.trc"))


# ------------------------------------------------------------- numpy views --


def test_frames_is_a_writable_zero_copy_view(take):
    f = take.frames
    assert f.shape == (take.n_frames, take.n_markers, 3) and f.dtype == np.float64
    assert f.flags["WRITEABLE"] and f.flags["C_CONTIGUOUS"]
    assert np.shares_memory(f, take.frames)
    f[3, 1, :] = np.nan
    assert take.position(3, "RKnee") is None
    assert take.position(4, "RKnee") == pytest.approx(tuple(f[4, 1]))
    take.restore_original()
    assert not np.isnan(f[3, 1]).any()  # the view saw the in-place restore


def test_original_time_frame_numbers_are_read_only(take):
    for arr in (take.original, take.time, take.frame_numbers):
        assert not arr.flags["WRITEABLE"]
        with pytest.raises(ValueError):
            arr[0] = 0


def test_view_keeps_the_take_alive():
    f = mstudio.load(str(TRC)).frames
    gc.collect()
    assert np.isfinite(f).sum() > 0
    assert f.base is not None


def test_frames_setter_copies_and_checks_shape(take):
    new = np.zeros_like(take.frames)
    view = take.frames
    take.frames = new
    assert view.sum() == 0  # copied into the existing buffer
    with pytest.raises(ValueError):
        take.frames = np.zeros((2, 2, 3))


def test_constructor_validates():
    with pytest.raises(ValueError):
        mstudio.Take(["a", "b"], 100.0, np.zeros((5, 3, 3)))
    with pytest.raises(ValueError):
        mstudio.Take(["a"], 0.0, np.zeros((5, 1, 3)))
    t = mstudio.Take(["a"], 50.0, np.zeros((5, 1, 3)))
    np.testing.assert_allclose(t.time, np.arange(5) / 50.0)
    t.markers = ["b"]
    assert t.index("b") == 0
    with pytest.raises(ValueError):
        t.markers = ["b", "c"]


def test_marker_can_be_index_or_name(take):
    assert take.position(0, 1) == take.position(0, "RKnee")
    with pytest.raises(ValueError):
        take.position(0, "Nope")
    with pytest.raises(ValueError):
        take.position(0, 999)


# ----------------------------------------------------------------- filters --


@pytest.mark.parametrize("case", MANIFEST["filter_cases"], ids=lambda c: f"{c['index']:02d}_{c['type']}")
def test_filters_match_golden(case, gapped):
    tag = f"filter_{case['index']:02d}_{case['type']}"
    tol = FILTER_TOL.get(case["type"], 1e-6)
    f = make_filter(case["type"], case["params"])
    full = mstudio.Take(MANIFEST["trc"]["markers"], case["fps"], golden("trc_frames"))
    full.filter_all(f)
    np.testing.assert_allclose(full.frames, golden(tag), atol=tol, equal_nan=True)
    gapped.filter_all(f)
    got, want = gapped.frames, golden(tag + "_gapped")
    if case["type"] == "median":
        # scipy.signal.medfilt's behaviour next to NaN is unspecified (docs/PHASE2_RESULTS.md);
        # like the Rust golden test, skip samples within one full kernel width of a NaN.
        w = int(case["params"]["kernel_size"]) | 1
        near_nan = np.zeros_like(want, dtype=bool)
        nan = np.isnan(golden("trc_frames_gapped"))
        for d in range(-w, w + 1):
            near_nan |= np.roll(nan, d, axis=0)
        got, want = np.where(near_nan, 0.0, got), np.where(near_nan, 0.0, want)
    np.testing.assert_allclose(got, want, atol=tol, equal_nan=True)


def test_filter_writes_back_only_the_range(take):
    before = take.frames.copy()
    start, end = take.filter("RKnee", mstudio.Filter.butterworth(), first=20, last=29)
    assert (start, end) == (20, 30)
    after = take.frames
    np.testing.assert_array_equal(after[:20], before[:20])
    np.testing.assert_array_equal(after[30:], before[30:])
    assert not np.array_equal(after[20:30, 1], before[20:30, 1])
    np.testing.assert_array_equal(after[20:30, 0], before[20:30, 0])  # other markers untouched


def test_filter1d_matches_take_filter(take):
    col = take.frames[:, 1, 0].copy()
    f = mstudio.Filter.gaussian(3.0)
    out = mstudio.filter1d(col, f, take.fps)
    take.filter("RKnee", f)
    np.testing.assert_array_equal(out, take.frames[:, 1, 0])


def test_invalid_filter_parameters_raise(take):
    with pytest.raises(ValueError):
        take.filter("RKnee", mstudio.Filter.butterworth(order=3))
    with pytest.raises(ValueError):
        take.filter("RKnee", mstudio.Filter.butterworth(cutoff_hz=100))  # >= Nyquist at 120 Hz


def test_filter_names():
    assert mstudio.Filter.loess().name == "LOESS"
    assert mstudio.Filter.kalman().name == "kalman"
    assert "Butterworth" in repr(mstudio.Filter.butterworth())


# ----------------------------------------------------------- interpolation --


@pytest.mark.parametrize("case", MANIFEST["interp_cases"], ids=lambda c: c["file"])
def test_interpolation_matches_golden(case, gapped):
    m = case["marker"]
    if case["method"] == "spline":
        # Documented deviation (docs/PHASE2_RESULTS.md): pandas' `spline` is a
        # smoothing spline; the port interpolates and so equals `cubic` at order 3.
        cubic = mstudio.Take(gapped.markers, gapped.fps, gapped.frames.copy())
        cubic.interpolate(m, "cubic", *case["range"])
        gapped.interpolate(m, "spline", *case["range"], order=3)
        np.testing.assert_allclose(gapped.frames, cubic.frames, atol=1e-9, equal_nan=True)
        return
    gapped.interpolate(m, case["method"], *case["range"], order=case["order"])
    np.testing.assert_allclose(gapped.frames[:, gapped.index(m), :], golden(case["file"]), atol=1e-6, equal_nan=True)


def test_interpolate_returns_touched_range(gapped):
    assert gapped.interpolate("RKnee", "linear", 0, 30) == (0, 0)  # gap is at 40..50
    start, end = gapped.interpolate("RKnee", "linear", 35, 55)
    assert (start, end) == (40, 51)
    assert not np.isnan(gapped.frames[:, 1]).any()


def test_interpolate1d_and_bad_method():
    v = np.array([0.0, np.nan, 2.0, np.nan, np.nan, 5.0])
    np.testing.assert_allclose(mstudio.interpolate1d(v), np.arange(6.0))
    with pytest.raises(ValueError):
        mstudio.interpolate1d(v, "bogus")
    assert "linear" in mstudio.INTERP_METHODS


@pytest.mark.parametrize("case", MANIFEST["pattern_cases"], ids=lambda c: c["file"])
def test_pattern_interpolation_matches_golden(case, gapped):
    gapped.pattern_interpolate(case["marker"], case["references"], *case["range"])
    np.testing.assert_allclose(gapped.frames[:, gapped.index(case["marker"]), :], golden(case["file"]), atol=1e-9, equal_nan=True)


def test_pattern_interpolation_rejects_self_reference(gapped):
    with pytest.raises(ValueError):
        gapped.pattern_interpolate("RKnee", ["RKnee"], 35, 55)
    with pytest.raises(ValueError):
        gapped.pattern_interpolate("RKnee", [], 35, 55)


# --------------------------------------------------------------- outliers --


@pytest.mark.parametrize("model", [m for m in MANIFEST["skeleton_pairs"] if MANIFEST["skeleton_pairs"][m]])
def test_outliers_match_golden(model, take, gapped):
    names = MANIFEST["skeleton_pairs"][model]
    pairs = mstudio.skeleton_pairs(model, take.markers)
    assert [(take.markers[a], take.markers[b]) for a, b in pairs] == [tuple(p) for p in names]
    flags = take.outliers(pairs, MANIFEST["outlier_threshold"])
    assert flags.dtype == np.bool_ and flags.shape == (take.n_markers, take.n_frames)
    np.testing.assert_array_equal(flags, golden(f"outliers_{model}").astype(bool))
    np.testing.assert_array_equal(gapped.outliers(names), golden(f"outliers_{model}_gapped").astype(bool))
    np.testing.assert_array_equal(mstudio.detect_outliers(take.frames, pairs, 0.3), flags)


def test_detect_outliers_range_check(take):
    with pytest.raises(ValueError):
        mstudio.detect_outliers(take.frames, [(0, 99)])


# ---------------------------------------------------------------- skeleton --


def test_skeleton_tables():
    assert "HALPE_26" in mstudio.skeleton_models()
    pairs = mstudio.skeleton_pair_names("HALPE_26")
    assert ("Hip", "RHip") in pairs
    with pytest.raises(ValueError):
        mstudio.skeleton_pair_names("NOPE")


def test_rename_generic_markers():
    t = mstudio.Take([f"Keypoint_{i}" for i in range(26)], 30.0, np.zeros((2, 26, 3)))
    assert t.has_generic_names()
    assert t.rename_markers("HALPE_26")
    assert "Neck" in t.markers and not t.has_generic_names()
    assert not t.rename_markers("HALPE_26")


def test_auto_segments_and_joints(take):
    segs = mstudio.auto_segments(take.markers)
    assert ("Thigh_R", take.index("RHip"), take.index("RKnee")) in segs
    joints = mstudio.auto_joints(take.markers)
    assert ("Knee_R", take.index("RHip"), take.index("RKnee"), take.index("RAnkle")) in joints


# ------------------------------------------------------------------ editing --


def test_clear_and_restore(take):
    assert take.clear("RKnee", 10, 12) == (10, 13)
    assert np.isnan(take.frames[10:13, 1]).all()
    assert take.position(10, "RKnee") is None
    assert take.restore_original() == (0, take.n_frames)
    np.testing.assert_array_equal(take.frames, take.original)


def test_copy_is_independent(take):
    c = take.copy()
    c.clear("RKnee")
    assert not np.isnan(take.frames[:, 1]).any()
    assert np.isnan(c.frames[:, 1]).all()


def test_bounds(take):
    lo, hi = take.bounds()
    assert lo == pytest.approx(tuple(np.nanmin(take.frames, axis=(0, 1))))
    assert hi == pytest.approx(tuple(np.nanmax(take.frames, axis=(0, 1))))
    assert mstudio.Take(["a"], 1.0, np.full((2, 1, 3), np.nan)).bounds() is None


# ---------------------------------------------------------------- writers --


def test_trc_round_trip_is_byte_exact(take, tmp_path):
    out = tmp_path / "rt.trc"
    take.save(str(out))
    head, body = out.read_bytes().split(b"\n", 1)
    ghead, gbody = (GOLDEN / "trc_roundtrip.trc").read_bytes().split(b"\n", 1)
    assert body == gbody  # everything but the file path in line 1
    assert head.startswith(b"PathFileType\t4\t(X/Y/Z)\t") and head.endswith(b"rt.trc")
    assert ghead.startswith(b"PathFileType\t4\t(X/Y/Z)\t")
    back = mstudio.load(str(out))
    np.testing.assert_array_equal(back.frames, take.frames)
    assert back.markers == take.markers


def test_c3d_round_trip(take, tmp_path):
    out = tmp_path / "rt.c3d"
    mstudio.save(str(out), take)
    back = mstudio.load(str(out))
    assert back.markers == take.markers and back.fps == take.fps
    np.testing.assert_allclose(back.frames, take.frames, atol=1e-6, equal_nan=True)


# ----------------------------------------------------------------- report --


def test_report(take, tmp_path):
    html = mstudio.build_report(take, title="T1", skeleton="HALPE_26")
    assert html.startswith("<!doctype html>") and "T1" in html and "RKnee" in html and "Knee_R" in html
    out = tmp_path / "r.html"
    mstudio.write_report(take, str(out), skeleton="HALPE_26")
    assert out.stat().st_size > 1_000_000  # Plotly is inlined
    with pytest.raises(ValueError):
        mstudio.build_report(take, skeleton="NOPE")


# ------------------------------------------------------------ threading --


def test_filter_releases_the_gil(take):
    """`filter_all` runs with the GIL released; a second thread must make progress meanwhile."""
    big = mstudio.Take(["m"], 120.0, np.random.default_rng(0).standard_normal((200_000, 1, 3)))
    ticks = []
    stop = threading.Event()

    def counter():
        while not stop.is_set():
            ticks.append(1)

    t = threading.Thread(target=counter)
    t.start()
    try:
        for _ in range(3):
            big.filter_all(mstudio.Filter.loess(30))
    finally:
        stop.set()
        t.join()
    assert len(ticks) > 100


def test_cli_version():
    out = subprocess.run([sys.executable, "-m", "mstudio", "--version"], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == f"mstudio {mstudio.__version__}"
