#!/usr/bin/env python
"""Live parity check: the `mstudio` bindings (Rust) against the Python oracle (MStudio v0.1.5).

`tests/golden/` pins the Rust crates to a snapshot of the oracle. This script
runs both implementations side by side on the same inputs, so it also catches
drift in the oracle's dependencies (numpy / scipy / pandas / statsmodels /
filterpy) and exercises the bindings' Take API end to end. It is what the
parity QA on Windows / macOS / Linux runs (docs/QA_CHECKLIST.md).

    .venv/bin/python scripts/check_parity.py [--json out.json] [--quiet]

Exit status 1 when any case exceeds its tolerance. Documented deviations
(docs/PHASE2_RESULTS.md) are reported as "deviation", not failures.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oracle_harness as H  # noqa: E402

import mstudio  # noqa: E402
from MStudio.core.outlier_detector import OutlierDetector  # noqa: E402
from MStudio.utils import dataProcessor, skeletons  # noqa: E402
from MStudio.utils.dataLoader import read_data_from_c3d, read_data_from_trc  # noqa: E402
from MStudio.utils.dataSaver import save_to_c3d, save_to_trc  # noqa: E402
from MStudio.utils.filtering import filter1d  # noqa: E402
from MStudio.utils.reportGenerator import ReportGenerator  # noqa: E402

ROOT = H.ROOT
TRC = ROOT / "tests" / "test.trc"
C3D = ROOT / "tests" / "test.c3d"

FILTER_CASES = [
    ("butterworth", {"order": 4, "cut_off_frequency": 10.0}),
    ("butterworth", {"order": 4, "cut_off_frequency": 6.0}),
    ("butterworth", {"order": 2, "cut_off_frequency": 6.0}),
    ("butterworth_on_speed", {"order": 4, "cut_off_frequency": 10.0}),
    ("butterworth_on_speed", {"order": 2, "cut_off_frequency": 6.0}),
    ("kalman", {"trust_ratio": 20.0, "smooth": 1.0}),
    ("kalman", {"trust_ratio": 20.0, "smooth": 0.0}),
    ("kalman", {"trust_ratio": 100.0, "smooth": 1.0}),
    ("gaussian", {"sigma_kernel": 3.0}),
    ("gaussian", {"sigma_kernel": 1.0}),
    ("LOESS", {"nb_values_used": 10.0}),
    ("LOESS", {"nb_values_used": 30.0}),
    ("median", {"kernel_size": 3.0}),
    ("median", {"kernel_size": 5.0}),
]
FILTER_TOL = {"kalman": 1e-5, "LOESS": 1e-5}
GAPS = {"RKnee": [[40, 50]], "LWrist": [[90, 95]], "RAnkle": [[20, 20]]}
INTERP_METHODS = ["linear", "polynomial", "spline", "nearest", "zero", "slinear", "quadratic", "cubic"]
INTERP_CASES = [("RKnee", 35, 55), ("LWrist", 85, 100), ("RAnkle", 15, 25)]
PATTERN_CASES = [
    ("RKnee", 35, 55, ["RHip"]),
    ("RKnee", 35, 55, ["RHip", "RAnkle"]),
    ("RKnee", 35, 55, ["RHip", "RAnkle", "LKnee"]),
    ("LWrist", 85, 100, ["LElbow", "LShoulder", "LIndex"]),
]
SKELETONS = ["HALPE_26", "COCO_17", "COCO_133", "BODY_25B", "BODY_25"]

rows: list[dict] = []


def record(section: str, case: str, diff: float, tol: float, *, deviation: bool = False, note: str = "", **extra):
    if deviation:
        status = "deviation"
    else:
        status = "ok" if diff <= tol else "FAIL"
    rows.append({"section": section, "case": case, "max_abs_diff": float(diff), "tol": tol, "status": status, "note": note, **extra})


def max_diff(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Max |a-b| over samples where both are finite; +inf if NaN patterns differ (outside `mask`)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = np.isnan(a), np.isnan(b)
    consider = np.ones(a.shape, dtype=bool) if mask is None else ~mask
    if np.any((na != nb) & consider):
        return float("inf")
    both = ~na & consider
    return float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0


def make_filter(ftype: str, p: dict) -> mstudio.Filter:
    return {
        "butterworth": lambda: mstudio.Filter.butterworth(int(p["order"]), p["cut_off_frequency"]),
        "butterworth_on_speed": lambda: mstudio.Filter.butterworth_on_speed(int(p["order"]), p["cut_off_frequency"]),
        "kalman": lambda: mstudio.Filter.kalman(p["trust_ratio"], bool(p["smooth"])),
        "gaussian": lambda: mstudio.Filter.gaussian(p["sigma_kernel"]),
        "LOESS": lambda: mstudio.Filter.loess(p["nb_values_used"]),
        "median": lambda: mstudio.Filter.median(p["kernel_size"]),
    }[ftype]()


def oracle_filter(df: pd.DataFrame, markers: list[str], ftype: str, params: dict, fps: float) -> np.ndarray:
    out = df.copy()
    config = {"filtering": {ftype: params}}
    for m in markers:
        for c in "XYZ":
            col = f"{m}_{c}"
            series = out[col]
            filtered = filter1d(series.copy(), config, ftype, fps)
            if isinstance(filtered, np.ndarray):
                filtered = pd.Series(filtered, index=series.index)
            out.loc[:, col] = filtered.astype(series.dtype)
    return H.frames_of(out, markers)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", type=Path, help="write the result table to this file")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    # ------------------------------------------------------------ loaders --
    _, trc_df, markers, fps = read_data_from_trc(str(TRC))
    py_frames = H.frames_of(trc_df, markers)
    rt = mstudio.load(str(TRC))
    record("load", "trc markers/fps", 0.0 if (rt.markers == markers and rt.fps == fps) else float("inf"), 0.0)
    record("load", "trc frames", max_diff(rt.frames, py_frames), 0.0)
    record("load", "trc time", max_diff(rt.time, trc_df["Time"].to_numpy(dtype=np.float64)), 0.0)
    _, c3d_df, c3d_markers, c3d_fps = read_data_from_c3d(str(C3D))
    rc = mstudio.load(str(C3D))
    record("load", "c3d markers/fps", 0.0 if (rc.markers == c3d_markers and rc.fps == c3d_fps) else float("inf"), 0.0)
    record("load", "c3d frames", max_diff(rc.frames, H.frames_of(c3d_df, c3d_markers)), 0.0)
    record("load", "c3d time", max_diff(rc.time, c3d_df["Time"].to_numpy(dtype=np.float64)), 1e-12)

    gapped_df = trc_df.copy()
    for m, ranges in GAPS.items():
        for a, b in ranges:
            for c in "XYZ":
                gapped_df.loc[a:b, f"{m}_{c}"] = np.nan
    gapped_frames = H.frames_of(gapped_df, markers)

    # ------------------------------------------------------------ filters --
    for ftype, params in FILTER_CASES:
        tol = FILTER_TOL.get(ftype, 1e-6)
        f = make_filter(ftype, params)
        for tag, df, frames in (("full", trc_df, py_frames), ("gapped", gapped_df, gapped_frames)):
            t0 = time.perf_counter()
            want = oracle_filter(df, markers, ftype, params, fps)
            t_py = time.perf_counter() - t0
            take = mstudio.Take(markers, fps, frames)
            t0 = time.perf_counter()
            take.filter_all(f)
            t_rs = time.perf_counter() - t0
            mask = None
            note = ""
            if ftype == "median" and tag == "gapped":
                w = int(params["kernel_size"]) | 1
                nan = np.isnan(frames)
                mask = np.zeros_like(nan)
                for d in range(-w, w + 1):
                    mask |= np.roll(nan, d, axis=0)
                note = "samples within a kernel width of NaN skipped (scipy medfilt NaN order unspecified)"
            # the per-column entry point must agree bit-for-bit with the Take path
            col = mstudio.filter1d(frames[:, 1, 0], f, fps)
            same = np.array_equal(col, take.frames[:, 1, 0], equal_nan=True)
            record(
                "filter",
                f"{ftype} {params} [{tag}]",
                max_diff(take.frames, want, mask),
                tol,
                note=note if same else "filter1d != Take.filter_all",
                py_ms=round(t_py * 1e3, 1),
                rs_ms=round(t_rs * 1e3, 2),
            )
            if not same:
                rows[-1]["status"] = "FAIL"

    # ------------------------------------------------------ interpolation --
    for marker, start, end in INTERP_CASES:
        for method in INTERP_METHODS:
            df = gapped_df.copy()
            dataProcessor.interpolate_selected_data(H.viewer_stub(df, marker, start, end, method, order="3"))
            want = H.frames_of(df, [marker])[:, 0, :]
            take = mstudio.Take(markers, fps, gapped_frames)
            take.interpolate(marker, method, start, end, order=3)
            got = take.frames[:, take.index(marker), :]
            record(
                "interp",
                f"{method} {marker} [{start},{end}]",
                max_diff(got, want),
                1e-6,
                deviation=(method == "spline"),
                note="pandas 'spline' is a smoothing spline; the port interpolates (== cubic)" if method == "spline" else "",
            )

    for marker, start, end, refs in PATTERN_CASES:
        df = gapped_df.copy()
        dataProcessor.interpolate_with_pattern(H.viewer_stub(df, marker, start, end, "pattern-based", pattern=refs))
        want = H.frames_of(df, [marker])[:, 0, :]
        take = mstudio.Take(markers, fps, gapped_frames)
        take.pattern_interpolate(marker, refs, start, end)
        got = take.frames[:, take.index(marker), :]
        record("pattern", f"{marker} refs={refs}", max_diff(got, want), 1e-9)

    # ----------------------------------------------------------- outliers --
    seq = OutlierDetector(threshold=0.3, use_parallel=False)
    par = OutlierDetector(threshold=0.3, use_parallel=True)
    for name in SKELETONS:
        model = getattr(skeletons, name)
        pairs = H.skeleton_pairs_of(model, trc_df)
        rs_pairs = mstudio.skeleton_pairs(name, markers)
        same_pairs = [(markers[a], markers[b]) for a, b in rs_pairs] == pairs
        record("skeleton", f"{name} resolved pairs ({len(pairs)})", 0.0 if same_pairs else float("inf"), 0.0)
        if not pairs:
            continue
        for tag, df, frames in (("full", trc_df, py_frames), ("gapped", gapped_df, gapped_frames)):
            want = np.stack([seq.detect_outliers(df, markers, pairs)[m] for m in markers]).astype(bool)
            want_par = np.stack([par.detect_outliers(df, markers, pairs)[m] for m in markers]).astype(bool)
            got = mstudio.Take(markers, fps, frames).outliers(rs_pairs)
            diff = float(np.sum(got != want))
            record("outliers", f"{name} [{tag}] flagged={int(want.sum())}", diff, 0.0, note="" if np.array_equal(want, want_par) else "oracle sequential != parallel")

    # ------------------------------------------------------------ writers --
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        save_to_trc(str(tmp / "py.trc"), trc_df, fps, markers, len(trc_df))
        rt.save(str(tmp / "rs.trc"))
        py_bytes = (tmp / "py.trc").read_bytes().split(b"\n", 1)[1]
        rs_bytes = (tmp / "rs.trc").read_bytes().split(b"\n", 1)[1]
        # `save_to_trc` opens the file in text mode, so the oracle writes the host's
        # newline (CRLF on Windows) while the port always writes LF. Same content.
        eol_only = py_bytes != rs_bytes and py_bytes.replace(b"\r\n", b"\n") == rs_bytes
        record(
            "write",
            "trc bytes (after the path line)",
            0.0 if (py_bytes == rs_bytes or eol_only) else float("inf"),
            0.0,
            deviation=eol_only,
            note="identical apart from line endings: the oracle writes the host's newline, the port always writes LF"
            if eol_only
            else "",
        )
        _, back_df, back_markers, back_fps = read_data_from_trc(str(tmp / "rs.trc"))
        record("write", "rust trc -> python loader", max_diff(H.frames_of(back_df, back_markers), py_frames), 0.0)
        back = mstudio.load(str(tmp / "py.trc"))
        record("write", "python trc -> rust loader", max_diff(back.frames, py_frames), 0.0)

        rt.save(str(tmp / "rs.c3d"))
        _, back_df, back_markers, _ = read_data_from_c3d(str(tmp / "rs.c3d"))
        record("write", "rust c3d -> python loader", max_diff(H.frames_of(back_df, back_markers), py_frames), 1e-6, note="f32 mm on disk")
        save_to_c3d(str(tmp / "py.c3d"), trc_df, fps, markers, len(trc_df))
        back = mstudio.load(str(tmp / "py.c3d"))
        record("write", "python c3d -> rust loader", max_diff(back.frames, py_frames), 1e-6, note="f32 mm on disk")
        _, py_back_df, _, _ = read_data_from_c3d(str(tmp / "py.c3d"))
        record("write", "c3d: rust writer == python writer (both read by python)", max_diff(H.frames_of(back_df, back_markers), H.frames_of(py_back_df, markers)), 0.0)

    # --------------------------------------------- segments / joints ------
    rg = ReportGenerator.__new__(ReportGenerator)  # only the pattern-matching helpers are used
    cols = set(trc_df.columns)
    for name in SKELETONS:
        model = getattr(skeletons, name)
        segs = ReportGenerator._get_standard_segments_from_skeleton(rg, model)
        segs = {k: v for k, v in segs.items() if all(f"{m}_X" in cols for m in v)}
        joints = ReportGenerator._get_standard_joints_from_skeleton(rg, model)
        joints = {k: v for k, v in joints.items() if all(f"{m}_X" in cols for m in v)}
        rs_segs = {n: [markers[a], markers[b]] for n, a, b in mstudio.auto_segments(markers)}
        rs_joints = {n: [markers[a], markers[v], markers[c]] for n, a, v, c in mstudio.auto_joints(markers)}
        # The oracle matches the patterns against the *model's node names* and then drops
        # segments whose markers are missing from the data, never trying the next pattern;
        # the port matches against the data's marker names. Everything the oracle finds
        # must be found identically; anything extra is a documented deviation.
        differ = [k for k in segs if segs[k] != rs_segs.get(k)] + [k for k in joints if joints[k] != rs_joints.get(k)]
        extra = sorted(set(rs_segs) - set(segs)) + sorted(set(rs_joints) - set(joints))
        record(
            "segments",
            f"{name}: oracle {len(segs)} segments / {len(joints)} joints",
            0.0 if not differ else float("inf"),
            0.0,
            deviation=(not differ and bool(extra)),
            note=(f"port also finds {extra} (matched on the data, not on the model's node list)" if extra and not differ else "")
            + (f" differ: {differ}" if differ else ""),
        )

    # ------------------------------------------------------------- report --
    fails = [r for r in rows if r["status"] == "FAIL"]
    if not args.quiet:
        w = max(len(r["case"]) for r in rows)
        print(f"{'section':10s} {'case':{w}s} {'max|diff|':>12s} {'tol':>8s} status  note")
        for r in rows:
            extra = f"  py {r['py_ms']} ms / rs {r['rs_ms']} ms" if "py_ms" in r else ""
            print(f"{r['section']:10s} {r['case']:{w}s} {r['max_abs_diff']:12.3e} {r['tol']:8.0e} {r['status']:9s} {r['note']}{extra}")
        print()
    env = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "mstudio": mstudio.__version__,
    }
    summary = {
        "env": env,
        "total": len(rows),
        "ok": sum(r["status"] == "ok" for r in rows),
        "deviation": sum(r["status"] == "deviation" for r in rows),
        "fail": len(fails),
    }
    print(json.dumps(summary))
    if args.json:
        args.json.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
