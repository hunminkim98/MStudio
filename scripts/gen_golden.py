#!/usr/bin/env python
"""
Generate golden reference outputs from the Python implementation (MStudio v0.1.5).

These files are the numerical contract for the Rust port
(docs/CROSS_PLATFORM_PLAN.md, Phase 0a). Every processing primitive the Rust
crates reimplement must reproduce these arrays within the tolerance recorded in
manifest.json.

Run from the repository root:

    .venv/bin/python scripts/gen_golden.py

Regenerate only when the Python oracle itself changes on purpose (e.g. a
Pose2Sim filter update); commit the regenerated files together with that change.
"""
from __future__ import annotations

import json
import platform
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "tests" / "golden"
OUT.mkdir(parents=True, exist_ok=True)

# The oracle modules import tkinter.messagebox at module level. Never let a real
# message box open while generating goldens: replace it with one that raises.
class _RaisingMessageBox:
    def __getattr__(self, name):
        def _raise(*args, **kwargs):
            raise RuntimeError(f"messagebox.{name} called during golden generation: {args} {kwargs}")
        return _raise

try:
    import tkinter  # noqa: F401
except Exception:  # headless interpreter without Tk: provide a stub module
    _tk = types.ModuleType("tkinter")
    _mb = types.ModuleType("tkinter.messagebox")
    _tk.messagebox = _mb
    sys.modules["tkinter"] = _tk
    sys.modules["tkinter.messagebox"] = _mb

import scipy  # noqa: E402
import statsmodels  # noqa: E402
import filterpy  # noqa: E402

from MStudio.utils.dataLoader import read_data_from_trc, read_data_from_c3d  # noqa: E402
from MStudio.utils.dataSaver import save_to_trc, save_to_c3d  # noqa: E402
from MStudio.utils.filtering import filter1d  # noqa: E402
from MStudio.utils import dataProcessor  # noqa: E402
from MStudio.utils.analysisMode import (  # noqa: E402
    calculate_distance,
    calculate_angle,
    calculate_arc_points,
    calculate_velocity,
    calculate_acceleration,
)
from MStudio.core.outlier_detector import OutlierDetector  # noqa: E402
from MStudio.utils import skeletons  # noqa: E402

dataProcessor.messagebox = _RaisingMessageBox()

manifest: dict = {
    "generated_with": {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
        "filterpy": filterpy.__version__,
    },
    "tolerance": {
        "default_abs": 1e-6,
        "notes": "Kalman (filterpy) and LOESS (statsmodels) may need 1e-5; everything else 1e-6.",
    },
    "files": {},
}


def save(name: str, arr: np.ndarray, description: str) -> None:
    arr = np.asarray(arr)
    np.save(OUT / f"{name}.npy", arr)
    manifest["files"][name] = {"shape": list(arr.shape), "dtype": str(arr.dtype), "description": description}
    print(f"  {name:48s} {str(arr.shape):20s} {arr.dtype}")


def frames_of(df: pd.DataFrame, markers: list[str]) -> np.ndarray:
    cols = [f"{m}_{c}" for m in markers for c in "XYZ"]
    return df[cols].to_numpy(dtype=np.float64).reshape(len(df), len(markers), 3)


# --------------------------------------------------------------------------- #
# 1. Loaders
# --------------------------------------------------------------------------- #
print("Loaders")
_, trc_df, trc_markers, trc_fps = read_data_from_trc(str(ROOT / "tests" / "test.trc"))
save("trc_frames", frames_of(trc_df, trc_markers), "tests/test.trc loaded: [frame, marker, xyz] meters")
save("trc_time", trc_df["Time"].to_numpy(dtype=np.float64), "tests/test.trc Time column")
manifest["trc"] = {"markers": trc_markers, "fps": float(trc_fps), "n_frames": int(len(trc_df))}

_, c3d_df, c3d_markers, c3d_fps = read_data_from_c3d(str(ROOT / "tests" / "test.c3d"))
save("c3d_frames", frames_of(c3d_df, c3d_markers), "tests/test.c3d loaded: [frame, marker, xyz] meters (mm/1000)")
save("c3d_time", c3d_df["Time"].to_numpy(dtype=np.float64), "tests/test.c3d Time column")
manifest["c3d"] = {"markers": c3d_markers, "fps": float(c3d_fps), "n_frames": int(len(c3d_df))}

# --------------------------------------------------------------------------- #
# 2. Writers (round-trip fixtures for the Rust writers)
# --------------------------------------------------------------------------- #
print("Writers")
save_to_trc(str(OUT / "trc_roundtrip.trc"), trc_df, trc_fps, trc_markers, len(trc_df))
manifest["files"]["trc_roundtrip.trc"] = {"description": "save_to_trc(test.trc data) byte-exact output"}
try:
    save_to_c3d(str(OUT / "c3d_roundtrip.c3d"), trc_df, trc_fps, trc_markers, len(trc_df))
    manifest["files"]["c3d_roundtrip.c3d"] = {"description": "save_to_c3d(test.trc data); compare by re-reading, not bytes"}
except Exception as e:  # keep going; note it
    manifest["files"]["c3d_roundtrip.c3d"] = {"description": f"NOT GENERATED: {e}"}
    print(f"  c3d writer failed: {e}")

# --------------------------------------------------------------------------- #
# 3. Synthetic gaps (deterministic) for interpolation + NaN-handling filters
# --------------------------------------------------------------------------- #
GAPS = {  # marker -> inclusive frame ranges set to NaN on all three axes
    "RKnee": [[40, 50]],
    "LWrist": [[90, 95]],
    "RAnkle": [[20, 20]],
}
gapped_df = trc_df.copy()
for m, ranges in GAPS.items():
    for a, b in ranges:
        for c in "XYZ":
            gapped_df.loc[a:b, f"{m}_{c}"] = np.nan
save("trc_frames_gapped", frames_of(gapped_df, trc_markers), "trc_frames with GAPS applied (NaN)")
manifest["gaps"] = GAPS

# --------------------------------------------------------------------------- #
# 4. Filters — mirrors utils/dataProcessor.filter_selected_data over the full
#    range, applied to every marker and axis.
# --------------------------------------------------------------------------- #
FILTER_CASES = [
    ("butterworth", {"order": 4, "cut_off_frequency": 10.0}),  # app default
    ("butterworth", {"order": 4, "cut_off_frequency": 6.0}),  # Pose2Sim default
    ("butterworth", {"order": 2, "cut_off_frequency": 6.0}),
    ("butterworth_on_speed", {"order": 4, "cut_off_frequency": 10.0}),  # app default
    ("butterworth_on_speed", {"order": 2, "cut_off_frequency": 6.0}),
    ("kalman", {"trust_ratio": 20.0, "smooth": 1.0}),  # app default
    ("kalman", {"trust_ratio": 20.0, "smooth": 0.0}),
    ("kalman", {"trust_ratio": 100.0, "smooth": 1.0}),
    ("gaussian", {"sigma_kernel": 3.0}),  # app default
    ("gaussian", {"sigma_kernel": 1.0}),
    ("LOESS", {"nb_values_used": 10.0}),  # app default
    ("LOESS", {"nb_values_used": 30.0}),
    ("median", {"kernel_size": 3.0}),  # app default
    ("median", {"kernel_size": 5.0}),
]


def apply_filter(df: pd.DataFrame, markers: list[str], ftype: str, params: dict, fps: float) -> np.ndarray:
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
    return frames_of(out, markers)


print("Filters")
manifest["filter_cases"] = []
for i, (ftype, params) in enumerate(FILTER_CASES):
    tag = f"filter_{i:02d}_{ftype}"
    save(tag, apply_filter(trc_df, trc_markers, ftype, params, trc_fps), f"{ftype} {params} on trc_frames")
    save(tag + "_gapped", apply_filter(gapped_df, trc_markers, ftype, params, trc_fps), f"{ftype} {params} on trc_frames_gapped")
    manifest["filter_cases"].append({"index": i, "type": ftype, "params": params, "fps": float(trc_fps)})

# --------------------------------------------------------------------------- #
# 5. Interpolation — runs the real utils/dataProcessor functions through a
#    stand-in for the TRCViewer object.
# --------------------------------------------------------------------------- #
class _Var:
    def __init__(self, value):
        self._v = value

    def get(self):
        return self._v


def make_viewer_stub(df: pd.DataFrame, marker: str, start: int, end: int, method: str, order: str = "3", pattern=()):
    stub = types.SimpleNamespace()
    stub.selection_data = {"start": start, "end": end}
    stub.marker_axes = []
    stub.interp_method_var = _Var(method)
    stub.order_var = _Var(order)
    stub.state_manager = types.SimpleNamespace(
        selection_state=types.SimpleNamespace(current_marker=marker, pattern_markers=list(pattern)),
        editing_state=types.SimpleNamespace(pattern_selection_mode=False),
    )
    stub.data_manager = types.SimpleNamespace(data=df)
    noop = lambda *a, **k: None  # noqa: E731
    stub.detect_outliers = noop
    stub.update_plot = noop
    stub.highlight_selection = noop
    stub.show_marker_plot = noop
    return stub


INTERP_METHODS = ["linear", "polynomial", "spline", "nearest", "zero", "slinear", "quadratic", "cubic"]
INTERP_CASES = [("RKnee", 35, 55), ("LWrist", 85, 100), ("RAnkle", 15, 25)]

print("Interpolation")
manifest["interp_cases"] = []
for marker, start, end in INTERP_CASES:
    for method in INTERP_METHODS:
        df = gapped_df.copy()
        stub = make_viewer_stub(df, marker, start, end, method, order="3")
        dataProcessor.interpolate_selected_data(stub)
        tag = f"interp_{method}_{marker}"
        save(tag, frames_of(df, [marker])[:, 0, :], f"{method}(order=3) on {marker} gaps within [{start},{end}]")
        manifest["interp_cases"].append({"marker": marker, "range": [start, end], "method": method, "order": 3, "file": tag})

PATTERN_CASES = [
    ("RKnee", 35, 55, ["RHip"]),
    ("RKnee", 35, 55, ["RHip", "RAnkle"]),
    ("RKnee", 35, 55, ["RHip", "RAnkle", "LKnee"]),
    ("LWrist", 85, 100, ["LElbow", "LShoulder", "LIndex"]),
]
manifest["pattern_cases"] = []
for marker, start, end, refs in PATTERN_CASES:
    df = gapped_df.copy()
    stub = make_viewer_stub(df, marker, start, end, "pattern-based", pattern=refs)
    dataProcessor.interpolate_with_pattern(stub)
    tag = f"pattern_{marker}_{len(refs)}ref"
    save(tag, frames_of(df, [marker])[:, 0, :], f"pattern-based on {marker} using {refs} within [{start},{end}]")
    manifest["pattern_cases"].append({"marker": marker, "range": [start, end], "references": refs, "file": tag})

# --------------------------------------------------------------------------- #
# 6. Skeleton pairs + outliers — mirrors TRCViewer.update_skeleton_pairs and
#    OutlierDetector(threshold=0.3), sequential path.
# --------------------------------------------------------------------------- #
def pairs_for(model, df: pd.DataFrame) -> list[tuple[str, str]]:
    pairs = []
    for node in model.descendants:
        if node.parent and f"{node.parent.name}_X" in df.columns and f"{node.name}_X" in df.columns:
            pairs.append((node.parent.name, node.name))
    return pairs


print("Skeletons + outliers")
manifest["skeleton_pairs"] = {}
manifest["outlier_threshold"] = 0.3
detector = OutlierDetector(threshold=0.3, use_parallel=False)
for model_name in ["HALPE_26", "COCO_17", "COCO_133", "BODY_25B", "BODY_25"]:
    pairs = pairs_for(getattr(skeletons, model_name), trc_df)
    manifest["skeleton_pairs"][model_name] = pairs
    if not pairs:
        continue
    outliers = detector.detect_outliers(trc_df, trc_markers, pairs)
    arr = np.stack([outliers[m] for m in trc_markers]).astype(np.uint8)
    save(f"outliers_{model_name}", arr, f"[marker, frame] outlier flags, threshold 0.3, pairs from {model_name}")
    outliers_g = detector.detect_outliers(gapped_df, trc_markers, pairs)
    save(f"outliers_{model_name}_gapped", np.stack([outliers_g[m] for m in trc_markers]).astype(np.uint8),
         f"same on trc_frames_gapped")

# --------------------------------------------------------------------------- #
# 7. Analysis primitives
# --------------------------------------------------------------------------- #
print("Analysis")
P = frames_of(trc_df, trc_markers)
idx = {m: i for i, m in enumerate(trc_markers)}
n = len(P)


def series(name: str) -> np.ndarray:
    return P[:, idx[name], :]


dist = np.full(n, np.nan)
ang = np.full(n, np.nan)
for f in range(n):
    d = calculate_distance(series("RHip")[f], series("RKnee")[f])
    a = calculate_angle(series("RHip")[f], series("RKnee")[f], series("RAnkle")[f])
    dist[f] = np.nan if d is None else d
    ang[f] = np.nan if a is None else a
save("analysis_distance_RHip_RKnee", dist, "calculate_distance(RHip, RKnee) per frame")
save("analysis_angle_RHip_RKnee_RAnkle", ang, "calculate_angle(RHip, RKnee(vertex), RAnkle) per frame, degrees")

vel = np.full((n, 3), np.nan)
for f in range(1, n - 1):
    v = calculate_velocity(series("RKnee")[f - 1], series("RKnee")[f], series("RKnee")[f + 1], trc_fps)
    if v is not None:
        vel[f] = v
acc = np.full((n, 3), np.nan)
for f in range(2, n - 2):
    a = calculate_acceleration(vel[f - 1], vel[f + 1], trc_fps)
    if a is not None:
        acc[f] = a
save("analysis_velocity_RKnee", vel, "calculate_velocity(prev, curr, next, fps) per frame; NaN at edges")
save("analysis_acceleration_RKnee", acc, "calculate_acceleration(vel[f-1], vel[f+1], fps); NaN at edges")

arc = calculate_arc_points(series("RKnee")[0], series("RHip")[0], series("RAnkle")[0])
if arc is not None:
    save("analysis_arc_RKnee_frame0", np.stack(arc), "calculate_arc_points(vertex=RKnee, RHip, RAnkle) frame 0, default radius/segments")

# --------------------------------------------------------------------------- #
with open(OUT / "manifest.json", "w", encoding="utf-8") as fh:
    json.dump(manifest, fh, indent=2)
print(f"\nWrote {len(manifest['files'])} entries to {OUT.relative_to(ROOT)}/")
