"""Import helpers for driving the Python oracle (MStudio v0.1.5) headlessly.

Shared by `check_parity.py`; `gen_golden.py` predates it and carries its own copy.
Importing this module makes `MStudio.*` importable from the repository root
and replaces the `tkinter.messagebox` calls the oracle makes with exceptions
(or a no-op for the "Save Successful" box), so nothing ever blocks on a dialog.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class RaisingMessageBox:
    def __getattr__(self, name):
        def _raise(*args, **kwargs):
            raise RuntimeError(f"messagebox.{name} called by the oracle: {args} {kwargs}")

        return _raise


class QuietSaverMessageBox(RaisingMessageBox):
    def showinfo(self, *args, **kwargs):
        return None


try:
    import tkinter  # noqa: F401
    import tkinter.filedialog  # noqa: F401  (reportGenerator imports it at module level)
    import tkinter.messagebox  # noqa: F401
except Exception:  # interpreter built without Tk: provide stub modules
    _tk = types.ModuleType("tkinter")
    for _name in ("messagebox", "filedialog"):
        _sub = types.ModuleType(f"tkinter.{_name}")
        setattr(_tk, _name, _sub)
        sys.modules[f"tkinter.{_name}"] = _sub
    sys.modules["tkinter"] = _tk

from MStudio.utils import dataProcessor  # noqa: E402
from MStudio.utils import dataSaver  # noqa: E402

dataProcessor.messagebox = RaisingMessageBox()
dataSaver.messagebox = QuietSaverMessageBox()


def frames_of(df: pd.DataFrame, markers: list[str]) -> np.ndarray:
    """`[frame, marker, xyz]` float64 from the oracle's wide DataFrame."""
    cols = [f"{m}_{c}" for m in markers for c in "XYZ"]
    return df[cols].to_numpy(dtype=np.float64).reshape(len(df), len(markers), 3)


def dataframe_of(frames: np.ndarray, markers: list[str], fps: float) -> pd.DataFrame:
    """The oracle's wide DataFrame (`Frame#`, `Time`, `<M>_X/_Y/_Z`) from an array."""
    n = frames.shape[0]
    data = {"Frame#": np.arange(1, n + 1), "Time": np.arange(n) / fps}
    for i, m in enumerate(markers):
        for k, c in enumerate("XYZ"):
            data[f"{m}_{c}"] = frames[:, i, k]
    return pd.DataFrame(data)


class _Var:
    def __init__(self, value):
        self._v = value

    def get(self):
        return self._v


def viewer_stub(df: pd.DataFrame, marker: str, start: int, end: int, method: str, order: str = "3", pattern=()):
    """Stand-in for `TRCViewer` with just what `dataProcessor.interpolate_*` touch."""
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


def skeleton_pairs_of(model, df: pd.DataFrame) -> list[tuple[str, str]]:
    """`TRCViewer.update_skeleton_pairs`: (parent, child) names present in the data."""
    pairs = []
    for node in model.descendants:
        if node.parent and f"{node.parent.name}_X" in df.columns and f"{node.name}_X" in df.columns:
            pairs.append((node.parent.name, node.name))
    return pairs
