#!/usr/bin/env python
"""
Cross-check the Rust writers against the Python ecosystem readers.

Converts tests/test.trc → C3D and tests/test.c3d → TRC with the Rust
`convert` example, reads both results back with the Python loaders of
MStudio v0.1.5 (the `c3d` package and pandas), and compares them with the
golden arrays. Run from the repository root with the Rust toolchain on PATH:

    .venv/bin/python scripts/check_c3d_interop.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from MStudio.utils.dataLoader import read_data_from_c3d, read_data_from_trc  # noqa: E402

OUT = ROOT / "target" / "interop"
OUT.mkdir(parents=True, exist_ok=True)
manifest = json.loads((ROOT / "tests" / "golden" / "manifest.json").read_text())


def convert(src: Path, dst: Path) -> None:
    subprocess.run(
        ["cargo", "run", "-q", "-p", "mstudio-io", "--example", "convert", "--", str(src), str(dst)],
        cwd=ROOT,
        check=True,
    )


def frames(df, markers):
    return df[[f"{m}_{c}" for m in markers for c in "XYZ"]].to_numpy().reshape(len(df), len(markers), 3)


ok = True

convert(ROOT / "tests" / "test.trc", OUT / "from_trc.c3d")
_, df, markers, fps = read_data_from_c3d(str(OUT / "from_trc.c3d"))
golden = np.load(ROOT / "tests" / "golden" / "trc_frames.npy")
diff = float(np.nanmax(np.abs(frames(df, markers) - golden)))
good = markers == manifest["trc"]["markers"] and fps == manifest["trc"]["fps"] and diff < 1e-6 and len(df) == golden.shape[0]
ok &= good
print(f"[{'OK' if good else 'FAIL'}] Rust C3D read by python c3d: {len(df)} frames, fps {fps}, max|diff| {diff:.2e}")

convert(ROOT / "tests" / "test.c3d", OUT / "from_c3d.trc")
_, df, markers, fps = read_data_from_trc(str(OUT / "from_c3d.trc"))
golden = np.load(ROOT / "tests" / "golden" / "c3d_frames.npy")
diff = float(np.nanmax(np.abs(frames(df, markers) - golden)))
good = markers == manifest["c3d"]["markers"] and fps == manifest["c3d"]["fps"] and diff < 1e-12 and len(df) == golden.shape[0]
ok &= good
print(f"[{'OK' if good else 'FAIL'}] Rust TRC read by pandas: {len(df)} frames, fps {fps}, max|diff| {diff:.2e}")

sys.exit(0 if ok else 1)
