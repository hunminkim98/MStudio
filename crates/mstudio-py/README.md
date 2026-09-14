# mstudio (Python package)

Python bindings for the MStudio Rust workspace: `Take` ↔ NumPy (zero-copy
views), the Pose2Sim filters, gap interpolation, pattern-based gap filling,
outlier detection, TRC/C3D/JSON I/O, the HTML analysis report and
`mstudio.run()` to open the desktop application.

```python
import mstudio

take = mstudio.load("tests/test.trc")
take.frames.shape                                   # (n_frames, n_markers, 3), meters, NaN = missing
take.filter("RKnee", mstudio.Filter.butterworth(order=4, cutoff_hz=10))
take.interpolate("LWrist", "cubic", first=85, last=100)
take.pattern_interpolate("RKnee", ["RHip", "RAnkle"], first=35, last=55)
flags = take.outliers(mstudio.skeleton_pairs("HALPE_26", take.markers))
take.save("out.trc")
mstudio.write_report(take, "report.html", skeleton="HALPE_26")
mstudio.run("tests/test.trc")                       # the desktop app, blocks until closed
```

Build locally (from the repository root, after `uv venv .venv` / `pip install maturin`):

```bash
.venv/bin/maturin develop --release -m crates/mstudio-py/Cargo.toml
.venv/bin/pytest crates/mstudio-py/tests -q
.venv/bin/python scripts/check_parity.py          # live comparison against the Python oracle
```
