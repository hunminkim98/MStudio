# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Communication

**Always explain in Korean (한국어).** All prose addressed to the user — analysis, plans, code reviews, progress updates, summaries, and answers to questions — is written in Korean, regardless of which language the user writes in.

English stays for everything that lives in the repository: code, identifiers, code comments, docstrings, commit messages, PR descriptions, and the Markdown files under `docs/`. Do not translate existing English documents.

## Commands

```bash
pip install -e .                      # install for development
mstudio                               # launch the GUI (entry point: MStudio.main:main)
python -m MStudio.main                # equivalent

pytest tests -q                       # run tests
pytest tests/test.py::test_import_main # run one test

flake8 MStudio --count --select=E9,F63,F7,F82 --show-source --statistics   # the blocking lint gate in CI
flake8 MStudio --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
```

Rust workspace (migration, see `docs/CROSS_PLATFORM_PLAN.md`; needs `~/.cargo/bin` on PATH):

```bash
cargo run --release -p mstudio-app -- tests/test.trc         # the v2 desktop app (binary: target/release/mstudio)
./target/release/mstudio tests/test.trc --demo --screenshot shot.png --exit-after 4   # headless-ish visual check
./target/release/mstudio tests/test.trc --selftest          # worker filter → delete+undo → HTML report, no dialogs
cargo build --release -p mstudio-spike                       # Phase 0 viewport spike
./target/release/mstudio-spike                               # opens tests/test.trc
./target/release/mstudio-spike --stress 300 50000 --bench 6 --screenshot shot.png   # prints one JSON stats line
.venv/bin/python scripts/gen_golden.py                       # regenerate tests/golden/ from the Python oracle (uv venv, py3.11)
cargo test --workspace                                       # 100+ tests incl. golden parity (core, io, processing)
RUSTFLAGS="-D warnings" cargo clippy --workspace --all-targets --locked   # what CI runs
cargo bench -p mstudio-processing                            # criterion: filters on 300 markers × 50 000 frames
.venv/bin/python scripts/check_c3d_interop.py                # Rust-written C3D/TRC read back by the Python loaders
```

The Rust crates are ports of the Python modules and must stay numerically identical to `tests/golden/` (see `docs/PHASE*_RESULTS.md` for the few documented deviations). When a Python oracle bug is fixed, fix it in Python first, add a test to `tests/test.py`, regenerate the goldens, then port.

Note: `pyproject.toml` sets `python_files = "test.py"`, so pytest only collects files named exactly `test.py`. A new test file named `test_foo.py` will be silently ignored — add tests to `tests/test.py` or change the setting.

CI (`.github/workflows/continuous-integration.yml`) runs on ubuntu/windows/macos-latest/macos-13 with Python 3.10 and 3.11 via conda.

## Architecture

MStudio is a Tkinter (CustomTkinter) desktop app for viewing and editing 3D motion-capture marker data, rendered with OpenGL via `pyopengltk`.

**Migration in progress** — see `docs/CROSS_PLATFORM_PLAN.md`. The target is a Rust core + wgpu renderer + egui UI with PyO3 bindings and HTML reports. The Python code described below is the currently shipped app and serves as the numerical test oracle (`tests/golden/`) for the port; do not refactor it beyond what the plan's Phase 0a needs.

### The god-object + delegation pattern

`MStudio/app.py` defines `TRCViewer(ctk.CTk)` — a ~1450-line class that is *the* application object. Most methods on it are thin wrappers that call a free function in `utils/` or `gui/` with `self` as the first argument:

```python
def open_file(self):          open_file(self)          # utils/dataLoader.py
def create_widgets(self):     create_widgets(self)     # gui/TRCviewerWidgets.py
def filter_selected_data(self): filter_selected_data(self)  # utils/dataProcessor.py
```

Those free functions are written *as if they were methods* — they read and mutate `self.data_manager`, `self.state_manager`, `self.marker_axes`, `self.selection_data`, etc. When editing anything in `utils/dataProcessor.py`, `utils/dataLoader.py`, `utils/viewToggles.py`, `gui/TRCviewerWidgets.py`, `gui/markerPlot.py`, or `gui/plotCreator.py`, treat `self` as the full `TRCViewer` and check `app.py` for the attributes involved. This is deliberate file-splitting, not a mixin — there's no interface to hold onto.

### Core components (`MStudio/core/`)

`TRCViewer.__init__` composes five stateful objects, and callbacks are wired in `_setup_core_callbacks()`:

- **`DataManager`** — owns `data` (the working `pd.DataFrame`), `original_data` (deep copy for non-destructive restore), `marker_names`, and `data_limits`. All data access should go through it, not through raw attributes on the viewer.
- **`StateManager`** — owns `ViewState` / `SelectionState` / `EditingState` dataclasses plus `skeleton_pairs` and `current_skeleton_model`. Fires registered view/selection/editing callbacks on change.
- **`AnimationController`** — playback clock, frame index, fps, loop. `TRCViewer.frame_idx` mirrors it; frame changes flow controller → `_on_frame_changed` → `update_plot()`.
- **`OutlierDetector`** — flags frames where a skeleton bone's length deviates beyond a threshold; has sequential and parallel paths.
- **`MarkerVisualSettings`** — marker/skeleton color, size, opacity; observer callbacks push straight into the GL renderer.

### Data model

Every loader (`utils/dataLoader.py`: TRC, C3D, and Pose2Sim/Sports2D JSON folders) normalizes to one wide DataFrame:

```
Frame# | Time | <Marker>_X | <Marker>_Y | <Marker>_Z | ...
```

Units are **meters** (C3D millimetres are divided by 1000 on load). This `<name>_X/_Y/_Z` column convention is assumed everywhere — filtering, interpolation, skeleton-pair validation, the GL renderer, and the report generator all build column names by string concatenation. `utils/dataSaver.py` writes back out to TRC or C3D.

### Skeletons

`utils/skeletons.py` defines every supported model (HALPE_26, COCO_133, BODY_25/25B/135, BLAZEPOSE, MPII, …) as `anytree` `Node` trees with `id` = keypoint index. Two distinct steps:

1. `DataManager.update_keypoint_names(model)` — renames DataFrame columns from numeric keypoint ids to the model's joint names.
2. `TRCViewer.update_skeleton_pairs()` — walks `model.descendants` and emits `(parent, child)` pairs, **skipping any pair whose columns are absent** from the data. Renderer and outlier detection both consume these pairs.

`utils/skeleton_config.py` holds regex-ish name patterns used to auto-identify segments and joints for analysis/reporting.

### Rendering

`gui/opengl/GLPlotCreator.py` (`MarkerGLFrame`, base OpenGL frame + camera) → `gui/opengl/GLMarkerRenderer.py` (`MarkerGLRenderer`, ~3000 lines: markers, skeleton, trajectories, marker-name labels, analysis overlays, and a `PickingTexture` for color-ID marker picking under the mouse). Only OpenGL rendering exists; matplotlib is used solely for the 2D per-marker coordinate plots in the side panel.

The one-way data push is `TRCViewer.update_plot()` → `gl_renderer.set_frame_data(data, frame_idx, marker_names, current_marker, show_names, show_trajectory, show_skeleton, coordinate_system, skeleton_pairs)` → `gl_renderer.update_plot()`. Renderer state is set through explicit `set_*` methods; it never reaches back into the viewer except via `_notify_marker_selected`.

**Coordinate systems:** Y-up (default) and Z-up. The underlying data is never transformed — the renderer applies a different X-axis rotation per system (`COORDINATE_X_ROTATION_Y_UP` / `_Z_UP`). `StateManager` is the source of truth (`view_state.is_z_up` / `coordinate_system`).

**Headless:** `app.py` selects the matplotlib `Agg` backend when `$DISPLAY` is unset so imports work in CI; the OpenGL renderer is only constructed when a plot is actually created.

### Filtering

`utils/filtering.py` is vendored from **Pose2Sim** (authored by David Pagnon, BSD-3) and keeps its Config.toml-shaped API: every filter takes a nested `config_dict` like `{'filtering': {'butterworth': {'order': 4, 'cut_off_frequency': 6}}}`, dispatched through `filter1d(col, config_dict, filter_type, frame_rate)`. `utils/dataProcessor.py` builds that dict from the Tk variables in `self.filter_params` and applies the filter per-axis over the selected frame range. Preserve this shape when adding filters, and keep upstream attribution intact.

### Performance

`utils/performance_utils.py` provides `PerformanceTimer`, `memoize`/`LRUCache`, `debounce`, `throttle`, and `AnimationPerformanceManager`. During playback the app takes a reduced-work path (`_update_display_during_animation`) that redraws the 3D view and only the timeline's current-frame indicator, skipping the full timeline redraw.
