# MStudio v2 — Rust core, wgpu viewport, 3-OS native

**Branch:** `feature/cross-platform-renderer`
**Status:** planning, revision 2 (2026-09-14)
**Decision:** rewrite as a **Rust core + wgpu renderer + egui UI**, exposed to Python through **PyO3**. Reports become **self-contained interactive HTML**. The existing Python code stays in-tree during the port as the **numerical test oracle**.

Revision 1 of this document proposed PySide6 + pygfx (Python). It was superseded after deciding that implementation effort is not a selection criterion; only the quality of the end result is. See §2.

---

## 1. Goals

| # | Goal | Measurable exit criterion |
|---|---|---|
| G1 | Runs natively on Windows, macOS (Apple Silicon + Intel), Linux | CI matrix green on all runners incl. an offscreen render test; one signed/notarized artifact per OS |
| G2 | Blender-level viewport smoothness | 60 fps (or display refresh) sustained with ≥ 300 markers × ≥ 50 000 frames; **< 0.1 ms CPU per frame** outside the GPU submit; camera drag never drops a frame |
| G3 | Feature parity with v0.1.5 | Every row in §7 checked, filter outputs match the Python oracle within `1e-6` |
| G4 | "High-end" feel | Cold start **< 0.5 s**; single binary **< 30 MB**; HiDPI-correct; dark/light theme; dockable panels |
| G5 | Stays a first-class citizen of the Pose2Sim / Sports2D Python ecosystem | `pip install mstudio` works (maturin wheel) and `import mstudio` exposes the core API |

**Non-goals for this milestone:** multi-person, gait mode, new analysis features. Parity first.

---

## 2. Why this stack

**What we compared** (details in the conversation record): PySide6 + pygfx (Python), PySide6 + moderngl, Tauri/Electron + three.js, Dear PyGui, forking rerun, Rust + wgpu.

**Why Rust + wgpu wins once effort is excluded**

| Axis | Python (PySide6 + pygfx) | Rust core | Winner |
|---|---|---|---|
| Viewport fps | 60 (≈ 1 ms Python/frame) | 60 (≈ 0.05 ms/frame) | tie — both GPU-bound |
| Cold start | ~2 s | ~0.2 s | Rust — user-perceivable |
| Install size | ~300 MB | ~10–30 MB | Rust — user-perceivable |
| Memory ceiling | pandas copies | explicit | Rust |
| Parallel filtering | GIL workarounds | rayon | Rust |
| GC jitter | cycle-GC pauses possible | none | Rust |
| pip installability | yes | yes (maturin, as rerun-sdk does) | tie |
| Researcher scripting | yes | yes via PyO3 | tie |

This is the **Blender architecture** (native core + Python API) and the **rerun architecture** (Rust core + Python SDK).

**Why not fork rerun.** Forking is a labor-saving move, and labor is not the criterion. Rerun's heart (`re_chunk_store`) is an append-only Arrow log; MStudio is a mutable editor. Removing the heart leaves `re_renderer` and `egui`, both of which are ordinary crates on crates.io. We depend on crates, we do not carry a 60-crate fork.

**Why egui, not Qt.** Rust Qt bindings are immature. egui + eframe is what rerun ships and it looks professional; Blender itself proves a custom-drawn UI can feel high-end. eframe's wgpu backend shares the GPU device with the viewport, so the 3D view and the UI are one swap chain with no compositing hop.

**Why HTML reports.** No Rust matplotlib exists. A self-contained HTML file with an inline chart library gives interactivity (hover, zoom, series toggle), single-file sharing, offline use, and PDF through the browser's print dialog — and removes the last Python dependency from the binary.

---

## 3. Performance architecture — the five rules

Enforced by benchmarks in §8, not by convention.

| Rule | In code |
|---|---|
| **R1 — Data lives on the GPU** | One `wgpu::Buffer` of `f32[N_frames × N_markers × 3]` uploaded at load. Frame advance = write one `u32` frame index into a uniform. Never re-upload per frame. |
| **R2 — Edits are range writes** | `Take` (see §4) is the editable CPU copy. An edit produces a dirty `[f0, f1)`; only that slice goes through `queue.write_buffer` at the matching byte offset. |
| **R3 — Playback follows the frame clock** | No timers. Each `eframe` frame asks `Playback::frame_at(Instant::now())`; the renderer writes the index. The event loop is driven by `request_repaint()` while playing and idles otherwise. |
| **R4 — Heavy work is off the UI thread** | Filtering, interpolation, outlier detection, report generation run on a `rayon` pool or a worker thread; results arrive via channel and are applied between frames. The viewport keeps rendering during a filter. |
| **R5 — One draw call per layer** | Markers: one instanced quad draw. Skeleton: one line-list draw from an index buffer. Trajectories: one line-strip draw with restart indices. Labels: projected to screen and drawn by egui's painter in a single mesh. |

Frame budget at 60 Hz = 16.7 ms. Target: CPU ≤ 0.1 ms, GPU ≤ 2 ms.

---

## 4. Workspace layout

Cargo workspace at repo root. The Python package survives as an oracle and as the thin `pip` entry.

```
Cargo.toml                       # workspace
crates/
├── mstudio-core/                # data model — no I/O, no GPU, no UI
│   ├── take.rs                  #   Take { frames: Array3<f32>, markers: Vec<String>, fps, original: Array3<f32> }
│   ├── skeleton.rs              #   12 models as static (parent, child, id) tables; pair resolution against marker names
│   ├── state.rs                 #   ViewState / SelectionState / EditingState (port of core/state_manager.py)
│   ├── playback.rs              #   frame_at(t), loop, fps  (port of core/animation_controller.py)
│   ├── visual.rs                #   marker/skeleton colors, sizes, presets (port of core/marker_visual_settings.py)
│   └── outliers.rs              #   bone-length outlier detection, rayon (port of core/outlier_detector.py)
├── mstudio-io/                  # TRC (tsv), C3D (evaluate `c3dio`, else port the reader), Pose2Sim/Sports2D JSON folders
├── mstudio-processing/          # filters + interpolation + analysis; pure functions on ndarray
│   ├── filters.rs               #   butterworth, butterworth_on_speed, kalman+RTS, gaussian, loess, median
│   ├── interp.rs                #   linear, nearest, zero, slinear, quadratic, cubic, polynomial, spline, pattern-based
│   └── analysis.rs              #   distance, segment angle vs axis, joint angle, arc points, velocity, acceleration
├── mstudio-render/              # wgpu only — no egui types leak in
│   ├── gpu_take.rs              #   R1/R2: the frame buffer + dirty-range writer
│   ├── markers.rs               #   instanced points pipeline, per-marker color/size/state SSBO
│   ├── skeleton.rs              #   line pipeline, outlier & torso styling
│   ├── trajectories.rs          #   windowed line strips
│   ├── grid.rs                  #   grid + axes, Y-up / Z-up as a root transform
│   ├── camera.rs                #   orbit / pan / zoom / fit, per-OS input normalization
│   ├── picking.rs               #   ID render target + readback (async, no stall)
│   └── analysis_overlay.rs      #   reference line, arc, axis label anchors
├── mstudio-report/              # HTML report: minijinja templates + inline data + vendored Plotly.js
├── mstudio-app/                 # eframe binary: docking, panels, timeline, marker plot, menus, shortcuts, dialogs (rfd)
└── mstudio-py/                  # PyO3 + maturin: `import mstudio` → Take, filters, interp, io, and `mstudio.run()`

MStudio/                         # existing Python app — UNCHANGED until Phase 6, then reduced to a thin launcher
tests/golden/                    # generated ONCE from the Python implementation; the Rust port must reproduce them
```

Crate dependency direction is strictly downward: `app → render, report, processing, io, core`; `py → everything except app` (plus `app` for `run()`).

**Key crates:** `wgpu`, `eframe`/`egui` (wgpu backend), `egui_dock`, `glam`, `ndarray`, `rayon`, `serde`/`serde_json`, `rfd` (native dialogs), `minijinja`, `opener`, `pyo3` + `numpy`, `maturin`. Evaluate in the spike: `re_renderer` (rerun's renderer crate) vs. hand-written pipelines; `sci-rs` for `butter`/`sosfiltfilt` parity with scipy.

---

## 5. Data model

```rust
pub struct Take {
    pub markers: Vec<String>,           // column order == GPU order
    pub fps: f32,
    pub frames: Array3<f64>,            // [n_frames, n_markers, 3], meters — f64 for oracle parity at 1e-6; f32 only on the GPU
    pub original: Array3<f64>,          // deep copy at load; `restore_original()` copies back
    pub time: Array1<f32>,
}
```

- Loaders in `mstudio-io` all produce a `Take`; units are meters (C3D mm ÷ 1000 as today).
- Column naming `<Marker>_X/_Y/_Z` is now an *export* concern only (TRC writer). Internally markers are indices.
- Skeleton keypoint renaming (`update_keypoint_names`) becomes `Take::rename_markers(&SkeletonModel)`.
- Edits return `DirtyRange { f0, f1 }`; the app forwards it to `GpuTake::write_range` (R2) and to outlier re-detection (R4).

---

## 6. Report design (HTML)

- `mstudio-report` renders one **self-contained `.html`**: inline CSS, inline vendored **Plotly.js basic bundle (~1 MB)**, inline JSON of the analysis results. Works offline, shareable as one file.
- Sections mirror the current PDF: dataset overview & quality, per-marker coordinates with stats, velocity/acceleration, segment angles, joint angles. Each chart: hover, zoom, series toggle, PNG export from Plotly's toolbar.
- Marker selector and frame-range slider at the top re-filter every chart client-side.
- `@media print` stylesheet → the user prints to PDF from the browser; this replaces the PdfPages output.
- App flow: **Report → Generate…** → `rfd` save dialog → write file → `opener::open()` in the default browser. Generation runs on a worker (R4).
- Skeleton segment/joint auto-detection reuses the patterns in `utils/skeleton_config.py`, ported into `mstudio-processing/analysis.rs`.

---

## 7. Phases

### Phase 0 — Oracle capture + Rust spike  (1 week) — **0a DONE, 0b macOS GO; see `PHASE0_RESULTS.md`**

Two independent tracks.

**0a — Golden files from Python (do this first, nothing else depends on Rust)**
- [x] For `tests/test.trc` and `tests/test.c3d`: dump loaded arrays, every filter with every parameter set in `filterUI.py`, every interpolation method on a fixed gap set, outlier maps for HALPE_26, and analysis values (distance/angles/velocity) to `tests/golden/*.npy` + `manifest.json`
- [x] Freeze them in git; they are the contract for the port

**0b — Spike in `crates/spike/`**
- [x] `eframe` window (wgpu backend), load `tests/test.trc`
- [x] `GpuTake` upload, instanced markers, skeleton line list, egui-painted labels, orbit camera
- [x] Playback through the frame clock (R3), on-screen fps + CPU-µs counters
- [x] ID-buffer picking, click selects a marker
- [x] Stress: 300 markers × 50 000 frames
- [x] Run on this Mac · [ ] one Windows machine · [ ] one Linux machine

**Go if:** display-refresh fps on all three with CPU < 0.1 ms/frame and picking is exact. If `re_renderer` was evaluated, decide here whether to adopt it.

### Phase 1 — Workspace + core + I/O  (1 week) — **DONE locally; see `PHASE1_RESULTS.md`**
- [x] Cargo workspace, CI matrix (ubuntu / windows / macos-14 arm / macos-13 x86) with `cargo test`, `clippy -D warnings`, `rustfmt --check`
- [x] `mstudio-core`: `Take`, skeleton tables (12 models), state, playback, visual settings, outliers
- [x] `mstudio-io`: TRC read/write, C3D read/write, JSON folder read; round-trip tests against golden arrays
- [x] Linux CI installs `mesa-vulkan-drivers` (lavapipe) so wgpu tests can run headless

### Phase 2 — Processing with oracle parity  (1–2 weeks) — **DONE; see `PHASE2_RESULTS.md`**
- [x] `filters.rs`: six filters; test each against golden output at `1e-6` (Butterworth via SOS + zero-phase; Kalman + RTS ported from filterpy semantics; LOESS ported from statsmodels' `lowess` defaults)
- [x] `interp.rs`: nine methods incl. pattern-based; golden parity
- [x] `analysis.rs`: distance, angles, arc, velocity, acceleration; golden parity
- [x] Benchmarks: full-take Butterworth on 300 markers × 50 000 frames < 200 ms on 8 cores

### Phase 3 — Renderer  (2 weeks)
- [ ] `mstudio-render` pipelines: markers (states: normal/selected/pattern/analysis/outlier-frame), skeleton (outlier recolor, torso width), trajectories (± window), grid/axes, Y-up/Z-up root transform
- [ ] Camera: orbit LMB, pan RMB **and** MMB, wheel zoom normalized (macOS delta, X11 buttons 4/5 handled by winit), reset, fit-to-data
- [ ] Picking: ID render target, async map + readback, no pipeline stall
- [ ] Analysis overlay: reference line with clickable axis cycle, angle arc, label anchors
- [ ] Offscreen render tests: render one frame to a texture, assert marker/skeleton pixel counts

### Phase 4 — App shell, playback, timeline  (1–2 weeks)
- [ ] `mstudio-app`: `egui_dock` layout (viewport center, panels right, timeline + marker plot bottom), menus, shortcuts (Space/Enter/Esc/←/→), `rfd` dialogs, icon, dark/light theme
- [ ] Timeline widget: frame/time tick modes, scrub, range selection, current-frame cursor
- [ ] Marker X/Y/Z plot with `egui_plot`: current-frame line, range select, pan/zoom
- [ ] Playback controls, loop, fps entry

### Phase 5 — Editing, panels, report  (2 weeks)
- [ ] Edit mode: delete range, restore original — dirty ranges to GPU (R2)
- [ ] Filter panel (6 filters, same parameter sets), interpolation panel (9 methods, pattern marker selection in viewport) — all on workers (R4)
- [ ] Skeleton model combo → rename → pairs → renderer + outlier re-detect
- [ ] Visual customization panel + presets; analysis mode + selected-marker list
- [ ] `mstudio-report`: templates, Plotly bundle, sections, print stylesheet, open-in-browser

### Phase 6 — Python bindings, parity QA, distribution  (1–2 weeks)
- [ ] `mstudio-py`: `Take` ↔ numpy zero-copy, filters/interp/io exposed, `mstudio.run()` launches the app in-process; maturin wheels for all OSes on CI
- [ ] `MStudio/` reduced to a thin launcher + deprecation shim; `mstudio` entry point unchanged for users
- [ ] §8 parity checklist walked on real Windows / macOS / Linux
- [ ] `cargo-dist`: standalone binaries; macOS notarization; Windows signing (or documented unsigned path)
- [ ] README rewrite (controls table: RMB or MMB pan), `CLAUDE.md` architecture section rewritten for the Rust workspace, version `0.2.0`

**Total ≈ 9–11 weeks** of calendar time with AI-assisted implementation; phases 2 and 3 run in parallel after Phase 1.

---

## 8. Feature parity checklist (v0.1.5 → v2)

| Area | v0.1.5 | v2 crate / module | Done |
|---|---|---|---|
| Open TRC / C3D / JSON folder (multi-select) | `utils/dataLoader.open_file` | `mstudio-io` + `app` dialogs | ☐ |
| Save As TRC / C3D | `utils/dataSaver.save_as` | `mstudio-io` | ☐ |
| Markers with size / opacity / color states | `GLMarkerRenderer._render_markers_immediate` | `render/markers.rs` | ☐ |
| Skeleton lines, outlier highlight, torso width | `_render_skeleton_immediate` | `render/skeleton.rs` | ☐ |
| Trajectories (± window) | `_render_trajectories_immediate` | `render/trajectories.rs` | ☐ |
| Marker name labels (toggle) | `_render_marker_names_immediate` (GLUT) | egui painter over projected positions | ☐ |
| Grid + axes, Y-up / Z-up | `GridUtils`, `set_coordinate_system` | `render/grid.rs` | ☐ |
| Orbit / pan / zoom / reset | `on_mouse_*`, `reset_view` | `render/camera.rs` | ☐ |
| Click-to-select (picking) | `PickingTexture`, `pick_marker` | `render/picking.rs` | ☐ |
| Analysis: distance, segment angle (axis cycle), joint angle + arc | `_render_analysis_immediate`, `analysisMode.py` | `processing/analysis.rs` + `render/analysis_overlay.rs` | ☐ |
| Play / pause / stop / loop / fps / prev / next | `AnimationController` + Tk `after` | `core/playback.rs` + frame clock | ☐ |
| Timeline (frame & time modes, scrub, range) | `update_timeline` | `app/timeline.rs` | ☐ |
| Marker X/Y/Z plot with range selection | `gui/markerPlot.py` (matplotlib) | `app/marker_plot.rs` (`egui_plot`) | ☐ |
| Edit mode: delete range, restore original | `toggle_edit_mode`, `delete_selected_data` | `app/panels/edit.rs` | ☐ |
| Filters ×6 with params | `dataProcessor.filter_selected_data` | `processing/filters.rs` + worker | ☐ |
| Interpolation ×9 incl. pattern-based | `interpolate_*` | `processing/interp.rs` + worker | ☐ |
| Outlier detection (threshold, parallel) | `OutlierDetector` | `core/outliers.rs` (rayon) | ☐ |
| 12 skeleton models + keypoint rename | `on_model_change`, `update_keypoint_names` | `core/skeleton.rs` | ☐ |
| Visual customization panel + presets | `TRCviewerWidgets._create_customization_*` | `app/panels/visual.rs` | ☐ |
| Analysis report | `reportGenerator.py` → PDF | `mstudio-report` → interactive HTML (+ print-to-PDF) | ☐ |
| Keyboard shortcuts | `app.py` binds | `app/shortcuts.rs` | ☐ |
| Resizable right panel | custom sizer | `egui_dock` | ☐ |
| Window icon | `iconbitmap(.ico)` | `eframe` `IconData` (PNG) | ☐ |
| Python API | — | `mstudio-py` (new; G5) | ☐ |

---

## 9. Testing strategy

| Layer | How | Where |
|---|---|---|
| `core`, `io`, `processing` | `cargo test` against `tests/golden/` at `1e-6`; property tests for round-trips | all OSes |
| `render` | offscreen `wgpu` (lavapipe on Linux CI, native elsewhere) → texture readback → pixel assertions | all OSes |
| Performance | `criterion` benches: `GpuTake::set_frame` (< 0.1 ms CPU), `write_range` 1 000 frames (< 1 ms), full-take Butterworth (< 200 ms); CI fails on regression > 20 % | all OSes |
| `app` | `egui_kittest` snapshot tests for panels and timeline | all OSes |
| `py` | pytest: `import mstudio`, load, filter, compare to golden; numpy zero-copy checks | all OSes |
| Manual | `docs/QA_CHECKLIST.md` on real Win / Mac / Linux before each release | release gate |

The golden files are the contract: **Phase 0a runs before any Rust is written**, so parity is provable and Pose2Sim numerical compatibility is preserved.

---

## 10. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Filter parity drift (LOESS, Kalman defaults differ subtly from statsmodels/filterpy) | Medium | Golden tests at `1e-6`; document any intentional deviation; keep Python oracle in-tree until v1.0 |
| Pose2Sim updates its filters upstream | Certain, slow | Golden regeneration script `scripts/regen_golden.py`; a filter change is a tracked task, not a surprise |
| egui look judged "too developer-tool" | Medium | Custom theme + `re_ui`-style polish pass in Phase 4; Blender-style visual language |
| `re_renderer` unsuitable as a standalone crate | Medium | Spike decides; hand-written pipelines are ~1 500 lines and fully ours |
| wgpu on old GPUs / VMs | Low | wgpu GL backend fallback; documented minimum |
| Python users lose `pip install` during the port | — | Not possible: `MStudio/` ships unchanged until Phase 6 |
| Signing/notarization friction on macOS/Windows | Medium | Unsigned builds documented for Phase 6; signing tracked as a release task |

---

## 11. Distribution

| Channel | Artifact | Tool |
|---|---|---|
| PyPI (`pip install mstudio`) | wheel containing the PyO3 module + app; `mstudio` console script calls `mstudio.run()` | `maturin` on CI for cp310–cp313 × 3 OSes |
| GitHub Releases | standalone binaries: `.dmg` (universal or arm64+x86), `.msi`/`.zip`, `.tar.gz`/AppImage | `cargo-dist` |
| conda-forge (later) | recipe wrapping the wheel | — |

---

## 12. Immediate next steps

1. **Phase 0a** — write `scripts/gen_golden.py`, generate and commit `tests/golden/` from the current Python code.
2. **Phase 0b** — Rust spike in `crates/spike/`; fps and CPU-µs numbers from this Mac first.
3. Go/no-go on `re_renderer` vs. own pipelines; then Phase 1.
