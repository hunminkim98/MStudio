# MStudio v2 — Cross-platform, Blender-level viewport

**Branch:** `feature/cross-platform-renderer`
**Status:** planning (2026-09-14)
**Decision:** keep Python; replace Tk + legacy OpenGL with **PySide6 (Qt 6) + pygfx (wgpu → Metal / Vulkan / DX12)**.

---

## 1. Goals

| # | Goal | Measurable exit criterion |
|---|---|---|
| G1 | Runs natively on Windows, macOS (Apple Silicon + Intel), Linux | CI matrix green on all four runners incl. an actual offscreen render test |
| G2 | Blender-level viewport smoothness | 60 fps sustained during playback with ≥ 150 markers × ≥ 20 000 frames; **< 1 ms Python time per frame** in the draw loop; camera drag never drops below display refresh |
| G3 | Feature parity with v0.1.5 | Every row in §6 checked off |
| G4 | "High-end" feel | HiDPI-correct on all OSes, dark/light native theme, dockable panels, native file dialogs, < 2 s cold start |

**Non-goals for this milestone:** multi-person, gait mode, new analysis features. Parity first.

---

## 2. Why this stack (summary of research)

- **pyopengltk has no macOS backend** (`darwin.py` is a stub) → the current app cannot run on Mac at all.
- macOS OpenGL is deprecated (2018) with no kill date; every OpenGL-based option (moderngl, VisPy, pyqtgraph.opengl) inherits that risk.
- **wgpu** targets Metal / Vulkan / DX12 — the same API class Blender 4.x uses for its viewport. **pygfx** is the scene graph on top: `Points`, `Line`, SDF `Text`, built-in picking, orbit controllers, screenshot-to-numpy.
- **PySide6** (LGPL, pip wheels for all 3 OSes) supplies docking, HiDPI, native dialogs, theming.
- The scientific half (scipy, statsmodels, filterpy, pandas, c3d, anytree, matplotlib reports, vendored Pose2Sim filters) is untouched. Nothing in `MStudio/core/` or `MStudio/utils/{filtering,analysisMode,skeletons,skeleton_config,dataLoader,dataSaver,reportGenerator}.py` needs a rewrite.

Fallback if the spike fails: **PySide6 + moderngl in `QOpenGLWidget`** (same shell, same GPU-resident architecture, hand-written shaders, macOS on OpenGL 4.1).

Pinned targets at time of writing: `PySide6==6.11.x`, `pygfx==0.17.x`, `wgpu==0.32.x`, `rendercanvas==2.7.x`, Python ≥ 3.10.

---

## 3. Performance architecture — the five rules

These are **hard constraints**, enforced by tests in §7, not guidelines.

| Rule | What it means in code |
|---|---|
| **R1 — Data lives on the GPU** | On load, build one `float32[N_frames, N_markers, 3]` array and upload it once. Frame advance = change a per-frame **offset/uniform**, never re-upload. |
| **R2 — No pandas in the frame loop** | `DataManager.data` (DataFrame) is the *editing* representation. A parallel `DataManager.array` (`np.ndarray`, C-contiguous) is the *render* representation. Edits go DataFrame → numpy slice → **partial buffer upload** of the touched frame range only. |
| **R3 — Draw-loop-driven playback** | `AnimationController` no longer owns a Tk `after()` timer. The canvas's `request_draw` callback asks the controller "what frame is it now?" using a monotonic clock; the controller returns the frame; the renderer sets the offset. vsync paces everything. |
| **R4 — UI thread never blocks** | Filtering, interpolation, outlier detection, report generation run in `QThreadPool` workers (numpy/scipy release the GIL). Progress via signals. The 3D view keeps rendering while a filter runs. |
| **R5 — No per-glyph text, no per-marker draw calls** | Marker names use pygfx SDF `Text` (one object per label, batched by pygfx). Markers are a single `Points` object with per-vertex color/size; skeleton is a single `Line` with segment breaks; trajectories are one `Line` with NaN breaks. |

**Frame budget at 60 fps = 16.7 ms.** Target split: Python ≤ 1 ms, GPU ≤ 4 ms, rest idle.

---

## 4. Target package layout

```
MStudio/
├── core/                    # UNCHANGED — DataManager, StateManager, AnimationController,
│   │                        #   OutlierDetector, MarkerVisualSettings
│   └── data_manager.py      #   + .array (np.ndarray view for the GPU), + .mark_dirty(frame_range)
├── io/                      # moved from utils/: dataLoader.py, dataSaver.py (no tkinter imports)
├── processing/              # moved from utils/: filtering.py (Pose2Sim, keep attribution),
│                            #   dataProcessor.py (pure functions, no `self`), analysisMode.py,
│                            #   skeletons.py, skeleton_config.py, reportGenerator.py (no tkinter)
├── render/                  # NEW — pygfx scene, no Qt imports
│   ├── scene.py             #   MarkerScene: builds/owns all WorldObjects
│   ├── buffers.py           #   GPU-resident frame buffer, partial upload
│   ├── markers.py           #   Points layer + color/size state machine
│   ├── skeleton.py          #   Line layer, pairs → index buffer, outlier highlight
│   ├── trajectories.py      #   Line layer, ring window around current frame
│   ├── labels.py            #   SDF text, follows marker positions
│   ├── analysis.py          #   distance/angle overlays, arc, reference line
│   ├── grid.py              #   grid + axes, Y-up / Z-up rotation on the scene root
│   ├── picking.py           #   pygfx pick events → marker name
│   └── camera.py            #   OrbitController wrapper, reset, fit-to-data
├── ui/                      # NEW — PySide6 widgets, no rendering code
│   ├── main_window.py       #   QMainWindow + dock layout + menus + shortcuts
│   ├── viewport.py          #   QRenderWidget host, wires input → render/
│   ├── timeline.py          #   custom QWidget: frames/time ticks, scrub, selection range
│   ├── marker_plot.py       #   matplotlib FigureCanvasQTAgg for X/Y/Z curves + range select
│   ├── panels/              #   filter, interpolation, skeleton model, visual settings, analysis
│   ├── workers.py           #   QRunnable wrappers for processing/
│   └── theme.py             #   Fusion + dark/light palette, fonts per OS
├── app.py                   # thin: QApplication, MainWindow, exec
└── main.py                  # entry point (unchanged signature)
```

The `self`-passing free-function pattern (`open_file(self)`, `filter_selected_data(self)`) is retired. Each becomes a pure function taking explicit arguments and returning results; the UI layer owns the glue.

---

## 5. Phases

### Phase 0 — Spike / go-no-go  (2–3 days)

Purpose: prove G2 on real hardware before committing.

- [ ] `PySide6` window with `QRenderWidget`, pygfx `WgpuRenderer`
- [ ] Load `tests/test.trc`; upload all frames as one buffer (R1)
- [ ] `Points` + `Line` skeleton (HALPE_26), `Text` labels, orbit camera
- [ ] Playback via `request_draw` (R3); on-screen fps + per-frame Python µs counter
- [ ] Click-to-select a marker via pygfx picking
- [ ] Run on: this Mac (Apple Silicon), one Windows box, one Linux box (or CI offscreen)
- [ ] Synthetic stress: 300 markers × 50 000 frames

**Go if:** 60 fps on all three with Python < 1 ms/frame and picking works. **No-go →** switch `render/` to moderngl + `QOpenGLWidget`, same interfaces.

Lives in `spike/` at repo root; deleted after Phase 2.

### Phase 1 — Foundation  (1 week)

- [ ] `pyproject.toml`: add `PySide6`, `pygfx`, `wgpu`, `rendercanvas`; drop `customtkinter`, `pyopengl*`, `pyopengltk`, `opencv-python`; `pyopengl-accelerate` gone
- [ ] `io/`, `processing/` moves; strip all `tkinter` imports from them (dialogs move to `ui/`)
- [ ] `ui/main_window.py` shell: central viewport, right dock (panels), bottom dock (timeline + marker plot), menus, shortcuts (Space/Enter/Esc/←/→)
- [ ] `ui/theme.py`: Fusion style, dark + light palettes, per-OS font stack
- [ ] HiDPI: `Qt.AA_EnableHighDpiScaling` defaults + `devicePixelRatio` passed to renderer
- [ ] CI: matrix stays; add `xvfb-run` on Linux; add tests: import every module, construct `MainWindow` offscreen (`QT_QPA_PLATFORM=offscreen`), render one frame with `rendercanvas` offscreen backend and assert non-black pixels
- [ ] `pytest` config: `python_files = "test_*.py"` (current `"test.py"` silently skips new tests)

### Phase 2 — Renderer  (2 weeks)

- [ ] `render/buffers.py`: `FrameBuffer(array)` with `set_frame(i)` (offset only) and `update_range(f0, f1)` (partial upload) — R1, R2
- [ ] `render/markers.py`: one `Points`; per-vertex color from state (normal / selected / pattern / analysis / outlier-frame); size & opacity from `MarkerVisualSettings`
- [ ] `render/skeleton.py`: pairs → index buffer; outlier segments recolored, torso pairs thicker; width/opacity/color from settings
- [ ] `render/trajectories.py`: ± `trajectory_length` window, one `Line` with NaN breaks
- [ ] `render/labels.py`: SDF `Text` per marker, toggle, follows positions each frame
- [ ] `render/grid.py`: grid + axes; Y-up / Z-up as a rotation on the scene root (data untouched)
- [ ] `render/camera.py`: orbit (LMB), pan (RMB **and** MMB), zoom (wheel, normalized across OSes — pygfx handles this), reset, fit-to-data
- [ ] `render/picking.py`: `pointer_down` → `pick_info` → marker name → `StateManager`
- [ ] `render/analysis.py`: 2-marker distance + segment angle vs axis (click axis cycle), 3-marker joint angle + arc, live text readout
- [ ] Bench test: assert `< 1 ms` Python per `set_frame` on the stress dataset (R1/R2 enforcement)

### Phase 3 — Playback & timeline  (1 week)

- [ ] `AnimationController`: remove Tk timer; add `frame_at(t_monotonic)`; loop; fps from file or user override — R3
- [ ] `ui/timeline.py`: painted `QWidget`; frame/time tick modes; scrub; selection range drag; current-frame cursor updated without full repaint
- [ ] Play/pause/stop, prev/next, loop checkbox, fps entry
- [ ] Keyboard shortcuts identical to v0.1.5 table in README

### Phase 4 — Editing & processing  (2 weeks)

- [ ] `ui/marker_plot.py`: matplotlib `FigureCanvasQTAgg` X/Y/Z panels, vertical current-frame line, range selection, pan/zoom; `draw_idle` only
- [ ] Edit mode toggle; delete selected range; restore original (`DataManager.original_data`)
- [ ] Filter panel: butterworth, butterworth_on_speed, kalman, gaussian, LOESS, median — same `config_dict` shape into `processing/filtering.py`
- [ ] Interpolation panel: linear, polynomial, spline, nearest, zero, slinear, quadratic, cubic, pattern-based (pattern marker selection in viewport)
- [ ] All of the above run in `ui/workers.py` `QRunnable`s — R4; on completion: DataFrame ← result, `DataManager.array` slice updated, `FrameBuffer.update_range()`, outlier re-detect (also worker)
- [ ] Skeleton model combo (12 models) → `update_keypoint_names` → pairs → renderer

### Phase 5 — Visual settings, analysis panel, report  (1 week)

- [ ] Customization panel: marker size/opacity, color presets, skeleton width/opacity/colors, reset — pushes to `render/` via existing `MarkerVisualSettings` callbacks
- [ ] Analysis mode toggle + selected-markers list panel
- [ ] Report export: `processing/reportGenerator.py` stripped of tkinter; file dialog in UI; runs in worker; matplotlib `Agg` explicitly (no `$DISPLAY` sniffing)
- [ ] Window icon via `QIcon` (PNG/ICNS/ICO set), not `iconbitmap`

### Phase 6 — Parity, removal, packaging  (1 week)

- [ ] §6 checklist fully green on all three OSes (manual QA script in `docs/QA_CHECKLIST.md`)
- [ ] Delete `MStudio/gui/`, `MStudio/utils/{viewToggles,viewReset,mouseHandler,performance_utils}.py`, `spike/`
- [ ] `mstudio` entry point → Qt app; README screenshots/controls table updated (RMB **or** MMB pan)
- [ ] Packaging: `pyside6-deploy` (Nuitka) or PyInstaller spec per OS; smoke-launch each artifact in CI
- [ ] Version bump to `0.2.0`; CHANGELOG

**Total ≈ 8–9 weeks** single developer, sequential. Phases 2 and 3 can overlap once Phase 1 lands.

---

## 6. Feature parity checklist (v0.1.5 → v2)

| Area | v0.1.5 (Tk/GL) | v2 location | Done |
|---|---|---|---|
| Open TRC / C3D / JSON folder (multi-select) | `utils/dataLoader.open_file` | `io/` + `ui/main_window` | ☐ |
| Save As TRC / C3D | `utils/dataSaver.save_as` | `io/` + `ui/main_window` | ☐ |
| 3D markers with size / opacity / color states | `GLMarkerRenderer._render_markers_immediate` | `render/markers.py` | ☐ |
| Skeleton lines, outlier highlight, torso width | `_render_skeleton_immediate`, `_cache_skeleton_geometry` | `render/skeleton.py` | ☐ |
| Trajectories (± length window) | `_render_trajectories_immediate` | `render/trajectories.py` | ☐ |
| Marker name labels (toggle) | `_render_marker_names_immediate` (GLUT) | `render/labels.py` (SDF) | ☐ |
| Grid + axes, Y-up / Z-up toggle | `GridUtils`, `set_coordinate_system` | `render/grid.py` | ☐ |
| Orbit / pan / zoom / reset view | `on_mouse_*`, `reset_view` | `render/camera.py` | ☐ |
| Click-to-select marker (picking) | `PickingTexture`, `pick_marker` | `render/picking.py` | ☐ |
| Analysis mode: distance, segment angle (axis cycle), joint angle + arc | `_render_analysis_immediate`, `analysisMode.py` | `render/analysis.py` | ☐ |
| Play / pause / stop / loop / fps / prev / next | `AnimationController` + Tk `after` | `AnimationController` + draw loop | ☐ |
| Timeline (frame & time modes, scrub, range select) | `update_timeline`, `_draw_*_ticks` | `ui/timeline.py` | ☐ |
| Marker X/Y/Z plot with range selection | `gui/markerPlot.py` (mpl in Tk) | `ui/marker_plot.py` (mpl in Qt) | ☐ |
| Edit mode: delete range, restore original | `toggle_edit_mode`, `delete_selected_data` | `ui/panels/edit.py` | ☐ |
| Filters ×6 with params | `dataProcessor.filter_selected_data` | `ui/panels/filter.py` + worker | ☐ |
| Interpolation ×9 incl. pattern-based | `interpolate_*`, `on_pattern_selection_confirm` | `ui/panels/interp.py` + worker | ☐ |
| Outlier detection (threshold, parallel) | `OutlierDetector` | unchanged + worker | ☐ |
| 12 skeleton models + keypoint rename | `on_model_change`, `update_keypoint_names` | `ui/panels/skeleton.py` | ☐ |
| Visual customization panel + presets | `TRCviewerWidgets._create_customization_*` | `ui/panels/visual.py` | ☐ |
| PDF analysis report | `reportGenerator.py` | `processing/` + worker | ☐ |
| Keyboard shortcuts (Space/Enter/Esc/←/→) | `app.py` binds | `ui/main_window.py` | ☐ |
| Resizable right panel | `start_resize` sizer | `QDockWidget` | ☐ |
| Window icon | `iconbitmap(.ico)` | `QIcon` multi-format | ☐ |

---

## 7. Testing strategy

| Layer | How | Where it runs |
|---|---|---|
| `core/`, `processing/`, `io/` | plain pytest, numeric golden files (filter outputs must match v0.1.5 bit-for-bit on `tests/test.trc`) | all OSes |
| `render/` | `rendercanvas` **offscreen** backend → numpy frame; assert pixel counts (markers drawn, skeleton drawn, label region non-empty) | all OSes, no display needed |
| Performance | `pytest-benchmark`: `FrameBuffer.set_frame` < 1 ms; `update_range` on 1 000 frames < 5 ms | all OSes; fails CI if regressed |
| `ui/` | `pytest-qt` with `QT_QPA_PLATFORM=offscreen`; construct `MainWindow`, load file, toggle every panel | all OSes; Linux under `xvfb-run` for the on-screen variant |
| Manual | `docs/QA_CHECKLIST.md` walked on real Win / Mac / Linux before each release | release gate |

Golden-file tests for filtering are added **before** Phase 1's file moves so the moves are provably behavior-preserving.

---

## 8. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| pygfx (beta, pre-1.0) API change | Medium | Pin exact version; `render/` is the only package importing pygfx; upgrade in a dedicated PR |
| pygfx sync picking too slow / imprecise at marker density | Low | Spike measures it; fallback = own ID-color pass (we already have that design) |
| wgpu unavailable on old GPUs / VMs (no Vulkan on Linux CI) | Medium | wgpu falls back to software (lavapipe) — install `mesa-vulkan-drivers` on CI; document minimum GPU |
| Qt bitmap-present path caps fps on some Linux compositors | Low | `present_method="screen"` toggle in settings; default "bitmap" |
| Cold-start time > 2 s | Medium | Lazy-import matplotlib/seaborn/statsmodels (report only), `PySide6-Essentials` instead of full `PySide6` |
| Two UIs coexisting during the port confuses users | Low | Tk app stays the shipped `mstudio` until Phase 6; v2 launched via `mstudio --v2` until then |

---

## 9. Dependency changes

```toml
# remove
customtkinter, pyopengl, pyopengl-accelerate, pyopengltk, opencv-python

# add
PySide6-Essentials>=6.11
pygfx>=0.17,<0.18
wgpu>=0.32,<0.33
rendercanvas>=2.7

# keep
numpy, pandas, scipy, statsmodels, filterpy, c3d, anytree, matplotlib, seaborn

# dev
pytest, pytest-qt, pytest-benchmark, flake8
```

---

## 10. Immediate next steps

1. Phase 0 spike in `spike/` — target: fps numbers from this Mac by end of week.
2. Golden-file tests for `filtering.py` / `dataProcessor.py` (pre-Phase 1 safety net).
3. Phase 1 `pyproject` + package moves.
