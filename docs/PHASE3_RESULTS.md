# Phase 3 results — renderer (`mstudio-render`)

Status: **done**. A wgpu-only viewport crate (no window, no egui types) that keeps
the take resident on the GPU and draws one instanced call per layer.

## What exists

```
crates/mstudio-render/src
├── gpu_take.rs           GpuTake: whole take in one storage buffer; write_range(DirtyRange) patches frames (R1/R2);
│                         pack_outliers: [marker, frame] map → bit-packed SSBO read per frame by the shader
├── shader.wgsl           vs_marker/fs_marker (instanced shaded discs, state colours, selected ×1.5, outlier colour),
│                         vs_skeleton (pairs SSBO → thick quads, outlier segments red & ×1.5 width),
│                         vs_segments (segment lists: grid, trajectories, analysis), thick_segment() screen-space expansion
├── segments.rs           Segment {a, b, width px, rgba}; SegmentBuffer grows on demand, own bind group
├── grid.rs               ground grid (every 5th line brighter) + XYZ axes, view-up space
├── trajectories.rs       ± window of the selected marker, past bright / future dimmed, gaps break the line
├── analysis_overlay.rs   2 markers: segment + reference-axis line + "d m" / "θ° vs X" labels;
│                         3 markers: arms + arc (processing::analysis::arc_points) + angle label
├── camera.rs             orbit / pan / zoom / fit_bounds; coordinate_model(Y-up | Z-up); project() for labels & picking
├── picking.rs            pick_marker (CPU projection, nearest depth then pointer), distance_to_segment_px (axis-cycle click)
└── lib.rs                Renderer { new(config), load_take, update_frames, set_skeleton_pairs, set_outliers,
                                     set_marker_states, set_grid, set_overlay, prepare(FrameParams), draw(&mut RenderPass) }
```

`RenderConfig { color_format, depth_format, msaa_samples }` lets the same renderer draw
into egui's swap chain (Phase 4) and into an offscreen texture (tests).

## Tests (all green on Apple M5 Max / Metal)

| Test | Checks |
|---|---|
| `tests/offscreen.rs` | headless device → 256² texture → readback: every marker lights up at `project()` of its position; the skeleton midpoint is lit; background dark; frame 9 differs from frame 0 (uniform-only advance); `clear_range` + `update_frames(DirtyRange::single(9))` removes exactly that marker in that frame and leaves frame 0 byte-identical; Z-up view renders |
| 10 unit tests | GPU byte layout & partial-range offsets, outlier bit packing, camera fit/projection/zoom, Z-up model, picking (nearest depth, radius, off-screen), segment builders, analysis overlay geometry and labels |

The offscreen test skips with a message when no adapter exists; CI installs
lavapipe on Linux, DX12 WARP serves Windows, Metal serves macOS runners.

## Per-frame cost

`prepare()` writes one 208-byte uniform block and (only when a buffer was
replaced) rebuilds one bind group. `draw()` is at most four draw calls:
grid segments, skeleton, overlay segments, markers. No CPU work scales with
the number of frames; per-marker CPU work exists only in the host's label
projection and picking (Phase 0: < 20 µs for 1 000 markers).

## Deviations from the plan

| Plan | Built | Why |
|---|---|---|
| `picking.rs`: ID render target + async readback | CPU projection | Phase 0 measured it exact and cheaper, with no readback latency |
| `markers.rs`, `skeleton.rs`, … as separate pipelines files | one WGSL module, three pipelines in `lib.rs` | the pipelines share bind groups and the thick-segment helper |
| torso pairs thicker | not implemented | no oracle for which pairs are "torso"; outlier styling and width/opacity settings are in |
| text labels | host draws them (`AnalysisOverlay::labels`, `project`) | egui's text pipeline batches better than a custom SDF atlas would at ≤ 300 labels |

## Next: Phase 4 — app shell (`mstudio-app`)

eframe + egui_dock: viewport (this renderer inside an egui paint callback),
timeline, marker plot, panels, shortcuts, file dialogs.
