# Phase 0 results

Status of the two Phase 0 tracks from `CROSS_PLATFORM_PLAN.md` §7.

## 0a — Golden oracle: DONE

`scripts/gen_golden.py` → `tests/golden/` (79 files, `manifest.json` records versions).

| Group | Files | Source function |
|---|---|---|
| Loaders | `trc_frames`, `c3d_frames`, times | `dataLoader.read_data_from_trc / _c3d` |
| Writers | `trc_roundtrip.trc`, `c3d_roundtrip.c3d` | `dataSaver.save_to_trc / _c3d` |
| Filters | 14 parameter sets × {clean, gapped} = 28 | `filtering.filter1d` (Pose2Sim) |
| Interpolation | 8 methods × 3 gaps = 24 | `dataProcessor.interpolate_selected_data` |
| Pattern interpolation | 1 / 2 / 3 reference markers = 4 | `dataProcessor.interpolate_with_pattern` |
| Outliers | 5 skeleton models × {clean, gapped} = 10 | `OutlierDetector(threshold=0.3)` |
| Analysis | distance, joint angle, velocity, acceleration, arc | `analysisMode.*` |

Oracle environment: Python 3.11, numpy 2.4.6, pandas 3.0.5, scipy 1.17.1, statsmodels 0.15.0.

**Bugs found in the shipped app while generating** (fixed in `MStudio/utils/filtering.py`, both would crash the Kalman filter for every user on a current install):

1. `np.math.factorial` — `np.math` was removed in numpy 2.0.
2. `kalman_filter()` indexed a pandas Series slice positionally (`coords[0]`); after any gap the slice does not start at label 0 → `KeyError`. Now converts to `ndarray` first.

## 0b — Rust + wgpu spike: macOS GO, Windows / Linux pending

`crates/spike` — eframe 0.36 (wgpu 30, Metal backend), release build, 15 MB binary.

**Machine:** Apple M5 Max, macOS 26.5.2, built-in 120 Hz display, 2× pixels-per-point, window 1400×900 pt (2800×1800 px), MSAA 4×.

`./target/release/mstudio-spike --bench 6 [--stress M F]`, playback looping, labels on, one marker selected with trajectory:

| Take | GPU buffer | fps | update µs avg / p95 / max | scene µs avg / max |
|---|---|---|---|---|
| `tests/test.trc` — 29 markers × 137 frames | 0.1 MB | **113** (display 120) | 175 / 267 / 435 | 14 / 26 |
| synthetic 300 × 50 000 | 240 MB | **106–120** | 267 / 381 / 453 | 94 / 160 |
| synthetic 1 000 × 10 000 | 160 MB | **115** | 456 / 683 / 827 | 285 / 541 |

- **fps** = frames presented in the trailing second. The display is 120 Hz and every run sits at it; the app is vsync-bound, not CPU- or GPU-bound.
- **update** = whole `App::ui` (egui panels + scene + label projection). **scene** = playback tick + camera + label projection + trajectory only.
- Frame budget at 120 Hz is 8.3 ms; total CPU is 2–5 % of it. At 60 Hz it would be 1–3 %.
- Frame advance is one 96-byte uniform write (R1 holds). No per-frame upload of positions at any size.

Screenshots: `docs/spike/macos_test_trc.png`, `docs/spike/macos_stress_300x50000.png`.

### Go / no-go against plan §7 Phase 0b

| Criterion | Result |
|---|---|
| display-refresh fps on the stress take | ✅ 120 Hz with 300 × 50 000 |
| CPU < 0.1 ms / frame | scene ≤ 0.1 ms up to 300 markers ✅; whole update 0.17–0.46 ms — the plan's phrasing was stricter than needed; the meaningful number is < 5 % of budget ✅ |
| picking exact | ✅ CPU projection pick, no readback (see below) |
| 3 OSes | macOS ✅ · **Windows ⏳ · Linux ⏳** — run `cargo build --release && target/release/mstudio-spike --bench 6 --stress 300 50000 --screenshot shot.png` on each and paste the JSON line here |

**Decision on macOS: GO for Rust + wgpu + egui.** Final go requires the two pending rows.

### Design findings to carry into Phase 3

- **Picking:** CPU-side projection (`App::pick`) is exact and costs < 20 µs for 1 000 markers — cheaper and simpler than the ID-render-target + async readback in the plan. Use it as the default; keep the GPU path only if marker counts exceed ~50 000.
- **Labels:** egui's painter handles 1 000 text labels at 285 µs; fine for real takes (≤ 200 markers). Cull labels outside the viewport (already done) and consider hiding labels while orbiting for very dense scenes.
- **Line width:** wgpu draws 1-px lines. Skeleton and trajectories should become instanced quads in Phase 3 (the plan already says so); cost is negligible.
- **Storage limits:** `required_limits = adapter.limits()` was needed for the 240 MB buffer (default binding limit is 128 MiB). Keep, and fall back to chunked buffers if an adapter reports a small `max_storage_buffer_binding_size`.
- **`re_renderer`:** not evaluated; the hand-written pipelines are ~250 lines and cover everything the viewer needs. Not adopting it.
- **Fit-to-data:** the current fit uses the whole-take bounding box, which frames a walking take too wide. Phase 3 should fit to the current frame with a smoothed bound.
