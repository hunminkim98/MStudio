# Parity QA checklist (v0.1.5 → v2)

Walk this on each OS with the release binary (`cargo build --release -p
mstudio-app`, or `python -m mstudio` from the wheel). The *automated* column
names the check that already covers the row without a person; the OS columns
record the last manual walk (date · commit · who).

Automated shorthand: **golden** = `cargo test --workspace` against
`tests/golden/`; **parity** = `scripts/check_parity.py`; **py** =
`pytest crates/mstudio-py/tests`; **offscreen** = `mstudio-render/tests/offscreen.rs`
(headless GPU pixel checks); **selftest** = `mstudio <file> --selftest`;
**shot** = `mstudio <file> [--demo|--play] --screenshot out.png --exit-after 4`.

| # | Area | How to check by hand | Automated | macOS arm64 | Windows | Linux |
|---|---|---|---|---|---|---|
| 1 | Open TRC / C3D | File ▸ Open…, pick `tests/test.trc`, then `tests/test.c3d`; status bar shows markers/frames/Hz | golden, parity, py, shot (both files) | ✅ 2026-09-14 via `shot` | ✅ 2026-09-14 CI | ✅ 2026-09-14 CI |
| 2 | Open JSON folder (Pose2Sim / Sports2D) | File ▸ Open JSON folder…; markers named `Keypoint_<i>` until a model is chosen | golden (`mstudio-io` json tests) | ☐ manual (no sample folder in repo) | ☐ | ☐ |
| 3 | Save As TRC / C3D | Save, reopen in MStudio and in the Python app | parity (byte-exact TRC, cross-read C3D), py | ✅ automated | ✅ 2026-09-14 CI | ✅ 2026-09-14 CI |
| 4 | Markers: size / opacity / colour states | Appearance sliders; selected = accent, pattern refs, analysis colours | offscreen, shot | ✅ shot (defaults) · ☐ sliders by hand | ☐ | ☐ |
| 5 | Skeleton lines, outlier highlight, torso width | Skeleton on; pick COCO_133 on `test.trc` → 2 outlier flags shown red | offscreen, shot (`2 outlier flags`) | ✅ shot | ☐ | ☐ |
| 6 | Trajectories (± window) | View ▸ Trajectory with a marker selected | offscreen, shot (`--demo`) | ✅ shot | ☐ | ☐ |
| 7 | Marker name labels | View ▸ Names | shot (`--demo`) | ✅ shot | ☐ | ☐ |
| 8 | Grid + axes, Y-up / Z-up | Toggle up axis; grid stays under the feet, triad follows | offscreen (both systems), shot | ✅ shot (Y-up) · ☐ Z-up by hand | ☐ | ☐ |
| 9 | Orbit / pan / zoom / reset | LMB drag, RMB or MMB drag, wheel, F | camera unit tests | ☐ | ☐ | ☐ |
| 10 | Click-to-select (picking) | Click a marker; Selection panel + marker plot follow; click empty space clears | picking unit tests | ☐ | ☐ | ☐ |
| 11 | Analysis: distance, segment angle (axis cycle), joint angle + arc | Analysis mode, click 2 then 3 markers, cycle the reference axis | golden (`analysis_*`) | ☐ | ☐ | ☐ |
| 12 | Play / pause / stop / loop / fps / prev / next | Space, Esc, ←/→, speed and fps boxes | `Playback` unit tests, shot (`--play` advances frames) | ✅ shot · ☐ controls by hand | ☐ | ☐ |
| 13 | Timeline (frames / time modes, scrub, range) | Drag the cursor; switch mode; drag a range in edit mode | — | ☐ | ☐ | ☐ |
| 14 | Marker X/Y/Z plot with range selection | Select a marker; drag a range; double-click clears; ctrl-scroll zooms | shot (`--demo` shows range 30–60) | ✅ shot · ☐ interaction | ☐ | ☐ |
| 15 | Edit mode: delete range, restore original, undo | ⌫ / Restore original / ⌘Z | selftest (delete + undo bit-exact), py | ✅ selftest | ✅ CI: data ops via bindings · ☐ undo in the app | ✅ CI: data ops via bindings · ☐ undo in the app |
| 16 | Filters ×6 with params | Edit tab ▸ Filter ▸ Apply on a range; compare with the Python app | golden, parity (14 cases), selftest (Butterworth via worker) | ✅ automated | ✅ 2026-09-14 CI | ✅ 2026-09-14 CI |
| 17 | Interpolation ×9 incl. pattern-based | Delete a range, interpolate; pattern-based needs reference markers clicked in the 3D view | golden, parity (24 + 4 cases) | ✅ automated · ☐ UI flow by hand | ✅ 2026-09-14 CI | ✅ 2026-09-14 CI |
| 18 | Outlier detection (threshold, parallel) | Skeleton panel shows the flag count; flagged markers red | golden, parity (5 models, sequential == parallel oracle) | ✅ automated | ✅ 2026-09-14 CI | ✅ 2026-09-14 CI |
| 19 | 12 skeleton models + keypoint rename | Pick each model in the combo; JSON data gets renamed | golden (`skeleton_tables`), py (`rename_markers`) | ✅ automated · ☐ combo by hand | ✅ 2026-09-14 CI | ✅ 2026-09-14 CI |
| 20 | Visual customization + presets | Appearance ▸ scheme buttons, reset | `VisualSettings` unit tests | ☐ | ☐ | ☐ |
| 21 | Analysis report | Edit tab ▸ Generate report…; browser opens; marker/time/axis controls re-filter charts; print to PDF | selftest (report written), report crate tests, py | ✅ selftest + browser check in Phase 5 | ✅ CI: report generated · ☐ opened in a browser | ✅ CI: report generated · ☐ opened in a browser |
| 22 | Keyboard shortcuts | Space, Enter, Esc, ←/→, F, ⌘O, ⇧⌘S, ⌘Z, ⌫ (Ctrl on Windows/Linux) | — | ☐ | ☐ | ☐ |
| 23 | Resizable / dockable panels | Drag splitters; drag a tab out and back | — | ☐ | ☐ | ☐ |
| 24 | Window icon | Dock / taskbar shows `Content/icon.ico` | — | ☐ | ☐ | ☐ |
| 25 | Native file dialogs (rfd) | Open…, Save As…, Generate report… dialogs appear and cancel cleanly | — | ☐ | ☐ | ☐ |
| 26 | `import mstudio` / `mstudio.run()` | `python -c "import mstudio; mstudio.run('tests/test.trc')"` | py (75 tests), parity, `run(screenshot=…, exit_after=…)` | ✅ 2026-09-14 | ✅ CI: import + 75 tests · ☐ `run()` window | ✅ CI: import + 75 tests · ☐ `run()` window |
| 27 | Performance: 300 markers × 50 000 frames plays at display rate | `mstudio-spike --stress 300 50000 --bench 6` and the app with a large TRC | spike JSON stats (Phase 0/4) | ✅ Phase 0/4 | ☐ | ☐ |

"✅ 2026-09-14 CI" = run #34830599666 (commit 47229e2): `cargo test
--workspace`, the 75 bindings tests and the 88-case parity check all green on
`ubuntu-latest` and `windows-latest`. Rows whose evidence is a screenshot or a
`--selftest` run were only exercised on macOS, because the CI matrix does not
launch a window.

Rows 9–11, 13, 20, 22–25 have no automated coverage and are the ones a person
must walk on every OS.

Intel macOS is not a target: `macos-13` never got a runner in three CI runs and
was removed from the matrix, so macOS here means Apple Silicon
(`docs/PHASE6_RESULTS.md`).
