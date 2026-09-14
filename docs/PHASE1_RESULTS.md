# Phase 1 results — workspace, core, I/O

Status: **done locally**, CI workflow added but not yet observed green on GitHub (first push of this branch will run it).

## What exists now

```
Cargo.toml                    workspace: mstudio-core, mstudio-io, spike; rustfmt.toml, clippy.toml
crates/mstudio-core           Take (f64 [frame, marker, xyz]), DirtyRange, Limits
                              skeleton: 15 models generated from skeletons.py (scripts/gen_skeletons.py), resolve_pairs, id_to_name
                              state: ViewState / SelectionState / EditingState / StateManager (marker indices, no callbacks)
                              playback: wall-clock frame clock (rule R3), loop, speed, seek
                              visual: marker/skeleton configs, clamps, 4 color schemes
                              outliers: bone-length detector, rayon over pairs, deterministic merge
crates/mstudio-io             trc (reader + byte-exact pandas-compatible writer, py_repr), c3d (via c3dio), json folders
                              examples/convert.rs: load/save CLI
.github/workflows/rust.yml    ubuntu / windows / macos-latest (arm64) / macos-13 (x86_64): fmt, clippy -D warnings, test, release build; lavapipe on Linux
```

## Parity with the Python oracle (all green)

| Test | Checks |
|---|---|
| `trc_reader_matches_oracle` | frames bit-exact, time column, marker list, fps for `tests/test.trc` |
| `trc_writer_is_byte_exact_with_pandas` | `write_trc` output == `save_to_trc` output byte for byte |
| `c3d_reader_matches_oracle` | frames bit-exact (f32 mm ÷ 1000 semantics), time, labels |
| `c3d_written_by_python_reads_back_the_trc_data` | Rust reads the Python-written C3D within 1e-6 |
| `c3d_round_trip_preserves_values_and_missing_samples` | Rust write → Rust read, NaN preserved via residual −1 |
| `skeleton_pairs_match_python_for_every_model_in_manifest` | HALPE_26, COCO_17, COCO_133, BODY_25B, BODY_25 pair lists identical |
| `outliers_match_python_for_every_model_clean_and_gapped` | 10 outlier maps identical |
| 29 unit tests | py_repr vs Python `repr`, JSON folder centring, playback clock, state rules, clamps |

Cross-check with the Python ecosystem: a C3D written by Rust is read by the Python `c3d` package and matches the TRC source (see the session log for the command; to repeat: `cargo run -p mstudio-io --example convert -- tests/test.trc /tmp/x.c3d` then load it with `MStudio.utils.dataLoader.read_data_from_c3d`).

## Deviations from the Python code (intentional)

| Where | Python v0.1.5 | Rust | Why |
|---|---|---|---|
| `Take.frames` dtype | float64 DataFrame | f64 ndarray (plan said f32) | oracle comparison at 1e-6 needs f64; GPU gets f32 at upload |
| TRC reader | `skiprows=6` | data starts after 5 header lines, blank lines skipped | Python's own writer emits no blank line, so a saved TRC lost its first frame on re-open. **Fixed in Python too** (`dataLoader.py`, `skiprows=5`). |
| C3D reader, dropped labels | shifts remaining columns | keeps column ↔ label alignment | Python indexes columns by the *filtered* label list; wrong whenever a label is blank or duplicated |
| Pattern markers | `set` (unordered) | `Vec` in selection order | deterministic; the golden was generated with an ordered list |
| `Playback.play` at last frame | stops immediately | restarts from 0 | what ▶ is expected to do; loop/off semantics otherwise identical |

## Bugs found in the shipped Python app during this phase

1. **Save As TRC → Open drops frame 1** (reader/writer blank-line mismatch). Fixed in `MStudio/utils/dataLoader.py`.
2. (Phase 0) Kalman filter crashed on numpy ≥ 2 and on any gapped marker. Fixed in `filtering.py`.

## Next: Phase 2 — processing with oracle parity

`mstudio-processing`: six filters, nine interpolation methods, pattern-based interpolation, analysis primitives — each against the 60 golden arrays already in `tests/golden/`.
