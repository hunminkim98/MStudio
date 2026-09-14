# Phase 6 results — Python bindings and parity QA

Scope of this step: the `mstudio-py` bindings and the parity QA of
`docs/CROSS_PLATFORM_PLAN.md` §7 Phase 6. Distribution (`cargo-dist`,
notarization/signing, switching the root `pyproject.toml` to the Rust wheel,
retiring the legacy Python tree, `crates/spike` removal, README rewrite) is
deliberately **not** in this step.

## What was built

| Item | Where | Notes |
|---|---|---|
| `mstudio-app` as a library | `crates/mstudio-app/src/lib.rs` | `LaunchOptions` + `run(opts)`; `main.rs` only parses the command line. Same binary as before. |
| `mstudio-py` crate | `crates/mstudio-py/` | PyO3 0.29 + rust-numpy 0.29, `abi3-py310` (one wheel per OS/arch for every Python ≥ 3.10). Mixed project: Rust extension `mstudio._native`, Python package in `python/mstudio/`. |
| `Take` class | `src/lib.rs` | `frames`, `original`, `time`, `frame_numbers` are **zero-copy NumPy views** (`PyArray::borrow_from_array`, the view owns a reference to the `Take`). Safe because every edit is in place; `frames` setter copies into the existing buffer. `original`/`time`/`frame_numbers` are read-only views. |
| Processing | `Take.filter / filter_all / interpolate / pattern_interpolate / clear / restore_original / outliers / rename_markers / save / copy`, module-level `filter1d`, `interpolate1d`, `detect_outliers` | Markers accepted by index or name. Every routine runs with the GIL released (`Python::detach`); `filter_all` is rayon-parallel. Returns the half-open `(start, end)` dirty range like the Rust API. |
| `Filter` class | static constructors `butterworth`, `butterworth_on_speed`, `kalman`, `gaussian`, `loess`, `median` with the app's defaults | `ProcessingError` → `ValueError`, `IoError` → `OSError`, `ReportError` → `RuntimeError`. |
| Skeletons / report | `skeleton_models`, `skeleton_pair_names`, `skeleton_pairs`, `auto_segments`, `auto_joints`, `build_report`, `write_report`, `open_in_browser` | |
| `mstudio.run(path=None, play=False, *, screenshot=None, exit_after=None)` | | Opens the desktop app **in-process** (main thread, once per process). The keyword-only knobs are for automated checks. |
| CLI | `python -m mstudio [file] [--play]`, console script `mstudio` in the wheel | |
| Tests | `crates/mstudio-py/tests/test_mstudio.py` — 75 tests | Goldens (loaders, 14 filters × full/gapped, 24 interpolation, 4 pattern, 5 skeleton outlier sets), TRC byte-exact round trip, C3D round trip, view semantics (shared memory, lifetime via `base`, read-only views, setter copy), error mapping, GIL release, CLI. |
| Live parity | `scripts/check_parity.py` (+ `scripts/oracle_harness.py`) | Runs the **installed Python oracle** and the bindings side by side on the same inputs: 88 cases, table + JSON, exit 1 on any failure. Catches drift in the oracle's dependencies that frozen goldens cannot. |
| CI | `.github/workflows/rust.yml` job `wheels` | On ubuntu / windows / macos-latest: `maturin build --release`, install the wheel + the oracle (`pip install .`), run the bindings tests and `check_parity.py`, upload the wheel and the parity JSON as artifacts. No publishing. |

`crates/mstudio-py/Cargo.toml` sets `test = false` for the cdylib: its tests are
Python, and a Rust test harness would have to link libpython. `cargo test
--workspace` therefore keeps working everywhere.

## Verification (macOS 26.5.2, Apple M5 Max, Python 3.11.15, numpy 2.4.6, pandas 3.0.5)

- `cargo fmt --check`, `cargo clippy --workspace --all-targets` with `-D warnings`, `cargo test --workspace`: 90 tests green.
- `maturin develop --release` → `pytest crates/mstudio-py/tests`: **75 passed**.
- `scripts/check_parity.py` (`docs/parity_macos_arm64.json`): **81 ok, 7 documented deviations, 0 failures**.

| Section | Cases | ok | deviation | FAIL | max abs diff over ok cases |
|---|---|---|---|---|---|
| load (TRC, C3D: frames, time, markers, fps) | 6 | 6 | 0 | 0 | 0 (bit-exact) |
| filter (14 cases × full/gapped) | 28 | 28 | 0 | 0 | 6.0e-14 |
| interpolation (8 methods × 3 gaps) | 24 | 21 | 3 | 0 | 4.4e-16 |
| pattern-based (4 cases) | 4 | 4 | 0 | 0 | 5.6e-16 |
| skeleton pair resolution (5 models) | 5 | 5 | 0 | 0 | identical |
| outliers (5 models × full/gapped, sequential and parallel oracle) | 10 | 10 | 0 | 0 | identical |
| writers (TRC bytes, TRC/C3D cross-read both ways) | 6 | 6 | 0 | 0 | 2.3e-07 (C3D stores f32 mm) |
| segment / joint auto-detection (5 models) | 5 | 1 | 4 | 0 | — |

Filter timing from the same run (29 markers × 137 frames, all axes; Python = Pose2Sim `filter1d` per column, Rust = `Take.filter_all`):

| filter | Python (ms) | Rust (ms) | speed-up |
|---|---|---|---|
| butterworth | 25.4 | 0.69 | ×37 |
| butterworth_on_speed | 29.6 | 0.25 | ×118 |
| kalman | 195.6 | 0.27 | ×724 |
| gaussian | 6.2 | 0.30 | ×21 |
| LOESS | 48.2 | 0.27 | ×179 |
| median | 6.9 | 0.22 | ×31 |

- Desktop app after the lib/bin split: `--selftest` (worker filter → delete + undo bit-exact → 1.66 MB report) exits 0; `--demo --screenshot`, `--play --screenshot`, `tests/test.c3d --screenshot` all render as in Phase 5.
- `mstudio.run("tests/test.trc", screenshot=..., exit_after=3)` from Python: window opens in-process, renders the take, saves the PNG and returns after the timeout (`docs/spike/macos_app_phase6_python_run.png`). One of three runs was captured playing with a moved camera; not reproducible, consistent with keyboard/mouse input reaching the focused window while it was up.

## Deviations found by the live parity check

| Case | What differs | Decision |
|---|---|---|
| `spline` interpolation (3 cases, up to 0.29 m) | Known since Phase 2: pandas' `spline` is a smoothing `UnivariateSpline(s=len(x))`; the port interpolates (== `cubic` at order 3). | Kept. Reported as *deviation*, not failure. |
| TRC line endings (Windows only) | `save_to_trc` opens the file in text mode, so on Windows the oracle writes CRLF while the port always writes LF. Content is byte-identical otherwise, and each writer's output is read back correctly by the other implementation. | Kept: platform-independent LF output is deterministic and keeps the committed byte-exact fixture valid on every OS. `check_parity.py` reports a line-ending-only difference as *deviation*. |
| Segment / joint auto-detection (HALPE_26, COCO_17, BODY_25B, BODY_25) | `reportGenerator.py` matches the name patterns against the **skeleton model's node list**, then drops segments whose markers are missing from the data without trying the next pattern (e.g. HALPE_26 has a `Head` node, `tests/test.trc` has `Nose` but no `Head` marker → the oracle loses the Head segment; BODY_25B loses Trunk). The port matches against the data's markers. Everything the oracle finds, the port finds with the same markers; the port additionally finds 3–6 segments/joints per model. COCO_133 is identical. | Kept (the port's result is what the pattern table intends). Documented here and in `check_parity.py`. |

## Deferred to the distribution step

- Root `pyproject.toml` still builds the legacy `MStudio` Tk app (setuptools); the wheel from `crates/mstudio-py` is a separate distribution named `mstudio`. On PyPI both names are the same project, so the switch has to happen together with the release. On case-insensitive file systems `MStudio/` and `mstudio/` cannot both live in one `site-packages`, so the legacy package cannot be a shim next to the new one; it will move to a `legacy/` (oracle-only) tree.
- `crates/spike` removal, `cargo-dist`, notarization / signing, README + `CLAUDE.md` architecture rewrite.

## What the first CI run found (run #34820740136, commit af5eb5b)

The branch had never been pushed, so this was the first time the Rust workflow
ran. Two real cross-platform defects surfaced that no amount of local macOS
testing would have caught.

**1. `MStudio` and `mstudio` are one distribution to pip** — *all four wheel jobs*.
The job installed the oracle with `pip install .` and then the wheel with
`pip install --no-index --find-links dist mstudio`. pip normalises
distribution names case-insensitively, so the second command found "MStudio
0.1.5 already satisfies mstudio" and installed nothing; `import mstudio` then
failed (collection error, pytest exit 2) or resolved to the legacy package on
the case-insensitive runners (test failures, exit 1). Reproduced locally in a
clean virtualenv.
Fix: the oracle is no longer installed as a package. `scripts/oracle_harness.py`
already puts the repository root on `sys.path`, so the job installs only the
oracle's third-party dependencies (`scripts/oracle-requirements.txt`) and the
wheel keeps the `mstudio` name to itself. The install step now prints
`mstudio.__file__` so a silent no-op cannot happen again. This is the same name
collision recorded under *Deferred to the distribution step*; it bites CI today.

**2. No `.gitattributes`, so Windows checked out the byte-exact fixtures with CRLF** — *the `windows-latest` test job*.
`trc_writer_is_byte_exact_with_pandas` failed with `got 121207 want 121349`:
the golden TRC is 121 207 bytes with 142 newlines, and 121 207 + 142 = 121 349,
i.e. one CR added per line by git's autocrlf on the runner. The Rust writer was
right; the fixture was mangled on checkout.
Fix: `.gitattributes` marks `tests/*.trc`, `tests/*.c3d`, `tests/golden/**` and
the vendored Plotly bundle as `-text`. The test's panic message now counts the
CR bytes and names `.gitattributes`, because `str::lines()` strips a trailing
`\r` and the per-line comparison therefore passes while the byte count does not.

The `ubuntu-latest` and `macos-latest` test jobs (fmt, clippy `-D warnings`,
`cargo test --workspace`, release build) passed on the first attempt; the
`macos-13` jobs were still queued when the failures were diagnosed.

**3. The oracle writes CRLF on Windows** — *the `wheel windows-latest` job of the
second run (aac9ad7)*. With the wheel finally installed, the Windows parity run
reported 80 ok / 7 deviations / 1 failure: `trc bytes (after the path line)`.
`save_to_trc` uses `open(path, "w")`, so on Windows every newline it writes
becomes CRLF, while the port always writes LF; every other row, including both
cross-reads of the written files, was identical. Classified as a deviation
(line-ending-only differences are detected explicitly, a content difference
still fails).

## Parity QA on other platforms

Run #34830599666 (commit 47229e2) is green on Linux, Windows and macOS arm64.
The `wheels` job's parity JSON is committed per platform
(`docs/parity_linux_x86_64.json`, `docs/parity_windows_x86_64.json`,
`docs/parity_macos_arm64.json`):

| Runner | Python | Wheel | Parity: ok / deviation / fail |
|---|---|---|---|
| `ubuntu-latest` (Linux x86_64, glibc 2.39) | 3.11.16 | `mstudio-0.2.0-cp310-abi3-manylinux_2_39_x86_64.whl` | 81 / 7 / **0** |
| `windows-latest` (AMD64) | 3.11.9 | `mstudio-0.2.0-cp310-abi3-win_amd64.whl` | 80 / 8 / **0** |
| `macos-latest` (arm64) | 3.11.9 | `mstudio-0.2.0-cp310-abi3-macosx_11_0_arm64.whl` | 81 / 7 / **0** |
| `macos-13` (Intel x86_64) | — | — | never ran; dropped from the matrix, see below |

The deviations are the same everywhere (3 × smoothing spline, 4 × segment
auto-detection); Windows adds the line-ending one. `fmt`, `clippy -D warnings`,
`cargo test --workspace` (90 tests), the release build and the 75 bindings
tests pass on all three.

The interactive rows of `docs/QA_CHECKLIST.md` still need a person at each
machine.

## Decision: Intel macOS is out of scope for v2

`macos-13` sat queued through all three runs without ever being assigned a
runner (over 40 minutes in the last one), so it could not gate anything, and
GitHub is winding these images down. It is removed from both matrices in
`.github/workflows/rust.yml`; macOS means Apple Silicon for the Rust
application.

What follows from that:

- No `macosx_x86_64` wheel and no Intel `.dmg`. Intel Mac users stay on the
  Python v0.1.5 app, whose own workflow (`continuous-integration.yml`) still
  tests `macos-13` and is left untouched.
- `docs/CROSS_PLATFORM_PLAN.md` G1 now reads "macOS (Apple Silicon)" and the
  distribution row asks `cargo-dist` for an arm64 `.dmg` only.
- Nothing in the code is Apple-Silicon-specific; the target is dropped for
  lack of CI, not for a technical reason. If Intel coverage is wanted later,
  cross-compiling `x86_64-apple-darwin` on the arm64 runner would build the
  artifacts, but they would ship untested.
