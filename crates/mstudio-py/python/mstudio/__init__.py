"""MStudio: motion-capture marker viewer / editor.

The numerical core (filters, interpolation, I/O, reports) and the desktop
application live in Rust; this package exposes them to Python.
"""

from ._native import (  # noqa: F401
    DEFAULT_OUTLIER_THRESHOLD,
    INTERP_METHODS,
    Filter,
    Take,
    __version__,
    auto_joints,
    auto_segments,
    build_report,
    detect_outliers,
    filter1d,
    interpolate1d,
    load,
    open_in_browser,
    run,
    save,
    skeleton_models,
    skeleton_pair_names,
    skeleton_pairs,
    write_report,
)

__all__ = [
    "DEFAULT_OUTLIER_THRESHOLD",
    "INTERP_METHODS",
    "Filter",
    "Take",
    "__version__",
    "auto_joints",
    "auto_segments",
    "build_report",
    "detect_outliers",
    "filter1d",
    "interpolate1d",
    "load",
    "open_in_browser",
    "run",
    "save",
    "skeleton_models",
    "skeleton_pair_names",
    "skeleton_pairs",
    "write_report",
]
