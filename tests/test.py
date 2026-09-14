import pytest
from unittest.mock import patch

def test_import_main():
    try:
        from MStudio.main import main
    except ImportError as e:
        pytest.fail(f"Importing MStudio.main failed: {e}")

@patch('MStudio.main.TRCViewer')
def test_main_smoke(mock_trc_viewer):
    from MStudio.main import main
    # Just check that main() can be called without crashing (no arguments)
    try:
        main()
    except Exception as e:
        pytest.fail(f"Calling main() failed: {e}")


def test_pattern_interpolation_recovers_a_rotating_rigid_body():
    """Regression for v0.1.5: Rotation.align_vectors arguments were reversed, so the
    reconstructed offset was rotated the wrong way whenever the references rotated."""
    import types
    import numpy as np
    import pandas as pd
    from MStudio.utils import dataProcessor

    n = 30
    base = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.3, 0.4, 0.5]])
    truth = np.zeros((n, 4, 3))
    for i in range(n):
        c, s = np.cos(i * 0.05), np.sin(i * 0.05)
        truth[i] = base @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T + np.array([i * 0.01, 0, 0.02 * i])
    df = pd.DataFrame({f"{name}_{ax}": truth[:, m, k] for m, name in enumerate(["R0", "R1", "R2", "T"]) for k, ax in enumerate("XYZ")})
    for ax in "XYZ":
        df.loc[10:19, f"T_{ax}"] = np.nan

    stub = types.SimpleNamespace(
        selection_data={"start": 8, "end": 22},
        marker_axes=[],
        update_plot=lambda *a, **k: None,
        state_manager=types.SimpleNamespace(
            selection_state=types.SimpleNamespace(current_marker="T", pattern_markers=["R0", "R1", "R2"]),
            editing_state=types.SimpleNamespace(pattern_selection_mode=False),
        ),
        data_manager=types.SimpleNamespace(data=df),
    )
    with patch.object(dataProcessor, "messagebox"):
        dataProcessor.interpolate_with_pattern(stub)
    got = df[["T_X", "T_Y", "T_Z"]].to_numpy()[10:20]
    assert np.abs(got - truth[10:20, 3]).max() < 1e-9
