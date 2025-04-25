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
