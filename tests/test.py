import pytest

def test_import_main():
    try:
        from MStudio import main
    except ImportError as e:
        pytest.fail(f"Importing MStudio.main failed: {e}")

def test_main_smoke():
    from MStudio import main
    # Just check that main() can be called without crashing (no arguments)
    try:
        main()
    except Exception as e:
        pytest.fail(f"Calling main() failed: {e}")
