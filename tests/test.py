import pytest

def test_import_main():
    try:
        from MEditor import main
    except ImportError as e:
        pytest.fail(f"Importing MEditor.main failed: {e}")

def test_main_smoke():
    from MEditor import main
    # Just check that main() can be called without crashing (no arguments)
    try:
        main()
    except Exception as e:
        pytest.fail(f"Calling main() failed: {e}")
