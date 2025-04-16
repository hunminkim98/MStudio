# MEditor
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)\
A comprehensive toolkit for editing and managing 2D and 3D markers in motion capture and biomechanical studies. Designed to be compatible with [Pose2Sim](https://github.com/perfanalytics/pose2sim), and [Sports2D](https://github.com/davidpagnon/Sports2D), providing seamless integration for marker data processing and analysis.

---

## 📦 Installation

**PyPI (recommended):**
```bash
pip install meditor
```

**From source:**
```bash
git clone https://github.com/hunminkim98/MEditor.git
cd MarkerStudio
pip install .
```

---

## 🚀 Quick Start

### The Easiest Way to Run MEditor (Recommended!)

Just open your terminal in the project root folder (where you see both `README.md` and the `MEditor/` directory) and run:

```bash
python -m MEditor.main
```

That’s it! This is the safest way to launch the app.

> **Heads up!**
> If you try to run `main.py` directly (like `python MEditor/main.py`), you might get an error like:
> `ModuleNotFoundError: No module named 'MEditor'`
> To avoid this, always use the command above from the root folder.

---

### Want to Run main.py Directly?

If you’re actively developing and want to quickly test changes, you can add these lines at the very top of `MEditor/main.py`:

```python
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
```

Now you can run:

```bash
python MEditor/main.py
```

---

## 📚 Documentation & Support
- [Issue Tracker](https://github.com/hunminkim98/MEditor/issues)

---

## Features

### 🎯 3D Marker Visualization

- Interactive 3D viewport with real-time marker display

- Customizable marker colors and sizes (TODO)

- Toggle marker labels visibility

- Coordinate system switching (Z-up/Y-up)

- Zoom and pan controls

### 🦴 Skeleton 

- Multiple pre-defined skeleton models:

  - BODY_25B
  - BODY_25
  - BODY_135
  - BLAZEPOSE
  - HALPE (26/68/136)
  - COCO (17/133)
  - MPII
- Toggle skeleton visibility
- Color-coded connections for outlier detection
### 📊 Data Analysis Tools
- Marker trajectory visualization
- Multi-axis coordinate plots
- Frame-by-frame navigation
- Timeline scrubbing with time/frame display modes
- Outlier detection and highlighting
### 🔧 Data Processing
- Multiple filtering options:
  - Butterworth filter
  - Butterworth on speed
  - Median filter
- Customizable filter parameters
- Pattern-based marker interpolation
- Interactive data selection and editing
### 💾 File Operations
- Import TRC/C3D files
- Export to TRC/C3D files
- Original data preservation

