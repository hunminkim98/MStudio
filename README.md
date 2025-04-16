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

### 1. Recommended: Run from the project root

```bash
# From the project root directory (the folder containing MEditor/ and README.md)
python -m MEditor.main
```

- This ensures all package imports work correctly.
- Do NOT run main.py directly from the MEditor/ subfolder, as this may cause ModuleNotFoundError.

### 2. Alternative: Directly run main.py (with sys.path workaround)

If you want to run main.py directly (not recommended for production), make sure the following lines are at the top of MEditor/main.py:

```python
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
```

Then you can run:

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

