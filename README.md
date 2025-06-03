<div align="center">

# 🎯 MStudio

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.2-green.svg)](https://pypi.org/project/MStudio/)

**Markerless/Marker-based Motion Capture Data Visualization & Editing Tool**

*Seamlessly visualize, analyze, and edit 3D marker data from motion capture systems*

[🚀 Quick Start](#-quick-start) • [📖 Features](#-features) • [💾 Installation](#-installation) • [🎮 Usage](#-usage) • [🤝 Support](#-support)

</div>

---

## 🌟 What is MStudio?

MStudio is a user-friendly application designed for researchers, biomechanists, and motion capture professionals. It provides an intuitive interface to:

- **Visualize** 3D marker trajectories in real-time
- **Analyze** movement patterns and joint&segment angles
- **Edit** and filter motion capture data
- **Export** processed data for further analysis

Compatible with popular tools like [Pose2Sim](https://github.com/perfanalytics/pose2sim) and [Sports2D](https://github.com/davidpagnon/Sports2D).

---

## 💾 Installation

### Option 1: Install from PyPI (Recommended)

```bash
# Create a virtual environment (recommended)
conda create -n mstudio python=3.10 -y
conda activate mstudio

# Install MStudio
pip install mstudio
```

### Option 2: Install from Source

```bash
git clone https://github.com/hunminkim98/MStudio.git
cd MStudio
pip install .
```

---

## 🚀 Quick Start

### Launch MStudio

```bash
mstudio
```

That's it! The application will open with an intuitive interface ready for your motion capture data.

### Load Your First Dataset

1. **File → Open** or drag & drop your TRC/C3D file
2. **Play** the animation using the spacebar
3. **Explore** your data with mouse controls:
   - **Left click + drag**: Rotate view
   - **Right click + drag**: Pan
   - **Scroll wheel**: Zoom

---

## 📖 Features

<table>
<tr>
<td width="50%">

### 🎯 **3D Visualization**
- **Real-time 3D rendering** of marker data
- **Interactive camera controls** (rotate, pan, zoom)
- **Coordinate system switching** (Y-up/Z-up)
- **Marker label toggle** for clarity
- **Smooth 60 FPS animation** playback

### 🦴 **Skeleton Models**
Support for multiple skeleton formats:
- **BODY_25B/25** (OpenPose)
- **BODY_135** (Full body)
- **BLAZEPOSE** (MediaPipe)
- **HALPE** (26/68/136 keypoints)
- **COCO** (17/133 keypoints)
- **MPII** (16 keypoints)

</td>
<td width="50%">

### 📊 **Analysis Tools**
- **Trajectory visualization** with customizable paths
- **Distance measurement** between any 2 markers
- **Segment angle calculation** (relative to horizontal reference)
- **Joint angle calculation** using 3-point method
- **Outlier detection** with visual highlighting
- **Frame-by-frame navigation** with timeline scrubbing
- **Multi-axis coordinate plots**

### 🔧 **Data Processing**
- **Advanced filtering**:
  - Butterworth filter
  - Butterworth on speed
  - Median filter
- **Smart interpolation** with pattern matching
- **Interactive editing** tools
- **Original data preservation**

</td>
</tr>
</table>



## 🎮 Usage

### Step-by-Step Tutorial

#### 1. **Load Your Data**
```bash
# Launch MStudio
mstudio

# In the application:
# File → Open → Select your .trc or .c3d file
# Or simply drag & drop your file into the window
```

#### 2. **Basic Navigation**
| Action | Control |
|--------|---------|
| **Play/Pause Animation** | `Spacebar` or `Enter` |
| **Stop Animation** | `Esc` |
| **Next/Previous Frame** | `→` / `←` Arrow keys |
| **Rotate View** | Left click + drag |
| **Pan View** | Right click + drag |
| **Zoom** | Mouse wheel |

#### 3. **Analysis Mode**
1. Click the **"Analysis"** button to activate
2. **Select markers** by left-clicking in the 3D view:
   - **2 markers**: Shows distance measurement and segment angle (relative to virtual horizontal line)
     - Left-click to cycle through reference axes (X, Y, Z)
   - **3 markers**: Shows joint angle (middle marker = vertex)
3. Results display directly in the 3D viewport

#### 4. **Data Processing**
- **Filter data**: Use the filter panel to apply Butterworth, median, or speed-based filters
- **Interpolate gaps**: Select problematic markers and use pattern-based interpolation
- **Export results**: File → Save As → Choose TRC or C3D format

---

## 🤝 Support

### Getting Help
- 📖 **Documentation**: [GitHub Wiki](https://github.com/hunminkim98/MStudio/wiki) *(Coming Soon)*
- 🐛 **Bug Reports**: [Issue Tracker](https://github.com/hunminkim98/MStudio/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/hunminkim98/MStudio/discussions)
- 📧 **Contact**: hunminkim98@gmail.com

### Contributing
We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) *(Coming Soon)*.

### Supported File Formats
- **Input**: TRC, C3D
- **Output**: TRC, C3D
- **Skeleton Models**: OpenPose, MediaPipe, COCO, HALPE, MPII

---

## 🔮 Roadmap

### ✅ **Completed Features**
- [x] **3D Visualization**: Real-time marker rendering with smooth 60 FPS animation
- [x] **Skeleton Support**: Multiple skeleton models (OpenPose, MediaPipe, COCO, etc.)
- [x] **Analysis Tools**: Distance measurement, joint angles, segment angles
- [x] **Data Processing**: Advanced filtering and interpolation
- [x] **File Support**: TRC/C3D import/export functionality

### 🚧 **In Development**
- [ ] **Multi-Selection**: Drag-and-drop selection for multiple markers
- [ ] **View Planes**: Click-to-set orthogonal views (inspired by OpenSim)
- [ ] **Customization**: Adjustable marker size, color, and opacity
- [ ] **Performance**: Enhanced rendering for large datasets

### 🎯 **Planned Features**
- [ ] **Multi-Person Support**: Simultaneous visualization and analysis of multiple subjects
- [ ] **Analysis Reports**: Export comprehensive analysis results to PDF format
- [ ] **Gait Analysis Mode**: Specialized tools and metrics for gait analysis

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Compatible with [Pose2Sim](https://github.com/perfanalytics/pose2sim) and [Sports2D](https://github.com/davidpagnon/Sports2D)
- Built with Python, OpenGL, and CustomTkinter
- Inspired by the biomechanics and motion capture community

---

