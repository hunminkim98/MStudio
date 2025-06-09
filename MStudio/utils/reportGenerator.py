"""
PDF Report Generator for MStudio Analysis Results

This module generates comprehensive biomechanical analysis reports in PDF format.
"""

import os
import logging
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
from tkinter import filedialog, messagebox
from anytree import PreOrderIter

from MStudio.utils.analysisMode import calculate_distance, calculate_angle, calculate_velocity, calculate_acceleration
from MStudio.utils.skeletons import (
    BODY_25B, BODY_25, BODY_135, BLAZEPOSE, HALPE_26, HALPE_68,
    HALPE_136, COCO_133, COCO, MPII, COCO_17
)
from MStudio.utils.skeleton_config import get_segment_patterns, get_joint_patterns

logger = logging.getLogger(__name__)

## AUTHORSHIP INFORMATION
__author__ = "HunMin Kim"
__copyright__ = ""
__credits__ = [""]
__license__ = ""
__maintainer__ = "HunMin Kim"
__email__ = "hunminkim98@gmail.com"
__status__ = "Development"


class ReportGenerator:
    """
    Generates comprehensive biomechanical analysis reports in PDF format.
    """
    
    def __init__(self, data_manager, state_manager, fps: float = 30.0, skeleton_model_name: str = "No skeleton"):
        """
        Initialize the report generator.

        Args:
            data_manager: DataManager instance containing marker data
            state_manager: StateManager instance containing analysis state
            fps: Frame rate of the data
            skeleton_model_name: Name of the currently selected skeleton model
        """
        self.data_manager = data_manager
        self.state_manager = state_manager
        self.fps = fps
        self.skeleton_model_name = skeleton_model_name

        # Define centralized style configuration
        self.style = {
            'colors': {
                # Axis colors for biomechanical analysis
                'X': '#E74C3C',  # Professional red for X-axis
                'Y': '#27AE60',  # Professional green for Y-axis
                'Z': '#3498DB',  # Professional blue for Z-axis
                'mean': '#2C3E50',  # Dark navy for mean lines
                'accent': '#8E44AD',  # Purple for accents

                # Section colors
                'section_1': '#1B365D',  # Marker Coordinates
                'section_2': '#2C5F2D',  # Velocity & Acceleration
                'section_3': '#4A4A4A',  # Segment Angles
                'section_4': '#5D1B36',  # Joint Angles

                # UI colors
                'primary_text': '#2C3E50',  # Primary text color
                'secondary_text': '#34495e',  # Secondary text color
                'light_text': '#5a5a5a',  # Light text color
                'muted_text': '#7f8c8d',  # Muted text color
                'background_light': '#FAFAFA',  # Light background
                'background_alt': '#F8F9FA',  # Alternate background
                'table_header': '#2C3E50',  # Table header background
                'table_header_text': 'white',  # Table header text
            },
            'fonts': {
                'family': 'serif',
                'serif': ['Times New Roman', 'DejaVu Serif'],
                'sizes': {
                    'tiny': 8,
                    'small': 9,
                    'normal': 10,
                    'medium': 12,
                    'large': 14,
                    'xlarge': 16,
                    'title': 23,
                    'cover': 24
                }
            },
            'plot': {
                'linewidth': 2.0,
                'linewidth_thick': 2.5,
                'linewidth_thin': 1.5,
                'alpha_main': 0.8,
                'alpha_fill': 0.1,
                'alpha_grid': 0.3,
                'grid_linewidth': 0.8,
                'axes_linewidth': 1.2,
                'dpi': 100,
                'savefig_dpi': 300
            },
            'table': {
                'fontsize': 8,
                'fontsize_medium': 10,
                'fontsize_large': 12,
                'scale_height': 2.0,
                'scale_height_large': 2.5,
                'col_width_marker': 0.12,
                'col_width_stat': 0.055,
                'col_width_joint': 0.16,
                'col_width_joint_name': 0.2
            }
        }

        # Set seaborn style for professional biomechanical reports
        sns.set_style("whitegrid", {
            'axes.linewidth': self.style['plot']['axes_linewidth'],
            'grid.linewidth': self.style['plot']['grid_linewidth'],
            'grid.alpha': self.style['plot']['alpha_grid']
        })
        sns.set_context("paper", font_scale=1.1)

        # Set matplotlib parameters for high-quality scientific figures
        plt.rcParams.update({
            'font.family': self.style['fonts']['family'],
            'font.serif': self.style['fonts']['serif'],
            'font.size': self.style['fonts']['sizes']['normal'],
            'axes.titlesize': self.style['fonts']['sizes']['medium'],
            'axes.labelsize': self.style['fonts']['sizes']['normal'],
            'xtick.labelsize': self.style['fonts']['sizes']['small'],
            'ytick.labelsize': self.style['fonts']['sizes']['small'],
            'legend.fontsize': self.style['fonts']['sizes']['small'],
            'figure.titlesize': self.style['fonts']['sizes']['large'],
            'figure.dpi': self.style['plot']['dpi'],
            'savefig.dpi': self.style['plot']['savefig_dpi'],
            'savefig.bbox': 'tight',
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'grid.alpha': self.style['plot']['alpha_grid']
        })

        # Initialize analysis results storage
        self.analysis_results = {}

    def _get_skeleton_model_by_name(self, model_name: str):
        """Get skeleton model object by name."""
        model_mapping = {
            'BODY_25B': BODY_25B,
            'BODY_25': BODY_25,
            'BODY_135': BODY_135,
            'BLAZEPOSE': BLAZEPOSE,
            'HALPE_26': HALPE_26,
            'HALPE_68': HALPE_68,
            'HALPE_136': HALPE_136,
            'COCO_133': COCO_133,
            'COCO': COCO,
            'MPII': MPII,
            'COCO_17': COCO_17
        }
        return model_mapping.get(model_name, None)

    def _extract_marker_names_from_skeleton(self, skeleton_model):
        """Extract all marker names from skeleton model."""
        if skeleton_model is None:
            return []

        marker_names = []
        for node in PreOrderIter(skeleton_model):
            if node.name and node.name not in marker_names:
                marker_names.append(node.name)
        return marker_names

    def _get_standard_segments_from_skeleton(self, skeleton_model):
        """Generate standard biomechanical segments from skeleton model."""
        if skeleton_model is None:
            return {}

        # Extract all available markers from skeleton
        available_markers = self._extract_marker_names_from_skeleton(skeleton_model)

        # Get standard segment patterns from configuration
        segment_patterns = get_segment_patterns()

        # Find available segments
        available_segments = {}
        for segment_name, marker_combinations in segment_patterns.items():
            for markers in marker_combinations:
                if all(marker in available_markers for marker in markers):
                    available_segments[segment_name] = markers
                    logger.info(f"Found segment {segment_name}: {markers}")
                    break  # Use the first available combination

        logger.info(f"Available segments: {list(available_segments.keys())}")
        return available_segments

    def _get_standard_joints_from_skeleton(self, skeleton_model):
        """Generate standard biomechanical joints from skeleton model."""
        if skeleton_model is None:
            return {}

        # Extract all available markers from skeleton
        available_markers = self._extract_marker_names_from_skeleton(skeleton_model)

        # Get standard joint patterns from configuration
        joint_patterns = get_joint_patterns()

        # Find available joints
        available_joints = {}
        for joint_name, marker_combinations in joint_patterns.items():
            for markers in marker_combinations:
                if all(marker in available_markers for marker in markers):
                    available_joints[joint_name] = markers
                    logger.info(f"Found joint {joint_name}: {markers}")
                    break  # Use the first available combination

        logger.info(f"Available joints: {list(available_joints.keys())}")
        return available_joints

    def _prepare_all_data(self):
        """Prepare all analysis data by calculating kinematics, segments, and joints."""
        logger.info("Preparing all analysis data...")

        # Calculate kinematics (velocity and acceleration)
        self._calculate_kinematics()

        # Calculate segment data
        self._calculate_segment_data()

        # Calculate joint data
        self._calculate_joint_data()

        logger.info("All analysis data prepared successfully.")

    def _calculate_kinematics(self):
        """Calculate velocity and acceleration for all markers."""
        logger.info("Calculating kinematics data...")

        velocity_data = {}
        acceleration_data = {}

        for marker in self.data_manager.marker_names:
            velocities_x, velocities_y, velocities_z = [], [], []
            accelerations_x, accelerations_y, accelerations_z = [], [], []

            # Calculate velocities for frames 1 to num_frames-2
            for frame in range(1, self.data_manager.num_frames - 1):
                try:
                    pos_prev = self.data_manager.data.loc[frame-1, [f'{marker}_X', f'{marker}_Y', f'{marker}_Z']].values
                    pos_curr = self.data_manager.data.loc[frame, [f'{marker}_X', f'{marker}_Y', f'{marker}_Z']].values
                    pos_next = self.data_manager.data.loc[frame+1, [f'{marker}_X', f'{marker}_Y', f'{marker}_Z']].values

                    vel = calculate_velocity(pos_prev, pos_curr, pos_next, self.fps)
                    if vel is not None:
                        velocities_x.append(vel[0])
                        velocities_y.append(vel[1])
                        velocities_z.append(vel[2])
                    else:
                        velocities_x.append(np.nan)
                        velocities_y.append(np.nan)
                        velocities_z.append(np.nan)

                except (KeyError, IndexError):
                    velocities_x.append(np.nan)
                    velocities_y.append(np.nan)
                    velocities_z.append(np.nan)

            # Calculate accelerations for frames 2 to num_frames-3
            for frame in range(2, self.data_manager.num_frames - 2):
                try:
                    # Get positions for velocity calculation
                    pos_prev2 = self.data_manager.data.loc[frame-2, [f'{marker}_X', f'{marker}_Y', f'{marker}_Z']].values
                    pos_prev = self.data_manager.data.loc[frame-1, [f'{marker}_X', f'{marker}_Y', f'{marker}_Z']].values
                    pos_curr = self.data_manager.data.loc[frame, [f'{marker}_X', f'{marker}_Y', f'{marker}_Z']].values
                    pos_next = self.data_manager.data.loc[frame+1, [f'{marker}_X', f'{marker}_Y', f'{marker}_Z']].values
                    pos_next2 = self.data_manager.data.loc[frame+2, [f'{marker}_X', f'{marker}_Y', f'{marker}_Z']].values

                    # Calculate velocities at frame-1 and frame+1
                    vel_prev = calculate_velocity(pos_prev2, pos_prev, pos_curr, self.fps)
                    vel_next = calculate_velocity(pos_curr, pos_next, pos_next2, self.fps)

                    if vel_prev is not None and vel_next is not None:
                        acc = calculate_acceleration(vel_prev, vel_next, self.fps)
                        if acc is not None:
                            accelerations_x.append(acc[0])
                            accelerations_y.append(acc[1])
                            accelerations_z.append(acc[2])
                        else:
                            accelerations_x.append(np.nan)
                            accelerations_y.append(np.nan)
                            accelerations_z.append(np.nan)
                    else:
                        accelerations_x.append(np.nan)
                        accelerations_y.append(np.nan)
                        accelerations_z.append(np.nan)

                except (KeyError, IndexError):
                    accelerations_x.append(np.nan)
                    accelerations_y.append(np.nan)
                    accelerations_z.append(np.nan)

            velocity_data[marker] = {
                'x': velocities_x,
                'y': velocities_y,
                'z': velocities_z
            }

            acceleration_data[marker] = {
                'x': accelerations_x,
                'y': accelerations_y,
                'z': accelerations_z
            }

        # Store in analysis results
        self.analysis_results['velocity'] = velocity_data
        self.analysis_results['acceleration'] = acceleration_data

        logger.info(f"Kinematics calculated for {len(velocity_data)} markers.")

    def _calculate_segment_data(self):
        """Calculate segment lengths and angles for all available segments."""
        logger.info("Calculating segment data...")

        # Get skeleton model and available segments
        skeleton_model = self._get_skeleton_model_by_name(self.skeleton_model_name)

        if skeleton_model is None:
            # Use user-defined skeleton pairs
            available_segments = {}
            if self.state_manager.skeleton_pairs:
                for pair in self.state_manager.skeleton_pairs:
                    marker1, marker2 = pair
                    segment_name = f"{marker1}-{marker2}"
                    if all(f"{marker}_X" in self.data_manager.data.columns for marker in [marker1, marker2]):
                        available_segments[segment_name] = [marker1, marker2]
        else:
            # Get standard segments from skeleton model
            available_segments = self._get_standard_segments_from_skeleton(skeleton_model)

            # Filter segments based on available markers in data
            filtered_segments = {}
            for segment_name, markers in available_segments.items():
                if all(f"{marker}_X" in self.data_manager.data.columns for marker in markers):
                    filtered_segments[segment_name] = markers

            available_segments = filtered_segments

            # Also include user-defined skeleton pairs if any
            if self.state_manager.skeleton_pairs:
                for pair in self.state_manager.skeleton_pairs:
                    marker1, marker2 = pair
                    segment_name = f"{marker1}-{marker2}"
                    if segment_name not in available_segments:  # Avoid duplicates
                        if all(f"{marker}_X" in self.data_manager.data.columns for marker in [marker1, marker2]):
                            available_segments[segment_name] = [marker1, marker2]

        # Calculate segment data
        segment_data = {}
        for segment_name, markers in available_segments.items():
            marker1, marker2 = markers
            lengths = []
            angles_x = []
            angles_y = []
            angles_z = []

            for frame in range(self.data_manager.num_frames):
                try:
                    pos1 = self.data_manager.data.loc[frame, [f'{marker1}_X', f'{marker1}_Y', f'{marker1}_Z']].values
                    pos2 = self.data_manager.data.loc[frame, [f'{marker2}_X', f'{marker2}_Y', f'{marker2}_Z']].values

                    if not (np.isnan(pos1).any() or np.isnan(pos2).any()):
                        # Calculate distance
                        distance = calculate_distance(pos1, pos2)
                        lengths.append(distance if distance is not None else np.nan)

                        # Calculate angles relative to each axis
                        # Angle with X-axis
                        x_axis = np.array([1.0, 0.0, 0.0])
                        ref_point_x = pos1 + x_axis
                        angle_x = calculate_angle(ref_point_x, pos1, pos2)
                        angles_x.append(angle_x if angle_x is not None else np.nan)

                        # Angle with Y-axis
                        y_axis = np.array([0.0, 1.0, 0.0])
                        ref_point_y = pos1 + y_axis
                        angle_y = calculate_angle(ref_point_y, pos1, pos2)
                        angles_y.append(angle_y if angle_y is not None else np.nan)

                        # Angle with Z-axis
                        z_axis = np.array([0.0, 0.0, 1.0])
                        ref_point_z = pos1 + z_axis
                        angle_z = calculate_angle(ref_point_z, pos1, pos2)
                        angles_z.append(angle_z if angle_z is not None else np.nan)
                    else:
                        lengths.append(np.nan)
                        angles_x.append(np.nan)
                        angles_y.append(np.nan)
                        angles_z.append(np.nan)

                except (KeyError, IndexError):
                    lengths.append(np.nan)
                    angles_x.append(np.nan)
                    angles_y.append(np.nan)
                    angles_z.append(np.nan)

            segment_data[segment_name] = {
                'lengths': lengths,
                'angles_x': angles_x,
                'angles_y': angles_y,
                'angles_z': angles_z
            }

        # Store in analysis results
        self.analysis_results['segments'] = segment_data
        logger.info(f"Segment data calculated for {len(segment_data)} segments.")

    def _calculate_joint_data(self):
        """Calculate joint angles for all available joints."""
        logger.info("Calculating joint data...")

        # Get skeleton model and available joints
        skeleton_model = self._get_skeleton_model_by_name(self.skeleton_model_name)

        if skeleton_model is None:
            available_joints = {}
            logger.warning(f"Skeleton model '{self.skeleton_model_name}' not found. Joint analysis will be limited.")
        else:
            # Get standard joints from skeleton model
            available_joints = self._get_standard_joints_from_skeleton(skeleton_model)

            # Filter joints based on available markers in data
            filtered_joints = {}
            for joint_name, markers in available_joints.items():
                if all(f"{marker}_X" in self.data_manager.data.columns for marker in markers):
                    filtered_joints[joint_name] = markers

            available_joints = filtered_joints

        # Calculate joint angles
        joint_angles = {}
        for joint_name, markers in available_joints.items():
            marker1, marker2, marker3 = markers
            angles = []

            for frame in range(self.data_manager.num_frames):
                try:
                    pos1 = self.data_manager.data.loc[frame, [f'{marker1}_X', f'{marker1}_Y', f'{marker1}_Z']].values
                    pos2 = self.data_manager.data.loc[frame, [f'{marker2}_X', f'{marker2}_Y', f'{marker2}_Z']].values
                    pos3 = self.data_manager.data.loc[frame, [f'{marker3}_X', f'{marker3}_Y', f'{marker3}_Z']].values

                    if not (np.isnan(pos1).any() or np.isnan(pos2).any() or np.isnan(pos3).any()):
                        angle = calculate_angle(pos1, pos2, pos3)
                        angles.append(angle if angle is not None else np.nan)
                    else:
                        angles.append(np.nan)

                except (KeyError, IndexError):
                    angles.append(np.nan)

            joint_angles[joint_name] = angles

        # Store in analysis results
        self.analysis_results['joints'] = joint_angles
        logger.info(f"Joint data calculated for {len(joint_angles)} joints.")

    def _create_statistics_table(self, pdf: PdfPages, data: list, col_labels: list,
                                title: str = "", figsize: tuple = (16, 10),
                                col_widths: list = None) -> None:
        """
        Create a standardized statistics table page.

        Args:
            pdf: PdfPages object to save the figure
            data: List of lists containing table data
            col_labels: List of column labels
            title: Optional title for the table
            figsize: Figure size tuple
            col_widths: List of column widths (optional)
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('off')

        if title:
            ax.set_title(title, fontsize=self.style['fonts']['sizes']['xlarge'],
                        fontweight='bold', pad=20, color=self.style['colors']['primary_text'])

        if data:
            # Use default column widths if not provided
            if col_widths is None:
                col_widths = [self.style['table']['col_width_marker']] + \
                           [self.style['table']['col_width_stat']] * (len(col_labels) - 1)

            table = ax.table(cellText=data,
                           colLabels=col_labels,
                           cellLoc='center',
                           loc='center',
                           colWidths=col_widths)
            table.auto_set_font_size(False)
            table.set_fontsize(self.style['table']['fontsize'])
            table.scale(1, self.style['table']['scale_height_large'])

            # Style the table headers
            for i in range(len(col_labels)):
                table[(0, i)].set_facecolor(self.style['colors']['table_header'])
                table[(0, i)].set_text_props(weight='bold', color=self.style['colors']['table_header_text'])

            # Add alternating row colors
            for i in range(1, len(data) + 1):
                if i % 2 == 0:
                    for j in range(len(col_labels)):
                        table[(i, j)].set_facecolor(self.style['colors']['background_alt'])
        else:
            ax.text(0.5, 0.5, 'No valid data available',
                   ha='center', va='center', fontsize=self.style['fonts']['sizes']['large'],
                   transform=ax.transAxes, color=self.style['colors']['muted_text'])

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _create_analysis_plot(self, axes, time_axis: np.ndarray, data_dict: dict,
                            title: str, ylabel: str, plot_type: str = 'single') -> None:
        """
        Create a standardized analysis plot with professional styling.

        Args:
            axes: Matplotlib axes object
            time_axis: Time axis data
            data_dict: Dictionary containing plot data
            title: Plot title
            ylabel: Y-axis label
            plot_type: 'single' for single line, 'multi' for multiple lines (X,Y,Z)
        """
        axes.set_title(title, fontweight='bold', color=self.style['colors']['primary_text'])
        axes.set_xlabel('Time (s)', fontweight='bold')
        axes.set_ylabel(ylabel, fontweight='bold')
        axes.grid(True, alpha=self.style['plot']['alpha_grid'],
                 linestyle='-', linewidth=self.style['plot']['grid_linewidth'])
        axes.set_facecolor(self.style['colors']['background_light'])

        if plot_type == 'single':
            # Single line plot (e.g., joint angles)
            data = data_dict.get('data', [])
            color = data_dict.get('color', self.style['colors']['accent'])
            label = data_dict.get('label', 'Data')

            if len(data) > 0:
                axes.plot(time_axis, data,
                         linewidth=self.style['plot']['linewidth_thick'],
                         alpha=self.style['plot']['alpha_main'],
                         color=color, label=label)

                # Add statistics
                valid_data = np.array(data)[~np.isnan(data)]
                if len(valid_data) > 0:
                    mean_val = np.mean(valid_data)
                    std_val = np.std(valid_data)

                    # Mean line
                    axes.axhline(mean_val, color=self.style['colors']['mean'],
                               linestyle='--', alpha=self.style['plot']['alpha_main'],
                               linewidth=self.style['plot']['linewidth_thin'],
                               label=f'Mean: {mean_val:.3f}')

                    # Confidence interval
                    axes.axhspan(mean_val - std_val, mean_val + std_val,
                               alpha=self.style['plot']['alpha_fill'], color=color)

                    axes.legend(fontsize=self.style['fonts']['sizes']['small'],
                              frameon=True, fancybox=True, shadow=True)
            else:
                axes.text(0.5, 0.5, 'No Data Available', ha='center', va='center',
                         transform=axes.transAxes, fontsize=self.style['fonts']['sizes']['medium'],
                         color=self.style['colors']['muted_text'], fontweight='bold')

        elif plot_type == 'multi':
            # Multi-line plot (e.g., X, Y, Z components)
            for axis_name in ['X', 'Y', 'Z']:
                data = data_dict.get(axis_name, [])
                color = self.style['colors'][axis_name]

                if len(data) > 0:
                    axes.plot(time_axis, data,
                             linewidth=self.style['plot']['linewidth_thick'],
                             alpha=self.style['plot']['alpha_main'],
                             color=color, label=f'{axis_name}-axis')

            axes.legend(fontsize=self.style['fonts']['sizes']['small'],
                       frameon=True, fancybox=True, shadow=True)

    def generate_report(self, output_path: Optional[str] = None) -> bool:
        """
        Generate a comprehensive analysis report.

        Args:
            output_path: Path to save the PDF report. If None, user will be prompted.

        Returns:
            bool: True if report was generated successfully, False otherwise
        """
        try:
            # Get output path if not provided
            if output_path is None:
                output_path = filedialog.asksaveasfilename(
                    title="Save Analysis Report",
                    defaultextension=".pdf",
                    filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
                )
                if not output_path:
                    return False

            # Validate data
            if not self.data_manager.has_data():
                messagebox.showerror("Error", "No data available for report generation.")
                return False

            # Prepare all analysis data first
            self._prepare_all_data()

            # Generate report
            with PdfPages(output_path) as pdf:
                self._create_cover_page(pdf)
                self._create_data_overview(pdf)

                # Marker Coordinates Analysis Section
                self._create_section_cover(pdf, "Marker Coordinates Analysis",
                                         "Detailed analysis of marker position data over time",
                                         "Section I", self.style['colors']['section_1'])
                self._create_coordinate_analysis(pdf)

                # Velocity & Acceleration Analysis Section
                self._create_section_cover(pdf, "Velocity & Acceleration Analysis",
                                         "Analysis of marker velocity and acceleration components",
                                         "Section II", self.style['colors']['section_2'])
                self._create_velocity_acceleration_analysis(pdf)

                # Segment Angles Analysis Section
                self._create_section_cover(pdf, "Segment Angles Analysis",
                                         "Analysis of skeletal segment orientations and angles",
                                         "Section III", self.style['colors']['section_3'])
                self._create_skeleton_analysis(pdf)

                # Joint Angles Analysis Section
                self._create_section_cover(pdf, "Joint Angles Analysis",
                                         "Analysis of joint angles and biomechanical parameters",
                                         "Section IV", self.style['colors']['section_4'])
                self._create_joint_analysis(pdf)

            messagebox.showinfo("Success", f"Analysis report saved to:\n{output_path}")
            logger.info(f"Analysis report generated: {output_path}")
            return True

        except Exception as e:
            error_msg = f"Error generating report: {e}"
            logger.error(error_msg, exc_info=True)
            messagebox.showerror("Error", error_msg)
            return False
    
    def _create_cover_page(self, pdf: PdfPages) -> None:
        """Create the cover page of the report."""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.8, 'MStudio Analysis Report',
                fontsize=self.style['fonts']['sizes']['cover'], fontweight='bold', ha='center', va='center')

        # Subtitle
        ax.text(0.5, 0.7, 'Comprehensive Biomechanical Analysis',
                fontsize=self.style['fonts']['sizes']['xlarge'], ha='center', va='center', style='italic')

        # Date and time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ax.text(0.5, 0.6, f'Generated on: {current_time}',
                fontsize=self.style['fonts']['sizes']['medium'], ha='center', va='center')

        # Data information
        data_info = [
            f"Number of Markers: {len(self.data_manager.marker_names)}",
            f"Number of Frames: {self.data_manager.num_frames}",
            f"Frame Rate: {self.fps:.1f} fps",
            f"Duration: {self.data_manager.num_frames / self.fps:.2f} seconds"
        ]

        for i, info in enumerate(data_info):
            ax.text(0.5, 0.45 - i*0.05, info,
                    fontsize=self.style['fonts']['sizes']['medium'], ha='center', va='center')

        # Skeleton model info
        ax.text(0.5, 0.2, f"Skeleton Model: {self.skeleton_model_name}",
                fontsize=self.style['fonts']['sizes']['medium'], ha='center', va='center', fontweight='bold')

        # Footer
        ax.text(0.5, 0.05, 'Generated by MStudio - Motion Capture Analysis Tool',
                fontsize=self.style['fonts']['sizes']['normal'], ha='center', va='center', style='italic', alpha=0.7)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _create_section_cover(self, pdf: PdfPages, title: str, subtitle: str, section_number: str, color: str) -> None:
        """Create an academic-style section cover page."""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')

        # Clean white background with subtle border
        border_rect = patches.Rectangle((0.05, 0.05), 0.9, 0.9,
                                      linewidth=2, edgecolor=color,
                                      facecolor='none', alpha=0.8)
        ax.add_patch(border_rect)

        # Header section with subtle background
        header_rect = patches.Rectangle((0.1, 0.75), 0.8, 0.15,
                                      facecolor=color, alpha=0.1)
        ax.add_patch(header_rect)

        # Section number in top left
        ax.text(0.15, 0.85, section_number,
                fontsize=16, fontweight='bold', ha='left', va='center',
                color=color, family='serif')

        # Main title - academic style
        ax.text(0.5, 0.6, title,
                fontsize=self.style['fonts']['sizes']['title'], fontweight='bold', ha='center', va='center',
                color=self.style['colors']['primary_text'], family=self.style['fonts']['family'])

        # Subtitle with academic formatting
        ax.text(0.5, 0.5, subtitle,
                fontsize=self.style['fonts']['sizes']['large'], ha='center', va='center',
                color=self.style['colors']['secondary_text'], family=self.style['fonts']['family'],
                style='italic')

        # Horizontal divider lines
        ax.plot([0.2, 0.8], [0.42, 0.42], color=color, linewidth=self.style['plot']['linewidth_thin'], alpha=self.style['plot']['alpha_main'])
        ax.plot([0.25, 0.75], [0.38, 0.38], color=color, linewidth=self.style['plot']['grid_linewidth'], alpha=0.6)

        # Academic footer
        ax.text(0.5, 0.2, 'Biomechanical Analysis Report',
                fontsize=self.style['fonts']['sizes']['medium'], ha='center', va='center',
                color=self.style['colors']['light_text'], family=self.style['fonts']['family'])

        ax.text(0.5, 0.15, 'MStudio Motion Capture Analysis System',
                fontsize=self.style['fonts']['sizes']['normal'], ha='center', va='center',
                color=self.style['colors']['muted_text'], family=self.style['fonts']['family'], style='italic')

        # Page number placeholder
        ax.text(0.85, 0.1, f'{section_number}',
                fontsize=10, ha='center', va='center',
                color=color, family='serif')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _create_data_overview(self, pdf: PdfPages) -> None:
        """Create data overview page."""
        # Dataset Statistics page only
        self._create_dataset_statistics_page(pdf)

    def _create_dataset_statistics_page(self, pdf: PdfPages) -> None:
        """Create dataset statistics page."""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        fig.suptitle('Dataset Statistics', fontsize=self.style['fonts']['sizes']['xlarge'],
                    fontweight='bold', color=self.style['colors']['primary_text'])

        ax.axis('off')

        # Calculate data quality statistics
        total_possible_points = len(self.data_manager.marker_names) * self.data_manager.num_frames * 3  # X, Y, Z
        actual_data_points = 0
        missing_data_points = 0

        for marker in self.data_manager.marker_names:
            for axis in ['X', 'Y', 'Z']:
                col_name = f"{marker}_{axis}"
                if col_name in self.data_manager.data.columns:
                    valid_count = (~self.data_manager.data[col_name].isnull()).sum()
                    actual_data_points += valid_count
                    missing_data_points += (self.data_manager.num_frames - valid_count)

        data_completeness_percent = (actual_data_points / total_possible_points * 100) if total_possible_points > 0 else 0

        stats_data = [
            ['Total Markers', len(self.data_manager.marker_names)],
            ['Total Frames', self.data_manager.num_frames],
            ['Frame Rate', f"{self.fps:.1f} fps"],
            ['Duration', f"{self.data_manager.num_frames / self.fps:.2f} s"],
            ['Skeleton Model', self.skeleton_model_name],
            ['Skeleton Pairs', len(self.state_manager.skeleton_pairs)],
            ['Data Completeness', f"{data_completeness_percent:.1f}%"],
            ['Valid Data Points', f"{actual_data_points:,}"],
            ['Missing Data Points', f"{missing_data_points:,}"],
            ['Data Size', f"{self.data_manager.data.shape[0]} × {self.data_manager.data.shape[1]}"]
        ]

        table = ax.table(cellText=stats_data,
                         colLabels=['Parameter', 'Value'],
                         cellLoc='left',
                         loc='center',
                         colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(self.style['table']['fontsize_large'])
        table.scale(1, self.style['table']['scale_height'])

        # Style the table with modern colors
        for i in range(len(stats_data) + 1):  # +1 for header
            for j in range(2):
                if i == 0:  # Header row
                    table[(i, j)].set_facecolor(self.style['colors']['table_header'])
                    table[(i, j)].set_text_props(weight='bold', color=self.style['colors']['table_header_text'])
                elif i % 2 == 0:  # Even rows
                    table[(i, j)].set_facecolor(self.style['colors']['background_alt'])

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)



    def _create_coordinate_analysis(self, pdf: PdfPages) -> None:
        """Create coordinate analysis pages with modern seaborn styling."""
        # Create pages for each marker's coordinate data
        markers_per_page = 6
        marker_chunks = [self.data_manager.marker_names[i:i+markers_per_page]
                        for i in range(0, len(self.data_manager.marker_names), markers_per_page)]

        for chunk_idx, marker_chunk in enumerate(marker_chunks):
            # Create figure with seaborn styling
            with sns.axes_style("whitegrid"):
                fig, axes = plt.subplots(len(marker_chunk), 3, figsize=(15, 2.5*len(marker_chunk)))
                if len(marker_chunk) == 1:
                    axes = axes.reshape(1, -1)

                fig.suptitle(f'Marker Coordinates (Page {chunk_idx + 1})',
                            fontsize=self.style['fonts']['sizes']['xlarge'],
                            fontweight='bold', color=self.style['colors']['primary_text'])

                time_axis = np.arange(self.data_manager.num_frames) / self.fps

                for i, marker in enumerate(marker_chunk):
                    for j, axis in enumerate(['X', 'Y', 'Z']):
                        col_name = f"{marker}_{axis}"
                        if col_name in self.data_manager.data.columns:
                            data = self.data_manager.data[col_name].values

                            # Use professional biomechanical colors
                            color = self.style['colors'][axis]

                            # Plot with enhanced styling
                            axes[i, j].plot(time_axis, data,
                                          linewidth=self.style['plot']['linewidth'],
                                          alpha=self.style['plot']['alpha_main'],
                                          color=color,
                                          label=f'{axis}-coordinate')

                            axes[i, j].set_title(f"{marker} - {axis} Coordinate",
                                                fontweight='bold',
                                                color=self.style['colors']['primary_text'])
                            axes[i, j].set_xlabel('Time (s)', fontweight='bold')
                            axes[i, j].set_ylabel(f'{axis} Position (m)', fontweight='bold')

                            # Enhanced grid styling
                            axes[i, j].grid(True, alpha=self.style['plot']['alpha_grid'],
                                          linestyle='-', linewidth=self.style['plot']['grid_linewidth'])
                            axes[i, j].set_facecolor(self.style['colors']['background_light'])

                            # Add professional statistics
                            valid_data = data[~np.isnan(data)]
                            if len(valid_data) > 0:
                                mean_val = np.mean(valid_data)
                                std_val = np.std(valid_data)

                                # Mean line with professional styling
                                axes[i, j].axhline(mean_val,
                                                 color=self.style['colors']['mean'],
                                                 linestyle='--',
                                                 alpha=self.style['plot']['alpha_main'],
                                                 linewidth=self.style['plot']['linewidth_thin'],
                                                 label=f'Mean: {mean_val:.3f}m')

                                # Add confidence interval
                                axes[i, j].axhspan(mean_val - std_val, mean_val + std_val,
                                                 alpha=self.style['plot']['alpha_fill'], color=color)

                                axes[i, j].legend(fontsize=self.style['fonts']['sizes']['tiny'],
                                                frameon=True, fancybox=True, shadow=True)
                        else:
                            axes[i, j].text(0.5, 0.5, 'No Data Available',
                                           ha='center', va='center',
                                           transform=axes[i, j].transAxes,
                                           fontsize=self.style['fonts']['sizes']['medium'],
                                           color=self.style['colors']['muted_text'],
                                           fontweight='bold')
                            axes[i, j].set_title(f"{marker} - {axis} Coordinate",
                                                fontweight='bold', color=self.style['colors']['primary_text'])
                            axes[i, j].set_facecolor(self.style['colors']['background_alt'])

                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight', facecolor='white')
                plt.close(fig)

    def _create_velocity_acceleration_analysis(self, pdf: PdfPages) -> None:
        """Create velocity and acceleration analysis pages."""
        # Calculate velocity and acceleration for all markers
        markers_per_page = 2  # Reduced to accommodate more subplots
        marker_chunks = [self.data_manager.marker_names[i:i+markers_per_page]
                        for i in range(0, len(self.data_manager.marker_names), markers_per_page)]

        for chunk_idx, marker_chunk in enumerate(marker_chunks):
            # Create 2 rows (velocity and acceleration) × 3 columns (X, Y, Z) for each marker
            with sns.axes_style("whitegrid"):
                fig, axes = plt.subplots(len(marker_chunk) * 2, 3, figsize=(15, 4*len(marker_chunk)))
                if len(marker_chunk) == 1:
                    axes = axes.reshape(2, 3)

                fig.suptitle(f'Velocity & Acceleration Components (Page {chunk_idx + 1})',
                            fontsize=self.style['fonts']['sizes']['xlarge'],
                            fontweight='bold', color=self.style['colors']['primary_text'])

                time_axis = np.arange(self.data_manager.num_frames) / self.fps

                for i, marker in enumerate(marker_chunk):
                    # Calculate velocity and acceleration components
                    velocities_x, velocities_y, velocities_z = [], [], []
                    accelerations_x, accelerations_y, accelerations_z = [], [], []

                    # Calculate velocities for frames 1 to num_frames-2
                    for frame in range(1, self.data_manager.num_frames - 1):
                        try:
                            pos_prev = self.data_manager.data.loc[frame-1, [f'{marker}_X', f'{marker}_Y', f'{marker}_Z']].values
                            pos_curr = self.data_manager.data.loc[frame, [f'{marker}_X', f'{marker}_Y', f'{marker}_Z']].values
                            pos_next = self.data_manager.data.loc[frame+1, [f'{marker}_X', f'{marker}_Y', f'{marker}_Z']].values

                            vel = calculate_velocity(pos_prev, pos_curr, pos_next, self.fps)
                            if vel is not None:
                                velocities_x.append(vel[0])
                                velocities_y.append(vel[1])
                                velocities_z.append(vel[2])
                            else:
                                velocities_x.append(np.nan)
                                velocities_y.append(np.nan)
                                velocities_z.append(np.nan)

                        except (KeyError, IndexError):
                            velocities_x.append(np.nan)
                            velocities_y.append(np.nan)
                            velocities_z.append(np.nan)

                    # Calculate accelerations for frames 2 to num_frames-3
                    for frame in range(2, self.data_manager.num_frames - 2):
                        try:
                            # Get positions for velocity calculation
                            pos_prev2 = self.data_manager.data.loc[frame-2, [f'{marker}_X', f'{marker}_Y', f'{marker}_Z']].values
                            pos_prev = self.data_manager.data.loc[frame-1, [f'{marker}_X', f'{marker}_Y', f'{marker}_Z']].values
                            pos_curr = self.data_manager.data.loc[frame, [f'{marker}_X', f'{marker}_Y', f'{marker}_Z']].values
                            pos_next = self.data_manager.data.loc[frame+1, [f'{marker}_X', f'{marker}_Y', f'{marker}_Z']].values
                            pos_next2 = self.data_manager.data.loc[frame+2, [f'{marker}_X', f'{marker}_Y', f'{marker}_Z']].values

                            # Calculate velocities at frame-1 and frame+1
                            vel_prev = calculate_velocity(pos_prev2, pos_prev, pos_curr, self.fps)
                            vel_next = calculate_velocity(pos_curr, pos_next, pos_next2, self.fps)

                            if vel_prev is not None and vel_next is not None:
                                acc = calculate_acceleration(vel_prev, vel_next, self.fps)
                                if acc is not None:
                                    accelerations_x.append(acc[0])
                                    accelerations_y.append(acc[1])
                                    accelerations_z.append(acc[2])
                                else:
                                    accelerations_x.append(np.nan)
                                    accelerations_y.append(np.nan)
                                    accelerations_z.append(np.nan)
                            else:
                                accelerations_x.append(np.nan)
                                accelerations_y.append(np.nan)
                                accelerations_z.append(np.nan)

                        except (KeyError, IndexError):
                            accelerations_x.append(np.nan)
                            accelerations_y.append(np.nan)
                            accelerations_z.append(np.nan)

                    # Plot velocity components (X, Y, Z) with professional styling
                    velocity_components = [
                        (velocities_x, 'X'),
                        (velocities_y, 'Y'),
                        (velocities_z, 'Z')
                    ]

                    for j, (vel_component, axis_name) in enumerate(velocity_components):
                        if vel_component:
                            # Create time axis for velocity
                            vel_array = np.array(vel_component)
                            vel_start_frame = 1
                            vel_end_frame = vel_start_frame + len(vel_array)
                            vel_time = time_axis[vel_start_frame:vel_end_frame]

                            # Ensure both arrays have the same length
                            min_length = min(len(vel_time), len(vel_array))
                            vel_time = vel_time[:min_length]
                            vel_array = vel_array[:min_length]

                            if len(vel_time) > 0 and len(vel_array) > 0:
                                # Use professional biomechanical colors
                                color = self.style['colors'][axis_name]

                                axes[i*2, j].plot(vel_time, vel_array,
                                                linewidth=self.style['plot']['linewidth'],
                                                color=color,
                                                alpha=self.style['plot']['alpha_main'],
                                                label=f'Velocity {axis_name}')

                                axes[i*2, j].set_title(f"{marker} - Velocity {axis_name}",
                                                      fontweight='bold',
                                                      color=self.style['colors']['primary_text'])
                                axes[i*2, j].set_xlabel('Time (s)', fontweight='bold')
                                axes[i*2, j].set_ylabel(f'Velocity {axis_name} (m/s)', fontweight='bold')
                                axes[i*2, j].grid(True, alpha=self.style['plot']['alpha_grid'],
                                                linestyle='-', linewidth=self.style['plot']['grid_linewidth'])
                                axes[i*2, j].set_facecolor(self.style['colors']['background_light'])

                                # Add professional statistics
                                valid_vel = vel_array[~np.isnan(vel_array)]
                                if len(valid_vel) > 0:
                                    mean_vel = np.mean(valid_vel)
                                    std_vel = np.std(valid_vel)

                                    axes[i*2, j].axhline(mean_vel,
                                                       color=self.style['colors']['mean'],
                                                       linestyle='--',
                                                       alpha=self.style['plot']['alpha_main'],
                                                       linewidth=self.style['plot']['linewidth_thin'],
                                                       label=f'Mean: {mean_vel:.3f} m/s')

                                    # Add confidence interval
                                    axes[i*2, j].axhspan(mean_vel - std_vel, mean_vel + std_vel,
                                                       alpha=self.style['plot']['alpha_fill'], color=color)

                                    axes[i*2, j].legend(fontsize=self.style['fonts']['sizes']['tiny'],
                                                      frameon=True, fancybox=True, shadow=True)
                            else:
                                axes[i*2, j].text(0.5, 0.5, 'Insufficient Data',
                                                 ha='center', va='center',
                                                 transform=axes[i*2, j].transAxes,
                                                 fontsize=self.style['fonts']['sizes']['medium'],
                                                 color=self.style['colors']['muted_text'],
                                                 fontweight='bold')
                                axes[i*2, j].set_title(f"{marker} - Velocity {axis_name}",
                                                      fontweight='bold', color=self.style['colors']['primary_text'])
                                axes[i*2, j].set_facecolor(self.style['colors']['background_alt'])

                    # Plot acceleration components (X, Y, Z) with professional styling
                    acceleration_components = [
                        (accelerations_x, 'X'),
                        (accelerations_y, 'Y'),
                        (accelerations_z, 'Z')
                    ]

                    for j, (acc_component, axis_name) in enumerate(acceleration_components):
                        if acc_component:
                            # Create time axis for acceleration
                            acc_array = np.array(acc_component)
                            acc_start_frame = 2
                            acc_end_frame = acc_start_frame + len(acc_array)
                            acc_time = time_axis[acc_start_frame:acc_end_frame]

                            # Ensure both arrays have the same length
                            min_length = min(len(acc_time), len(acc_array))
                            acc_time = acc_time[:min_length]
                            acc_array = acc_array[:min_length]

                            if len(acc_time) > 0 and len(acc_array) > 0:
                                # Use professional biomechanical colors
                                color = self.style['colors'][axis_name]

                                axes[i*2+1, j].plot(acc_time, acc_array,
                                                  linewidth=self.style['plot']['linewidth'],
                                                  color=color,
                                                  alpha=self.style['plot']['alpha_main'],
                                                  label=f'Acceleration {axis_name}')

                                axes[i*2+1, j].set_title(f"{marker} - Acceleration {axis_name}",
                                                        fontweight='bold',
                                                        color=self.style['colors']['primary_text'])
                                axes[i*2+1, j].set_xlabel('Time (s)', fontweight='bold')
                                axes[i*2+1, j].set_ylabel(f'Acceleration {axis_name} (m/s²)', fontweight='bold')
                                axes[i*2+1, j].grid(True, alpha=self.style['plot']['alpha_grid'],
                                                  linestyle='-', linewidth=self.style['plot']['grid_linewidth'])
                                axes[i*2+1, j].set_facecolor(self.style['colors']['background_light'])

                                # Add professional statistics
                                valid_acc = acc_array[~np.isnan(acc_array)]
                                if len(valid_acc) > 0:
                                    mean_acc = np.mean(valid_acc)
                                    std_acc = np.std(valid_acc)

                                    axes[i*2+1, j].axhline(mean_acc,
                                                         color=self.style['colors']['mean'],
                                                         linestyle='--',
                                                         alpha=self.style['plot']['alpha_main'],
                                                         linewidth=self.style['plot']['linewidth_thin'],
                                                         label=f'Mean: {mean_acc:.3f} m/s²')

                                    # Add confidence interval
                                    axes[i*2+1, j].axhspan(mean_acc - std_acc, mean_acc + std_acc,
                                                         alpha=self.style['plot']['alpha_fill'], color=color)

                                    axes[i*2+1, j].legend(fontsize=self.style['fonts']['sizes']['tiny'],
                                                        frameon=True, fancybox=True, shadow=True)
                            else:
                                axes[i*2+1, j].text(0.5, 0.5, 'Insufficient Data',
                                                   ha='center', va='center',
                                                   transform=axes[i*2+1, j].transAxes,
                                                   fontsize=self.style['fonts']['sizes']['medium'],
                                                   color=self.style['colors']['muted_text'],
                                                   fontweight='bold')
                                axes[i*2+1, j].set_title(f"{marker} - Acceleration {axis_name}",
                                                        fontweight='bold', color=self.style['colors']['primary_text'])
                                axes[i*2+1, j].set_facecolor(self.style['colors']['background_alt'])

                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight', facecolor='white')
                plt.close(fig)

        # Create velocity and acceleration statistics tables
        self._create_velocity_acceleration_statistics_tables(pdf)

    def _create_velocity_acceleration_statistics_tables(self, pdf: PdfPages) -> None:
        """Create velocity and acceleration statistics tables using pre-calculated data."""
        # Get pre-calculated data
        velocity_data = self.analysis_results.get('velocity', {})
        acceleration_data = self.analysis_results.get('acceleration', {})

        # Prepare velocity statistics data
        velocity_stats_data = []
        for marker, data in velocity_data.items():
            row_data = [marker]

            # Calculate statistics for each axis
            for axis_name, velocities in [('X', data['x']), ('Y', data['y']), ('Z', data['z'])]:
                valid_velocities = np.array(velocities)[~np.isnan(velocities)]
                if len(valid_velocities) > 0:
                    mean_vel = np.mean(valid_velocities)
                    std_vel = np.std(valid_velocities)
                    min_vel = np.min(valid_velocities)
                    max_vel = np.max(valid_velocities)
                    range_vel = max_vel - min_vel

                    # Add formatted statistics for this axis
                    row_data.extend([
                        f"{mean_vel:.3f}",
                        f"{std_vel:.3f}",
                        f"{min_vel:.3f}",
                        f"{max_vel:.3f}",
                        f"{range_vel:.3f}"
                    ])
                else:
                    # No valid data for this axis
                    row_data.extend(['N/A', 'N/A', 'N/A', 'N/A', 'N/A'])

            velocity_stats_data.append(row_data)

        # Create velocity statistics table using helper function
        if velocity_stats_data:
            col_headers = ['Marker']
            for axis in ['X-axis', 'Y-axis', 'Z-axis']:
                col_headers.extend([
                    f'{axis}\nMean (m/s)',
                    f'{axis}\nStd (m/s)',
                    f'{axis}\nMin (m/s)',
                    f'{axis}\nMax (m/s)',
                    f'{axis}\nRange (m/s)'
                ])

            col_widths = [self.style['table']['col_width_marker']] + [self.style['table']['col_width_stat']] * 15
            self._create_statistics_table(pdf, velocity_stats_data, col_headers,
                                        title="", col_widths=col_widths)
        else:
            self._create_statistics_table(pdf, [], [], title="")

        # Prepare acceleration statistics data
        acceleration_stats_data = []
        for marker, data in acceleration_data.items():
            row_data = [marker]

            # Calculate statistics for each axis
            for axis_name, accelerations in [('X', data['x']), ('Y', data['y']), ('Z', data['z'])]:
                valid_accelerations = np.array(accelerations)[~np.isnan(accelerations)]
                if len(valid_accelerations) > 0:
                    mean_acc = np.mean(valid_accelerations)
                    std_acc = np.std(valid_accelerations)
                    min_acc = np.min(valid_accelerations)
                    max_acc = np.max(valid_accelerations)
                    range_acc = max_acc - min_acc

                    # Add formatted statistics for this axis
                    row_data.extend([
                        f"{mean_acc:.3f}",
                        f"{std_acc:.3f}",
                        f"{min_acc:.3f}",
                        f"{max_acc:.3f}",
                        f"{range_acc:.3f}"
                    ])
                else:
                    # No valid data for this axis
                    row_data.extend(['N/A', 'N/A', 'N/A', 'N/A', 'N/A'])

            acceleration_stats_data.append(row_data)

        # Create acceleration statistics table using helper function
        if acceleration_stats_data:
            col_headers = ['Marker']
            for axis in ['X-axis', 'Y-axis', 'Z-axis']:
                col_headers.extend([
                    f'{axis}\nMean (m/s²)',
                    f'{axis}\nStd (m/s²)',
                    f'{axis}\nMin (m/s²)',
                    f'{axis}\nMax (m/s²)',
                    f'{axis}\nRange (m/s²)'
                ])

            col_widths = [self.style['table']['col_width_marker']] + [self.style['table']['col_width_stat']] * 15
            self._create_statistics_table(pdf, acceleration_stats_data, col_headers,
                                        title="", col_widths=col_widths)
        else:
            self._create_statistics_table(pdf, [], [], title="")

    def _create_skeleton_analysis(self, pdf: PdfPages) -> None:
        """Create skeleton segment analysis pages using pre-calculated data."""
        # Get pre-calculated segment data
        segment_data = self.analysis_results.get('segments', {})

        if not segment_data:
            # Create a page indicating no segment data
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')
            ax.text(0.5, 0.5, 'No Segment Analysis Available\n\nSegment analysis requires specific marker combinations that are not present in the current dataset.',
                   ha='center', va='center', fontsize=self.style['fonts']['sizes']['xlarge'],
                   transform=ax.transAxes, color=self.style['colors']['muted_text'])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            return

        time_axis = np.arange(self.data_manager.num_frames) / self.fps

        # Create segment angle analysis pages (similar to joint analysis)
        segments_per_page = 6
        segment_names = list(segment_data.keys())
        segment_chunks = [segment_names[i:i+segments_per_page]
                         for i in range(0, len(segment_names), segments_per_page)]

        for chunk_idx, segment_chunk in enumerate(segment_chunks):
            rows = (len(segment_chunk) + 1) // 2
            with sns.axes_style("whitegrid"):
                fig, axes = plt.subplots(rows, 2, figsize=(15, 4*rows))
                if rows == 1:
                    axes = axes.reshape(1, -1)

                fig.suptitle(f'Segment Angles (Page {chunk_idx + 1})',
                            fontsize=self.style['fonts']['sizes']['xlarge'],
                            fontweight='bold', color=self.style['colors']['primary_text'])

                for i, segment_name in enumerate(segment_chunk):
                    row = i // 2
                    col = i % 2

                    data = segment_data[segment_name]
                    angles_x = np.array(data['angles_x'])
                    angles_y = np.array(data['angles_y'])
                    angles_z = np.array(data['angles_z'])

                    # Plot all three angle components with professional styling
                    axes[row, col].plot(time_axis, angles_x,
                                      linewidth=self.style['plot']['linewidth_thick'],
                                      alpha=self.style['plot']['alpha_main'],
                                      color=self.style['colors']['X'],
                                      label='X-axis')
                    axes[row, col].plot(time_axis, angles_y,
                                      linewidth=self.style['plot']['linewidth_thick'],
                                      alpha=self.style['plot']['alpha_main'],
                                      color=self.style['colors']['Y'],
                                      label='Y-axis')
                    axes[row, col].plot(time_axis, angles_z,
                                      linewidth=self.style['plot']['linewidth_thick'],
                                      alpha=self.style['plot']['alpha_main'],
                                      color=self.style['colors']['Z'],
                                      label='Z-axis')

                    axes[row, col].set_title(f"{segment_name} Segment Angles",
                                           fontweight='bold', color=self.style['colors']['primary_text'])
                    axes[row, col].set_xlabel('Time (s)', fontweight='bold')
                    axes[row, col].set_ylabel('Angle (degrees)', fontweight='bold')
                    axes[row, col].grid(True, alpha=self.style['plot']['alpha_grid'],
                                      linestyle='-', linewidth=self.style['plot']['grid_linewidth'])
                    axes[row, col].set_facecolor(self.style['colors']['background_light'])
                    axes[row, col].legend(fontsize=self.style['fonts']['sizes']['small'],
                                        frameon=True, fancybox=True, shadow=True)

                    # Add statistics text box
                    stats_text_lines = []
                    for axis_name, angles in [('X', angles_x), ('Y', angles_y), ('Z', angles_z)]:
                        valid_data = angles[~np.isnan(angles)]
                        if len(valid_data) > 0:
                            mean_angle = np.mean(valid_data)
                            std_angle = np.std(valid_data)
                            min_angle = np.min(valid_data)
                            max_angle = np.max(valid_data)
                            range_angle = max_angle - min_angle
                            stats_text_lines.append(f'{axis_name}: μ={mean_angle:.1f}°, σ={std_angle:.1f}°, R={range_angle:.1f}°')

                    if stats_text_lines:
                        stats_text = '\n'.join(stats_text_lines)
                        axes[row, col].text(0.02, 0.98, stats_text, transform=axes[row, col].transAxes,
                                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                                          fontsize=8)

                # Hide empty subplots
                for i in range(len(segment_chunk), rows * 2):
                    row = i // 2
                    col = i % 2
                    axes[row, col].axis('off')

                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight', facecolor='white')
                plt.close(fig)

        # Prepare segment angle statistics data
        stats_data = []
        for segment_name, data in segment_data.items():
            angles_x = np.array(data['angles_x'])
            angles_y = np.array(data['angles_y'])
            angles_z = np.array(data['angles_z'])

            row_data = [segment_name]

            # Calculate statistics for each axis
            for axis_name, angles in [('X', angles_x), ('Y', angles_y), ('Z', angles_z)]:
                valid_angles = angles[~np.isnan(angles)]
                if len(valid_angles) > 0:
                    mean_angle = np.mean(valid_angles)
                    std_angle = np.std(valid_angles)
                    min_angle = np.min(valid_angles)
                    max_angle = np.max(valid_angles)
                    range_angle = max_angle - min_angle

                    # Add formatted statistics for this axis
                    row_data.extend([
                        f"{mean_angle:.1f}°",
                        f"{std_angle:.1f}°",
                        f"{min_angle:.1f}°",
                        f"{max_angle:.1f}°",
                        f"{range_angle:.1f}°"
                    ])
                else:
                    # No valid data for this axis
                    row_data.extend(['N/A', 'N/A', 'N/A', 'N/A', 'N/A'])

            stats_data.append(row_data)

        # Create segment angle statistics table using helper function
        if stats_data:
            col_headers = ['Segment']
            for axis in ['X-axis', 'Y-axis', 'Z-axis']:
                col_headers.extend([
                    f'{axis}\nMean',
                    f'{axis}\nStd',
                    f'{axis}\nMin',
                    f'{axis}\nMax',
                    f'{axis}\nRange'
                ])

            col_widths = [self.style['table']['col_width_marker']] + [self.style['table']['col_width_stat']] * 15
            self._create_statistics_table(pdf, stats_data, col_headers,
                                        title="", col_widths=col_widths)
        else:
            self._create_statistics_table(pdf, [], [], title="")

    def _create_joint_analysis(self, pdf: PdfPages) -> None:
        """Create joint angle analysis pages using pre-calculated data."""
        # Get pre-calculated joint data
        joint_angles = self.analysis_results.get('joints', {})

        if not joint_angles:
            # Create a page indicating no joint data
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')
            ax.text(0.5, 0.5, 'No Joint Analysis Available\n\nJoint analysis requires specific marker combinations that are not present in the current dataset.',
                   ha='center', va='center', fontsize=self.style['fonts']['sizes']['xlarge'],
                   transform=ax.transAxes, color=self.style['colors']['muted_text'])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            return

        time_axis = np.arange(self.data_manager.num_frames) / self.fps

        # Create joint angle plots
        joints_per_page = 6
        joint_names = list(joint_angles.keys())
        joint_chunks = [joint_names[i:i+joints_per_page]
                       for i in range(0, len(joint_names), joints_per_page)]

        for chunk_idx, joint_chunk in enumerate(joint_chunks):
            rows = (len(joint_chunk) + 1) // 2
            with sns.axes_style("whitegrid"):
                fig, axes = plt.subplots(rows, 2, figsize=(12, 4*rows))
                if rows == 1:
                    axes = axes.reshape(1, -1)

                fig.suptitle(f'Joint Angles (Page {chunk_idx + 1})',
                            fontsize=self.style['fonts']['sizes']['xlarge'],
                            fontweight='bold', color=self.style['colors']['primary_text'])

                for i, joint_name in enumerate(joint_chunk):
                    row = i // 2
                    col = i % 2

                    angles = joint_angles[joint_name]
                    valid_angles = np.array(angles)

                    # Use professional biomechanical styling for joint angles
                    axes[row, col].plot(time_axis, angles,
                                      linewidth=self.style['plot']['linewidth_thick'],
                                      alpha=self.style['plot']['alpha_main'],
                                      color=self.style['colors']['accent'],
                                      label=f'{joint_name} Angle')

                    axes[row, col].set_title(f"{joint_name} Joint Angle",
                                           fontweight='bold', color=self.style['colors']['primary_text'])
                    axes[row, col].set_xlabel('Time (s)', fontweight='bold')
                    axes[row, col].set_ylabel('Angle (degrees)', fontweight='bold')
                    axes[row, col].grid(True, alpha=self.style['plot']['alpha_grid'],
                                      linestyle='-', linewidth=self.style['plot']['grid_linewidth'])
                    axes[row, col].set_facecolor(self.style['colors']['background_light'])

                    # Add professional statistics
                    valid_data = valid_angles[~np.isnan(valid_angles)]
                    if len(valid_data) > 0:
                        mean_angle = np.mean(valid_data)
                        std_angle = np.std(valid_data)
                        min_angle = np.min(valid_data)
                        max_angle = np.max(valid_data)

                        # Professional mean line
                        axes[row, col].axhline(mean_angle,
                                             color=self.style['colors']['mean'],
                                             linestyle='--', alpha=self.style['plot']['alpha_main'],
                                             linewidth=self.style['plot']['linewidth_thin'],
                                             label=f'Mean: {mean_angle:.1f}°')

                        # Add confidence interval
                        axes[row, col].axhspan(mean_angle - std_angle, mean_angle + std_angle,
                                             alpha=self.style['plot']['alpha_fill'],
                                             color=self.style['colors']['accent'])

                        axes[row, col].legend(fontsize=self.style['fonts']['sizes']['small'],
                                            frameon=True, fancybox=True, shadow=True)

                        # Professional statistics text box
                        stats_text = f'Mean: {mean_angle:.1f}°\nStd: {std_angle:.1f}°\nRange: {min_angle:.1f}° - {max_angle:.1f}°'
                        axes[row, col].text(0.02, 0.98, stats_text, transform=axes[row, col].transAxes,
                                          verticalalignment='top',
                                          bbox=dict(boxstyle='round,pad=0.5',
                                                  facecolor='white',
                                                  edgecolor=self.style['colors']['primary_text'],
                                                  alpha=0.9),
                                          fontsize=self.style['fonts']['sizes']['tiny'], fontweight='bold')

                # Hide empty subplots
                for i in range(len(joint_chunk), rows * 2):
                    row = i // 2
                    col = i % 2
                    axes[row, col].axis('off')

                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight', facecolor='white')
                plt.close(fig)

        # Prepare joint angle statistics data
        stats_data = []
        for joint_name, angles in joint_angles.items():
            valid_angles = np.array(angles)[~np.isnan(angles)]
            if len(valid_angles) > 0:
                mean_angle = np.mean(valid_angles)
                std_angle = np.std(valid_angles)
                min_angle = np.min(valid_angles)
                max_angle = np.max(valid_angles)
                range_angle = max_angle - min_angle
                stats_data.append([
                    joint_name,
                    f"{mean_angle:.1f}°",
                    f"{std_angle:.1f}°",
                    f"{min_angle:.1f}°",
                    f"{max_angle:.1f}°",
                    f"{range_angle:.1f}°"
                ])

        # Create joint angle statistics table using helper function
        col_labels = ['Joint', 'Mean', 'Std', 'Min', 'Max', 'Range']
        col_widths = [self.style['table']['col_width_joint_name']] + [self.style['table']['col_width_joint']] * 5
        self._create_statistics_table(pdf, stats_data, col_labels,
                                    title="Joint Angle Statistics",
                                    figsize=(11, 8.5), col_widths=col_widths)
