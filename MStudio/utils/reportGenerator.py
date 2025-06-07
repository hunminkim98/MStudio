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
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
from tkinter import filedialog, messagebox

from MStudio.utils.analysisMode import calculate_distance, calculate_angle, calculate_velocity, calculate_acceleration

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
        
        # Set matplotlib style for professional reports
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14,
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })
    
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
            
            # Generate report
            with PdfPages(output_path) as pdf:
                self._create_cover_page(pdf)
                self._create_data_overview(pdf)
                self._create_coordinate_analysis(pdf)
                self._create_velocity_acceleration_analysis(pdf)
                self._create_skeleton_analysis(pdf)
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
                fontsize=24, fontweight='bold', ha='center', va='center')
        
        # Subtitle
        ax.text(0.5, 0.7, 'Comprehensive Biomechanical Analysis', 
                fontsize=16, ha='center', va='center', style='italic')
        
        # Date and time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ax.text(0.5, 0.6, f'Generated on: {current_time}', 
                fontsize=12, ha='center', va='center')
        
        # Data information
        data_info = [
            f"Number of Markers: {len(self.data_manager.marker_names)}",
            f"Number of Frames: {self.data_manager.num_frames}",
            f"Frame Rate: {self.fps:.1f} fps",
            f"Duration: {self.data_manager.num_frames / self.fps:.2f} seconds"
        ]
        
        for i, info in enumerate(data_info):
            ax.text(0.5, 0.45 - i*0.05, info, 
                    fontsize=12, ha='center', va='center')
        
        # Skeleton model info
        ax.text(0.5, 0.2, f"Skeleton Model: {self.skeleton_model_name}",
                fontsize=12, ha='center', va='center', fontweight='bold')
        
        # Footer
        ax.text(0.5, 0.05, 'Generated by MStudio - Motion Capture Analysis Tool', 
                fontsize=10, ha='center', va='center', style='italic', alpha=0.7)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_data_overview(self, pdf: PdfPages) -> None:
        """Create data overview page."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Data Overview', fontsize=16, fontweight='bold')
        
        # 1. Data completeness heatmap
        data_completeness = ~self.data_manager.data.isnull()
        marker_completeness = []
        
        for marker in self.data_manager.marker_names:
            marker_cols = [f"{marker}_{axis}" for axis in ['X', 'Y', 'Z']]
            if all(col in self.data_manager.data.columns for col in marker_cols):
                completeness = data_completeness[marker_cols].all(axis=1).mean()
                marker_completeness.append(completeness)
            else:
                marker_completeness.append(0.0)
        
        ax1.barh(range(len(self.data_manager.marker_names)), marker_completeness)
        ax1.set_yticks(range(len(self.data_manager.marker_names)))
        ax1.set_yticklabels(self.data_manager.marker_names, fontsize=8)
        ax1.set_xlabel('Data Completeness')
        ax1.set_title('Data Completeness by Marker')
        ax1.set_xlim(0, 1)
        
        # 2. Frame-by-frame data availability
        frame_completeness = []
        for frame in range(min(1000, self.data_manager.num_frames)):  # Sample first 1000 frames
            frame_data = self.data_manager.data.iloc[frame]
            completeness = (~frame_data.isnull()).mean()
            frame_completeness.append(completeness)
        
        ax2.plot(frame_completeness)
        ax2.set_xlabel('Frame Number')
        ax2.set_ylabel('Data Completeness')
        ax2.set_title('Data Completeness Over Time')
        ax2.set_ylim(0, 1)
        
        # 3. Marker distribution in 3D space (first frame)
        if self.data_manager.num_frames > 0:
            positions = []
            for marker in self.data_manager.marker_names:
                try:
                    x = self.data_manager.data.loc[0, f"{marker}_X"]
                    y = self.data_manager.data.loc[0, f"{marker}_Y"] 
                    z = self.data_manager.data.loc[0, f"{marker}_Z"]
                    if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                        positions.append([x, y, z])
                except KeyError:
                    continue
            
            if positions:
                positions = np.array(positions)
                ax3.scatter(positions[:, 0], positions[:, 1], alpha=0.7)
                ax3.set_xlabel('X (m)')
                ax3.set_ylabel('Y (m)')
                ax3.set_title('Marker Positions (Frame 0, X-Y View)')
                ax3.grid(True, alpha=0.3)
        
        # 4. Basic statistics table
        ax4.axis('off')
        stats_data = [
            ['Total Markers', len(self.data_manager.marker_names)],
            ['Total Frames', self.data_manager.num_frames],
            ['Frame Rate', f"{self.fps:.1f} fps"],
            ['Duration', f"{self.data_manager.num_frames / self.fps:.2f} s"],
            ['Skeleton Pairs', len(self.state_manager.skeleton_pairs)],
            ['Data Size', f"{self.data_manager.data.shape[0]} × {self.data_manager.data.shape[1]}"]
        ]
        
        table = ax4.table(cellText=stats_data, 
                         colLabels=['Parameter', 'Value'],
                         cellLoc='left',
                         loc='center',
                         colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax4.set_title('Dataset Statistics', fontweight='bold', pad=20)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _create_coordinate_analysis(self, pdf: PdfPages) -> None:
        """Create coordinate analysis pages."""
        # Create pages for each marker's coordinate data
        markers_per_page = 6
        marker_chunks = [self.data_manager.marker_names[i:i+markers_per_page]
                        for i in range(0, len(self.data_manager.marker_names), markers_per_page)]

        for chunk_idx, marker_chunk in enumerate(marker_chunks):
            fig, axes = plt.subplots(len(marker_chunk), 3, figsize=(15, 2.5*len(marker_chunk)))
            if len(marker_chunk) == 1:
                axes = axes.reshape(1, -1)

            fig.suptitle(f'Marker Coordinates Analysis (Page {chunk_idx + 1})',
                        fontsize=16, fontweight='bold')

            time_axis = np.arange(self.data_manager.num_frames) / self.fps

            for i, marker in enumerate(marker_chunk):
                for j, axis in enumerate(['X', 'Y', 'Z']):
                    col_name = f"{marker}_{axis}"
                    if col_name in self.data_manager.data.columns:
                        data = self.data_manager.data[col_name].values
                        axes[i, j].plot(time_axis, data, linewidth=1.5, alpha=0.8)
                        axes[i, j].set_title(f"{marker} - {axis} Coordinate")
                        axes[i, j].set_xlabel('Time (s)')
                        axes[i, j].set_ylabel(f'{axis} Position (m)')
                        axes[i, j].grid(True, alpha=0.3)

                        # Add statistics
                        valid_data = data[~np.isnan(data)]
                        if len(valid_data) > 0:
                            mean_val = np.mean(valid_data)
                            std_val = np.std(valid_data)
                            axes[i, j].axhline(mean_val, color='red', linestyle='--', alpha=0.7,
                                             label=f'Mean: {mean_val:.3f}m')
                            axes[i, j].legend(fontsize=8)
                    else:
                        axes[i, j].text(0.5, 0.5, 'No Data', ha='center', va='center',
                                       transform=axes[i, j].transAxes)
                        axes[i, j].set_title(f"{marker} - {axis} Coordinate")

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    def _create_velocity_acceleration_analysis(self, pdf: PdfPages) -> None:
        """Create velocity and acceleration analysis pages."""
        # Calculate velocity and acceleration for all markers
        markers_per_page = 4
        marker_chunks = [self.data_manager.marker_names[i:i+markers_per_page]
                        for i in range(0, len(self.data_manager.marker_names), markers_per_page)]

        for chunk_idx, marker_chunk in enumerate(marker_chunks):
            fig, axes = plt.subplots(len(marker_chunk), 2, figsize=(12, 3*len(marker_chunk)))
            if len(marker_chunk) == 1:
                axes = axes.reshape(1, -1)

            fig.suptitle(f'Velocity & Acceleration Analysis (Page {chunk_idx + 1})',
                        fontsize=16, fontweight='bold')

            time_axis = np.arange(self.data_manager.num_frames) / self.fps

            for i, marker in enumerate(marker_chunk):
                # Calculate velocity and acceleration
                velocities = []
                accelerations = []

                # Calculate velocities for frames 1 to num_frames-2
                for frame in range(1, self.data_manager.num_frames - 1):
                    try:
                        pos_prev = self.data_manager.data.loc[frame-1, [f'{marker}_X', f'{marker}_Y', f'{marker}_Z']].values
                        pos_curr = self.data_manager.data.loc[frame, [f'{marker}_X', f'{marker}_Y', f'{marker}_Z']].values
                        pos_next = self.data_manager.data.loc[frame+1, [f'{marker}_X', f'{marker}_Y', f'{marker}_Z']].values

                        vel = calculate_velocity(pos_prev, pos_curr, pos_next, self.fps)
                        if vel is not None:
                            vel_magnitude = np.linalg.norm(vel)
                            velocities.append(vel_magnitude)
                        else:
                            velocities.append(np.nan)

                    except (KeyError, IndexError):
                        velocities.append(np.nan)

                # Calculate accelerations for frames 2 to num_frames-3 (need velocity at frame-1 and frame+1)
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
                                acc_magnitude = np.linalg.norm(acc)
                                accelerations.append(acc_magnitude)
                            else:
                                accelerations.append(np.nan)
                        else:
                            accelerations.append(np.nan)

                    except (KeyError, IndexError):
                        accelerations.append(np.nan)

                # Plot velocity
                if velocities:
                    # Make sure time axis and velocity data have the same length
                    vel_array = np.array(velocities)
                    # Create time axis that matches velocity data length
                    vel_start_frame = 1  # velocity starts from frame 1
                    vel_end_frame = vel_start_frame + len(vel_array)
                    vel_time = time_axis[vel_start_frame:vel_end_frame]

                    # Ensure both arrays have the same length
                    min_length = min(len(vel_time), len(vel_array))
                    vel_time = vel_time[:min_length]
                    vel_array = vel_array[:min_length]

                    if len(vel_time) > 0 and len(vel_array) > 0:
                        axes[i, 0].plot(vel_time, vel_array, linewidth=1.5, color='blue', alpha=0.8)
                        axes[i, 0].set_title(f"{marker} - Velocity Magnitude")
                        axes[i, 0].set_xlabel('Time (s)')
                        axes[i, 0].set_ylabel('Velocity (m/s)')
                        axes[i, 0].grid(True, alpha=0.3)

                        # Add statistics
                        valid_vel = vel_array[~np.isnan(vel_array)]
                        if len(valid_vel) > 0:
                            mean_vel = np.mean(valid_vel)
                            max_vel = np.max(valid_vel)
                            axes[i, 0].axhline(mean_vel, color='red', linestyle='--', alpha=0.7,
                                             label=f'Mean: {mean_vel:.3f} m/s')
                            axes[i, 0].legend(fontsize=8)
                    else:
                        axes[i, 0].text(0.5, 0.5, 'Insufficient data for velocity',
                                       ha='center', va='center', transform=axes[i, 0].transAxes)
                        axes[i, 0].set_title(f"{marker} - Velocity Magnitude")

                # Plot acceleration
                if accelerations:
                    # Make sure time axis and acceleration data have the same length
                    acc_array = np.array(accelerations)
                    # Create time axis that matches acceleration data length
                    acc_start_frame = 2  # acceleration starts from frame 2
                    acc_end_frame = acc_start_frame + len(acc_array)
                    acc_time = time_axis[acc_start_frame:acc_end_frame]

                    # Ensure both arrays have the same length
                    min_length = min(len(acc_time), len(acc_array))
                    acc_time = acc_time[:min_length]
                    acc_array = acc_array[:min_length]

                    if len(acc_time) > 0 and len(acc_array) > 0:
                        axes[i, 1].plot(acc_time, acc_array, linewidth=1.5, color='red', alpha=0.8)
                        axes[i, 1].set_title(f"{marker} - Acceleration Magnitude")
                        axes[i, 1].set_xlabel('Time (s)')
                        axes[i, 1].set_ylabel('Acceleration (m/s²)')
                        axes[i, 1].grid(True, alpha=0.3)

                        # Add statistics
                        valid_acc = acc_array[~np.isnan(acc_array)]
                        if len(valid_acc) > 0:
                            mean_acc = np.mean(valid_acc)
                            max_acc = np.max(valid_acc)
                            axes[i, 1].axhline(mean_acc, color='blue', linestyle='--', alpha=0.7,
                                             label=f'Mean: {mean_acc:.3f} m/s²')
                            axes[i, 1].legend(fontsize=8)
                    else:
                        axes[i, 1].text(0.5, 0.5, 'Insufficient data for acceleration',
                                       ha='center', va='center', transform=axes[i, 1].transAxes)
                        axes[i, 1].set_title(f"{marker} - Acceleration Magnitude")

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    def _create_skeleton_analysis(self, pdf: PdfPages) -> None:
        """Create skeleton segment analysis pages."""
        if not self.state_manager.skeleton_pairs:
            # Create a page indicating no skeleton data
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')
            ax.text(0.5, 0.5, 'No Skeleton Model Selected\n\nSkeleton analysis requires a skeleton model to be selected.',
                   ha='center', va='center', fontsize=16, transform=ax.transAxes)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            return

        # Segment length analysis
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Skeleton Segment Analysis', fontsize=16, fontweight='bold')

        # Calculate segment lengths over time
        segment_lengths = {}
        segment_angles_x = {}
        segment_angles_y = {}
        segment_angles_z = {}

        time_axis = np.arange(self.data_manager.num_frames) / self.fps

        for pair in self.state_manager.skeleton_pairs:
            marker1, marker2 = pair
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
                        segment_vector = pos2 - pos1

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

            segment_lengths[f"{marker1}-{marker2}"] = lengths
            segment_angles_x[f"{marker1}-{marker2}"] = angles_x
            segment_angles_y[f"{marker1}-{marker2}"] = angles_y
            segment_angles_z[f"{marker1}-{marker2}"] = angles_z

        # Plot segment lengths
        ax = axes[0, 0]
        for segment_name, lengths in segment_lengths.items():
            valid_lengths = np.array(lengths)
            if not np.all(np.isnan(valid_lengths)):
                ax.plot(time_axis, lengths, label=segment_name, alpha=0.7, linewidth=1)
        ax.set_title('Segment Lengths Over Time')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Length (m)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

        # Plot segment angles relative to X-axis
        ax = axes[0, 1]
        for segment_name, angles in segment_angles_x.items():
            valid_angles = np.array(angles)
            if not np.all(np.isnan(valid_angles)):
                ax.plot(time_axis, angles, label=segment_name, alpha=0.7, linewidth=1)
        ax.set_title('Segment Angles Relative to X-Axis')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (degrees)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

        # Plot segment angles relative to Y-axis
        ax = axes[1, 0]
        for segment_name, angles in segment_angles_y.items():
            valid_angles = np.array(angles)
            if not np.all(np.isnan(valid_angles)):
                ax.plot(time_axis, angles, label=segment_name, alpha=0.7, linewidth=1)
        ax.set_title('Segment Angles Relative to Y-Axis')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (degrees)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

        # Plot segment angles relative to Z-axis
        ax = axes[1, 1]
        for segment_name, angles in segment_angles_z.items():
            valid_angles = np.array(angles)
            if not np.all(np.isnan(valid_angles)):
                ax.plot(time_axis, angles, label=segment_name, alpha=0.7, linewidth=1)
        ax.set_title('Segment Angles Relative to Z-Axis')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (degrees)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Create segment statistics table
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        ax.set_title('Segment Length Statistics', fontsize=16, fontweight='bold', pad=20)

        # Prepare statistics data
        stats_data = []
        for segment_name, lengths in segment_lengths.items():
            valid_lengths = np.array(lengths)[~np.isnan(lengths)]
            if len(valid_lengths) > 0:
                mean_length = np.mean(valid_lengths)
                std_length = np.std(valid_lengths)
                min_length = np.min(valid_lengths)
                max_length = np.max(valid_lengths)
                stats_data.append([
                    segment_name,
                    f"{mean_length:.4f}",
                    f"{std_length:.4f}",
                    f"{min_length:.4f}",
                    f"{max_length:.4f}"
                ])

        if stats_data:
            table = ax.table(cellText=stats_data,
                           colLabels=['Segment', 'Mean (m)', 'Std (m)', 'Min (m)', 'Max (m)'],
                           cellLoc='center',
                           loc='center',
                           colWidths=[0.3, 0.175, 0.175, 0.175, 0.175])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
        else:
            ax.text(0.5, 0.5, 'No valid segment data available',
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _create_joint_analysis(self, pdf: PdfPages) -> None:
        """Create joint angle analysis pages."""
        # Define common joint triplets for different skeleton models
        joint_definitions = {
            'Knee_R': ['RHip', 'RKnee', 'RAnkle'],
            'Knee_L': ['LHip', 'LKnee', 'LAnkle'],
            'Elbow_R': ['RShoulder', 'RElbow', 'RWrist'],
            'Elbow_L': ['LShoulder', 'LElbow', 'LWrist'],
            'Hip_R': ['Neck', 'RHip', 'RKnee'],
            'Hip_L': ['Neck', 'LHip', 'LKnee'],
            'Shoulder_R': ['Neck', 'RShoulder', 'RElbow'],
            'Shoulder_L': ['Neck', 'LShoulder', 'LElbow'],
            'Ankle_R': ['RKnee', 'RAnkle', 'RBigToe'],
            'Ankle_L': ['LKnee', 'LAnkle', 'LBigToe']
        }

        # Filter joint definitions based on available markers
        available_joints = {}
        for joint_name, markers in joint_definitions.items():
            if all(f"{marker}_X" in self.data_manager.data.columns for marker in markers):
                available_joints[joint_name] = markers

        if not available_joints:
            # Create a page indicating no joint data
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')
            ax.text(0.5, 0.5, 'No Joint Analysis Available\n\nJoint analysis requires specific marker combinations that are not present in the current dataset.',
                   ha='center', va='center', fontsize=16, transform=ax.transAxes)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            return

        # Calculate joint angles over time
        joint_angles = {}
        time_axis = np.arange(self.data_manager.num_frames) / self.fps

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

        # Create joint angle plots
        joints_per_page = 6
        joint_names = list(available_joints.keys())
        joint_chunks = [joint_names[i:i+joints_per_page]
                       for i in range(0, len(joint_names), joints_per_page)]

        for chunk_idx, joint_chunk in enumerate(joint_chunks):
            rows = (len(joint_chunk) + 1) // 2
            fig, axes = plt.subplots(rows, 2, figsize=(12, 4*rows))
            if rows == 1:
                axes = axes.reshape(1, -1)

            fig.suptitle(f'Joint Angle Analysis (Page {chunk_idx + 1})',
                        fontsize=16, fontweight='bold')

            for i, joint_name in enumerate(joint_chunk):
                row = i // 2
                col = i % 2

                angles = joint_angles[joint_name]
                valid_angles = np.array(angles)

                axes[row, col].plot(time_axis, angles, linewidth=2, alpha=0.8, color='blue')
                axes[row, col].set_title(f"{joint_name} Joint Angle")
                axes[row, col].set_xlabel('Time (s)')
                axes[row, col].set_ylabel('Angle (degrees)')
                axes[row, col].grid(True, alpha=0.3)

                # Add statistics
                valid_data = valid_angles[~np.isnan(valid_angles)]
                if len(valid_data) > 0:
                    mean_angle = np.mean(valid_data)
                    std_angle = np.std(valid_data)
                    min_angle = np.min(valid_data)
                    max_angle = np.max(valid_data)

                    axes[row, col].axhline(mean_angle, color='red', linestyle='--', alpha=0.7,
                                         label=f'Mean: {mean_angle:.1f}°')
                    axes[row, col].legend(fontsize=8)

                    # Add text box with statistics
                    stats_text = f'Mean: {mean_angle:.1f}°\nStd: {std_angle:.1f}°\nRange: {min_angle:.1f}° - {max_angle:.1f}°'
                    axes[row, col].text(0.02, 0.98, stats_text, transform=axes[row, col].transAxes,
                                      verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                                      fontsize=8)

            # Hide empty subplots
            for i in range(len(joint_chunk), rows * 2):
                row = i // 2
                col = i % 2
                axes[row, col].axis('off')

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # Create joint angle statistics table
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        ax.set_title('Joint Angle Statistics', fontsize=16, fontweight='bold', pad=20)

        # Prepare statistics data
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

        if stats_data:
            table = ax.table(cellText=stats_data,
                           colLabels=['Joint', 'Mean', 'Std', 'Min', 'Max', 'Range'],
                           cellLoc='center',
                           loc='center',
                           colWidths=[0.2, 0.16, 0.16, 0.16, 0.16, 0.16])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
        else:
            ax.text(0.5, 0.5, 'No valid joint angle data available',
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
