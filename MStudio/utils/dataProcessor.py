"""
This module provides data processing functionality for marker data in the TRCViewer application.
"""
import numpy as np
import pandas as pd
from tkinter import messagebox
from Pose2Sim.filtering import filter1d
import logging

logger = logging.getLogger(__name__)

## AUTHORSHIP INFORMATION
__author__ = "HunMin Kim"
__copyright__ = ""
__credits__ = [""]
__license__ = ""
# from importlib.metadata import version
# __version__ = version('MStudio')
__maintainer__ = "HunMin Kim"
__email__ = "hunminkim98@gmail.com"
__status__ = "Development"


## Filtering
def filter_selected_data(self):
    """
    Apply the selected filter to the currently displayed marker data.
    If a specific range is selected, only that range is filtered.
    Otherwise, the entire data range is filtered.
    """
    try:
        # save current selection area
        current_selection = None
        if hasattr(self, 'selection_data'):
            current_selection = {
                'start': self.selection_data.get('start'),
                'end': self.selection_data.get('end')
            }

        # If no selection, use entire range
        if self.selection_data.get('start') is None or self.selection_data.get('end') is None:
            start_frame = 0
            end_frame = len(self.data) - 1
        else:
            start_frame = int(min(self.selection_data['start'], self.selection_data['end']))
            end_frame = int(max(self.selection_data['start'], self.selection_data['end']))

        # Store current view states
        view_states = []
        for ax in self.marker_axes:
            view_states.append({
                'xlim': ax.get_xlim(),
                'ylim': ax.get_ylim()
            })

        # Get filter parameters
        filter_type = self.filter_type_var.get()

        if filter_type == 'butterworth' or filter_type == 'butterworth_on_speed':
            try:
                cutoff_freq = float(self.filter_params[filter_type]['cut_off_frequency'].get())
                filter_order = int(self.filter_params[filter_type]['order'].get())
                
                if cutoff_freq <= 0:
                    messagebox.showerror("Input Error", "Hz must be greater than 0")
                    return
                if filter_order < 1:
                    messagebox.showerror("Input Error", "Order must be at least 1")
                    return
                    
            except ValueError:
                messagebox.showerror("Input Error", "Please enter valid numbers for Hz and Order")
                return

            # Create config dict for Pose2Sim
            config_dict = {
                'filtering': {
                    filter_type: {
                        'order': filter_order,
                        'cut_off_frequency': cutoff_freq
                    }
                }
            }
        else:
            config_dict = {
                'filtering': {
                    filter_type: {k: float(v.get()) for k, v in self.filter_params[filter_type].items()}
                }
            }

        # Get frame rate and apply filter
        frame_rate = float(self.fps_var.get())
        
        for coord in ['X', 'Y', 'Z']:
            col_name = f'{self.current_marker}_{coord}'
            series = self.data[col_name]
            
            # Apply Pose2Sim filter
            filtered_data = filter1d(series, config_dict, filter_type, frame_rate)
            
            # Update data
            self.data[col_name] = filtered_data

        # Update plots
        self.detect_outliers()
        self.show_marker_plot(self.current_marker)

        # Restore view states
        for ax, view_state in zip(self.marker_axes, view_states):
            ax.set_xlim(view_state['xlim'])
            ax.set_ylim(view_state['ylim'])

        # Restore selection if it existed
        if current_selection and current_selection['start'] is not None:
            self.selection_data['start'] = current_selection['start']
            self.selection_data['end'] = current_selection['end']
            self.highlight_selection()

        self.update_plot()

        # No need to focus on edit_window as it's integrated now
        # Just update the edit button if needed when not in edit mode
        if not self.is_editing and hasattr(self, 'edit_button') and self.edit_button and self.edit_button.winfo_exists():
            self.edit_button.configure(fg_color="#555555")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during filtering: {str(e)}")
        logger.error("Detailed error: %s", e, exc_info=True)

## Interpolation
def interpolate_selected_data(self):
    """
    Interpolate missing data points for the currently selected marker within a selected frame range.
    Supports various interpolation methods including pattern-based, linear, polynomial, and spline.
    """
    if self.selection_data['start'] is None or self.selection_data['end'] is None:
        return

    view_states = []
    for ax in self.marker_axes:
        view_states.append({
            'xlim': ax.get_xlim(),
            'ylim': ax.get_ylim()
        })

    current_selection = {
        'start': self.selection_data['start'],
        'end': self.selection_data['end']
    }

    start_frame = int(min(self.selection_data['start'], self.selection_data['end']))
    end_frame = int(max(self.selection_data['start'], self.selection_data['end']))

    method = self.interp_method_var.get()
    
    if method == 'pattern-based':
        self.interpolate_with_pattern()
    else:
        order = None
        if method in ['polynomial', 'spline']:
            try:
                order = self.order_var.get()
            except:
                messagebox.showerror("Error", "Please enter a valid order number")
                return

        for coord in ['X', 'Y', 'Z']:
            col_name = f'{self.current_marker}_{coord}'
            series = self.data[col_name]

            self.data.loc[start_frame:end_frame, col_name] = np.nan

            interp_kwargs = {}
            if order is not None:
                interp_kwargs['order'] = order

            try:
                self.data[col_name] = series.interpolate(method=method, **interp_kwargs)
            except Exception as e:
                messagebox.showerror("Interpolation Error", f"Error interpolating {coord} with method '{method}': {e}")
                return

    self.detect_outliers()
    self.show_marker_plot(self.current_marker)

    for ax, view_state in zip(self.marker_axes, view_states):
        ax.set_xlim(view_state['xlim'])
        ax.set_ylim(view_state['ylim'])

    self.update_plot()

    self.selection_data['start'] = current_selection['start']
    self.selection_data['end'] = current_selection['end']
    self.highlight_selection()

def interpolate_with_pattern(self):
    """
    Pattern-based interpolation using reference markers to interpolate target marker.
    This method uses spatial relationships between markers to estimate missing positions.
    """
    try:
        print(f"\nStarting pattern-based interpolation:")
        print(f"Target marker to interpolate: {self.current_marker}")
        print(f"Reference markers: {list(self.pattern_markers)}")
        
        reference_markers = list(self.pattern_markers)
        if not reference_markers:
            print("Error: No reference markers selected")
            messagebox.showerror("Error", "Please select reference markers")
            return

        start_frame = int(min(self.selection_data['start'], self.selection_data['end']))
        end_frame = int(max(self.selection_data['start'], self.selection_data['end']))
        print(f"Frame range for interpolation: {start_frame} to {end_frame}")
        
        # search for valid frames in entire dataset
        print("\nSearching for valid target marker data...")
        all_valid_frames = []
        for frame in range(len(self.data)):
            if not any(pd.isna(self.data.loc[frame, f'{self.current_marker}_{coord}']) 
                      for coord in ['X', 'Y', 'Z']):
                all_valid_frames.append(frame)
        
        if not all_valid_frames:
            logger.error("Error: No valid data found for target marker in entire dataset")
            messagebox.showerror("Error", "No valid data found for target marker in entire dataset")
            return
            
        logger.info("Found %d valid frames for target marker", len(all_valid_frames))
        logger.info("Valid frames range: %d to %d", min(all_valid_frames), max(all_valid_frames))
        
        # find the closest valid frame
        closest_frame = min(all_valid_frames, 
                          key=lambda x: min(abs(x - start_frame), abs(x - end_frame)))
        logger.info("Using frame %d as reference frame", closest_frame)
        
        # Get initial positions using closest valid frame
        target_pos_init = np.array([
            self.data.loc[closest_frame, f'{self.current_marker}_X'],
            self.data.loc[closest_frame, f'{self.current_marker}_Y'],
            self.data.loc[closest_frame, f'{self.current_marker}_Z']
        ])
        logger.info("Initial target position: %s", target_pos_init)
        
        # Calculate initial distances and positions
        marker_distances = {}
        marker_positions_init = {}
        
        logger.info("Calculating initial distances:")
        for ref_marker in reference_markers:
            ref_pos = np.array([
                self.data.loc[closest_frame, f'{ref_marker}_X'],
                self.data.loc[closest_frame, f'{ref_marker}_Y'],
                self.data.loc[closest_frame, f'{ref_marker}_Z']
            ])
            marker_positions_init[ref_marker] = ref_pos
            marker_distances[ref_marker] = np.linalg.norm(target_pos_init - ref_pos)
            logger.info("%s: Initial position: %s, Distance from target: %.3f", ref_marker, ref_pos, marker_distances[ref_marker])
        
        # Interpolate missing frames
        logger.info("Starting frame interpolation:")
        interpolated_count = 0
        frames = range(start_frame, end_frame + 1)
        for frame in frames:
            # Check if target marker needs interpolation
            if any(pd.isna(self.data.loc[frame, f'{self.current_marker}_{coord}']) 
                  for coord in ['X', 'Y', 'Z']):
                
                weighted_pos = np.zeros(3)
                total_weight = 0
                
                # Use each reference marker to estimate position
                for ref_marker in reference_markers:
                    current_ref_pos = np.array([
                        self.data.loc[frame, f'{ref_marker}_X'],
                        self.data.loc[frame, f'{ref_marker}_Y'],
                        self.data.loc[frame, f'{ref_marker}_Z']
                    ])
                    
                    # Calculate expected position based on initial distance
                    init_distance = marker_distances[ref_marker]
                    init_direction = target_pos_init - marker_positions_init[ref_marker]
                    init_unit_vector = init_direction / np.linalg.norm(init_direction)
                    
                    # Weight based on initial distance
                    weight = 1.0 / (init_distance + 1e-6)
                    weighted_pos += weight * (current_ref_pos + init_unit_vector * init_distance)
                    total_weight += weight
                
                # Calculate final interpolated position
                interpolated_pos = weighted_pos / total_weight
                
                # Update target marker position
                self.data.loc[frame, f'{self.current_marker}_X'] = interpolated_pos[0]
                self.data.loc[frame, f'{self.current_marker}_Y'] = interpolated_pos[1]
                self.data.loc[frame, f'{self.current_marker}_Z'] = interpolated_pos[2]
                
                interpolated_count += 1
                
                if frame % 10 == 0:
                    logger.info("  Interpolated position: %s", interpolated_pos)
            
            elif frame % 10 == 0:
                logger.info("Skipping frame %d (valid data exists)", frame)
        
        logger.info("Interpolation completed successfully")
        logger.info("Total frames interpolated: %d", interpolated_count)
        
        # end pattern-based mode and initialize
        self.pattern_selection_mode = False
        self.pattern_markers.clear()
        
        # update UI
        self.update_plot()
        self.show_marker_plot(self.current_marker)
        
    except Exception as e:
        logger.error("FATAL ERROR during interpolation: %s", e, exc_info=True)
        messagebox.showerror("Interpolation Error", f"Error during pattern-based interpolation: {str(e)}")
    finally:
        # reset mouse events and UI state
        logger.info("Resetting mouse events and UI state")
        self.disconnect_mouse_events()
        self.connect_mouse_events()

def on_pattern_selection_confirm(self):
    """Process pattern selection confirmation"""
    try:
        logger.info("Pattern selection confirmation:")
        logger.info("Selected markers: %s", self.pattern_markers)
        
        if not self.pattern_markers:
            logger.error("Error: No markers selected")
            messagebox.showwarning("No Selection", "Please select at least one pattern marker")
            return
        
        logger.info("Starting interpolation")
        self.interpolate_selected_data()
        
        # pattern selection window is closed in interpolate_with_pattern
        
    except Exception as e:
        logger.error("Error in pattern selection confirmation: %s", e, exc_info=True)
        
        # initialize related variables if error occurs
        if hasattr(self, 'pattern_window'):
            delattr(self, 'pattern_window')
        self._selected_markers_list = None
