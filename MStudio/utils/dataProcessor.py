"""
This module provides data processing functionality for marker data in the TRCViewer application.
"""
import numpy as np
import pandas as pd
from tkinter import messagebox
from MStudio.utils.filtering import *
import logging
from .filtering import filter1d
from scipy.spatial.transform import Rotation # Import Rotation

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
            filtered_series = filter1d(series.copy(), config_dict, filter_type, frame_rate)
            
            # Update data only within the selected range, casting to original dtype to avoid warnings
            original_dtype = self.data[col_name].dtype
            self.data.loc[start_frame:end_frame, col_name] = filtered_series.loc[start_frame:end_frame].astype(original_dtype)

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
            original_series = self.data[col_name] # No need for copy() if we update self.data directly

            # 1. Identify NaN indices *within* the selected range
            nan_indices_in_range = self.data.loc[start_frame:end_frame, col_name].isnull()
            target_indices = nan_indices_in_range[nan_indices_in_range].index

            if not target_indices.empty: # Proceed only if there are NaNs in the selected range
                interp_kwargs = {}
                if method in ['polynomial', 'spline']:
                    try:
                        # Ensure order is an integer for polynomial/spline
                        interp_kwargs['order'] = int(order)
                    except (ValueError, TypeError):
                        messagebox.showerror("Interpolation Error", f"Invalid order '{order}' for {method} interpolation. Please enter an integer.")
                        return # Stop processing for this coordinate
                
                try:
                    # 2. Perform full interpolation on the series to get potential values
                    fully_interpolated_series = original_series.interpolate(method=method, limit_direction='both', **interp_kwargs)

                    # 3. Selective update: Update the original data only at the target NaN indices
                    self.data.loc[target_indices, col_name] = fully_interpolated_series.loc[target_indices]
                    
                except Exception as e:
                    messagebox.showerror("Interpolation Error", f"Error interpolating {coord} with method '{method}': {e}")
                    logger.error(f"Interpolation failed for {col_name}, method={method}, kwargs={interp_kwargs}: {e}", exc_info=True)
                    # Optionally continue to the next coordinate or return
                    return # Stop if one coordinate fails

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
        reference_markers = list(self.pattern_markers)
        if len(reference_markers) < 3:
            messagebox.showerror("Error", "Please select at least 3 reference markers for robust pattern-based interpolation.")
            return

        start_frame = int(min(self.selection_data['start'], self.selection_data['end']))
        end_frame = int(max(self.selection_data['start'], self.selection_data['end']))
        logger.info(f"Frame range for interpolation: {start_frame} to {end_frame}")
        
        # --- Pre-extract data into NumPy arrays for performance ---
        try:
            target_cols = [f'{self.current_marker}_{c}' for c in 'XYZ']
            ref_cols = [f'{m}_{c}' for m in reference_markers for c in 'XYZ']
            
            # Get data as NumPy arrays. Use .copy() for target data as we'll modify it.
            target_data_np = self.data[target_cols].values.copy() 
            ref_data_np = self.data[ref_cols].values
            num_frames_total = len(target_data_np)
            num_ref_markers = len(reference_markers)
            
            # Get original dtypes from DataFrame to cast back later
            original_dtypes = {c: self.data[f'{self.current_marker}_{c}'].dtype for c in 'XYZ'}
            
        except KeyError as e:
             messagebox.showerror("Error", f"Marker data column not found: {e}")
             return
        except Exception as e:
             messagebox.showerror("Error", f"Failed to extract data into NumPy arrays: {e}")
             logger.error("NumPy data extraction failed: %s", e, exc_info=True)
             return
             
        # search for valid frames in entire dataset (using NumPy)
        logger.info("Searching for valid target marker data...")
        # Check rows where NONE of the XYZ coords are NaN
        valid_target_mask = ~np.isnan(target_data_np).any(axis=1)
        all_valid_frames = np.where(valid_target_mask)[0]
        
        if not all_valid_frames.size > 0:
            logger.error("Error: No valid data found for target marker in entire dataset")
            messagebox.showerror("Error", "No valid data found for target marker in entire dataset")
            return
            
        logger.info("Found %d valid frames for target marker", len(all_valid_frames))
        logger.info("Valid frames range: %d to %d", min(all_valid_frames), max(all_valid_frames))
        
        # find the closest valid frame
        closest_frame = min(all_valid_frames, 
                          key=lambda x: min(abs(x - start_frame), abs(x - end_frame)))
        logger.info("Using frame %d as reference frame", closest_frame)
        
        try:
            # Reshape ref_data_np row for this frame into (num_ref_markers, 3)
            P0 = ref_data_np[closest_frame].reshape(num_ref_markers, 3)
            if np.isnan(P0).any():
                messagebox.showerror("Error", f"Reference markers have NaN values in the chosen reference frame ({closest_frame}). Cannot proceed.")
                return
            p0_centroid = P0.mean(axis=0)
            P0_centered = P0 - p0_centroid
            
            # Target position from target_data_np
            target_pos_init = target_data_np[closest_frame]
            if np.isnan(target_pos_init).any():
                 messagebox.showerror("Error", f"Target marker has NaN values in the chosen reference frame ({closest_frame}). Cannot proceed.")
                 return
            target_rel = target_pos_init - p0_centroid
        except KeyError as e:
            messagebox.showerror("Error", f"Marker data not found in reference frame {closest_frame}: {e}")
            return
        except Exception as e:
            messagebox.showerror("Error", f"Error during initial state calculation: {e}")
            logger.error("Error calculating initial state: %s", e, exc_info=True)
            return
            
        logger.info("Initial state calculated.")
        logger.info(f"  P0 Centroid: {p0_centroid}")
        logger.info(f"  Target Relative Vector: {target_rel}")
        
        # Interpolate missing frames using Kabsch/Procrustes
        logger.info("Starting frame interpolation using rigid body transformation:")
        interpolated_count = 0
        # Iterate through the selected frame range
        for frame in range(start_frame, end_frame + 1):
            # Check if target marker needs interpolation (using NumPy array)
            if np.isnan(target_data_np[frame]).any():
                
                # --- Current Frame Calculation (using NumPy) ---
                try:
                    # Reshape ref_data_np row for the current frame
                    Q = ref_data_np[frame].reshape(num_ref_markers, 3)
                    # Check if *any* reference marker is NaN in the current frame
                    if np.isnan(Q).any():
                        logger.warning(f"Skipping frame {frame}: NaN value found in reference markers.")
                        continue # Skip this frame if reference data is missing
                    
                    q_centroid = Q.mean(axis=0)
                    Q_centered = Q - q_centroid
                except KeyError as e:
                    logger.warning(f"Skipping frame {frame}: Marker data not found: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing frame {frame}: {e}", exc_info=True)
                    continue
                    
                # --- Find Optimal Rotation and Transform Target ---
                try:
                    # Find rotation that aligns initial centered points (P0_centered) to current centered points (Q_centered)
                    R_opt, _ = Rotation.align_vectors(P0_centered, Q_centered)
                    
                    # Apply rotation to the initial relative target vector and add current centroid
                    target_est = R_opt.apply(target_rel) + q_centroid
                    
                except Exception as e:
                    logger.error(f"Error during alignment/transformation for frame {frame}: {e}", exc_info=True)
                    continue # Skip frame if alignment fails
                
                try:
                    target_data_np[frame, 0] = np.array(target_est[0]).astype(original_dtypes['X'])
                    target_data_np[frame, 1] = np.array(target_est[1]).astype(original_dtypes['Y'])
                    target_data_np[frame, 2] = np.array(target_est[2]).astype(original_dtypes['Z'])
                    interpolated_count += 1
                except Exception as e:
                     logger.error(f"Error updating NumPy array for frame {frame}: {e}", exc_info=True)

                if frame % 100 == 0: # Log every 100 frames
                    logger.debug(f"  Frame {frame}: Interpolated position: {target_est}")
            
            elif frame % 100 == 0: 
                logger.debug("Skipping frame %d (valid data exists)", frame)
        
        logger.info("Interpolation loop completed.")
        logger.info(f"Total frames processed in range: {end_frame - start_frame + 1}")
        logger.info(f"Total frames interpolated: {interpolated_count}")

        try:
             self.data[target_cols] = target_data_np
             logger.info("DataFrame updated with interpolated data.")
        except Exception as e:
             messagebox.showerror("Error", f"Failed to update DataFrame with results: {e}")
             logger.error("DataFrame update failed: %s", e, exc_info=True)

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
