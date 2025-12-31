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
            end_frame = len(self.data_manager.data) - 1
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
        
        current_marker = self.state_manager.selection_state.current_marker
        for coord in ['X', 'Y', 'Z']:
            col_name = f'{current_marker}_{coord}'
            series = self.data_manager.data[col_name]

            # Apply Pose2Sim filter
            filtered_series = filter1d(series.copy(), config_dict, filter_type, frame_rate)

            # Update data only within the selected range, casting to original dtype to avoid warnings
            original_dtype = self.data_manager.data[col_name].dtype

            # Handle case where filter1d returns numpy array instead of pandas Series
            if isinstance(filtered_series, np.ndarray):
                # Convert numpy array back to pandas Series with original index
                filtered_series = pd.Series(filtered_series, index=series.index)

            self.data_manager.data.loc[start_frame:end_frame, col_name] = filtered_series.loc[start_frame:end_frame].astype(original_dtype)

        # Update plots
        self.detect_outliers()
        self.show_marker_plot(current_marker)

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
        if not self.state_manager.editing_state.is_editing and hasattr(self, 'edit_button') and self.edit_button and self.edit_button.winfo_exists():
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

        current_marker = self.state_manager.selection_state.current_marker
        for coord in ['X', 'Y', 'Z']:
            col_name = f'{current_marker}_{coord}'
            original_series = self.data_manager.data[col_name] # No need for copy() if we update self.data directly

            # 1. Identify NaN indices *within* the selected range
            nan_indices_in_range = self.data_manager.data.loc[start_frame:end_frame, col_name].isnull()
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
                    self.data_manager.data.loc[target_indices, col_name] = fully_interpolated_series.loc[target_indices]
                    
                except Exception as e:
                    messagebox.showerror("Interpolation Error", f"Error interpolating {coord} with method '{method}': {e}")
                    logger.error(f"Interpolation failed for {col_name}, method={method}, kwargs={interp_kwargs}: {e}", exc_info=True)
                    # Optionally continue to the next coordinate or return
                    return # Stop if one coordinate fails

        self.detect_outliers()
        self.show_marker_plot(current_marker)

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
    Handles 1, 2, or 3+ reference markers.
    """
    try:
        reference_markers = list(self.state_manager.selection_state.pattern_markers)
        num_ref_markers = len(reference_markers)

        if num_ref_markers == 0:
            messagebox.showerror("Error", "Please select at least 1 reference marker for pattern-based interpolation.")
            return
        
        effective_mode = num_ref_markers # Will be 1, 2, or 3 (representing 3+)

        start_frame = int(min(self.selection_data['start'], self.selection_data['end']))
        end_frame = int(max(self.selection_data['start'], self.selection_data['end']))
        logger.info(f"Frame range for interpolation: {start_frame} to {end_frame}")
        
        # --- Pre-extract data into NumPy arrays for performance ---
        try:
            current_marker = self.state_manager.selection_state.current_marker
            target_cols = [f'{current_marker}_{c}' for c in 'XYZ']
            ref_cols = [f'{m}_{c}' for m in reference_markers for c in 'XYZ']

            target_data_np = self.data_manager.data[target_cols].values.copy()
            ref_data_np = self.data_manager.data[ref_cols].values
            num_frames_total = len(target_data_np)

            original_dtypes = {c: self.data_manager.data[f'{current_marker}_{c}'].dtype for c in 'XYZ'}
            
        except KeyError as e:
             messagebox.showerror("Error", f"Marker data column not found: {e}")
             return
        except Exception as e:
             messagebox.showerror("Error", f"Failed to extract data into NumPy arrays: {e}")
             logger.error("NumPy data extraction failed: %s", e, exc_info=True)
             return
             
        logger.info("Searching for a valid reference frame for target and all selected reference markers...")
        valid_target_mask = ~np.isnan(target_data_np).any(axis=1)
        
        # Reshape ref_data_np to check NaNs per marker: (num_frames, num_ref_markers, 3)
        ref_data_reshaped_for_nan_check = ref_data_np.reshape(num_frames_total, num_ref_markers, 3)
        valid_ref_mask_per_marker = ~np.isnan(ref_data_reshaped_for_nan_check).any(axis=2) # True if marker is valid
        valid_all_refs_mask = valid_ref_mask_per_marker.all(axis=1) # True if all ref markers are valid for that frame
        
        combined_valid_mask = valid_target_mask & valid_all_refs_mask
        all_valid_frames = np.where(combined_valid_mask)[0]
        
        if not all_valid_frames.size > 0:
            logger.error("Error: No frame found where target and ALL selected reference markers have valid data simultaneously.")
            messagebox.showerror("Error", "No frame found where target and ALL selected reference markers have valid data simultaneously.")
            return
            
        closest_frame = min(all_valid_frames, 
                          key=lambda x: min(abs(x - start_frame), abs(x - end_frame)))
        logger.info(f"Using frame {closest_frame} as reference for initial state calculation.")
        
        # --- Initial State Calculation (conditional) ---
        target_pos_init = target_data_np[closest_frame]
        ref_markers_at_closest_frame = ref_data_np[closest_frame].reshape(num_ref_markers, 3)

        if num_ref_markers == 1:
            _1marker_ref_pos_init = ref_markers_at_closest_frame[0]
            _1marker_offset_vector = target_pos_init - _1marker_ref_pos_init
            logger.info(f"1-Marker Mode Initialized: Offset Vector: {_1marker_offset_vector}")
            effective_mode = 1
        elif num_ref_markers == 2:
            _2marker_P1_init, _2marker_P2_init = ref_markers_at_closest_frame[0], ref_markers_at_closest_frame[1]
            _2marker_v_ref_init = _2marker_P2_init - _2marker_P1_init
            _2marker_norm_v_ref_init = np.linalg.norm(_2marker_v_ref_init)
            if np.isclose(_2marker_norm_v_ref_init, 0):
                logger.warning("2-Marker Mode: Reference markers are coincident at initial frame. Falling back to 1-marker logic using the first selected reference marker.")
                effective_mode = 1
                _1marker_ref_pos_init = _2marker_P1_init 
                _1marker_offset_vector = target_pos_init - _1marker_ref_pos_init
                logger.info(f"Fallback to 1-Marker Mode Initialized: Offset Vector: {_1marker_offset_vector}")
            else:
                effective_mode = 2
                _2marker_v_target_rel_to_P1_init = target_pos_init - _2marker_P1_init
                logger.info(f"2-Marker Mode Initialized: v_ref_init={_2marker_v_ref_init}, v_target_rel_to_P1_init={_2marker_v_target_rel_to_P1_init}, norm_v_ref_init={_2marker_norm_v_ref_init}")
        else: # num_ref_markers >= 3
            effective_mode = 3 # Representing 3+
            _3plus_P0_init_coords = ref_markers_at_closest_frame
            _3plus_p0_centroid = _3plus_P0_init_coords.mean(axis=0)
            _3plus_P0_centered = _3plus_P0_init_coords - _3plus_p0_centroid
            _3plus_target_rel_to_centroid = target_pos_init - _3plus_p0_centroid
            logger.info(f">=3-Marker Mode Initialized: Centroid={_3plus_p0_centroid}, Target Relative to Centroid={_3plus_target_rel_to_centroid}")
            
        logger.info("Starting frame interpolation using effective_mode: %d", effective_mode)
        interpolated_count = 0
        
        for frame in range(start_frame, end_frame + 1):
            if np.isnan(target_data_np[frame]).any(): # If target marker needs interpolation
                current_ref_positions_flat = ref_data_np[frame]
                
                # Check NaNs for *actually used* reference markers based on original num_ref_markers
                if np.isnan(current_ref_positions_flat[:num_ref_markers*3]).any():
                     logger.warning(f"Skipping frame {frame}: NaN in current data for one of the {num_ref_markers} originally selected reference markers.")
                     continue
                
                all_current_ref_positions = current_ref_positions_flat.reshape(num_ref_markers, 3)
                target_est = None

                if effective_mode == 1:
                    current_ref_marker_pos = all_current_ref_positions[0] # Uses the first selected marker
                    target_est = current_ref_marker_pos + _1marker_offset_vector
                elif effective_mode == 2:
                    P1_curr, P2_curr = all_current_ref_positions[0], all_current_ref_positions[1]
                    v_ref_curr = P2_curr - P1_curr
                    norm_v_ref_curr = np.linalg.norm(v_ref_curr)

                    if np.isclose(norm_v_ref_curr, 0):
                        logger.warning(f"Frame {frame}: In 2-Marker mode, current reference markers are coincident. Skipping interpolation for this frame.")
                        continue
                    if np.isclose(_2marker_norm_v_ref_init, 0): # Should have been caught by fallback, but as safeguard
                        logger.error(f"Frame {frame}: In 2-Marker mode, initial reference vector norm is zero. This should not happen. Skipping.")
                        continue

                    scale = norm_v_ref_curr / _2marker_norm_v_ref_init 
                    try:
                        # Reshape for align_vectors which expects (N,3)
                        R_opt, _ = Rotation.align_vectors(_2marker_v_ref_init.reshape(1,-1), v_ref_curr.reshape(1,-1))
                        target_est = P1_curr + R_opt.apply(scale * _2marker_v_target_rel_to_P1_init)
                    except Exception as e:
                        logger.error(f"Error during 2-marker alignment/transformation for frame {frame}: {e}", exc_info=True)
                        continue
                elif effective_mode >= 3: # Handles 3+ markers
                    Q_curr_coords = all_current_ref_positions
                    q_centroid = Q_curr_coords.mean(axis=0)
                    Q_centered = Q_curr_coords - q_centroid
                    try:
                        R_opt, _ = Rotation.align_vectors(_3plus_P0_centered, Q_centered)
                        target_est = R_opt.apply(_3plus_target_rel_to_centroid) + q_centroid
                    except Exception as e:
                        logger.error(f"Error during >=3-marker alignment/transformation for frame {frame}: {e}", exc_info=True)
                        continue
                
                if target_est is not None:
                    try:
                        target_data_np[frame, 0] = np.array(target_est[0]).astype(original_dtypes['X'])
                        target_data_np[frame, 1] = np.array(target_est[1]).astype(original_dtypes['Y'])
                        target_data_np[frame, 2] = np.array(target_est[2]).astype(original_dtypes['Z'])
                        interpolated_count += 1
                    except Exception as e:
                         logger.error(f"Error updating NumPy array for frame {frame} with estimated value: {e}", exc_info=True)

                if frame % 100 == 0 and target_est is not None:
                    logger.debug(f"  Frame {frame}: Interpolated position: {target_est}")
            
            elif frame % 100 == 0: 
                logger.debug(f"Skipping frame {frame} (target data already valid)")
        
        logger.info("Interpolation loop completed.")
        logger.info(f"Total frames processed in range: {end_frame - start_frame + 1}")
        logger.info(f"Total frames interpolated with new values: {interpolated_count}")

        try:
             self.data_manager.data[target_cols] = target_data_np
             logger.info("DataFrame updated with interpolated data.")
        except Exception as e:
             messagebox.showerror("Error", f"Failed to update DataFrame with results: {e}")
             logger.error("DataFrame update failed: %s", e, exc_info=True)

        # end pattern-based mode and initialize
        self.state_manager.editing_state.pattern_selection_mode = False
        self.state_manager.selection_state.pattern_markers.clear()
        
        # update UI
        self.update_plot()
        # Ensure marker plot is updated only if a marker is selected and graph is visible
        current_marker = self.state_manager.selection_state.current_marker
        if current_marker and hasattr(self, 'graph_frame') and self.graph_frame.winfo_ismapped():
            self.show_marker_plot(current_marker)
        
    except Exception as e:
        logger.error("FATAL ERROR during pattern-based interpolation: %s", e, exc_info=True)
        messagebox.showerror("Interpolation Error", f"An unexpected error occurred during pattern-based interpolation: {str(e)}")
    finally:
        # reset mouse events and UI state (optional, depending on desired UX)
        logger.info("Pattern-based interpolation process finished.")
        # self.disconnect_mouse_events() # Consider if this is always needed
        # self.connect_mouse_events()    # Or if it disrupts other interactions

def on_pattern_selection_confirm(self):
    """Process pattern selection confirmation"""
    try:
        logger.info("Pattern selection confirmation:")
        logger.info("Selected markers: %s", self.state_manager.selection_state.pattern_markers)

        if not self.state_manager.selection_state.pattern_markers:
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

