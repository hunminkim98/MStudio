"""
This module provides interpolation functionality for marker data in the TRCViewer application.
These functions were extracted from the main class to improve code organization.
"""

import numpy as np
import pandas as pd
from tkinter import messagebox

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

    # Update edit button state if it exists
    if hasattr(self, 'edit_button'):
        self.edit_button.configure(fg_color="#555555")

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
            print("Error: No valid data found for target marker in entire dataset")
            messagebox.showerror("Error", "No valid data found for target marker in entire dataset")
            return
            
        print(f"Found {len(all_valid_frames)} valid frames for target marker")
        print(f"Valid frames range: {min(all_valid_frames)} to {max(all_valid_frames)}")
        
        # find the closest valid frame
        closest_frame = min(all_valid_frames, 
                          key=lambda x: min(abs(x - start_frame), abs(x - end_frame)))
        print(f"\nUsing frame {closest_frame} as reference frame")
        
        # Get initial positions using closest valid frame
        target_pos_init = np.array([
            self.data.loc[closest_frame, f'{self.current_marker}_X'],
            self.data.loc[closest_frame, f'{self.current_marker}_Y'],
            self.data.loc[closest_frame, f'{self.current_marker}_Z']
        ])
        print(f"Initial target position: {target_pos_init}")
        
        # Calculate initial distances and positions
        marker_distances = {}
        marker_positions_init = {}
        
        print("\nCalculating initial distances:")
        for ref_marker in reference_markers:
            ref_pos = np.array([
                self.data.loc[closest_frame, f'{ref_marker}_X'],
                self.data.loc[closest_frame, f'{ref_marker}_Y'],
                self.data.loc[closest_frame, f'{ref_marker}_Z']
            ])
            marker_positions_init[ref_marker] = ref_pos
            marker_distances[ref_marker] = np.linalg.norm(target_pos_init - ref_pos)
            print(f"{ref_marker}:")
            print(f"  Initial position: {ref_pos}")
            print(f"  Distance from target: {marker_distances[ref_marker]:.3f}")
        
        # Interpolate missing frames
        print("\nStarting frame interpolation:")
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
                    print(f"  Interpolated position: {interpolated_pos}")
            
            elif frame % 10 == 0:
                print(f"\nSkipping frame {frame} (valid data exists)")
        
        print(f"\nInterpolation completed successfully")
        print(f"Total frames interpolated: {interpolated_count}")
        
        # end pattern-based mode and initialize
        self.pattern_selection_mode = False
        self.pattern_markers.clear()
        
        # update UI
        self.update_plot()
        self.show_marker_plot(self.current_marker)
        
    except Exception as e:
        print(f"\nFATAL ERROR during interpolation: {e}")
        print("Traceback:")
        import traceback
        traceback.print_exc()
        messagebox.showerror("Interpolation Error", f"Error during pattern-based interpolation: {str(e)}")
    finally:
        # reset mouse events and UI state
        print("\nResetting mouse events and UI state")
        self.disconnect_mouse_events()
        self.connect_mouse_events()

def on_interp_method_change(self, choice):
    """Interpolation method change processing"""
    if choice != 'pattern-based':
        # initialize pattern markers
        self.pattern_markers.clear()
        self.pattern_selection_mode = False
        
        # update screen
        self.update_plot()
        self.canvas.draw_idle()
    else:
        # activate pattern selection mode when pattern-based is selected
        self.pattern_selection_mode = True
        messagebox.showinfo("Pattern Selection", 
            "Right-click markers to select/deselect them as reference patterns.\n"
            "Selected markers will be shown in red.")
    
    # change Order input field state only if EditWindow is open
    if hasattr(self, 'edit_window') and self.edit_window:
        if choice in ['polynomial', 'spline']:
            self.edit_window.order_entry.configure(state='normal')
            self.edit_window.order_label.configure(state='normal')
        else:
            self.edit_window.order_entry.configure(state='disabled')
            self.edit_window.order_label.configure(state='disabled')

def on_pattern_selection_confirm(self):
    """Process pattern selection confirmation"""
    try:
        print("\nPattern selection confirmation:")
        print(f"Selected markers: {self.pattern_markers}")
        
        if not self.pattern_markers:
            print("Error: No markers selected")
            messagebox.showwarning("No Selection", "Please select at least one pattern marker")
            return
        
        print("Starting interpolation")
        self.interpolate_selected_data()
        
        # pattern selection window is closed in interpolate_with_pattern
        
    except Exception as e:
        print(f"Error in pattern selection confirmation: {e}")
        import traceback
        traceback.print_exc()
        
        # initialize related variables if error occurs
        if hasattr(self, 'pattern_window'):
            delattr(self, 'pattern_window')
        self._selected_markers_list = None
