"""
This module provides filtering functionality for marker data in the TRCViewer application.
These functions were extracted from the main class to improve code organization.
"""

from tkinter import messagebox
from Pose2Sim.filtering import filter1d

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

        if filter_type == 'butterworth':
            try:
                cutoff_freq = float(self.filter_params['butterworth']['cut_off_frequency'].get())
                filter_order = int(self.filter_params['butterworth']['order'].get())
                
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
                    'butterworth': {
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

        if hasattr(self, 'edit_window') and self.edit_window:
            self.edit_window.focus()
            # update edit_button state
            if hasattr(self, 'edit_button'):
                self.edit_button.configure(fg_color="#555555")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during filtering: {str(e)}")
        print(f"Detailed error: {e}")
        import traceback
        traceback.print_exc()
