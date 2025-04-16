"""
This module provides UI components for filtering functionality in the TRCViewer application.
These functions were extracted from the main class to improve code organization.
"""

import customtkinter as ctk

## AUTHORSHIP INFORMATION
__author__ = "HunMin Kim"
__copyright__ = ""
__credits__ = [""]
__license__ = ""
# from importlib.metadata import version
# __version__ = version('MEditor')
__maintainer__ = "HunMin Kim"
__email__ = "hunminkim98@gmail.com"
__status__ = "Development"

def build_filter_parameter_widgets(parent_frame: ctk.CTkFrame, filter_type: str, filter_params_vars: dict):
    """
    Builds the specific parameter entry widgets for the selected filter type
    into the provided parent frame.

    Args:
        parent_frame: The CTkFrame to build the widgets into.
        filter_type: The name of the selected filter (e.g., 'butterworth').
        filter_params_vars: The dictionary containing the StringVars for filter parameters.
                            Expected structure: {'filter_name': {'param_name': ctk.StringVar(), ...}, ...}
    """
    label_width = 60  # 너비를 80에서 60으로 줄임
    entry_width = 60  # Consistent entry width
    
    # Common styling parameters
    label_style = {"width": 40, "anchor": 'e'}  # 기본 라벨 너비를 조정
    label_pack = {"side": 'left', "padx": (0, 4)}  # 왼쪽 여백을 0으로 유지
    entry_pack = {"side": 'left', "padx": (0, 10)}

    if filter_type == 'butterworth' or filter_type == 'butterworth_on_speed':
        # Order 라벨에 대해 너비를 더 줄이고 패딩 조정
        order_label = ctk.CTkLabel(parent_frame, text="Order:", width=40, anchor='e')
        order_label.pack(side='left', padx=(0, 4))
        
        ctk.CTkEntry(parent_frame, textvariable=filter_params_vars[filter_type]['order'], width=entry_width).pack(**entry_pack)
        ctk.CTkLabel(parent_frame, text="Cutoff (Hz):", **label_style).pack(**label_pack)
        ctk.CTkEntry(parent_frame, textvariable=filter_params_vars[filter_type]['cut_off_frequency'], width=entry_width).pack(**entry_pack)
    elif filter_type == 'kalman':
        # 첫 번째 라벨 (Trust Ratio)을 왼쪽으로 이동
        trust_label = ctk.CTkLabel(parent_frame, text="Trust Ratio:", width=40, anchor='e')
        trust_label.pack(side='left', padx=(0, 4))
        
        ctk.CTkEntry(parent_frame, textvariable=filter_params_vars['kalman']['trust_ratio'], width=entry_width).pack(**entry_pack)
        ctk.CTkLabel(parent_frame, text="Smooth:", **label_style).pack(**label_pack)
        ctk.CTkEntry(parent_frame, textvariable=filter_params_vars['kalman']['smooth'], width=entry_width).pack(**entry_pack)
    elif filter_type == 'gaussian':
        # 첫 번째 라벨 (Sigma Kernel)을 왼쪽으로 이동
        kernel_label = ctk.CTkLabel(parent_frame, text="Sigma Kernel:", width=40, anchor='e')
        kernel_label.pack(side='left', padx=(0, 4))
        
        ctk.CTkEntry(parent_frame, textvariable=filter_params_vars['gaussian']['sigma_kernel'], width=entry_width).pack(**entry_pack)
    elif filter_type == 'LOESS':
        # 첫 번째 라벨 (Values Used)을 왼쪽으로 이동
        values_label = ctk.CTkLabel(parent_frame, text="Values Used:", width=40, anchor='e')
        values_label.pack(side='left', padx=(0, 4))
        
        ctk.CTkEntry(parent_frame, textvariable=filter_params_vars['LOESS']['nb_values_used'], width=entry_width).pack(**entry_pack)
    elif filter_type == 'median':
        # 첫 번째 라벨 (Kernel Size)을 왼쪽으로 이동
        kernel_label = ctk.CTkLabel(parent_frame, text="Kernel Size:", width=40, anchor='e')
        kernel_label.pack(side='left', padx=(0, 4))
        
        ctk.CTkEntry(parent_frame, textvariable=filter_params_vars['median']['kernel_size'], width=entry_width).pack(**entry_pack)
    # Add other filter types if needed following the pattern

def on_filter_type_change(self, choice):
    """
    Updates the filter parameters UI based on the selected filter type.
    This function likely belongs to a class like EditWindow.
    """
    # Destroy the old parameter frame if it exists
    if hasattr(self, 'current_params_frame') and self.current_params_frame:
        widgets_to_destroy = list(self.current_params_frame.winfo_children())
        for widget in widgets_to_destroy:
             widget.destroy()
        self.current_params_frame.destroy() # Destroy the frame itself

    # Create a new frame for the parameters
    # Assumes self.filter_params_frame exists on the parent object (e.g., EditWindow)
    self.current_params_frame = ctk.CTkFrame(self.filter_params_frame, fg_color="transparent") # Make frame transparent
    self.current_params_frame.pack(side='left', fill='x', expand=True, padx=5) # Allow expansion

    # Get current values from filter parameters
    current_values = {}
    if hasattr(self, 'parent') and hasattr(self.parent, 'filter_params') and choice in self.parent.filter_params:
        for param, var in self.parent.filter_params[choice].items():
            current_values[param] = var.get()
    
    # Recreate StringVar objects for the selected filter type to avoid widget reference issues
    if hasattr(self, 'parent') and hasattr(self.parent, 'filter_params') and choice in self.parent.filter_params:
        for param in self.parent.filter_params[choice]:
            # Save the current value
            value = current_values.get(param, self.parent.filter_params[choice][param].get())
            # Create a new StringVar with the same value
            self.parent.filter_params[choice][param] = ctk.StringVar(value=value)

    # Build the widgets using the reusable function
    # Assumes filter parameters are stored in self.parent.filter_params
    if hasattr(self, 'parent') and hasattr(self.parent, 'filter_params'):
        build_filter_parameter_widgets(self.current_params_frame, choice, self.parent.filter_params)
    else:
        print("Error: Could not find filter parameters (self.parent.filter_params).")
