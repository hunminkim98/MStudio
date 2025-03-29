"""
This module provides UI components for filtering functionality in the TRCViewer application.
These functions were extracted from the main class to improve code organization.
"""

import customtkinter as ctk

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
    label_width = 80 # Consistent label width
    entry_width = 60 # Consistent entry width

    if filter_type == 'butterworth':
        ctk.CTkLabel(parent_frame, text="Order:", width=label_width, anchor='w').pack(side='left', padx=(5,2))
        ctk.CTkEntry(parent_frame, textvariable=filter_params_vars['butterworth']['order'], width=entry_width).pack(side='left', padx=(0,5))
        ctk.CTkLabel(parent_frame, text="Cutoff (Hz):", width=label_width, anchor='w').pack(side='left', padx=(5,2))
        ctk.CTkEntry(parent_frame, textvariable=filter_params_vars['butterworth']['cut_off_frequency'], width=entry_width).pack(side='left', padx=(0,5))
    elif filter_type == 'kalman':
        ctk.CTkLabel(parent_frame, text="Trust Ratio:", width=label_width, anchor='w').pack(side='left', padx=(5,2))
        ctk.CTkEntry(parent_frame, textvariable=filter_params_vars['kalman']['trust_ratio'], width=entry_width).pack(side='left', padx=(0,5))
        ctk.CTkLabel(parent_frame, text="Smooth:", width=label_width, anchor='w').pack(side='left', padx=(5,2))
        ctk.CTkEntry(parent_frame, textvariable=filter_params_vars['kalman']['smooth'], width=entry_width).pack(side='left', padx=(0,5))
    elif filter_type == 'gaussian':
        ctk.CTkLabel(parent_frame, text="Sigma Kernel:", width=label_width, anchor='w').pack(side='left', padx=(5,2))
        ctk.CTkEntry(parent_frame, textvariable=filter_params_vars['gaussian']['sigma_kernel'], width=entry_width).pack(side='left', padx=(0,5))
    elif filter_type == 'LOESS':
        ctk.CTkLabel(parent_frame, text="Values Used:", width=label_width, anchor='w').pack(side='left', padx=(5,2))
        ctk.CTkEntry(parent_frame, textvariable=filter_params_vars['LOESS']['nb_values_used'], width=entry_width).pack(side='left', padx=(0,5))
    elif filter_type == 'median':
        ctk.CTkLabel(parent_frame, text="Kernel Size:", width=label_width, anchor='w').pack(side='left', padx=(5,2))
        ctk.CTkEntry(parent_frame, textvariable=filter_params_vars['median']['kernel_size'], width=entry_width).pack(side='left', padx=(0,5))
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

    # Build the widgets using the reusable function
    # Assumes filter parameters are stored in self.parent.filter_params
    if hasattr(self, 'parent') and hasattr(self.parent, 'filter_params'):
        build_filter_parameter_widgets(self.current_params_frame, choice, self.parent.filter_params)
    else:
        print("Error: Could not find filter parameters (self.parent.filter_params).")
