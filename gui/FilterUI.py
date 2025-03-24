"""
This module provides UI components for filtering functionality in the TRCViewer application.
These functions were extracted from the main class to improve code organization.
"""

import customtkinter as ctk

def on_filter_type_change(self, choice):
    """
    Updates the filter parameters UI based on the selected filter type.
    """
    if self.current_params_frame:
        self.current_params_frame.destroy()
    
    self.current_params_frame = ctk.CTkFrame(self.filter_params_frame)
    self.current_params_frame.pack(side='left', padx=5)
    
    if choice == 'butterworth':
        order_label = ctk.CTkLabel(self.current_params_frame, text="Order:")
        order_label.pack(side='left', padx=2)
        order_entry = ctk.CTkEntry(self.current_params_frame, 
                                 textvariable=self.parent.filter_params['butterworth']['order'],
                                 width=50)
        order_entry.pack(side='left', padx=2)

        cutoff_label = ctk.CTkLabel(self.current_params_frame, text="Cutoff (Hz):")
        cutoff_label.pack(side='left', padx=2)
        cutoff_entry = ctk.CTkEntry(self.current_params_frame,
                                  textvariable=self.parent.filter_params['butterworth']['cut_off_frequency'],
                                  width=50)
        cutoff_entry.pack(side='left', padx=2)

    elif choice == 'kalman':
        trust_label = ctk.CTkLabel(self.current_params_frame, text="Trust Ratio:")
        trust_label.pack(side='left', padx=2)
        trust_entry = ctk.CTkEntry(self.current_params_frame,
                                 textvariable=self.parent.filter_params['kalman']['trust_ratio'],
                                 width=50)
        trust_entry.pack(side='left', padx=2)

        smooth_label = ctk.CTkLabel(self.current_params_frame, text="Smooth:")
        smooth_label.pack(side='left', padx=2)
        smooth_entry = ctk.CTkEntry(self.current_params_frame,
                                  textvariable=self.parent.filter_params['kalman']['smooth'],
                                  width=50)
        smooth_entry.pack(side='left', padx=2)

    elif choice == 'gaussian':
        kernel_label = ctk.CTkLabel(self.current_params_frame, text="Sigma Kernel:")
        kernel_label.pack(side='left', padx=2)
        kernel_entry = ctk.CTkEntry(self.current_params_frame,
                                  textvariable=self.parent.filter_params['gaussian']['sigma_kernel'],
                                  width=50)
        kernel_entry.pack(side='left', padx=2)

    elif choice == 'LOESS':
        values_label = ctk.CTkLabel(self.current_params_frame, text="Values Used:")
        values_label.pack(side='left', padx=2)
        values_entry = ctk.CTkEntry(self.current_params_frame,
                                  textvariable=self.parent.filter_params['LOESS']['nb_values_used'],
                                  width=50)
        values_entry.pack(side='left', padx=2)

    elif choice == 'median':
        kernel_label = ctk.CTkLabel(self.current_params_frame, text="Kernel Size:")
        kernel_label.pack(side='left', padx=2)
        kernel_entry = ctk.CTkEntry(self.current_params_frame,
                                  textvariable=self.parent.filter_params['median']['kernel_size'],
                                  width=50)
        kernel_entry.pack(side='left', padx=2)
