import customtkinter as ctk

class EditWindow(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        
        # Always display on top
        self.attributes('-topmost', True)
        
        # Window settings
        self.title("Edit Options")
        self.geometry("1230x120")  # 창 크기만 수정
        self.resizable(False, False)
        
        # Main frame
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Filter parameters frame
        self.filter_params_frame = ctk.CTkFrame(self.main_frame)
        self.filter_params_frame.pack(side='left', padx=5)
        
        # Filter type selection
        self.filter_type_frame = ctk.CTkFrame(self.filter_params_frame, fg_color="transparent")
        self.filter_type_frame.pack(side='left', padx=5)
        
        filter_type_label = ctk.CTkLabel(self.filter_type_frame, text="Filter:")
        filter_type_label.pack(side='left', padx=2)
        
        self.filter_type_combo = ctk.CTkComboBox(
            self.filter_type_frame,
            values=['kalman', 'butterworth', 'butterworth_on_speed', 'gaussian', 'LOESS', 'median'],
            variable=parent.filter_type_var,
            command=self.on_filter_type_change)
        self.filter_type_combo.pack(side='left')
        
        # Buttons frame
        self.button_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.button_frame.pack(side='left', padx=5)
        
        # Edit buttons
        button_style = {"width": 80, "fg_color": "#333333", "hover_color": "#444444"}
        
        buttons = [
            ("Filter", parent.filter_selected_data),
            ("Delete", parent.delete_selected_data),
            ("Interpolate", parent.interpolate_selected_data),
            ("Restore", parent.restore_original_data)
        ]
        
        for text, command in buttons:
            btn = ctk.CTkButton(
                self.button_frame,
                text=text,
                command=command,
                **button_style
            )
            btn.pack(side='left', padx=5)
        
        # Interpolation method frame
        self.interp_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.interp_frame.pack(side='left', padx=5)
        
        interp_label = ctk.CTkLabel(self.interp_frame, text="Interpolation:")
        interp_label.pack(side='left', padx=5)
        
        self.interp_combo = ctk.CTkComboBox(
            self.interp_frame,
            values=parent.interp_methods,
            variable=parent.interp_method_var,
            command=parent.on_interp_method_change)
        self.interp_combo.pack(side='left', padx=5)
        
        # Order frame
        self.order_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.order_frame.pack(side='left', padx=5)
        
        self.order_label = ctk.CTkLabel(self.order_frame, text="Order:")
        self.order_label.pack(side='left', padx=5)
        
        self.order_entry = ctk.CTkEntry(
            self.order_frame,
            textvariable=parent.order_var,
            width=50
        )
        self.order_entry.pack(side='left', padx=5)
        
        # Initialize state based on current interpolation method
        if parent.interp_method_var.get() not in ['polynomial', 'spline']:
            self.order_entry.configure(state='disabled')
            self.order_label.configure(state='disabled')
        
        # Create initial filter parameter UI
        self.current_params_frame = None
        self.on_filter_type_change(parent.filter_type_var.get())
        
        # Handle window close event
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def on_closing(self):
        self.parent.edit_window = None
        self.destroy()
    
    def on_filter_type_change(self, choice):
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

            cutoff_label = ctk.CTkLabel(self.current_params_frame, text="Cutoff Frequency (Hz):")
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

            smooth_label = ctk.CTkLabel(self.current_params_frame, text="Smoothing:")
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
