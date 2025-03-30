import customtkinter as ctk

class EditWindow(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        
        # Always display on top
        self.attributes('-topmost', True)
        
        # Window settings
        self.title("Edit Options")
        self.geometry("1230x130")  # Slightly increased height for better spacing
        self.resizable(False, False)
        
        # Main frame with grid layout
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill='both', expand=True, padx=15, pady=15)
        
        # Set up main grid - fixed column widths for better alignment
        self.main_frame.grid_columnconfigure(0, weight=0, minsize=100)  # Label column
        self.main_frame.grid_columnconfigure(1, weight=0, minsize=160)  # ComboBox column
        self.main_frame.grid_columnconfigure(2, weight=0, minsize=80)   # Label column
        self.main_frame.grid_columnconfigure(3, weight=0, minsize=70)   # Entry column
        self.main_frame.grid_columnconfigure(4, weight=0, minsize=100)  # Label column
        self.main_frame.grid_columnconfigure(5, weight=0, minsize=70)   # Entry column
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)
        
        # Filter type section
        filter_label = ctk.CTkLabel(self.main_frame, text="Filter:", anchor="e")
        filter_label.grid(row=0, column=0, padx=(10, 5), pady=5, sticky="e")
        
        self.filter_type_combo = ctk.CTkComboBox(
            self.main_frame,
            values=['kalman', 'butterworth', 'butterworth_on_speed', 'gaussian', 'LOESS', 'median'],
            variable=parent.filter_type_var,
            width=150,
            command=self.on_filter_type_change)
        self.filter_type_combo.grid(row=0, column=1, padx=(0, 10), pady=5, sticky="w")
        
        # Order section
        order_label = ctk.CTkLabel(self.main_frame, text="Order:")
        order_label.grid(row=0, column=2, padx=(20, 5), pady=5, sticky="e")
        
        order_entry = ctk.CTkEntry(
            self.main_frame,
            textvariable=parent.order_var,
            width=50)
        order_entry.grid(row=0, column=3, padx=(0, 20), pady=5, sticky="w")
        
        # Cutoff section (only visible for butterworth)
        self.cutoff_label = ctk.CTkLabel(self.main_frame, text="Cutoff (Hz):")
        self.cutoff_label.grid(row=0, column=4, padx=(20, 5), pady=5, sticky="e")
        
        self.cutoff_entry = ctk.CTkEntry(
            self.main_frame,
            textvariable=parent.filter_params['butterworth']['cut_off_frequency'],
            width=50)
        self.cutoff_entry.grid(row=0, column=5, padx=(0, 10), pady=5, sticky="w")
        
        # Interpolation section
        interp_label = ctk.CTkLabel(self.main_frame, text="Interpolation:")
        interp_label.grid(row=1, column=0, padx=(10, 5), pady=5, sticky="e")
        
        self.interp_combo = ctk.CTkComboBox(
            self.main_frame,
            values=parent.interp_methods,
            variable=parent.interp_method_var,
            width=150,
            command=parent.on_interp_method_change)
        self.interp_combo.grid(row=1, column=1, padx=(0, 20), pady=5, sticky="w")
        
        # Order for interpolation
        self.order_interp_label = ctk.CTkLabel(self.main_frame, text="Order:")
        self.order_interp_label.grid(row=1, column=2, padx=(20, 5), pady=5, sticky="e")
        
        self.order_entry = ctk.CTkEntry(
            self.main_frame,
            textvariable=parent.order_var,
            width=50)
        self.order_entry.grid(row=1, column=3, padx=(0, 20), pady=5, sticky="w")
        
        # Button frame
        button_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        button_frame.grid(row=2, column=0, columnspan=6, pady=10, sticky="ew")
        
        # Configure button spacing
        for i in range(5):
            button_frame.grid_columnconfigure(i, weight=1, uniform="buttons")
        
        # Buttons
        button_style = {"width": 100, "fg_color": "#333333", "hover_color": "#444444"}
        
        buttons = [
            ("Filter", parent.filter_selected_data),
            ("Delete", parent.delete_selected_data),
            ("Interpolate", parent.interpolate_selected_data),
            ("Restore", parent.restore_original_data),
            ("Done", self.on_closing)
        ]
        
        for i, (text, command) in enumerate(buttons):
            btn = ctk.CTkButton(
                button_frame,
                text=text,
                command=command,
                **button_style
            )
            btn.grid(row=0, column=i, padx=10, pady=5)
        
        # Initialize state based on current interpolation method
        if parent.interp_method_var.get() not in ['polynomial', 'spline']:
            self.order_interp_label.configure(state='disabled')
            self.order_entry.configure(state='disabled')
        
        # Initialize filter params based on current filter type
        self.update_filter_params(parent.filter_type_var.get())
        
        # Handle window close event
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def on_closing(self):
        self.parent.edit_window = None
        self.destroy()
    
    def on_filter_type_change(self, choice):
        self.update_filter_params(choice)
    
    def update_filter_params(self, choice):
        # Hide all filter-specific parameters
        self.cutoff_label.grid_remove()
        self.cutoff_entry.grid_remove()
        
        # Show parameters based on filter type
        if choice == 'butterworth' or choice == 'butterworth_on_speed':
            self.cutoff_label.grid()
            self.cutoff_entry.grid()
            # Update textvariable to the right filter
            if choice == 'butterworth':
                self.cutoff_entry.configure(textvariable=self.parent.filter_params['butterworth']['cut_off_frequency'])
            else:
                self.cutoff_entry.configure(textvariable=self.parent.filter_params['butterworth_on_speed']['cut_off_frequency'])
        
        # Update the order entry's textvariable
        if choice in self.parent.filter_params and 'order' in self.parent.filter_params[choice]:
            self.main_frame.children['!ctkentry'].configure(
                textvariable=self.parent.filter_params[choice]['order']
            )
