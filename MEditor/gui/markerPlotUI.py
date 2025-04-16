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

def build_marker_plot_buttons(viewer, parent_frame):
    """Helper method to build buttons for the marker plot, adjusting height for edit mode."""
    # Destroy existing button frame contents if they exist
    for widget in parent_frame.winfo_children():
        widget.destroy()

    button_style = {
        "width": 80, "height": 28, "fg_color": "#3B3B3B", "hover_color": "#4B4B4B",
        "text_color": "#FFFFFF", "corner_radius": 6, "border_width": 1, "border_color": "#555555"
    }

    if viewer.is_editing:
        # --- Build Edit Controls (Multi-row) ---
        parent_frame.configure(height=90) # Increase height for edit mode with interpolation

        # Main container for edit controls
        viewer.edit_controls_frame = ctk.CTkFrame(parent_frame, fg_color="transparent")
        viewer.edit_controls_frame.pack(fill='both', expand=True, padx=5, pady=2)

        # Top Row Frame
        top_row_frame = ctk.CTkFrame(viewer.edit_controls_frame, fg_color="transparent")
        top_row_frame.pack(side='top', fill='x', pady=(0, 5)) # Add padding below

        # 1. Filter Type Frame (in top row)
        filter_type_frame = ctk.CTkFrame(top_row_frame, fg_color="transparent")
        filter_type_frame.pack(side='left', padx=(0, 10)) # Add padding to the right

        # Filter label with increased padding
        filter_label = ctk.CTkLabel(filter_type_frame, text="Filter:", width=50, anchor="e")
        filter_label.pack(side='left', padx=(3, 8))

        viewer.filter_type_combo = ctk.CTkComboBox(
            filter_type_frame,
            width=150, # Adjust width if needed
            values=['kalman', 'butterworth', 'butterworth_on_speed', 'gaussian', 'LOESS', 'median'],
            variable=viewer.filter_type_var,
            command=viewer._on_filter_type_change_in_panel
        )
        viewer.filter_type_combo.pack(side='left', padx=(42, 0))

        # 2. Dynamic Filter Parameters Container (in top row)
        viewer.filter_params_container = ctk.CTkFrame(top_row_frame, fg_color="transparent")
        viewer.filter_params_container.pack(side='left', fill='x', expand=True)
        # Initial population of parameters based on current filter type
        viewer._build_filter_param_widgets(viewer.filter_type_var.get())

        # Middle Row Frame for Interpolation
        middle_row_frame = ctk.CTkFrame(viewer.edit_controls_frame, fg_color="transparent")
        middle_row_frame.pack(side='top', fill='x', pady=(0, 5))

        # Interpolation Method Frame
        interp_frame = ctk.CTkFrame(middle_row_frame, fg_color="transparent")
        interp_frame.pack(side='left', padx=(0, 10))

        # Interpolation label with consistent styling
        interp_label = ctk.CTkLabel(interp_frame, text="Interpolation:", width=90, anchor="e")
        interp_label.pack(side='left', padx=(5, 8))

        # Interpolation ComboBox
        viewer.interp_method_combo = ctk.CTkComboBox(
            interp_frame,
            width=150,
            values=viewer.interp_methods,
            variable=viewer.interp_method_var,
            command=viewer._on_interp_method_change_in_panel
        )
        viewer.interp_method_combo.pack(side='left')

        # Interpolation Order Frame
        interp_order_frame = ctk.CTkFrame(middle_row_frame, fg_color="transparent")
        interp_order_frame.pack(side='left')

        # Order Label and Entry - consistent styling with other labels
        viewer.interp_order_label = ctk.CTkLabel(interp_order_frame, text="Order:", width=40, anchor='e')
        viewer.interp_order_label.pack(side='left', padx=(0, 4))

        viewer.interp_order_entry = ctk.CTkEntry(interp_order_frame, textvariable=viewer.order_var, width=60)
        viewer.interp_order_entry.pack(side='left', padx=(0, 0))

        # Set initial state based on current method
        current_method = viewer.interp_method_var.get()
        if current_method not in ['polynomial', 'spline']:
            viewer.interp_order_label.configure(state='disabled')
            viewer.interp_order_entry.configure(state='disabled')

        # Bottom Row Frame
        bottom_row_frame = ctk.CTkFrame(viewer.edit_controls_frame, fg_color="transparent")
        bottom_row_frame.pack(side='top', fill='x')

        # 3. Action Buttons Frame (in bottom row)
        action_buttons_frame = ctk.CTkFrame(bottom_row_frame, fg_color="transparent")
        action_buttons_frame.pack(side='left', padx=(15, 0)) # 왼쪽에 20px 패딩 추가
        action_buttons = [
            ("Filter", viewer.filter_selected_data),
            ("Delete", viewer.delete_selected_data),
            ("Interpolate", viewer.interpolate_selected_data),
            ("Restore", viewer.restore_original_data)
        ]
        # Use smaller width for action buttons if needed
        action_button_style = {**button_style, "width": 80, "height": 28}
        for text, command in action_buttons:
            btn = ctk.CTkButton(action_buttons_frame, text=text, command=command, **action_button_style)
            btn.pack(side='left', padx=3)

        # 4. Done Button (in bottom row, packed to the right)
        done_button = ctk.CTkButton(
            bottom_row_frame, text="Done", command=viewer.toggle_edit_mode, **button_style
        )
        # Pack Done button to the far right of the bottom row
        done_button.pack(side='right', padx=(10, 0)) # Add padding to the left

    else:
        # --- Build View Controls (Single Row) ---
        parent_frame.configure(height=40) # Set default height for view mode

        reset_button = ctk.CTkButton(
            parent_frame, text="Reset View", command=viewer.reset_graph_view, **button_style
        )
        reset_button.pack(side='right', padx=5, pady=5)

        viewer.edit_button = ctk.CTkButton(
            parent_frame, text="Edit", command=viewer.toggle_edit_mode, **button_style
        )
        viewer.edit_button.pack(side='right', padx=5, pady=5) 