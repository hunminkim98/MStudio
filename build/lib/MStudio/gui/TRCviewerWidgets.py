import customtkinter as ctk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from MStudio.utils.viewToggles import (
    toggle_coordinates,
    toggle_marker_names,
    toggle_trajectory,
    toggle_animation,
    toggle_analysis_mode
)
from MStudio.gui.customization_toolbar import CustomizationToolbar

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


def create_widgets(self):
    """
    Creates all widgets for the TRCViewer application.
    This function was extracted from the main class to improve code organization.
    """
    button_frame = ctk.CTkFrame(self)
    button_frame.pack(pady=10, padx=10, fill='x')

    button_style = {
        "fg_color": "#333333",
        "hover_color": "#444444"
    }

    left_button_frame = ctk.CTkFrame(button_frame, fg_color="transparent")
    left_button_frame.pack(side='left', fill='x')

    self.reset_view_button = ctk.CTkButton(
        left_button_frame,
        text="🎥",
        width=30,
        command=self.reset_main_view,
        **button_style
    )
    self.reset_view_button.pack(side='left', padx=5)

    self.open_button = ctk.CTkButton(
        left_button_frame,
        text="Open File",
        command=self.open_file,
        **button_style
    )
    self.open_button.pack(side='left', padx=5)

    self.coord_button = ctk.CTkButton(
        button_frame,
        text="Switch to Y-up" if self.state_manager.view_state.is_z_up else "Switch to Z-up",
        command=lambda: toggle_coordinates(self),
        **button_style
    )
    self.coord_button.pack(side='left', padx=5)

    self.names_button = ctk.CTkButton(
        button_frame,
        text="Hide Names",
        command=lambda: toggle_marker_names(self),
        **button_style
    )
    self.names_button.pack(side='left', padx=5)

    self.trajectory_button = ctk.CTkButton(
        button_frame,
        text="Show Trajectory",
        command=lambda: toggle_trajectory(self),
        **button_style
    )
    self.trajectory_button.pack(side='left', padx=5)

    # Analysis button (connect to the function from viewToggles)
    self.analysis_button = ctk.CTkButton(
        button_frame,
        text="Analysis",
        command=lambda: toggle_analysis_mode(self),
        **button_style
    )
    self.analysis_button.pack(side='left', padx=5)

    self.save_button = ctk.CTkButton(
        button_frame,
        text="Save As...",
        command=self.save_as,
        **button_style
    )
    self.save_button.pack(side='left', padx=5)

    # Export Report button
    report_button_style = {
        "fg_color": "#2E8B57",  # Sea green color
        "hover_color": "#3CB371",
        "text_color": "white",
        "font": ("Arial", 11, "bold")
    }

    self.export_report_button = ctk.CTkButton(
        button_frame,
        text="📊 Export Report",
        command=self.export_analysis_report,
        width=120,
        **report_button_style
    )
    self.export_report_button.pack(side='left', padx=5)

    self.model_var = ctk.StringVar(value='No skeleton')
    self.model_combo = ctk.CTkComboBox(
        button_frame,
        values=list(self.available_models.keys()),
        variable=self.model_var,
        command=self.on_model_change
    )
    self.model_combo.pack(side='left', padx=5)

    # Add customization button to the main button frame
    customize_button_style = {
        "fg_color": "#CC5529",  # Darker orange color for visibility
        "hover_color": "#E6612F",
        "text_color": "white",
        "font": ("Arial", 12, "bold")
    }

    self.customize_button = ctk.CTkButton(
        button_frame,
        text="🎨 Customize",
        command=lambda: _toggle_customization_panel(self),
        width=120,
        **customize_button_style
    )
    self.customize_button.pack(side='left', padx=5)

    # Pack control_frame FIRST to ensure timeline is always visible
    self.control_frame = ctk.CTkFrame(
        self,
        border_width=1,
        fg_color="#1A1A1A"  # background color
    )
    self.control_frame.pack(fill='x', padx=10, pady=(0, 10))

    # Then pack main_content
    self.main_content = ctk.CTkFrame(self)
    self.main_content.pack(fill='both', expand=True, padx=10, pady=(0, 10))

    self.view_frame = ctk.CTkFrame(self.main_content, fg_color="black")
    self.view_frame.pack(side='left', fill='both', expand=True)

    self.right_panel = ctk.CTkFrame(self.main_content, fg_color="black")
    self.right_panel.pack_forget()  # initially hidden
    self.right_panel.pack_propagate(False)  # fixed size

    self.graph_frame = ctk.CTkFrame(self.right_panel, fg_color="black")

    viewer_top_frame = ctk.CTkFrame(self.view_frame)
    viewer_top_frame.pack(fill='x', pady=(5, 0))

    self.title_label = ctk.CTkLabel(viewer_top_frame, text="", font=("Arial", 14))
    self.title_label.pack(side='left', expand=True)

    canvas_container = ctk.CTkFrame(self.view_frame)
    canvas_container.pack(fill='both', expand=True)

    self.canvas_frame = ctk.CTkFrame(canvas_container)
    self.canvas_frame.pack(expand=True, fill='both')
    self.canvas_frame.pack_propagate(False)

    # Create customization toolbar - add to main button frame for guaranteed visibility
    _create_customization_button(self)

    # control button style
    control_style = {
        "width": 30,
        "fg_color": "#333333",
        "hover_color": "#444444"
    }

    # control button frame
    button_frame = ctk.CTkFrame(self.control_frame, fg_color="transparent")
    button_frame.pack(side='left', padx=5)

    # play control buttons
    self.play_pause_button = ctk.CTkButton(
        button_frame,
        text="▶",
        command=lambda: toggle_animation(self),
        **control_style
    )
    self.play_pause_button.pack(side='left', padx=2)

    self.stop_button = ctk.CTkButton(
        button_frame,
        text="■",
        command=self.stop_animation,
        # state='disabled',
        **control_style
    )
    self.stop_button.pack(side='left', padx=2)

    # loop checkbox style
    checkbox_style = {
        "width": 60,
        "fg_color": "#1A1A1A",  # transparent instead of background color
        "border_color": "#666666",  # border color
        "hover_color": "#1A1A1A",  # hover color
        "checkmark_color": "#00A6FF",  # checkmark color
        "border_width": 2  # border width
    }

    # loop checkbox
    self.loop_var = ctk.BooleanVar(value=False)
    self.loop_checkbox = ctk.CTkCheckBox(
        button_frame,
        text="Loop",
        variable=self.loop_var,
        command=self._on_loop_checkbox_changed,  # BUG FIX: Connect to callback
        text_color="#FFFFFF",
        **checkbox_style
    )
    self.loop_checkbox.pack(side='left', padx=5)

    # timeline menu frame
    timeline_menu_frame = ctk.CTkFrame(self.control_frame, fg_color="transparent")
    timeline_menu_frame.pack(side='left', padx=(5, 10))

    # current frame/time display label
    self.current_info_label = ctk.CTkLabel(
        timeline_menu_frame,
        text="0.00s",
        font=("Arial", 14),
        text_color="#FFFFFF"
    )
    self.current_info_label.pack(side='left', padx=5)

    # mode selection button frame
    mode_frame = ctk.CTkFrame(timeline_menu_frame, fg_color="#222222", corner_radius=6)
    mode_frame.pack(side='left', padx=2)

    # time/frame mode button
    button_style = {
        "width": 60,
        "height": 24,
        "corner_radius": 4,
        "font": ("Arial", 11),
        "fg_color": "transparent",
        "text_color": "#888888",
        "hover_color": "#333333"
    }

    self.timeline_display_var = ctk.StringVar(value="time")
    
    self.time_btn = ctk.CTkButton(
        mode_frame,
        text="Time",
        command=lambda: self.change_timeline_mode("time"),
        **button_style
    )
    self.time_btn.pack(side='left', padx=2, pady=2)

    self.frame_btn = ctk.CTkButton(
        mode_frame,
        text="Frame",
        command=lambda: self.change_timeline_mode("frame"),
        **button_style
    )
    self.frame_btn.pack(side='left', padx=2, pady=2)

    # timeline figure
    self.timeline_fig = Figure(figsize=(5, 0.8), facecolor='black')
    self.timeline_ax = self.timeline_fig.add_subplot(111)
    self.timeline_ax.set_facecolor('black')
    
    # timeline canvas
    self.timeline_canvas = FigureCanvasTkAgg(self.timeline_fig, master=self.control_frame)
    self.timeline_canvas.get_tk_widget().pack(fill='x', expand=True, padx=5, pady=5)
    
    # timeline event connection
    self.timeline_canvas.mpl_connect('button_press_event', self.mouse_handler.on_timeline_click)
    self.timeline_canvas.mpl_connect('motion_notify_event', self.mouse_handler.on_timeline_drag)
    self.timeline_canvas.mpl_connect('button_release_event', self.mouse_handler.on_timeline_release)
    
    self.timeline_dragging = False

    # initial timeline mode
    self.change_timeline_mode("time")

    self.marker_label = ctk.CTkLabel(self, text="")
    self.marker_label.pack(pady=5)


def _create_customization_button(self):
    """Create customization button - this is now handled in the main button frame"""
    # Initialize customization panel state
    self.customization_panel_visible = False


def _toggle_customization_panel(self):
    """Toggle the customization panel visibility"""
    if not hasattr(self, 'customization_panel_visible'):
        self.customization_panel_visible = False

    if self.customization_panel_visible:
        # Hide panel
        _hide_customization_panel(self)
    else:
        # Show panel
        _show_customization_panel(self)


def _show_customization_panel(self):
    """Show the customization panel"""
    # Alternative approach: Create a new window instead of overlay
    if hasattr(self, 'customization_window') and self.customization_window.winfo_exists():
        self.customization_window.destroy()

    # Create a new toplevel window for the customization panel
    self.customization_window = ctk.CTkToplevel(self)
    self.customization_window.title("Marker & Skeleton Customization")
    self.customization_window.geometry("500x450+100+100")  # width x height + x_offset + y_offset
    self.customization_window.configure(fg_color="#1E1E1E")

    # Make it stay on top
    self.customization_window.attributes("-topmost", True)
    self.customization_window.resizable(False, False)

    # Create panel content in the window
    _create_customization_panel_content_window(self)

    # Handle window close event
    def on_window_close():
        _hide_customization_panel(self)

    self.customization_window.protocol("WM_DELETE_WINDOW", on_window_close)

    # Update button text and state
    self.customize_button.configure(text="✕ Close", fg_color="#666666")
    self.customization_panel_visible = True


def _hide_customization_panel(self):
    """Hide the customization panel"""
    if hasattr(self, 'customization_window') and self.customization_window.winfo_exists():
        self.customization_window.destroy()

    # Update button text and state
    self.customize_button.configure(text="🎨 Customize", fg_color="#CC5529")
    self.customization_panel_visible = False


def _create_customization_panel_content_window(self):
    """Create the content for the customization panel window"""

    # Title
    title_label = ctk.CTkLabel(
        self.customization_window,
        text="🎨 Marker & Skeleton Customization",
        font=("Arial", 16, "bold"),
        text_color="white"
    )
    title_label.pack(pady=(15, 10))

    # Check if marker_visual_settings exists
    if not hasattr(self, 'marker_visual_settings'):
        error_label = ctk.CTkLabel(
            self.customization_window,
            text="⚠️ Marker visual settings not available",
            font=("Arial", 12),
            text_color="#FF6B6B"
        )
        error_label.pack(pady=20)
        return

    # Create scrollable frame for content
    scrollable_frame = ctk.CTkScrollableFrame(self.customization_window, fg_color="transparent")
    scrollable_frame.pack(fill="both", expand=True, padx=15, pady=10)

    # Marker section
    marker_section = ctk.CTkFrame(scrollable_frame, fg_color="#2A2A2A", corner_radius=8)
    marker_section.pack(fill="x", pady=(0, 10))

    marker_title = ctk.CTkLabel(
        marker_section,
        text="🔵 Marker Settings",
        font=("Arial", 14, "bold"),
        text_color="#4A9EFF"
    )
    marker_title.pack(pady=(10, 5))

    marker_content = ctk.CTkFrame(marker_section, fg_color="transparent")
    marker_content.pack(fill="x", padx=10, pady=(0, 10))

    # Size control
    size_frame = ctk.CTkFrame(marker_content, fg_color="transparent")
    size_frame.pack(fill="x", pady=5)

    size_label = ctk.CTkLabel(size_frame, text="Marker Size:", font=("Arial", 12))
    size_label.pack(side="left")

    self.size_slider = ctk.CTkSlider(
        size_frame,
        from_=1.0,
        to=20.0,
        width=150,
        command=lambda v: _on_size_change(self, v)
    )
    self.size_slider.set(self.marker_visual_settings.get_marker_size())
    self.size_slider.pack(side="right", padx=(10, 0))

    self.size_value_label = ctk.CTkLabel(
        size_frame,
        text=f"{self.marker_visual_settings.get_marker_size():.1f}",
        font=("Arial", 10),
        width=30
    )
    self.size_value_label.pack(side="right", padx=(5, 10))

    # Opacity control
    opacity_frame = ctk.CTkFrame(marker_content, fg_color="transparent")
    opacity_frame.pack(fill="x", pady=5)

    opacity_label = ctk.CTkLabel(opacity_frame, text="Marker Opacity:", font=("Arial", 12))
    opacity_label.pack(side="left")

    self.opacity_slider = ctk.CTkSlider(
        opacity_frame,
        from_=0.1,
        to=1.0,
        width=150,
        command=lambda v: _on_opacity_change(self, v)
    )
    self.opacity_slider.set(self.marker_visual_settings.get_opacity())
    self.opacity_slider.pack(side="right", padx=(10, 0))

    self.opacity_value_label = ctk.CTkLabel(
        opacity_frame,
        text=f"{self.marker_visual_settings.get_opacity():.2f}",
        font=("Arial", 10),
        width=30
    )
    self.opacity_value_label.pack(side="right", padx=(5, 10))

    # Skeleton section
    skeleton_section = ctk.CTkFrame(scrollable_frame, fg_color="#2A2A2A", corner_radius=8)
    skeleton_section.pack(fill="x", pady=(0, 10))

    skeleton_title = ctk.CTkLabel(
        skeleton_section,
        text="🦴 Skeleton Settings",
        font=("Arial", 14, "bold"),
        text_color="#FF9A4A"
    )
    skeleton_title.pack(pady=(10, 5))

    skeleton_content = ctk.CTkFrame(skeleton_section, fg_color="transparent")
    skeleton_content.pack(fill="x", padx=10, pady=(0, 10))

    # Skeleton line width control
    skel_width_frame = ctk.CTkFrame(skeleton_content, fg_color="transparent")
    skel_width_frame.pack(fill="x", pady=5)

    skel_width_label = ctk.CTkLabel(skel_width_frame, text="Line Width:", font=("Arial", 12))
    skel_width_label.pack(side="left")

    self.skel_width_slider = ctk.CTkSlider(
        skel_width_frame,
        from_=0.5,
        to=5.0,
        width=150,
        command=lambda v: _on_skeleton_width_change(self, v)
    )
    self.skel_width_slider.set(self.marker_visual_settings.get_skeleton_line_width())
    self.skel_width_slider.pack(side="right", padx=(10, 0))

    self.skel_width_value_label = ctk.CTkLabel(
        skel_width_frame,
        text=f"{self.marker_visual_settings.get_skeleton_line_width():.1f}",
        font=("Arial", 10),
        width=30
    )
    self.skel_width_value_label.pack(side="right", padx=(5, 10))

    # Skeleton opacity control
    skel_opacity_frame = ctk.CTkFrame(skeleton_content, fg_color="transparent")
    skel_opacity_frame.pack(fill="x", pady=5)

    skel_opacity_label = ctk.CTkLabel(skel_opacity_frame, text="Skeleton Opacity:", font=("Arial", 12))
    skel_opacity_label.pack(side="left")

    self.skel_opacity_slider = ctk.CTkSlider(
        skel_opacity_frame,
        from_=0.1,
        to=1.0,
        width=150,
        command=lambda v: _on_skeleton_opacity_change(self, v)
    )
    self.skel_opacity_slider.set(self.marker_visual_settings.get_skeleton_opacity())
    self.skel_opacity_slider.pack(side="right", padx=(10, 0))

    self.skel_opacity_value_label = ctk.CTkLabel(
        skel_opacity_frame,
        text=f"{self.marker_visual_settings.get_skeleton_opacity():.2f}",
        font=("Arial", 10),
        width=30
    )
    self.skel_opacity_value_label.pack(side="right", padx=(5, 10))

    # Skeleton color controls
    skel_color_frame = ctk.CTkFrame(skeleton_content, fg_color="transparent")
    skel_color_frame.pack(fill="x", pady=5)

    skel_color_label = ctk.CTkLabel(skel_color_frame, text="Skeleton Colors:", font=("Arial", 12))
    skel_color_label.pack(side="left")

    skel_color_buttons_frame = ctk.CTkFrame(skel_color_frame, fg_color="transparent")
    skel_color_buttons_frame.pack(side="right", padx=(10, 0))

    # Normal skeleton color button
    self.skel_normal_color_btn = ctk.CTkButton(
        skel_color_buttons_frame,
        text="Normal",
        width=60,
        height=24,
        command=lambda: _open_skeleton_color_picker(self, "normal"),
        fg_color=_rgb_to_hex(self.marker_visual_settings.get_skeleton_normal_color()),
        hover_color=_rgb_to_hex(self.marker_visual_settings.get_skeleton_normal_color()),
        font=("Arial", 9)
    )
    self.skel_normal_color_btn.pack(side="left", padx=(0, 5))

    # Outlier skeleton color button
    self.skel_outlier_color_btn = ctk.CTkButton(
        skel_color_buttons_frame,
        text="Outlier",
        width=60,
        height=24,
        command=lambda: _open_skeleton_color_picker(self, "outlier"),
        fg_color=_rgb_to_hex(self.marker_visual_settings.get_skeleton_outlier_color()),
        hover_color=_rgb_to_hex(self.marker_visual_settings.get_skeleton_outlier_color()),
        font=("Arial", 9)
    )
    self.skel_outlier_color_btn.pack(side="left")

    # Global controls section
    global_section = ctk.CTkFrame(scrollable_frame, fg_color="#2A2A2A", corner_radius=8)
    global_section.pack(fill="x", pady=(0, 10))

    global_title = ctk.CTkLabel(
        global_section,
        text="🌐 Global Settings",
        font=("Arial", 14, "bold"),
        text_color="#4AFF9A"
    )
    global_title.pack(pady=(10, 5))

    global_content = ctk.CTkFrame(global_section, fg_color="transparent")
    global_content.pack(fill="x", padx=10, pady=(0, 10))

    # Color presets
    preset_frame = ctk.CTkFrame(global_content, fg_color="transparent")
    preset_frame.pack(fill="x", pady=5)

    preset_label = ctk.CTkLabel(preset_frame, text="Color Theme:", font=("Arial", 12))
    preset_label.pack(side="left")

    self.preset_combo = ctk.CTkComboBox(
        preset_frame,
        values=self.marker_visual_settings.get_color_schemes(),
        width=120,
        command=lambda v: _on_preset_change(self, v)
    )
    self.preset_combo.set("Default")
    self.preset_combo.pack(side="right", padx=(10, 0))

    # Reset button
    reset_button = ctk.CTkButton(
        global_content,
        text="Reset to Defaults",
        command=lambda: _reset_settings(self),
        fg_color="#666666",
        hover_color="#777777",
        width=150
    )
    reset_button.pack(pady=(10, 5))


# Helper functions for customization panel
def _on_size_change(self, value):
    """Handle size slider change"""
    size = float(value)
    self.marker_visual_settings.set_marker_size(size)
    self.size_value_label.configure(text=f"{size:.1f}")


def _on_opacity_change(self, value):
    """Handle opacity slider change"""
    opacity = float(value)
    self.marker_visual_settings.set_opacity(opacity)
    self.opacity_value_label.configure(text=f"{opacity:.2f}")


def _on_preset_change(self, scheme_name):
    """Handle preset selection change"""
    self.marker_visual_settings.apply_color_scheme(scheme_name)

    # Update skeleton color buttons
    if hasattr(self, 'skel_normal_color_btn'):
        color = _rgb_to_hex(self.marker_visual_settings.get_skeleton_normal_color())
        self.skel_normal_color_btn.configure(fg_color=color, hover_color=color)

    if hasattr(self, 'skel_outlier_color_btn'):
        color = _rgb_to_hex(self.marker_visual_settings.get_skeleton_outlier_color())
        self.skel_outlier_color_btn.configure(fg_color=color, hover_color=color)


def _on_skeleton_width_change(self, value):
    """Handle skeleton width slider change"""
    width = float(value)
    self.marker_visual_settings.set_skeleton_line_width(width)
    self.skel_width_value_label.configure(text=f"{width:.1f}")


def _on_skeleton_opacity_change(self, value):
    """Handle skeleton opacity slider change"""
    opacity = float(value)
    self.marker_visual_settings.set_skeleton_opacity(opacity)
    self.skel_opacity_value_label.configure(text=f"{opacity:.2f}")


def _rgb_to_hex(rgb):
    """Convert RGB tuple (0-1 range) to hex string"""
    try:
        # Ensure values are in valid range [0, 1]
        r, g, b = [max(0.0, min(1.0, float(c))) for c in rgb]
        # Convert to 0-255 range and format as hex
        r_int, g_int, b_int = [int(c * 255) for c in [r, g, b]]
        hex_color = f"#{r_int:02x}{g_int:02x}{b_int:02x}"
        return hex_color
    except Exception:
        return "#808080"  # Default gray color


def _open_skeleton_color_picker(self, color_type):
    """Open color picker for skeleton colors"""
    from tkinter import colorchooser

    current_color = None
    if color_type == "normal":
        current_color = self.marker_visual_settings.get_skeleton_normal_color()
    elif color_type == "outlier":
        current_color = self.marker_visual_settings.get_skeleton_outlier_color()

    if current_color:
        # Convert RGB to hex for color chooser
        hex_color = _rgb_to_hex(current_color)
        color = colorchooser.askcolor(color=hex_color, title=f"Choose {color_type} skeleton color")

        if color[0]:  # If user didn't cancel
            rgb_color = tuple(c/255.0 for c in color[0])  # Convert to 0-1 range

            if color_type == "normal":
                self.marker_visual_settings.set_skeleton_normal_color(rgb_color)
                self.skel_normal_color_btn.configure(
                    fg_color=_rgb_to_hex(rgb_color),
                    hover_color=_rgb_to_hex(rgb_color)
                )
            elif color_type == "outlier":
                self.marker_visual_settings.set_skeleton_outlier_color(rgb_color)
                self.skel_outlier_color_btn.configure(
                    fg_color=_rgb_to_hex(rgb_color),
                    hover_color=_rgb_to_hex(rgb_color)
                )




def _reset_settings(self):
    """Reset all settings to defaults"""
    self.marker_visual_settings.reset_to_defaults()

    # Update marker UI controls
    if hasattr(self, 'size_slider'):
        self.size_slider.set(self.marker_visual_settings.get_marker_size())
        self.size_value_label.configure(text=f"{self.marker_visual_settings.get_marker_size():.1f}")

    if hasattr(self, 'opacity_slider'):
        self.opacity_slider.set(self.marker_visual_settings.get_opacity())
        self.opacity_value_label.configure(text=f"{self.marker_visual_settings.get_opacity():.2f}")

    # Update skeleton UI controls
    if hasattr(self, 'skel_width_slider'):
        self.skel_width_slider.set(self.marker_visual_settings.get_skeleton_line_width())
        self.skel_width_value_label.configure(text=f"{self.marker_visual_settings.get_skeleton_line_width():.1f}")

    if hasattr(self, 'skel_opacity_slider'):
        self.skel_opacity_slider.set(self.marker_visual_settings.get_skeleton_opacity())
        self.skel_opacity_value_label.configure(text=f"{self.marker_visual_settings.get_skeleton_opacity():.2f}")

    # Update skeleton color buttons
    if hasattr(self, 'skel_normal_color_btn'):
        color = _rgb_to_hex(self.marker_visual_settings.get_skeleton_normal_color())
        self.skel_normal_color_btn.configure(fg_color=color, hover_color=color)

    if hasattr(self, 'skel_outlier_color_btn'):
        color = _rgb_to_hex(self.marker_visual_settings.get_skeleton_outlier_color())
        self.skel_outlier_color_btn.configure(fg_color=color, hover_color=color)

    if hasattr(self, 'preset_combo'):
        self.preset_combo.set("Default")
