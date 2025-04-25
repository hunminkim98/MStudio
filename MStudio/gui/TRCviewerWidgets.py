import customtkinter as ctk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from MStudio.utils.viewToggles import toggle_coordinates

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
        text="Open TRC File",
        command=self.open_file,
        **button_style
    )
    self.open_button.pack(side='left', padx=5)

    self.coord_button = ctk.CTkButton(
        button_frame,
        text="Switch to Y-up" if self.is_z_up else "Switch to Z-up",
        command=lambda: toggle_coordinates(self),
        **button_style
    )
    self.coord_button.pack(side='left', padx=5)

    self.names_button = ctk.CTkButton(
        button_frame,
        text="Hide Names",
        command=self.toggle_marker_names,
        **button_style
    )
    self.names_button.pack(side='left', padx=5)

    self.trajectory_button = ctk.CTkButton(
        button_frame,
        text="Show Trajectory",
        command=self.toggle_trajectory,
        **button_style
    )
    self.trajectory_button.pack(side='left', padx=5)

    # Analysis button (connect to the method in TRCViewer)
    self.analysis_button = ctk.CTkButton(
        button_frame,
        text="Analysis",
        command=self.toggle_analysis_mode,
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

    self.model_var = ctk.StringVar(value='No skeleton')
    self.model_combo = ctk.CTkComboBox(
        button_frame,
        values=list(self.available_models.keys()),
        variable=self.model_var,
        command=self.on_model_change
    )
    self.model_combo.pack(side='left', padx=5)

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

    self.control_frame = ctk.CTkFrame(
        self,
        border_width=1,  
        fg_color="#1A1A1A"  # background color
    )
    self.control_frame.pack(fill='x', padx=10, pady=(0, 10))

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
        command=self.toggle_animation,
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
