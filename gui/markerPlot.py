import numpy as np
import customtkinter as ctk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from .markerPlotUI import build_marker_plot_buttons

def show_marker_plot(self, marker_name):
    """
    Creates and displays a detailed plot for a specific marker.
    This function was extracted from the main class to improve code organization.
    
    Note: 이 함수는 OpenGL 렌더링 모드에서도 matplotlib을 사용하여 그래프를 표시합니다.
    """
    # Save current states
    was_editing = getattr(self, 'is_editing', False)
    
    # Save previous filter parameters if they exist
    prev_filter_params = None
    if hasattr(self, 'filter_params'):
        prev_filter_params = {
            filter_type: {
                param: var.get() for param, var in params.items()
            } for filter_type, params in self.filter_params.items()
        }
    prev_filter_type = getattr(self, 'filter_type_var', None)
    if prev_filter_type:
        prev_filter_type = prev_filter_type.get()
    
    # Save previous interpolation parameters
    prev_interp_method = getattr(self, 'interp_method_var', None)
    if prev_interp_method:
        prev_interp_method = prev_interp_method.get()
    prev_order = getattr(self, 'order_var', None)
    if prev_order:
        prev_order = prev_order.get()

    if not self.graph_frame.winfo_ismapped():
        # display right panel
        self.right_panel.pack(side='right', fill='both')
        
        # initial width setting (1/3 of the window width)
        initial_width = self.winfo_width() // 3
        self.right_panel.configure(width=initial_width)
        
        # create and configure Sizer
        if not hasattr(self, 'sizer') or self.sizer is None:
            self.sizer = ctk.CTkFrame(self.main_content, width=5, height=self.main_content.winfo_height(),
                                    fg_color="#666666", bg_color="black")
            self.sizer.pack(side='left', fill='y')
            self.sizer.pack_propagate(False)
            
            # Sizer bindings
            self.sizer.bind('<Enter>', lambda e: (
                self.sizer.configure(fg_color="#888888"),
                self.sizer.configure(cursor="sb_h_double_arrow")
            ))
            self.sizer.bind('<Leave>', lambda e: self.sizer.configure(fg_color="#666666"))
            self.sizer.bind('<Button-1>', self.start_resize)
            self.sizer.bind('<B1-Motion>', self.do_resize)
            self.sizer.bind('<ButtonRelease-1>', self.stop_resize)
    
        self.graph_frame.pack(fill='both', expand=True)

    for widget in self.graph_frame.winfo_children():
        widget.destroy()

    self.marker_plot_fig = Figure(figsize=(6, 8), facecolor='black')
    self.marker_plot_fig.patch.set_facecolor('black')

    self.current_marker = marker_name

    self.marker_axes = []
    self.marker_lines = []
    coords = ['X', 'Y', 'Z']

    if not hasattr(self, 'outliers') or marker_name not in self.outliers:
        self.outliers = {marker_name: np.zeros(len(self.data), dtype=bool)}

    outlier_frames = np.where(self.outliers[marker_name])[0]

    for i, coord in enumerate(coords):
        ax = self.marker_plot_fig.add_subplot(3, 1, i+1)
        ax.set_facecolor('black')

        data = self.data[f'{marker_name}_{coord}']
        frames = np.arange(len(data))

        ax.plot(frames[~self.outliers[marker_name]],
                data[~self.outliers[marker_name]],
                color='white',
                label='Normal')

        if len(outlier_frames) > 0:
            ax.plot(frames[self.outliers[marker_name]],
                    data[self.outliers[marker_name]],
                    'ro',
                    markersize=3,
                    label='Outlier')

        ax.set_title(f'{marker_name} - {coord}', color='white')
        ax.grid(True, color='gray', alpha=0.3)
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')

        self.marker_axes.append(ax)

        if len(outlier_frames) > 0:
            ax.legend(facecolor='black',
                    labelcolor='white',
                    loc='upper right',
                    bbox_to_anchor=(1.0, 1.0))

    # initialize current frame display line
    self.marker_lines = []  # initialize existing lines
    for ax in self.marker_axes:
        line = ax.axvline(x=self.frame_idx, color='red', linestyle='--', alpha=0.8)
        self.marker_lines.append(line)

    self.marker_plot_fig.tight_layout()

    self.marker_canvas = FigureCanvasTkAgg(self.marker_plot_fig, master=self.graph_frame)
    self.marker_canvas.draw()

    # Force layout update *after* canvas is drawn, *before* button frame
    self.graph_frame.update_idletasks()

    self.initial_graph_limits = []
    for ax in self.marker_axes:
        self.initial_graph_limits.append({
            'x': ax.get_xlim(),
            'y': ax.get_ylim()
        })

    self.marker_canvas.mpl_connect('scroll_event', self.mouse_handler.on_marker_scroll)
    self.marker_canvas.mpl_connect('button_press_event', self.mouse_handler.on_marker_mouse_press)
    self.marker_canvas.mpl_connect('button_release_event', self.mouse_handler.on_marker_mouse_release)
    self.marker_canvas.mpl_connect('motion_notify_event', self.mouse_handler.on_marker_mouse_move)

    # Create and pack the button frame first at the bottom
    # Height will be set dynamically by _build_marker_plot_buttons
    button_frame = ctk.CTkFrame(self.graph_frame, fg_color="#1A1A1A")
    button_frame.pack_propagate(False) # Prevent frame from shrinking
    button_frame.pack(fill='x', padx=5, pady=(5, 10), side='bottom')

    # --- Delegate button building to the main class method ---
    # This will now also set the height of button_frame
    build_marker_plot_buttons(self, button_frame)

    # Pack the canvas LAST, filling the remaining space at the top
    self.marker_canvas.get_tk_widget().pack(side='top', fill='both', expand=True)

    # Initialize filter parameters if not already present
    if not hasattr(self, 'filter_params'):
        self.filter_params = {
            'butterworth': {
                'order': ctk.StringVar(value="4"),
                'cut_off_frequency': ctk.StringVar(value="10")
            },
            'butterworth_on_speed': {
                'order': ctk.StringVar(value="4"),
                'cut_off_frequency': ctk.StringVar(value="10")
            },
            'kalman': {
                'trust_ratio': ctk.StringVar(value="20"),
                'smooth': ctk.StringVar(value="1")
            },
            'gaussian': {
                'sigma_kernel': ctk.StringVar(value="3")
            },
            'LOESS': {
                'nb_values_used': ctk.StringVar(value="10")
            },
            'median': {
                'kernel_size': ctk.StringVar(value="3")
            }
        }
    
    # Restore previous parameter values if they exist
    if prev_filter_params:
        for filter_type, params in prev_filter_params.items():
            for param, value in params.items():
                self.filter_params[filter_type][param].set(value)

    # Backwards compatibility for filter parameters
    self.hz_var = self.filter_params['butterworth']['cut_off_frequency']
    self.filter_order_var = self.filter_params['butterworth']['order']

    # Restore interpolation parameters
    if prev_interp_method:
        self.interp_method_var.set(prev_interp_method)
    if prev_order:
        self.order_var.set(prev_order)

    self.selection_data = {
        'start': None,
        'end': None,
        'rects': [],
        'current_ax': None,
        'rect': None
    }

    self.connect_mouse_events()

    # Restore edit state if it was active
    if was_editing and not self.is_editing:
        # Schedule with a small delay to avoid UI glitches
        self.after(10, self.toggle_edit_mode)
        
    # Force update of the layout to help ensure widgets are drawn
    self.graph_frame.update_idletasks()
