"""
Customization Toolbar for MStudio

This module provides a customization toolbar inspired by OpenSim UI for adjusting
marker visual properties like size, color, and opacity.
"""

import logging
import customtkinter as ctk
from tkinter import colorchooser
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class CustomizationToolbar(ctk.CTkFrame):
    """
    Customization toolbar for marker visual settings
    Inspired by OpenSim UI design
    """
    
    def __init__(self, parent, marker_settings, **kwargs):
        """
        Initialize the customization toolbar
        
        Args:
            parent: Parent widget
            marker_settings: MarkerVisualSettings instance
        """
        super().__init__(parent, **kwargs)
        
        self.marker_settings = marker_settings
        self.is_expanded = False
        
        # Configure frame appearance
        self.configure(
            fg_color="#2B2B2B",
            corner_radius=6,
            border_width=1,
            border_color="#404040"
        )

        # Create toolbar content
        self._create_toolbar()

        # Start expanded (since we're only creating this when needed)
        self._expand_toolbar()
        
        logger.info("CustomizationToolbar initialized")
    
    def _create_toolbar(self):
        """Create toolbar UI elements"""

        # Expandable content frame (no toggle button here since parent handles it)
        self.content_frame = ctk.CTkFrame(
            self,
            fg_color="transparent"
        )
        
        # Size controls
        self._create_size_controls()
        
        # Color controls
        self._create_color_controls()
        
        # Opacity controls
        self._create_opacity_controls()
        
        # Preset controls
        self._create_preset_controls()
    
    def _create_size_controls(self):
        """Create marker size adjustment controls"""
        size_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        size_frame.pack(side="left", padx=5, pady=2)
        
        # Size label
        size_label = ctk.CTkLabel(
            size_frame,
            text="Size:",
            font=("Arial", 10),
            width=30
        )
        size_label.pack(side="top")
        
        # Size slider
        self.size_slider = ctk.CTkSlider(
            size_frame,
            from_=1.0,
            to=20.0,
            width=80,
            height=16,
            command=self._on_size_change
        )
        self.size_slider.set(self.marker_settings.get_marker_size())
        self.size_slider.pack(side="top", pady=2)
        
        # Size value label
        self.size_value_label = ctk.CTkLabel(
            size_frame,
            text=f"{self.marker_settings.get_marker_size():.1f}",
            font=("Arial", 9),
            width=30
        )
        self.size_value_label.pack(side="top")
    
    def _create_color_controls(self):
        """Create color adjustment controls"""
        color_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        color_frame.pack(side="left", padx=5, pady=2)
        
        # Color label
        color_label = ctk.CTkLabel(
            color_frame,
            text="Colors:",
            font=("Arial", 10)
        )
        color_label.pack(side="top")
        
        # Color buttons frame
        color_buttons_frame = ctk.CTkFrame(color_frame, fg_color="transparent")
        color_buttons_frame.pack(side="top", pady=2)
        
        # Normal color button
        self.normal_color_btn = ctk.CTkButton(
            color_buttons_frame,
            text="●",
            width=24,
            height=24,
            font=("Arial", 12),
            command=lambda: self._open_color_picker("normal"),
            fg_color=self._rgb_to_hex(self.marker_settings.get_normal_color()),
            hover_color=self._rgb_to_hex(self.marker_settings.get_normal_color())
        )
        self.normal_color_btn.pack(side="left", padx=1)
        
        # Selected color button
        self.selected_color_btn = ctk.CTkButton(
            color_buttons_frame,
            text="●",
            width=24,
            height=24,
            font=("Arial", 12),
            command=lambda: self._open_color_picker("selected"),
            fg_color=self._rgb_to_hex(self.marker_settings.get_selected_color()),
            hover_color=self._rgb_to_hex(self.marker_settings.get_selected_color())
        )
        self.selected_color_btn.pack(side="left", padx=1)
        
        # Pattern color button
        self.pattern_color_btn = ctk.CTkButton(
            color_buttons_frame,
            text="●",
            width=24,
            height=24,
            font=("Arial", 12),
            command=lambda: self._open_color_picker("pattern"),
            fg_color=self._rgb_to_hex(self.marker_settings.get_pattern_color()),
            hover_color=self._rgb_to_hex(self.marker_settings.get_pattern_color())
        )
        self.pattern_color_btn.pack(side="left", padx=1)
    
    def _create_opacity_controls(self):
        """Create opacity adjustment controls"""
        opacity_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        opacity_frame.pack(side="left", padx=5, pady=2)
        
        # Opacity label
        opacity_label = ctk.CTkLabel(
            opacity_frame,
            text="Opacity:",
            font=("Arial", 10),
            width=40
        )
        opacity_label.pack(side="top")
        
        # Opacity slider
        self.opacity_slider = ctk.CTkSlider(
            opacity_frame,
            from_=0.1,
            to=1.0,
            width=80,
            height=16,
            command=self._on_opacity_change
        )
        self.opacity_slider.set(self.marker_settings.get_opacity())
        self.opacity_slider.pack(side="top", pady=2)
        
        # Opacity value label
        self.opacity_value_label = ctk.CTkLabel(
            opacity_frame,
            text=f"{self.marker_settings.get_opacity():.2f}",
            font=("Arial", 9),
            width=30
        )
        self.opacity_value_label.pack(side="top")
    
    def _create_preset_controls(self):
        """Create preset controls"""
        preset_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        preset_frame.pack(side="left", padx=5, pady=2)
        
        # Preset label
        preset_label = ctk.CTkLabel(
            preset_frame,
            text="Presets:",
            font=("Arial", 10)
        )
        preset_label.pack(side="top")
        
        # Preset dropdown
        self.preset_combo = ctk.CTkComboBox(
            preset_frame,
            values=self.marker_settings.get_color_schemes(),
            width=100,
            height=24,
            command=self._on_preset_change,
            font=("Arial", 9)
        )
        self.preset_combo.set("Default")
        self.preset_combo.pack(side="top", pady=2)
        
        # Reset button
        self.reset_btn = ctk.CTkButton(
            preset_frame,
            text="Reset",
            width=50,
            height=20,
            font=("Arial", 9),
            command=self._reset_settings,
            fg_color="#666666",
            hover_color="#777777"
        )
        self.reset_btn.pack(side="top", pady=2)
    
    def _expand_toolbar(self):
        """Expand the toolbar to show all controls"""
        self.content_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.is_expanded = True
        logger.debug("Customization toolbar expanded")

    def _collapse_toolbar(self):
        """Collapse the toolbar to show only toggle button"""
        self.content_frame.pack_forget()
        self.is_expanded = False
        logger.debug("Customization toolbar collapsed")
    
    def _on_size_change(self, value):
        """Handle size slider change"""
        size = float(value)
        self.marker_settings.set_marker_size(size)
        self.size_value_label.configure(text=f"{size:.1f}")
    
    def _on_opacity_change(self, value):
        """Handle opacity slider change"""
        opacity = float(value)
        self.marker_settings.set_opacity(opacity)
        self.opacity_value_label.configure(text=f"{opacity:.2f}")
    
    def _on_preset_change(self, scheme_name):
        """Handle preset selection change"""
        self.marker_settings.apply_color_scheme(scheme_name)
        self._update_color_buttons()
    
    def _reset_settings(self):
        """Reset all settings to defaults"""
        self.marker_settings.reset_to_defaults()
        self._update_all_controls()
    
    def _open_color_picker(self, color_type):
        """Open color picker for specified color type"""
        current_color = None
        if color_type == "normal":
            current_color = self.marker_settings.get_normal_color()
        elif color_type == "selected":
            current_color = self.marker_settings.get_selected_color()
        elif color_type == "pattern":
            current_color = self.marker_settings.get_pattern_color()
        
        if current_color:
            # Convert RGB to hex for color chooser
            hex_color = self._rgb_to_hex(current_color)
            color = colorchooser.askcolor(color=hex_color, title=f"Choose {color_type} marker color")
            
            if color[0]:  # If user didn't cancel
                rgb_color = tuple(c/255.0 for c in color[0])  # Convert to 0-1 range
                
                if color_type == "normal":
                    self.marker_settings.set_normal_color(rgb_color)
                elif color_type == "selected":
                    self.marker_settings.set_selected_color(rgb_color)
                elif color_type == "pattern":
                    self.marker_settings.set_pattern_color(rgb_color)
                
                self._update_color_buttons()
    
    def _rgb_to_hex(self, rgb):
        """Convert RGB tuple (0-1 range) to hex string"""
        r, g, b = [int(c * 255) for c in rgb]
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def _update_color_buttons(self):
        """Update color button appearances"""
        self.normal_color_btn.configure(
            fg_color=self._rgb_to_hex(self.marker_settings.get_normal_color()),
            hover_color=self._rgb_to_hex(self.marker_settings.get_normal_color())
        )
        self.selected_color_btn.configure(
            fg_color=self._rgb_to_hex(self.marker_settings.get_selected_color()),
            hover_color=self._rgb_to_hex(self.marker_settings.get_selected_color())
        )
        self.pattern_color_btn.configure(
            fg_color=self._rgb_to_hex(self.marker_settings.get_pattern_color()),
            hover_color=self._rgb_to_hex(self.marker_settings.get_pattern_color())
        )
    
    def _update_all_controls(self):
        """Update all control values to match current settings"""
        self.size_slider.set(self.marker_settings.get_marker_size())
        self.size_value_label.configure(text=f"{self.marker_settings.get_marker_size():.1f}")
        
        self.opacity_slider.set(self.marker_settings.get_opacity())
        self.opacity_value_label.configure(text=f"{self.marker_settings.get_opacity():.2f}")
        
        self._update_color_buttons()
        self.preset_combo.set("Default")
