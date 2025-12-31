"""
Marker Visual Settings Manager

This module manages visual settings for marker rendering including size, color, and opacity.
"""

import logging
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def validate_color(color: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Validate and clamp color values to [0.0, 1.0] range"""
    return tuple(max(0.0, min(1.0, c)) for c in color)


@dataclass
class MarkerVisualConfig:
    """Configuration for marker visual properties"""
    size: float = 5.0
    color_normal: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # White
    color_selected: Tuple[float, float, float] = (1.0, 0.9, 0.4)  # Light yellow
    color_pattern: Tuple[float, float, float] = (1.0, 0.0, 0.0)  # Red
    opacity: float = 1.0

@dataclass
class SkeletonVisualConfig:
    """Configuration for skeleton visual properties"""
    line_width: float = 2.0
    color_normal: Tuple[float, float, float] = (0.7, 0.7, 0.7)  # Gray
    color_outlier: Tuple[float, float, float] = (1.0, 0.0, 0.0)  # Red
    opacity: float = 0.8
    
    def __post_init__(self):
        """Validate configuration values"""
        self.size = max(1.0, min(20.0, self.size))  # Size between 1-20
        self.opacity = max(0.1, min(1.0, self.opacity))  # Opacity between 0.1-1.0

        # Ensure color values are in valid range [0.0, 1.0]
        self.color_normal = self._validate_color(self.color_normal)
        self.color_selected = self._validate_color(self.color_selected)
        self.color_pattern = self._validate_color(self.color_pattern)

    def _validate_color(self, color: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Validate and clamp color values to [0.0, 1.0] range"""
        return validate_color(color)


# Add validation to SkeletonVisualConfig
SkeletonVisualConfig.__post_init__ = lambda self: (
    setattr(self, 'line_width', max(0.5, min(5.0, self.line_width))),
    setattr(self, 'opacity', max(0.1, min(1.0, self.opacity))),
    setattr(self, 'color_normal', validate_color(self.color_normal)),
    setattr(self, 'color_outlier', validate_color(self.color_outlier))
)


class MarkerVisualSettings:
    """
    Manager for marker visual settings with callback support for real-time updates
    """
    
    def __init__(self):
        """Initialize with default settings"""
        self.config = MarkerVisualConfig()
        self.skeleton_config = SkeletonVisualConfig()
        self._change_callbacks: List[Callable] = []
        
        # Predefined color schemes
        self.color_schemes = {
            "Default": {
                "marker_normal": (1.0, 1.0, 1.0),
                "marker_selected": (1.0, 0.9, 0.4),
                "marker_pattern": (1.0, 0.0, 0.0),
                "skeleton_normal": (0.7, 0.7, 0.7),
                "skeleton_outlier": (1.0, 0.0, 0.0)
            },
            "Blue Theme": {
                "marker_normal": (0.7, 0.8, 1.0),
                "marker_selected": (0.0, 0.5, 1.0),
                "marker_pattern": (1.0, 0.0, 0.5),
                "skeleton_normal": (0.5, 0.7, 1.0),
                "skeleton_outlier": (1.0, 0.3, 0.3)
            },
            "Green Theme": {
                "marker_normal": (0.7, 1.0, 0.7),
                "marker_selected": (0.0, 1.0, 0.0),
                "marker_pattern": (1.0, 0.5, 0.0),
                "skeleton_normal": (0.5, 0.9, 0.5),
                "skeleton_outlier": (1.0, 0.4, 0.0)
            },
            "Warm Theme": {
                "marker_normal": (1.0, 0.9, 0.7),
                "marker_selected": (1.0, 0.6, 0.0),
                "marker_pattern": (1.0, 0.0, 0.0),
                "skeleton_normal": (0.9, 0.7, 0.5),
                "skeleton_outlier": (1.0, 0.2, 0.2)
            }
        }
        
        logger.info("MarkerVisualSettings initialized with default configuration")
    
    def add_change_callback(self, callback: Callable) -> None:
        """Add a callback to be called when settings change"""
        if callback not in self._change_callbacks:
            self._change_callbacks.append(callback)
            logger.debug(f"Added change callback: {callback.__name__}")
    
    def remove_change_callback(self, callback: Callable) -> None:
        """Remove a change callback"""
        if callback in self._change_callbacks:
            self._change_callbacks.remove(callback)
            logger.debug(f"Removed change callback: {callback.__name__}")
    
    def _notify_change(self) -> None:
        """Notify all registered callbacks of setting changes"""
        for callback in self._change_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in change callback {callback.__name__}: {e}")
    
    # Size settings
    def set_marker_size(self, size: float) -> None:
        """Set marker size (1.0 - 20.0)"""
        old_size = self.config.size
        self.config.size = max(1.0, min(20.0, size))
        if old_size != self.config.size:
            logger.info(f"Marker size changed: {old_size} -> {self.config.size}")
            self._notify_change()
    
    def get_marker_size(self) -> float:
        """Get current marker size"""
        return self.config.size
    
    # Opacity settings
    def set_opacity(self, opacity: float) -> None:
        """Set marker opacity (0.1 - 1.0)"""
        old_opacity = self.config.opacity
        self.config.opacity = max(0.1, min(1.0, opacity))
        if old_opacity != self.config.opacity:
            logger.info(f"Marker opacity changed: {old_opacity} -> {self.config.opacity}")
            self._notify_change()
    
    def get_opacity(self) -> float:
        """Get current marker opacity"""
        return self.config.opacity
    
    # Color settings
    def set_normal_color(self, color: Tuple[float, float, float]) -> None:
        """Set normal marker color"""
        old_color = self.config.color_normal
        self.config.color_normal = self.config._validate_color(color)
        if old_color != self.config.color_normal:
            logger.info(f"Normal marker color changed: {old_color} -> {self.config.color_normal}")
            self._notify_change()
    
    def set_selected_color(self, color: Tuple[float, float, float]) -> None:
        """Set selected marker color"""
        old_color = self.config.color_selected
        self.config.color_selected = self.config._validate_color(color)
        if old_color != self.config.color_selected:
            logger.info(f"Selected marker color changed: {old_color} -> {self.config.color_selected}")
            self._notify_change()
    
    def set_pattern_color(self, color: Tuple[float, float, float]) -> None:
        """Set pattern marker color"""
        old_color = self.config.color_pattern
        self.config.color_pattern = self.config._validate_color(color)
        if old_color != self.config.color_pattern:
            logger.info(f"Pattern marker color changed: {old_color} -> {self.config.color_pattern}")
            self._notify_change()
    
    def apply_color_scheme(self, scheme_name: str) -> None:
        """Apply a predefined color scheme"""
        if scheme_name in self.color_schemes:
            scheme = self.color_schemes[scheme_name]
            # Apply marker colors
            self.config.color_normal = scheme["marker_normal"]
            self.config.color_selected = scheme["marker_selected"]
            self.config.color_pattern = scheme["marker_pattern"]
            # Apply skeleton colors
            self.skeleton_config.color_normal = scheme["skeleton_normal"]
            self.skeleton_config.color_outlier = scheme["skeleton_outlier"]
            logger.info(f"Applied color scheme: {scheme_name}")
            self._notify_change()
        else:
            logger.warning(f"Unknown color scheme: {scheme_name}")
    
    def get_color_schemes(self) -> List[str]:
        """Get list of available color schemes"""
        return list(self.color_schemes.keys())
    
    # Getter methods for renderer
    def get_normal_color(self) -> Tuple[float, float, float]:
        """Get normal marker color"""
        return self.config.color_normal
    
    def get_selected_color(self) -> Tuple[float, float, float]:
        """Get selected marker color"""
        return self.config.color_selected
    
    def get_pattern_color(self) -> Tuple[float, float, float]:
        """Get pattern marker color"""
        return self.config.color_pattern
    
    # Configuration management
    def get_config(self) -> MarkerVisualConfig:
        """Get current configuration"""
        return self.config
    
    def set_config(self, config: MarkerVisualConfig) -> None:
        """Set configuration"""
        old_config = self.config
        self.config = config
        if old_config != self.config:
            logger.info("Marker visual configuration updated")
            self._notify_change()
    
    def reset_to_defaults(self) -> None:
        """Reset all settings to default values"""
        old_config = self.config
        old_skeleton_config = self.skeleton_config
        self.config = MarkerVisualConfig()
        self.skeleton_config = SkeletonVisualConfig()
        if old_config != self.config or old_skeleton_config != self.skeleton_config:
            logger.info("Marker and skeleton visual settings reset to defaults")
            self._notify_change()

    # Skeleton-specific methods
    def set_skeleton_line_width(self, width: float) -> None:
        """Set skeleton line width (0.5 - 5.0)"""
        old_width = self.skeleton_config.line_width
        self.skeleton_config.line_width = max(0.5, min(5.0, width))
        if old_width != self.skeleton_config.line_width:
            logger.info(f"Skeleton line width changed: {old_width} -> {self.skeleton_config.line_width}")
            self._notify_change()

    def get_skeleton_line_width(self) -> float:
        """Get current skeleton line width"""
        return self.skeleton_config.line_width

    def set_skeleton_opacity(self, opacity: float) -> None:
        """Set skeleton opacity (0.1 - 1.0)"""
        old_opacity = self.skeleton_config.opacity
        self.skeleton_config.opacity = max(0.1, min(1.0, opacity))
        if old_opacity != self.skeleton_config.opacity:
            logger.info(f"Skeleton opacity changed: {old_opacity} -> {self.skeleton_config.opacity}")
            self._notify_change()

    def get_skeleton_opacity(self) -> float:
        """Get current skeleton opacity"""
        return self.skeleton_config.opacity

    def set_skeleton_normal_color(self, color: Tuple[float, float, float]) -> None:
        """Set normal skeleton color"""
        old_color = self.skeleton_config.color_normal
        self.skeleton_config.color_normal = validate_color(color)
        if old_color != self.skeleton_config.color_normal:
            logger.info(f"Normal skeleton color changed: {old_color} -> {self.skeleton_config.color_normal}")
            self._notify_change()

    def set_skeleton_outlier_color(self, color: Tuple[float, float, float]) -> None:
        """Set outlier skeleton color"""
        old_color = self.skeleton_config.color_outlier
        self.skeleton_config.color_outlier = validate_color(color)
        if old_color != self.skeleton_config.color_outlier:
            logger.info(f"Outlier skeleton color changed: {old_color} -> {self.skeleton_config.color_outlier}")
            self._notify_change()

    def get_skeleton_normal_color(self) -> Tuple[float, float, float]:
        """Get normal skeleton color"""
        return self.skeleton_config.color_normal

    def get_skeleton_outlier_color(self) -> Tuple[float, float, float]:
        """Get outlier skeleton color"""
        return self.skeleton_config.color_outlier
    
    def __str__(self) -> str:
        """String representation of current settings"""
        return (f"MarkerVisualSettings("
                f"marker_size={self.config.size}, "
                f"marker_opacity={self.config.opacity}, "
                f"marker_normal_color={self.config.color_normal}, "
                f"marker_selected_color={self.config.color_selected}, "
                f"marker_pattern_color={self.config.color_pattern}, "
                f"skeleton_line_width={self.skeleton_config.line_width}, "
                f"skeleton_opacity={self.skeleton_config.opacity}, "
                f"skeleton_normal_color={self.skeleton_config.color_normal}, "
                f"skeleton_outlier_color={self.skeleton_config.color_outlier})")
