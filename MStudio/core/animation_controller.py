"""
Animation Controller for MStudio application.

This module handles all animation-related functionality including playback control,
frame management, and timeline operations.
"""

import logging
import time
from typing import Optional, Callable, Any

logger = logging.getLogger(__name__)


class AnimationController:
    """
    Controls animation playback and frame management.
    
    This class handles:
    - Play/pause/stop functionality
    - Frame navigation
    - Animation timing and speed control
    - Loop management
    """
    
    def __init__(self, parent_widget: Any):
        """
        Initialize the animation controller.
        
        Args:
            parent_widget: The parent widget (usually TRCViewer) for scheduling callbacks
        """
        self.parent = parent_widget
        
        # Animation state
        self.is_playing: bool = False
        self.frame_idx: int = 0
        self.num_frames: int = 0
        self.animation_job: Optional[str] = None
        
        # Animation settings
        self.fps: float = 60.0
        self.loop_enabled: bool = False

        # Performance optimization
        self._last_frame_time: float = 0.0
        self._target_frame_time: float = 1.0 / 60.0  # Target time per frame

        # Callbacks
        self.frame_update_callback: Optional[Callable[[int], None]] = None
        self.animation_state_callback: Optional[Callable[[bool], None]] = None
        
    def set_data_info(self, num_frames: int, fps: float = 60.0) -> None:
        """
        Set the data information for animation.
        
        Args:
            num_frames: Total number of frames in the data
            fps: Frames per second for playback
        """
        self.num_frames = num_frames
        self.fps = fps
        self.frame_idx = min(self.frame_idx, max(0, num_frames - 1))
        logger.info(f"Animation data set: {num_frames} frames at {fps} FPS")
        
    def set_frame_update_callback(self, callback: Callable[[int], None]) -> None:
        """Set the callback function for frame updates."""
        self.frame_update_callback = callback
        
    def set_animation_state_callback(self, callback: Callable[[bool], None]) -> None:
        """Set the callback function for animation state changes."""
        self.animation_state_callback = callback
        
    def play(self) -> None:
        """Start animation playback."""
        if self.num_frames <= 1:
            logger.warning("Cannot play animation: insufficient frames")
            return
            
        if not self.is_playing:
            self.is_playing = True
            self._notify_state_change()
            self._schedule_next_frame()
            logger.info("Animation started")
            
    def pause(self) -> None:
        """Pause animation playback."""
        if self.is_playing:
            self.is_playing = False
            self._cancel_scheduled_frame()
            self._notify_state_change()
            logger.info("Animation paused")
            
    def stop(self) -> None:
        """Stop animation and return to first frame."""
        was_playing = self.is_playing
        self.is_playing = False
        self._cancel_scheduled_frame()
        
        # Return to first frame
        self.set_frame(0)
        
        if was_playing:
            self._notify_state_change()
        logger.info("Animation stopped")
        
    def toggle_play_pause(self) -> None:
        """Toggle between play and pause states."""
        if self.is_playing:
            self.pause()
        else:
            self.play()
            
    def set_frame(self, frame_idx: int, from_external: bool = False) -> None:
        """
        Set the current frame index.

        Args:
            frame_idx: The frame index to set
            from_external: True if this is called from external UI interaction (timeline drag, etc.)
        """
        if self.num_frames == 0:
            return

        # Clamp frame index to valid range
        old_frame = self.frame_idx
        self.frame_idx = max(0, min(frame_idx, self.num_frames - 1))

        # If this is an external frame change during playback, we need to handle it specially
        if from_external and self.is_playing:
            # Cancel current animation job and reschedule from new position
            self._cancel_scheduled_frame()
            self._schedule_next_frame()
            logger.debug(f"Frame manually changed to {self.frame_idx} during playback")

        # Notify if frame actually changed
        if old_frame != self.frame_idx and self.frame_update_callback:
            self.frame_update_callback(self.frame_idx)
            
    def next_frame(self) -> None:
        """Move to the next frame."""
        if self.frame_idx < self.num_frames - 1:
            self.set_frame(self.frame_idx + 1)
        elif self.loop_enabled and self.num_frames > 1:
            self.set_frame(0)
            
    def prev_frame(self) -> None:
        """Move to the previous frame."""
        if self.frame_idx > 0:
            self.set_frame(self.frame_idx - 1)
        elif self.loop_enabled and self.num_frames > 1:
            self.set_frame(self.num_frames - 1)
            
    def set_fps(self, fps: float) -> None:
        """
        Set the animation frame rate.

        Args:
            fps: Frames per second
        """
        self.fps = max(1.0, fps)  # Minimum 1 FPS
        self._target_frame_time = 1.0 / self.fps
        logger.info(f"Animation FPS set to {self.fps}")
        
    def set_loop(self, enabled: bool) -> None:
        """
        Enable or disable animation looping.
        
        Args:
            enabled: Whether to enable looping
        """
        self.loop_enabled = enabled
        logger.info(f"Animation loop {'enabled' if enabled else 'disabled'}")
        
    def get_current_time(self) -> float:
        """
        Get the current time in seconds based on frame index and FPS.
        
        Returns:
            Current time in seconds
        """
        return self.frame_idx / self.fps if self.fps > 0 else 0.0
        
    def set_time(self, time_seconds: float) -> None:
        """
        Set the current time in seconds.
        
        Args:
            time_seconds: Time in seconds
        """
        frame_idx = int(time_seconds * self.fps)
        self.set_frame(frame_idx)
        
    def get_progress(self) -> float:
        """
        Get the animation progress as a value between 0 and 1.
        
        Returns:
            Progress value (0.0 to 1.0)
        """
        if self.num_frames <= 1:
            return 0.0
        return self.frame_idx / (self.num_frames - 1)
        
    def set_progress(self, progress: float) -> None:
        """
        Set the animation progress.
        
        Args:
            progress: Progress value between 0.0 and 1.0
        """
        progress = max(0.0, min(1.0, progress))
        frame_idx = int(progress * (self.num_frames - 1)) if self.num_frames > 1 else 0
        self.set_frame(frame_idx)
        
    def _schedule_next_frame(self) -> None:
        """Schedule the next frame update with optimized timing."""
        if not self.is_playing:
            return

        # Calculate delay based on FPS with better precision
        delay_ms = max(1, int(1000.0 / self.fps))  # Minimum 1ms delay

        # OPTIMIZATION: Always use timed scheduling to avoid blocking mouse events
        # after_idle() saturates the event loop and blocks camera controls
        # Use precise timing for all frame rates to maintain smooth camera interaction
        self.animation_job = self.parent.after(delay_ms, self._animate_step)
        
    def _cancel_scheduled_frame(self) -> None:
        """Cancel any scheduled frame update."""
        if self.animation_job:
            self.parent.after_cancel(self.animation_job)
            self.animation_job = None
            
    def _animate_step(self) -> None:
        """Execute one animation step with frame rate limiting."""
        if not self.is_playing:
            return

        # Frame rate limiting for smooth animation
        current_time = time.time()
        if current_time - self._last_frame_time < self._target_frame_time:
            # Too early for next frame, reschedule
            self._schedule_next_frame()
            return

        self._last_frame_time = current_time

        # Move to next frame
        if self.frame_idx < self.num_frames - 1:
            self.set_frame(self.frame_idx + 1)
            self._schedule_next_frame()
        else:
            # End of animation
            if self.loop_enabled:
                self.set_frame(0)
                self._schedule_next_frame()
            else:
                self.stop()
                
    def _notify_state_change(self) -> None:
        """Notify about animation state changes."""
        if self.animation_state_callback:
            self.animation_state_callback(self.is_playing)
            
    def cleanup(self) -> None:
        """Clean up resources and cancel any pending animations."""
        self.pause()
        self._cancel_scheduled_frame()
        logger.info("Animation controller cleaned up")
        
    def get_state_info(self) -> dict:
        """
        Get current animation state information.
        
        Returns:
            Dictionary containing current state information
        """
        return {
            'is_playing': self.is_playing,
            'frame_idx': self.frame_idx,
            'num_frames': self.num_frames,
            'fps': self.fps,
            'loop_enabled': self.loop_enabled,
            'current_time': self.get_current_time(),
            'progress': self.get_progress()
        }
