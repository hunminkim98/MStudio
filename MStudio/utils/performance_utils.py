"""
Performance utilities for MStudio application.

This module provides utilities for performance optimization including
caching, memoization, and efficient data operations.
"""

import logging
import time
import functools
from typing import Any, Callable, Dict, Optional, Tuple
import threading
import weakref

logger = logging.getLogger(__name__)


class PerformanceTimer:
    """Context manager for timing code execution."""
    
    def __init__(self, name: str, log_level: int = logging.INFO):
        self.name = name
        self.log_level = log_level
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        duration = self.end_time - self.start_time
        logger.log(self.log_level, f"{self.name} took {duration:.4f} seconds")
        
    @property
    def duration(self) -> Optional[float]:
        """Get the duration of the timed operation."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class LRUCache:
    """Thread-safe LRU (Least Recently Used) cache implementation."""
    
    def __init__(self, max_size: int = 128):
        self.max_size = max_size
        self.cache: Dict[Any, Any] = {}
        self.access_order: Dict[Any, int] = {}
        self.access_counter = 0
        self.lock = threading.RLock()
        
    def get(self, key: Any, default: Any = None) -> Any:
        """Get a value from the cache."""
        with self.lock:
            if key in self.cache:
                self.access_counter += 1
                self.access_order[key] = self.access_counter
                return self.cache[key]
            return default
            
    def put(self, key: Any, value: Any) -> None:
        """Put a value into the cache."""
        with self.lock:
            if key in self.cache:
                self.cache[key] = value
                self.access_counter += 1
                self.access_order[key] = self.access_counter
            else:
                if len(self.cache) >= self.max_size:
                    self._evict_lru()
                self.cache[key] = value
                self.access_counter += 1
                self.access_order[key] = self.access_counter
                
    def _evict_lru(self) -> None:
        """Evict the least recently used item."""
        if not self.cache:
            return
            
        lru_key = min(self.access_order.keys(), key=lambda k: self.access_order[k])
        del self.cache[lru_key]
        del self.access_order[lru_key]
        
    def clear(self) -> None:
        """Clear the cache."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.access_counter = 0
            
    def size(self) -> int:
        """Get the current cache size."""
        return len(self.cache)


def memoize(max_size: int = 128):
    """
    Decorator for memoizing function results with LRU cache.
    
    Args:
        max_size: Maximum number of cached results
    """
    def decorator(func: Callable) -> Callable:
        cache = LRUCache(max_size)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from arguments
            key = (args, tuple(sorted(kwargs.items())))
            
            # Try to get from cache
            result = cache.get(key)
            if result is not None:
                return result
                
            # Compute and cache result
            result = func(*args, **kwargs)
            cache.put(key, result)
            return result
            
        wrapper.cache = cache
        wrapper.cache_clear = cache.clear
        wrapper.cache_info = lambda: {'size': cache.size(), 'max_size': max_size}
        return wrapper
    return decorator


def debounce(wait_time: float):
    """
    Decorator that debounces function calls.
    
    Args:
        wait_time: Time to wait before executing the function
    """
    def decorator(func: Callable) -> Callable:
        timer = None
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal timer
            
            def call_func():
                func(*args, **kwargs)
                
            if timer:
                timer.cancel()
            timer = threading.Timer(wait_time, call_func)
            timer.start()
            
        return wrapper
    return decorator


def throttle(min_interval: float):
    """
    Decorator that throttles function calls to a minimum interval.

    Args:
        min_interval: Minimum time between function calls
    """
    def decorator(func: Callable) -> Callable:
        last_called = [0.0]
        lock = threading.Lock()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with lock:
                now = time.time()
                if now - last_called[0] >= min_interval:
                    last_called[0] = now
                    return func(*args, **kwargs)

        return wrapper
    return decorator


def animation_optimized(func: Callable) -> Callable:
    """
    Decorator to optimize functions for animation performance.

    This decorator adds frame rate limiting and reduces redundant calls
    during animation playback.
    """
    last_call_time = [0.0]
    min_interval = 1.0 / 120.0  # Max 120 FPS

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        current_time = time.time()
        if current_time - last_call_time[0] >= min_interval:
            last_call_time[0] = current_time
            return func(*args, **kwargs)
        # Skip call if too frequent
        return None

    return wrapper


def smooth_camera_controls(func: Callable) -> Callable:
    """
    Decorator to ensure smooth camera controls during animation.

    This decorator bypasses frame rate limiting for mouse interactions
    to maintain responsive camera movement during animation playback.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check if this is a mouse interaction during animation
        if len(args) > 0 and hasattr(args[0], '_context_active'):
            # Force immediate execution for camera controls
            return func(*args, **kwargs)
        else:
            # Normal execution for non-camera operations
            return func(*args, **kwargs)

    return wrapper


class AnimationPerformanceManager:
    """
    Manager class for optimizing animation performance.

    Handles context switching, frame rate limiting, and resource management
    during animation playback to ensure smooth 60 FPS performance.
    """

    def __init__(self):
        self.animation_active = False
        self.last_frame_time = 0.0
        self.target_fps = 60.0
        self.context_cache = {}

    def set_animation_active(self, active: bool) -> None:
        """Set whether animation is currently active."""
        self.animation_active = active
        if not active:
            # Clear context cache when animation stops
            self.context_cache.clear()

    def should_render_frame(self) -> bool:
        """Check if a new frame should be rendered based on target FPS."""
        current_time = time.time()
        target_interval = 1.0 / self.target_fps

        if current_time - self.last_frame_time >= target_interval:
            self.last_frame_time = current_time
            return True
        return False

    def optimize_for_animation(self, renderer) -> None:
        """Apply animation-specific optimizations to a renderer."""
        if hasattr(renderer, '_context_active'):
            renderer._context_active = True
        if hasattr(renderer, '_pending_redraw'):
            renderer._pending_redraw = False


class WeakMethodCache:
    """Cache that holds weak references to methods to avoid memory leaks."""
    
    def __init__(self, max_size: int = 64):
        self.max_size = max_size
        self.cache: Dict[Tuple, Any] = {}
        self.weak_refs: Dict[Tuple, weakref.ref] = {}
        self.access_order: Dict[Tuple, int] = {}
        self.access_counter = 0
        self.lock = threading.RLock()
        
    def get_method_key(self, obj: Any, method_name: str, args: Tuple, kwargs: Dict) -> Tuple:
        """Generate a cache key for a method call."""
        obj_id = id(obj)
        return (obj_id, method_name, args, tuple(sorted(kwargs.items())))
        
    def get(self, obj: Any, method_name: str, args: Tuple, kwargs: Dict) -> Any:
        """Get a cached method result."""
        key = self.get_method_key(obj, method_name, args, kwargs)
        
        with self.lock:
            # Check if object still exists
            if key in self.weak_refs:
                ref = self.weak_refs[key]
                if ref() is None:  # Object was garbage collected
                    self._remove_key(key)
                    return None
                    
            if key in self.cache:
                self.access_counter += 1
                self.access_order[key] = self.access_counter
                return self.cache[key]
                
        return None
        
    def put(self, obj: Any, method_name: str, args: Tuple, kwargs: Dict, result: Any) -> None:
        """Cache a method result."""
        key = self.get_method_key(obj, method_name, args, kwargs)
        
        with self.lock:
            if len(self.cache) >= self.max_size:
                self._evict_lru()
                
            self.cache[key] = result
            self.weak_refs[key] = weakref.ref(obj, lambda ref: self._remove_key(key))
            self.access_counter += 1
            self.access_order[key] = self.access_counter
            
    def _remove_key(self, key: Tuple) -> None:
        """Remove a key from all caches."""
        self.cache.pop(key, None)
        self.weak_refs.pop(key, None)
        self.access_order.pop(key, None)
        
    def _evict_lru(self) -> None:
        """Evict the least recently used item."""
        if not self.cache:
            return
            
        lru_key = min(self.access_order.keys(), key=lambda k: self.access_order[k])
        self._remove_key(lru_key)
        
    def clear(self) -> None:
        """Clear all caches."""
        with self.lock:
            self.cache.clear()
            self.weak_refs.clear()
            self.access_order.clear()
            self.access_counter = 0


# Removed unused cached_method decorator to reduce dead code


class BatchProcessor:
    """Utility for processing items in batches to improve performance."""
    
    def __init__(self, batch_size: int = 100, process_func: Optional[Callable] = None):
        self.batch_size = batch_size
        self.process_func = process_func
        self.pending_items = []
        self.lock = threading.Lock()
        
    def add_item(self, item: Any) -> None:
        """Add an item to be processed."""
        with self.lock:
            self.pending_items.append(item)
            if len(self.pending_items) >= self.batch_size:
                self._process_batch()
                
    def flush(self) -> None:
        """Process all pending items."""
        with self.lock:
            if self.pending_items:
                self._process_batch()
                
    def _process_batch(self) -> None:
        """Process the current batch of items."""
        if not self.pending_items or not self.process_func:
            return
            
        try:
            batch = self.pending_items.copy()
            self.pending_items.clear()
            self.process_func(batch)
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            
    def set_process_func(self, func: Callable) -> None:
        """Set the function to process batches."""
        self.process_func = func


# Removed unused profile_memory_usage decorator to reduce dead code
