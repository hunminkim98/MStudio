"""
Outlier Detection for MStudio application.

This module provides optimized algorithms for detecting outliers in motion capture data
based on skeleton constraints and temporal consistency.
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)


class OutlierDetector:
    """
    Optimized outlier detection for motion capture data.
    
    This class provides efficient algorithms to detect outliers based on:
    - Skeleton bone length consistency
    - Temporal smoothness
    - Statistical thresholds
    """
    
    def __init__(self, threshold: float = 0.3, use_parallel: bool = True):
        """
        Initialize the outlier detector.
        
        Args:
            threshold: Relative change threshold for outlier detection
            use_parallel: Whether to use parallel processing for large datasets
        """
        self.threshold = threshold
        self.use_parallel = use_parallel
        self._cache = {}
        self._cache_lock = threading.Lock()
        
    def detect_outliers(self, data: pd.DataFrame, marker_names: List[str], 
                       skeleton_pairs: List[Tuple[str, str]]) -> Dict[str, np.ndarray]:
        """
        Detect outliers in the motion capture data.
        
        Args:
            data: DataFrame containing marker coordinate data
            marker_names: List of marker names
            skeleton_pairs: List of (parent, child) marker pairs defining skeleton
            
        Returns:
            Dictionary mapping marker names to boolean arrays indicating outliers
        """
        if not skeleton_pairs or data is None or data.empty:
            return {marker: np.zeros(len(data), dtype=bool) for marker in marker_names}
            
        logger.info(f"Detecting outliers for {len(data)} frames with {len(skeleton_pairs)} skeleton pairs")
        
        # Initialize outlier arrays
        outliers = {marker: np.zeros(len(data), dtype=bool) for marker in marker_names}
        
        try:
            if self.use_parallel and len(data) > 1000:
                outliers = self._detect_outliers_parallel(data, marker_names, skeleton_pairs, outliers)
            else:
                outliers = self._detect_outliers_sequential(data, skeleton_pairs, outliers)
                
            # Log statistics
            total_outliers = sum(np.sum(outlier_array) for outlier_array in outliers.values())
            logger.info(f"Detected {total_outliers} outliers across all markers")
            
        except Exception as e:
            logger.error("Error in outlier detection: %s", e, exc_info=True)
            
        return outliers
        
    def _detect_outliers_sequential(self, data: pd.DataFrame, 
                                  skeleton_pairs: List[Tuple[str, str]], 
                                  outliers: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Sequential outlier detection implementation."""
        
        # Pre-compute bone lengths for all pairs and frames
        bone_lengths = self._compute_bone_lengths(data, skeleton_pairs)
        
        # Detect outliers based on bone length changes
        for pair_idx, (parent, child) in enumerate(skeleton_pairs):
            lengths = bone_lengths[pair_idx]
            
            # Skip if we don't have enough data
            if len(lengths) < 2:
                continue
                
            # Compute relative changes
            length_changes = np.abs(np.diff(lengths)) / (lengths[:-1] + 1e-8)  # Add small epsilon
            
            # Find outliers
            outlier_frames = np.where(length_changes > self.threshold)[0] + 1  # +1 because diff reduces length
            
            # Mark both parent and child as outliers
            for frame in outlier_frames:
                if frame < len(outliers[parent]):
                    outliers[parent][frame] = True
                if frame < len(outliers[child]):
                    outliers[child][frame] = True
                    
        return outliers
        
    def _detect_outliers_parallel(self, data: pd.DataFrame, marker_names: List[str],
                                skeleton_pairs: List[Tuple[str, str]], 
                                outliers: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Parallel outlier detection implementation for large datasets."""
        
        # Split skeleton pairs into chunks for parallel processing
        chunk_size = max(1, len(skeleton_pairs) // 4)  # Use 4 threads
        pair_chunks = [skeleton_pairs[i:i + chunk_size] 
                      for i in range(0, len(skeleton_pairs), chunk_size)]
        
        def process_chunk(pairs_chunk):
            chunk_outliers = {marker: np.zeros(len(data), dtype=bool) for marker in marker_names}
            return self._detect_outliers_sequential(data, pairs_chunk, chunk_outliers)
            
        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            chunk_results = list(executor.map(process_chunk, pair_chunks))
            
        # Combine results
        for chunk_result in chunk_results:
            for marker in marker_names:
                outliers[marker] = np.logical_or(outliers[marker], chunk_result[marker])
                
        return outliers
        
    def _compute_bone_lengths(self, data: pd.DataFrame, 
                            skeleton_pairs: List[Tuple[str, str]]) -> List[np.ndarray]:
        """
        Efficiently compute bone lengths for all pairs and frames.
        
        Args:
            data: DataFrame containing marker data
            skeleton_pairs: List of (parent, child) pairs
            
        Returns:
            List of numpy arrays containing bone lengths for each pair
        """
        bone_lengths = []
        
        for parent, child in skeleton_pairs:
            try:
                # Get coordinates for both markers
                parent_coords = data[[f'{parent}_X', f'{parent}_Y', f'{parent}_Z']].values
                child_coords = data[[f'{child}_X', f'{child}_Y', f'{child}_Z']].values
                
                # Compute distances using vectorized operations
                distances = np.linalg.norm(child_coords - parent_coords, axis=1)
                bone_lengths.append(distances)
                
            except KeyError as e:
                logger.warning(f"Missing coordinate data for pair ({parent}, {child}): {e}")
                bone_lengths.append(np.array([]))
                
        return bone_lengths
        
    def detect_statistical_outliers(self, data: pd.DataFrame, marker_names: List[str],
                                  z_threshold: float = 3.0) -> Dict[str, np.ndarray]:
        """
        Detect outliers using statistical methods (Z-score).
        
        Args:
            data: DataFrame containing marker data
            marker_names: List of marker names
            z_threshold: Z-score threshold for outlier detection
            
        Returns:
            Dictionary mapping marker names to boolean arrays indicating outliers
        """
        outliers = {marker: np.zeros(len(data), dtype=bool) for marker in marker_names}
        
        try:
            for marker in marker_names:
                for axis in ['X', 'Y', 'Z']:
                    col_name = f'{marker}_{axis}'
                    if col_name in data.columns:
                        values = data[col_name].values
                        
                        # Skip if all values are NaN
                        if np.all(np.isnan(values)):
                            continue
                            
                        # Compute Z-scores
                        mean_val = np.nanmean(values)
                        std_val = np.nanstd(values)
                        
                        if std_val > 0:
                            z_scores = np.abs((values - mean_val) / std_val)
                            axis_outliers = z_scores > z_threshold
                            outliers[marker] = np.logical_or(outliers[marker], axis_outliers)
                            
        except Exception as e:
            logger.error("Error in statistical outlier detection: %s", e, exc_info=True)
            
        return outliers
        
    def smooth_outliers(self, data: pd.DataFrame, outliers: Dict[str, np.ndarray],
                       method: str = 'linear') -> pd.DataFrame:
        """
        Smooth detected outliers using interpolation.
        
        Args:
            data: DataFrame containing marker data
            outliers: Dictionary of outlier boolean arrays
            method: Interpolation method ('linear', 'cubic', etc.)
            
        Returns:
            DataFrame with smoothed outliers
        """
        smoothed_data = data.copy()
        
        try:
            for marker, outlier_mask in outliers.items():
                if not np.any(outlier_mask):
                    continue
                    
                for axis in ['X', 'Y', 'Z']:
                    col_name = f'{marker}_{axis}'
                    if col_name in smoothed_data.columns:
                        # Set outliers to NaN and interpolate
                        values = smoothed_data[col_name].copy()
                        values[outlier_mask] = np.nan
                        smoothed_data[col_name] = values.interpolate(method=method)
                        
        except Exception as e:
            logger.error("Error smoothing outliers: %s", e, exc_info=True)
            
        return smoothed_data
        
    def set_threshold(self, threshold: float) -> None:
        """Set the outlier detection threshold."""
        self.threshold = max(0.0, threshold)
        logger.info(f"Outlier detection threshold set to {self.threshold}")
        
    def clear_cache(self) -> None:
        """Clear the internal cache."""
        with self._cache_lock:
            self._cache.clear()
            logger.debug("Outlier detector cache cleared")
