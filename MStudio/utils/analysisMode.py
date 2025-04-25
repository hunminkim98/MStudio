"""
Provides functions for analysis mode calculations (distance, angle) in MStudio.
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)

## AUTHORSHIP INFORMATION
__author__ = "Your Name / HunMin Kim" # Adjust authorship as needed
__copyright__ = ""
__credits__ = [""]
__license__ = ""
# from importlib.metadata import version
# __version__ = version('MStudio')
__maintainer__ = "HunMin Kim"
__email__ = "hunminkim98@gmail.com"
__status__ = "Development"

def calculate_distance(marker_pos1: np.ndarray, marker_pos2: np.ndarray) -> float | None:
    """
    Calculates the Euclidean distance between two 3D points.

    Args:
        marker_pos1: NumPy array representing the coordinates of the first marker [x, y, z].
        marker_pos2: NumPy array representing the coordinates of the second marker [x, y, z].

    Returns:
        The Euclidean distance in meters, or None if input is invalid.
    """
    try:
        if not isinstance(marker_pos1, np.ndarray) or marker_pos1.shape != (3,) or \
           not isinstance(marker_pos2, np.ndarray) or marker_pos2.shape != (3,):
            logger.warning("Invalid input format for distance calculation.")
            return None
        
        distance = np.linalg.norm(marker_pos1 - marker_pos2)
        # Assuming the input coordinates are already in meters.
        # If they are in mm, divide by 1000. Adjust if necessary.
        return float(distance) 
    except Exception as e:
        logger.error(f"Error calculating distance: {e}", exc_info=True)
        return None

def calculate_angle(marker_pos1: np.ndarray, marker_pos2: np.ndarray, marker_pos3: np.ndarray) -> float | None:
    """
    Calculates the angle in degrees formed by three 3D points (pos2 is the vertex).

    Args:
        marker_pos1: NumPy array for the first point [x, y, z].
        marker_pos2: NumPy array for the vertex point [x, y, z].
        marker_pos3: NumPy array for the third point [x, y, z].

    Returns:
        The angle in degrees, or None if input is invalid or points are collinear.
    """
    try:
        if not isinstance(marker_pos1, np.ndarray) or marker_pos1.shape != (3,) or \
           not isinstance(marker_pos2, np.ndarray) or marker_pos2.shape != (3,) or \
           not isinstance(marker_pos3, np.ndarray) or marker_pos3.shape != (3,):
            logger.warning("Invalid input format for angle calculation.")
            return None

        # Create vectors from the vertex point (marker_pos2)
        vec_a = marker_pos1 - marker_pos2
        vec_b = marker_pos3 - marker_pos2

        # Calculate norms (lengths) of the vectors
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        # Avoid division by zero if vectors have zero length
        if norm_a == 0 or norm_b == 0:
            logger.warning("Cannot calculate angle with zero-length vector.")
            return None

        # Calculate dot product
        dot_product = np.dot(vec_a, vec_b)

        # Calculate cosine of the angle, clamping value to [-1, 1] for numerical stability
        cos_theta = np.clip(dot_product / (norm_a * norm_b), -1.0, 1.0)

        # Calculate angle in radians and convert to degrees
        angle_rad = np.arccos(cos_theta)
        angle_deg = np.degrees(angle_rad)

        return float(angle_deg)
    except Exception as e:
        logger.error(f"Error calculating angle: {e}", exc_info=True)
        return None

# Example usage (for testing, can be removed later)
if __name__ == '__main__':
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.0, 0.0, 0.0])
    p3 = np.array([1.0, 1.0, 0.0])

    dist = calculate_distance(p1, p2)
    print(f"Distance between p1 and p2: {dist}") # Expected: 1.0

    angle = calculate_angle(p1, p2, p3)
    print(f"Angle at p2 (p1-p2-p3): {angle}") # Expected: 90.0
    
    p4 = np.array([2.0, 0.0, 0.0])
    angle_collinear = calculate_angle(p1, p2, p4)
    print(f"Angle at p2 (p1-p2-p4): {angle_collinear}") # Expected: 180.0 or 0.0 depending on direction

    angle_zero = calculate_angle(p1, p1, p3)
    print(f"Angle at p1 (p1-p1-p3): {angle_zero}") # Expected: None 