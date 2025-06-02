"""
Provides functions for analysis mode calculations (distance, angle) in MStudio.
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)

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

def calculate_arc_points(vertex: np.ndarray, p1: np.ndarray, p3: np.ndarray, radius: float = 0.05, num_segments: int = 20) -> list[np.ndarray] | None:
    """
    Calculates points along a 3D arc defined by three points (vertex is the center).

    Args:
        vertex: The vertex of the angle (center of the arc). Shape (3,).
        p1: The first point defining the angle. Shape (3,).
        p3: The third point defining the angle. Shape (3,).
        radius: The desired radius of the arc.
        num_segments: The number of line segments to approximate the arc.

    Returns:
        A list of NumPy arrays representing points along the arc, or None if calculation fails.
    """
    try:
        v1 = p1 - vertex
        v3 = p3 - vertex
        norm_v1 = np.linalg.norm(v1)
        norm_v3 = np.linalg.norm(v3)

        if norm_v1 == 0 or norm_v3 == 0:
            logger.warning("Cannot calculate arc with zero-length vector.")
            return None

        v1_norm = v1 / norm_v1
        v3_norm = v3 / norm_v3

        # Calculate the angle between the vectors
        dot_product = np.clip(np.dot(v1_norm, v3_norm), -1.0, 1.0)
        angle_rad = np.arccos(dot_product)

        # If angle is very small or close to 180, don't draw arc (or points are collinear)
        if np.isclose(angle_rad, 0.0) or np.isclose(angle_rad, np.pi):
            return None 

        # Calculate the normal vector to the plane of the arc
        cross_product = np.cross(v1_norm, v3_norm)
        norm_cross = np.linalg.norm(cross_product)
        if np.isclose(norm_cross, 0.0):
             return None # Vectors are collinear
        plane_normal = cross_product / norm_cross

        # Calculate the axis perpendicular to v1_norm within the plane
        axis2 = np.cross(plane_normal, v1_norm)
        axis2_norm = np.linalg.norm(axis2)
        if np.isclose(axis2_norm, 0.0):
            return None # Should not happen if vectors are not collinear
        axis2 = axis2 / axis2_norm # Normalized second axis in the plane
        
        # Generate points along the arc
        arc_points = []
        for i in range(num_segments + 1):
            phi = angle_rad * (i / num_segments)
            # Point on circle in the plane = R * (axis1*cos(phi) + axis2*sin(phi))
            point_on_arc = vertex + radius * (v1_norm * np.cos(phi) + axis2 * np.sin(phi))
            arc_points.append(point_on_arc)

        return arc_points

    except Exception as e:
        logger.error(f"Error calculating arc points: {e}", exc_info=True)
        return None

def calculate_velocity(pos_prev: np.ndarray, pos_curr: np.ndarray, pos_next: np.ndarray, frame_rate: float) -> np.ndarray | None:
    """
    Calculates the velocity vector of a marker at the current frame using central difference.

    Args:
        pos_prev: Position vector [x, y, z] at the previous frame (i-1).
        pos_curr: Position vector [x, y, z] at the current frame (i).
        pos_next: Position vector [x, y, z] at the next frame (i+1).
        frame_rate: The frame rate of the data (frames per second).

    Returns:
        A NumPy array representing the velocity vector [vx, vy, vz] in units per second,
        or None if input is invalid or calculation fails.
    """
    try:
        if not all(isinstance(p, np.ndarray) and p.shape == (3,) for p in [pos_prev, pos_curr, pos_next]) or frame_rate <= 0:
            logger.warning("Invalid input for velocity calculation.")
            return None
            
        # Check for NaN values in positions used for calculation
        if np.isnan(pos_prev).any() or np.isnan(pos_next).any():
             logger.debug("NaN values encountered in adjacent frames, cannot calculate velocity.")
             return None 

        # Central difference formula: v(i) ≈ (pos(i+1) - pos(i-1)) / (2 * dt)
        dt = 1.0 / frame_rate
        velocity_vector = (pos_next - pos_prev) / (2.0 * dt)

        return velocity_vector

    except Exception as e:
        logger.error(f"Error calculating velocity: {e}", exc_info=True)
        return None

def calculate_acceleration(vel_prev: np.ndarray, vel_next: np.ndarray, frame_rate: float) -> np.ndarray | None:
    """
    Calculates the acceleration vector using central difference on velocity vectors.

    Args:
        vel_prev: Velocity vector [vx, vy, vz] at the previous frame (i-1).
        vel_next: Velocity vector [vx, vy, vz] at the next frame (i+1).
        frame_rate: The frame rate of the data (frames per second).

    Returns:
        A NumPy array representing the acceleration vector [ax, ay, az] in units/s²,
        or None if input is invalid or calculation fails.
    """
    try:
        if not isinstance(vel_prev, np.ndarray) or vel_prev.shape != (3,) or \
           not isinstance(vel_next, np.ndarray) or vel_next.shape != (3,) or frame_rate <= 0:
            logger.warning("Invalid input for acceleration calculation.")
            return None
            
        # Check for NaN values in velocities
        if np.isnan(vel_prev).any() or np.isnan(vel_next).any():
             logger.debug("NaN values encountered in adjacent velocities, cannot calculate acceleration.")
             return None

        # Central difference formula: a(i) ≈ (vel(i+1) - vel(i-1)) / (2 * dt)
        dt = 1.0 / frame_rate
        acceleration_vector = (vel_next - vel_prev) / (2.0 * dt)

        return acceleration_vector

    except Exception as e:
        logger.error(f"Error calculating acceleration: {e}", exc_info=True)
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