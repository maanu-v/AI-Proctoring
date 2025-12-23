"""
Utility functions for working with face landmarks.
These functions can be used by other models for various facial analysis tasks.
"""

import numpy as np
from typing import Tuple, List
from src.models.face.face_landmarks import FaceLandmarks

def calculate_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1: First point (x, y) or (x, y, z)
        point2: Second point (x, y) or (x, y, z)
    
    Returns:
        Euclidean distance
    """
    return np.linalg.norm(point1 - point2)

def calculate_eye_aspect_ratio(eye_landmarks: np.ndarray) -> float:
    """
    Calculate Eye Aspect Ratio (EAR) for blink detection.
    
    Args:
        eye_landmarks: Array of eye landmark points (6 points expected)
    
    Returns:
        Eye Aspect Ratio value
    """
    if len(eye_landmarks) < 6:
        return 0.0
    
    # Vertical distances
    v1 = calculate_distance(eye_landmarks[1], eye_landmarks[5])
    v2 = calculate_distance(eye_landmarks[2], eye_landmarks[4])
    
    # Horizontal distance
    h = calculate_distance(eye_landmarks[0], eye_landmarks[3])
    
    # EAR formula
    ear = (v1 + v2) / (2.0 * h)
    return ear

def calculate_mouth_aspect_ratio(mouth_landmarks: np.ndarray) -> float:
    """
    Calculate Mouth Aspect Ratio (MAR) for mouth open detection.
    
    Args:
        mouth_landmarks: Array of mouth landmark points
    
    Returns:
        Mouth Aspect Ratio value
    """
    if len(mouth_landmarks) < 20:
        return 0.0
    
    # Vertical distances (top to bottom)
    v1 = calculate_distance(mouth_landmarks[2], mouth_landmarks[10])
    v2 = calculate_distance(mouth_landmarks[4], mouth_landmarks[8])
    
    # Horizontal distance (left to right)
    h = calculate_distance(mouth_landmarks[0], mouth_landmarks[6])
    
    # MAR formula
    mar = (v1 + v2) / (2.0 * h)
    return mar

def get_iris_position(iris_landmarks: np.ndarray, eye_landmarks: np.ndarray) -> Tuple[float, float]:
    """
    Calculate normalized iris position within the eye.
    
    Args:
        iris_landmarks: Array of iris landmark points
        eye_landmarks: Array of eye landmark points
    
    Returns:
        Tuple of (horizontal_ratio, vertical_ratio) where 0.5 is center
    """
    if iris_landmarks is None or len(iris_landmarks) == 0:
        return (0.5, 0.5)
    
    # Get iris center
    iris_center = iris_landmarks.mean(axis=0)
    
    # Get eye boundaries
    eye_left = eye_landmarks[:, 0].min()
    eye_right = eye_landmarks[:, 0].max()
    eye_top = eye_landmarks[:, 1].min()
    eye_bottom = eye_landmarks[:, 1].max()
    
    # Calculate normalized position
    h_ratio = (iris_center[0] - eye_left) / (eye_right - eye_left) if eye_right != eye_left else 0.5
    v_ratio = (iris_center[1] - eye_top) / (eye_bottom - eye_top) if eye_bottom != eye_top else 0.5
    
    return (h_ratio, v_ratio)

def estimate_head_pose(face_landmarks: FaceLandmarks) -> Tuple[float, float, float]:
    """
    Estimate head pose (pitch, yaw, roll) from face landmarks.
    
    Args:
        face_landmarks: FaceLandmarks object
    
    Returns:
        Tuple of (pitch, yaw, roll) in degrees
    """
    # Key points for head pose estimation
    # Nose tip, chin, left eye corner, right eye corner, left mouth corner, right mouth corner
    image_points = face_landmarks.to_pixel_coords(normalized=False)
    
    # 3D model points (generic face model)
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ], dtype=np.float64)
    
    # Corresponding 2D image points (landmark indices)
    landmark_indices = [1, 152, 33, 263, 61, 291]  # nose tip, chin, eye corners, mouth corners
    image_points_subset = np.array([
        image_points[idx] for idx in landmark_indices
    ], dtype=np.float64)
    
    # Camera matrix (assuming centered camera)
    focal_length = face_landmarks.image_width
    center = (face_landmarks.image_width / 2, face_landmarks.image_height / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Assuming no lens distortion
    dist_coeffs = np.zeros((4, 1))
    
    # Solve PnP
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points_subset,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        return (0.0, 0.0, 0.0)
    
    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    # Calculate Euler angles
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6
    
    if not singular:
        pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw = np.arctan2(-rotation_matrix[2, 0], sy)
        roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        pitch = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        yaw = np.arctan2(-rotation_matrix[2, 0], sy)
        roll = 0
    
    # Convert to degrees
    pitch = np.degrees(pitch)
    yaw = np.degrees(yaw)
    roll = np.degrees(roll)
    
    return (pitch, yaw, roll)

def get_face_bounding_box(face_landmarks: FaceLandmarks, padding: float = 0.1) -> Tuple[int, int, int, int]:
    """
    Get bounding box around the face with optional padding.
    
    Args:
        face_landmarks: FaceLandmarks object
        padding: Padding ratio (0.1 = 10% padding)
    
    Returns:
        Tuple of (x, y, width, height)
    """
    pixel_coords = face_landmarks.to_pixel_coords(normalized=False)
    
    x_min = int(pixel_coords[:, 0].min())
    x_max = int(pixel_coords[:, 0].max())
    y_min = int(pixel_coords[:, 1].min())
    y_max = int(pixel_coords[:, 1].max())
    
    width = x_max - x_min
    height = y_max - y_min
    
    # Add padding
    pad_w = int(width * padding)
    pad_h = int(height * padding)
    
    x_min = max(0, x_min - pad_w)
    y_min = max(0, y_min - pad_h)
    x_max = min(face_landmarks.image_width, x_max + pad_w)
    y_max = min(face_landmarks.image_height, y_max + pad_h)
    
    return (x_min, y_min, x_max - x_min, y_max - y_min)

def is_face_centered(face_landmarks: FaceLandmarks, threshold: float = 0.2) -> bool:
    """
    Check if face is centered in the frame.
    
    Args:
        face_landmarks: FaceLandmarks object
        threshold: Maximum deviation from center (0.2 = 20%)
    
    Returns:
        True if face is centered, False otherwise
    """
    pixel_coords = face_landmarks.to_pixel_coords(normalized=False)
    
    # Get face center
    face_center_x = pixel_coords[:, 0].mean()
    face_center_y = pixel_coords[:, 1].mean()
    
    # Get image center
    image_center_x = face_landmarks.image_width / 2
    image_center_y = face_landmarks.image_height / 2
    
    # Calculate deviation
    deviation_x = abs(face_center_x - image_center_x) / face_landmarks.image_width
    deviation_y = abs(face_center_y - image_center_y) / face_landmarks.image_height
    
    return deviation_x < threshold and deviation_y < threshold


# Import cv2 for head pose estimation
import cv2
