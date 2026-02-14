# src/engine/face/head_pose.py

import numpy as np
import cv2
import math

class HeadPoseEstimator:
    def __init__(self, yaw_threshold=30, pitch_threshold=25, roll_threshold=25):
        """
        Initialize HeadPoseEstimator with thresholds.
        Default values are provided as fallback, but should be overridden by config.
        """
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.roll_threshold = roll_threshold

    def extract_pose(self, result):
        """
        Extract yaw, pitch, roll from MediaPipe result.
        Returns list of pose dicts (one per face).
        """
        if not result.facial_transformation_matrixes:
            return []

        poses = []

        for matrix in result.facial_transformation_matrixes:
            mat = np.array(matrix).reshape(4, 4)

            R = mat[:3, :3]

            sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
            singular = sy < 1e-6

            if not singular:
                pitch = np.arctan2(R[2, 1], R[2, 2])
                yaw = np.arctan2(-R[2, 0], sy)
                roll = np.arctan2(R[1, 0], R[0, 0])
            else:
                pitch = np.arctan2(-R[1, 2], R[1, 1])
                yaw = np.arctan2(-R[2, 0], sy)
                roll = 0

            poses.append({
                "pitch": np.degrees(pitch),
                "yaw": np.degrees(yaw),
                "roll": np.degrees(roll)
            })

        return poses

    def classify_direction(self, pose):
        """
        Classify head direction based on thresholds.
        """
        yaw = pose["yaw"]
        pitch = pose["pitch"]

        if yaw > self.yaw_threshold:
            return "Looking Right"
        elif yaw < -self.yaw_threshold:
            return "Looking Left"
        elif pitch > self.pitch_threshold:
            return "Looking Down"
        elif pitch < -self.pitch_threshold:
            return "Looking Up"
        else:
            return "Forward"

    def _get_rotation_matrix(self, pitch, yaw, roll):
        """
        Calculate rotation matrix from Euler angles (in degrees).
        """
        pitch = np.radians(pitch)
        yaw = np.radians(yaw)
        roll = np.radians(roll)

        Rx = np.array([[1, 0, 0],
                       [0, np.cos(pitch), -np.sin(pitch)],
                       [0, np.sin(pitch), np.cos(pitch)]])

        Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                       [0, 1, 0],
                       [-np.sin(yaw), 0, np.cos(yaw)]])

        Rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                       [np.sin(roll), np.cos(roll), 0],
                       [0, 0, 1]])

        # Assume ZYX order for rotation matrix composition
        R = Rz @ Ry @ Rx
        return R

    def draw_axes(self, image, pitch, yaw, roll, tx, ty, size=50):
        """
        Draw 3D axes on the image centered at (tx, ty).
        """
        R = self._get_rotation_matrix(pitch, yaw, roll)

        # Axis points in 3D
        # X - Red (Right), Y - Green (Down), Z - Blue (Forward)
        # Note: Coordinate system here assumes standard 3D convention
        # X: Right, Y: Down, Z: Forward (Screen into viewer or vice versa)
        
        # Determine axis endpoints
        # x_axis: (size, 0, 0)
        # y_axis: (0, size, 0)
        # z_axis: (0, 0, size)

        # Project to 2D (orthographic projection for simplicity)
        # x' = R * x
        
        # Coordinate system alignment for drawing:
        # OpenCV Image: X right, Y down.
        # Head Pose: Pitch (X), Yaw (Y), Roll (Z).
        
        xAxis = np.array([size, 0, 0])
        yAxis = np.array([0, size, 0])
        zAxis = np.array([0, 0, size])
        
        # Apply rotation
        draw_x = R @ xAxis
        draw_y = R @ yAxis
        draw_z = R @ zAxis
        
        # Draw lines
        # X-Axis (Red) - Pitch
        cv2.line(image, (int(tx), int(ty)), (int(tx + draw_x[0]), int(ty + draw_x[1])), (0, 0, 255), 2)
        # Y-Axis (Green) - Yaw
        cv2.line(image, (int(tx), int(ty)), (int(tx + draw_y[0]), int(ty + draw_y[1])), (0, 255, 0), 2)
        # Z-Axis (Blue) - Roll/Forward
        cv2.line(image, (int(tx), int(ty)), (int(tx + draw_z[0]), int(ty + draw_z[1])), (255, 0, 0), 2)
        
        return image
