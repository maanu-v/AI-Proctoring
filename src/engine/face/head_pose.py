# src/engine/face/head_pose.py

import numpy as np

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
