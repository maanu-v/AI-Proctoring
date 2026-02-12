import cv2
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)

class HeadPoseEstimator:
    def __init__(self):
        # 3D model points (generic face model)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])

    def estimate(self, face_landmarks, image_shape):
        """
        Estimates head pose (yaw, pitch, roll) from face landmarks.
        """
        img_h, img_w, _ = image_shape
        
        # 2D image points from landmarks
        # Indices: Nose=1, Chin=152, Left Eye=33, Right Eye=263, Left Mouth=61, Right Mouth=291
        # Note: MediaPipe landmarks are normalized [0,1], need to scale to image size
        
        # Face Mesh Landmark Indices map to 3D model points
        # These indices are standard for MediaPipe Face Mesh (468 points)
        # 1: Nose tip
        # 152: Chin
        # 263: Left eye left corner (from observer perspective, actually Right Eye in mesh term? Let's verify)
        # 33: Right eye right corner 
        # 291: Right Mouth corner
        # 61: Left Mouth corner
        
        # Correct Mapping based on standard MediaPipe topology:
        # Nose Tip: 1
        # Chin: 199 or 152 (152 is bottom of chin)
        # Left Eye Outer: 33
        # Right Eye Outer: 263
        # Left Mouth Corner: 61
        # Right Mouth Corner: 291
        
        image_points = np.array([
            (face_landmarks[1].x * img_w, face_landmarks[1].y * img_h),     # Nose tip
            (face_landmarks[152].x * img_w, face_landmarks[152].y * img_h), # Chin
            (face_landmarks[33].x * img_w, face_landmarks[33].y * img_h),   # Left eye left corner
            (face_landmarks[263].x * img_w, face_landmarks[263].y * img_h), # Right eye right corner
            (face_landmarks[61].x * img_w, face_landmarks[61].y * img_h),   # Left Mouth corner
            (face_landmarks[291].x * img_w, face_landmarks[291].y * img_h)  # Right mouth corner
        ], dtype="double")

        # Camera internals
        focal_length = img_w
        center = (img_w / 2, img_h / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        
        dist_coeffs = np.zeros((4, 1)) # Assuming no lens distortion
        
        # PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points, 
            image_points, 
            camera_matrix, 
            dist_coeffs, 
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return None, None, None, None

        # Rotation Matrix
        rmat, _ = cv2.Rodrigues(rotation_vector)
        
        # Euler Angles
        # Decompose rotation matrix to get Euler angles
        # We use decomposition compatible with typical head pose definitions
        # proj_matrix = np.hstack((rmat, translation_vector))
        # euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6] 
        
        # Manual calculation for better control over conventions
        sy = np.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(rmat[2, 1], rmat[2, 2])
            y = np.arctan2(-rmat[2, 0], sy)
            z = np.arctan2(rmat[1, 0], rmat[0, 0])
        else:
            x = np.arctan2(-rmat[1, 2], rmat[1, 1])
            y = np.arctan2(-rmat[2, 0], sy)
            z = 0
            
        # Convert to degrees
        pitch = x * 180 / np.pi
        yaw = y * 180 / np.pi
        roll = z * 180 / np.pi
        
        return (pitch, yaw, roll), rotation_vector, translation_vector, camera_matrix

    def draw_axis(self, img, rotation_vector, translation_vector, camera_matrix, dist_coeffs=None):
        if dist_coeffs is None:
            dist_coeffs = np.zeros((4, 1))

        # Project 3D axis points to 2D
        # Axis length 100
        axis_length = 50.0
        axis_points = np.array([
            (axis_length, 0.0, 0.0),   # X axis (Red) - Pitch
            (0.0, axis_length, 0.0),   # Y axis (Green) - Yaw
            (0.0, 0.0, axis_length)    # Z axis (Blue) - Roll (Forward)
        ])
        
        # Project 3D points to 2D image plane
        imgpts, _ = cv2.projectPoints(axis_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        imgpts = np.int32(imgpts).reshape(-1, 2)
        
        # Nose tip as origin
        nose_tip_3d = np.array([(0.0, 0.0, 0.0)])
        nose_imgs, _ = cv2.projectPoints(nose_tip_3d, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        nose_pt = tuple(np.int32(nose_imgs.reshape(-1, 2)[0]))

        # Draw lines
        # X axis (Pitch) - Red
        cv2.line(img, nose_pt, tuple(imgpts[0]), (0, 0, 255), 3)
        # Y axis (Yaw) - Green
        cv2.line(img, nose_pt, tuple(imgpts[1]), (0, 255, 0), 3)
        # Z axis (Roll/Forward) - Blue
        cv2.line(img, nose_pt, tuple(imgpts[2]), (255, 0, 0), 3)
        
        return img
        
    def get_orientation_label(self, pitch, yaw, roll):
        # Thresholds
        YAW_THRESH = 20
        PITCH_THRESH = 15
        
        label = "Forward"
        
        if yaw < -YAW_THRESH:
            label = "Looking RIGHT"
        elif yaw > YAW_THRESH:
            label = "Looking LEFT"
        elif pitch < -PITCH_THRESH:
            label = "Looking DOWN"
        elif pitch > PITCH_THRESH:
            label = "Looking UP"
            
        return label
