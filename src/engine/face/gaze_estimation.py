import numpy as np
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class GazeEstimator:
    # Left Eye Indices (Anatomical Left = Person's Left)
    # Iris 468-472 are spatially near eye corners 33/133
    LEFT_EYE_IRIS = [468, 469, 470, 471, 472]
    LEFT_EYE_OUTER = 33
    LEFT_EYE_INNER = 133
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145

    # Right Eye Indices (Anatomical Right = Person's Right)
    # Iris 473-477 are spatially near eye corners 362/263
    RIGHT_EYE_IRIS = [473, 474, 475, 476, 477]
    RIGHT_EYE_OUTER = 362
    RIGHT_EYE_INNER = 263
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374

    def __init__(self):
        self.horizontal_ratio = 0.5
        self.vertical_ratio = 0.5
        self.center_h = 0.5
        self.center_v = 0.5
        logger.info("Gaze Estimator initialized.")

    def set_calibration(self, center_h, center_v):
        """Sets the calibrated center point for gaze."""
        self.center_h = center_h
        self.center_v = center_v
        logger.info(f"Gaze Calibrated: H={center_h:.2f}, V={center_v:.2f}")

    def _get_iris_center(self, landmarks, iris_indices, width, height):
        """Calculates the center of the iris."""
        points = []
        for idx in iris_indices:
            point = landmarks[idx]
            points.append((point.x * width, point.y * height))
        
        points = np.array(points)
        center = np.mean(points, axis=0)
        return center

    def _get_gaze_ratio(self, landmarks, iris_center, outer_idx, inner_idx, width, height):
        """
        Calculates the horizontal gaze ratio using Vector Projection.
        Returns 0.0 (Outer) to 1.0 (Inner).
        """
        outer = np.array([landmarks[outer_idx].x * width, landmarks[outer_idx].y * height])
        inner = np.array([landmarks[inner_idx].x * width, landmarks[inner_idx].y * height])

        # Vectors
        eye_vector = inner - outer
        iris_vector = iris_center - outer
        
        # Projection: dot(iris, eye) / dot(eye, eye)
        # This gives distance along the eye axis normalized by eye length
        eye_len_sq = np.dot(eye_vector, eye_vector)
        
        if eye_len_sq == 0:
            return 0.5

        projection = np.dot(iris_vector, eye_vector) / eye_len_sq
        return np.clip(projection, 0.0, 1.0)

    def _get_vertical_gaze_ratio(self, landmarks, iris_center, top_idx, bottom_idx, width, height):
        """
        Calculates the vertical gaze ratio using relative position.
        Returns 0.0 (Top) to 1.0 (Bottom).
        """
        top = np.array([landmarks[top_idx].x * width, landmarks[top_idx].y * height])
        bottom = np.array([landmarks[bottom_idx].x * width, landmarks[bottom_idx].y * height])
        
        # Simple relative Y position, projected onto vertical axis approx
        # Since heads can tilt, true projection is better, but simple relative Y 
        # is decent if we assume eye is mostly vertical relative to itself.
        # Let's use projection for robustness against tilt too!
        
        eye_vector = bottom - top
        iris_vector = iris_center - top
        
        eye_len_sq = np.dot(eye_vector, eye_vector)
        if eye_len_sq == 0:
            return 0.5
            
        projection = np.dot(iris_vector, eye_vector) / eye_len_sq
        return np.clip(projection, 0.0, 1.0)

    def estimate_gaze(self, landmarks, width, height):
        """
        Estimates the gaze direction based on facial landmarks.
        Returns: direction (str), horizontal_ratio (float), vertical_ratio (float)
        """
        try:
            # LEFT EYE (Patient's Left)
            # Outer=33, Inner=133
            left_iris_center = self._get_iris_center(landmarks, self.LEFT_EYE_IRIS, width, height)
            left_h = self._get_gaze_ratio(landmarks, left_iris_center, self.LEFT_EYE_OUTER, self.LEFT_EYE_INNER, width, height)
            left_v = self._get_vertical_gaze_ratio(landmarks, left_iris_center, self.LEFT_EYE_TOP, self.LEFT_EYE_BOTTOM, width, height)

            # RIGHT EYE (Patient's Right)
            # Outer=362, Inner=263
            right_iris_center = self._get_iris_center(landmarks, self.RIGHT_EYE_IRIS, width, height)
            right_h_raw = self._get_gaze_ratio(landmarks, right_iris_center, self.RIGHT_EYE_OUTER, self.RIGHT_EYE_INNER, width, height)
            right_v = self._get_vertical_gaze_ratio(landmarks, right_iris_center, self.RIGHT_EYE_TOP, self.RIGHT_EYE_BOTTOM, width, height)
            
            # Logic Alignment
            # _get_gaze_ratio returns 0 (Outer) to 1 (Inner).
            
            # Left Eye: Outer=Left(33), Inner=Right(133). 
            # Looking Left (to 33) -> Near 0.
            # Looking Right (to 133) -> Near 1.
            # So Left_H: Low=Left, High=Right.
            
            # Right Eye: Outer=Right(362), Inner=Left(263).
            # Looking Left (to 263) -> Near 1 (Inner).
            # Looking Right (to 362) -> Near 0 (Outer).
            # So Right_H_Raw: High=Left, Low=Right.
            
            # Invert Right to match Left (Low=Left, High=Right)
            right_h = 1.0 - right_h_raw
            
            # Average
            avg_h = (left_h + right_h) / 2.0
            avg_v = (left_v + right_v) / 2.0
            
            # Smoothing
            alpha = config.gaze.smoothing_factor
            self.horizontal_ratio = (alpha * avg_h) + ((1 - alpha) * self.horizontal_ratio)
            self.vertical_ratio = (alpha * avg_v) + ((1 - alpha) * self.vertical_ratio)
            
            # Direction Logic with Calibration
            # Thresholds shift relative to calibrated center:
            # thresh = calibrated_center + (config_threshold - 0.5)
            thresh_left = self.center_h + (config.gaze.horizontal_threshold_left - 0.5)
            thresh_right = self.center_h + (config.gaze.horizontal_threshold_right - 0.5)
            
            thresh_up = self.center_v + (config.gaze.vertical_threshold_up - 0.5)
            thresh_down = self.center_v + (config.gaze.vertical_threshold_down - 0.5)
            
            direction = "Center"
            
            if self.horizontal_ratio < thresh_left:
                direction = "Left"
            elif self.horizontal_ratio > thresh_right:
                direction = "Right"
                
            v_dir = ""
            if self.vertical_ratio < thresh_up:
                v_dir = "Up"
            elif self.vertical_ratio > thresh_down:
                v_dir = "Down"
                
            if v_dir:
                if direction == "Center":
                    direction = v_dir
                else:
                    direction = f"{direction}-{v_dir}"

            return direction, self.horizontal_ratio, self.vertical_ratio


        except Exception as e:
            logger.error(f"Gaze estimation error: {e}")
            return "Unknown", 0.5, 0.5
