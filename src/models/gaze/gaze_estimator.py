import numpy as np
import time
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, field
from collections import deque
from src.models.face.face_landmarks import FaceLandmarks
from src.utils.logger import logger
import yaml
import os

@dataclass
class GazeMetrics:
    """Container for gaze tracking metrics."""
    is_looking_at_screen: bool
    horizontal_ratio: float  # 0.0 (left) to 1.0 (right), 0.5 is center
    vertical_ratio: float    # 0.0 (up) to 1.0 (down), 0.5 is center
    confidence: float
    timestamp: float = field(default_factory=time.time)

@dataclass
class GazeViolation:
    """Container for a gaze violation event."""
    start_time: float
    end_time: Optional[float] = None
    duration: float = 0.0
    direction: str = ""  # "left", "right", "up", "down", "away"
    
    def update_duration(self):
        """Update duration if end_time is set."""
        if self.end_time:
            self.duration = self.end_time - self.start_time

@dataclass
class GazeStatistics:
    """Statistics for gaze tracking session."""
    total_violations: int = 0
    current_violation: Optional[GazeViolation] = None
    violation_history: List[GazeViolation] = field(default_factory=list)
    total_looking_away_time: float = 0.0
    session_start_time: float = field(default_factory=time.time)
    
    def get_session_duration(self) -> float:
        """Get total session duration in seconds."""
        return time.time() - self.session_start_time
    
    def get_attention_percentage(self) -> float:
        """Get percentage of time looking at screen."""
        session_duration = self.get_session_duration()
        if session_duration == 0:
            return 100.0
        return max(0.0, (1.0 - self.total_looking_away_time / session_duration) * 100.0)
    
    def get_average_violation_duration(self) -> float:
        """Get average duration of violations."""
        if not self.violation_history:
            return 0.0
        return sum(v.duration for v in self.violation_history) / len(self.violation_history)


class GazeEstimator:
    """
    Gaze estimation using iris landmarks from MediaPipe Face Mesh.
    Tracks eye movements and determines if user is looking at the screen.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Gaze Estimator.
        
        Args:
            config_path: Path to thresholds configuration file
        """
        # Load configuration
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "configs", "thresholds.yaml"
            )
        
        self.config = self._load_config(config_path)
        
        # Thresholds
        self.h_center_min = self.config['gaze']['horizontal_center_min']
        self.h_center_max = self.config['gaze']['horizontal_center_max']
        self.v_center_min = self.config['gaze']['vertical_center_min']
        self.v_center_max = self.config['gaze']['vertical_center_max']
        self.violation_min_duration = self.config['gaze']['violation_min_duration']
        self.warning_duration = self.config['gaze']['warning_duration']
        
        # Smoothing
        self.smoothing_window = self.config['gaze']['smoothing_window']
        self.h_ratio_history = deque(maxlen=self.smoothing_window)
        self.v_ratio_history = deque(maxlen=self.smoothing_window)
        
        # Statistics
        self.statistics = GazeStatistics()
        self.looking_away_start_time: Optional[float] = None
        
        logger.info("GazeEstimator initialized")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}. Using defaults.")
            return {
                'gaze': {
                    'horizontal_center_min': 0.35,
                    'horizontal_center_max': 0.65,
                    'vertical_center_min': 0.35,
                    'vertical_center_max': 0.65,
                    'violation_min_duration': 2.0,
                    'warning_duration': 1.0,
                    'smoothing_window': 5
                }
            }
    
    def _get_iris_position(
        self,
        iris_landmarks: np.ndarray,
        eye_landmarks: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calculate normalized iris position within the eye.
        
        Args:
            iris_landmarks: Array of iris landmark points (5 points)
            eye_landmarks: Array of eye landmark points (6 points)
        
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
        
        # Clamp to [0, 1]
        h_ratio = max(0.0, min(1.0, h_ratio))
        v_ratio = max(0.0, min(1.0, v_ratio))
        
        return (h_ratio, v_ratio)
    
    def estimate(self, face_landmarks: FaceLandmarks) -> Optional[GazeMetrics]:
        """
        Estimate gaze direction from face landmarks.
        
        Args:
            face_landmarks: FaceLandmarks object with iris landmarks
        
        Returns:
            GazeMetrics object or None if estimation fails
        """
        # Get eye landmarks
        from src.models.face.face_landmarks import FaceLandmarkDetector
        detector = FaceLandmarkDetector()
        eyes = detector.get_eye_landmarks(face_landmarks)
        
        # Check if iris landmarks are available
        if eyes['left_iris'] is None or eyes['right_iris'] is None:
            logger.warning("Iris landmarks not available. Enable refine_landmarks=True")
            return None
        
        # Get iris positions for both eyes
        left_h, left_v = self._get_iris_position(eyes['left_iris'], eyes['left_eye'])
        right_h, right_v = self._get_iris_position(eyes['right_iris'], eyes['right_eye'])
        
        # Average both eyes
        h_ratio = (left_h + right_h) / 2.0
        v_ratio = (left_v + right_v) / 2.0
        
        # Apply smoothing
        self.h_ratio_history.append(h_ratio)
        self.v_ratio_history.append(v_ratio)
        
        smoothed_h = np.mean(self.h_ratio_history)
        smoothed_v = np.mean(self.v_ratio_history)
        
        # Determine if looking at screen
        is_looking_at_screen = (
            self.h_center_min <= smoothed_h <= self.h_center_max and
            self.v_center_min <= smoothed_v <= self.v_center_max
        )
        
        # Update statistics
        self._update_statistics(is_looking_at_screen, smoothed_h, smoothed_v)
        
        # Calculate confidence (how centered the gaze is)
        h_center_dist = abs(smoothed_h - 0.5) * 2  # 0 at center, 1 at edges
        v_center_dist = abs(smoothed_v - 0.5) * 2
        confidence = 1.0 - (h_center_dist + v_center_dist) / 2.0
        
        return GazeMetrics(
            is_looking_at_screen=is_looking_at_screen,
            horizontal_ratio=smoothed_h,
            vertical_ratio=smoothed_v,
            confidence=confidence
        )
    
    def _update_statistics(self, is_looking_at_screen: bool, h_ratio: float, v_ratio: float):
        """Update gaze statistics based on current state."""
        current_time = time.time()
        
        if not is_looking_at_screen:
            # User is looking away
            if self.looking_away_start_time is None:
                # Start of looking away
                self.looking_away_start_time = current_time
                
                # Determine direction
                direction = self._get_gaze_direction(h_ratio, v_ratio)
                
                # Start new violation
                self.statistics.current_violation = GazeViolation(
                    start_time=current_time,
                    direction=direction
                )
            else:
                # Continue looking away
                duration = current_time - self.looking_away_start_time
                
                # Update current violation duration
                if self.statistics.current_violation:
                    self.statistics.current_violation.end_time = current_time
                    self.statistics.current_violation.update_duration()
        else:
            # User is looking at screen
            if self.looking_away_start_time is not None:
                # End of looking away
                duration = current_time - self.looking_away_start_time
                
                # Only count as violation if duration exceeds threshold
                if duration >= self.violation_min_duration:
                    if self.statistics.current_violation:
                        self.statistics.current_violation.end_time = current_time
                        self.statistics.current_violation.update_duration()
                        
                        # Add to history
                        self.statistics.violation_history.append(self.statistics.current_violation)
                        self.statistics.total_violations += 1
                        self.statistics.total_looking_away_time += duration
                        
                        logger.info(f"Gaze violation recorded: {duration:.2f}s looking {self.statistics.current_violation.direction}")
                
                # Reset
                self.looking_away_start_time = None
                self.statistics.current_violation = None
    
    def _get_gaze_direction(self, h_ratio: float, v_ratio: float) -> str:
        """Determine gaze direction from ratios."""
        directions = []
        
        if h_ratio < self.h_center_min:
            directions.append("left")
        elif h_ratio > self.h_center_max:
            directions.append("right")
        
        if v_ratio < self.v_center_min:
            directions.append("up")
        elif v_ratio > self.v_center_max:
            directions.append("down")
        
        if not directions:
            return "away"
        
        return "-".join(directions)
    
    def get_statistics(self) -> Dict:
        """Get current gaze statistics as dictionary."""
        current_looking_away_duration = 0.0
        if self.looking_away_start_time is not None:
            current_looking_away_duration = time.time() - self.looking_away_start_time
        
        return {
            'total_violations': self.statistics.total_violations,
            'total_looking_away_time': self.statistics.total_looking_away_time,
            'current_looking_away_duration': current_looking_away_duration,
            'session_duration': self.statistics.get_session_duration(),
            'attention_percentage': self.statistics.get_attention_percentage(),
            'average_violation_duration': self.statistics.get_average_violation_duration(),
            'is_currently_looking_away': self.looking_away_start_time is not None,
            'current_violation_direction': self.statistics.current_violation.direction if self.statistics.current_violation else None
        }
    
    def reset_statistics(self):
        """Reset all statistics (e.g., when starting a new session)."""
        self.statistics = GazeStatistics()
        self.looking_away_start_time = None
        self.h_ratio_history.clear()
        self.v_ratio_history.clear()
        logger.info("Gaze statistics reset")
    
    def is_warning_state(self) -> bool:
        """Check if user has been looking away long enough to show warning."""
        if self.looking_away_start_time is None:
            return False
        
        duration = time.time() - self.looking_away_start_time
        return duration >= self.warning_duration
