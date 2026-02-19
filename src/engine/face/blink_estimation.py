# src/engine/face/blink_estimation.py

import numpy as np
import time
from collections import deque
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BlinkEstimator:
    """
    Detects eye blinks using Eye Aspect Ratio (EAR) from MediaPipe face landmarks.
    Tracks blink events, computes blink rate, and exposes per-frame features
    suitable for ML model training.

    EAR Formula (Soukupová & Čech):
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    where p1, p4 are the eye corners and p2-p6 are upper/lower lid landmarks.

    Features:
        - Exponential smoothing on EAR to reduce landmark jitter
        - Adaptive threshold calibrated from user's baseline open-eye EAR
        - Persistent long closure flag with timestamp for scoring
        - Blink interval variance for behavioral anomaly detection
    """

    # MediaPipe Face Mesh landmark indices for each eye
    # Left Eye (person's left)
    LEFT_EYE = {
        "corner_outer": 33,
        "corner_inner": 133,
        "upper": [159, 160, 161],  # Upper eyelid (outer to inner)
        "lower": [144, 145, 153],  # Lower eyelid (outer to inner)
    }

    # Right Eye (person's right)
    RIGHT_EYE = {
        "corner_outer": 362,
        "corner_inner": 263,
        "upper": [386, 385, 384],  # Upper eyelid (outer to inner)
        "lower": [380, 374, 381],  # Lower eyelid (outer to inner)
    }

    def __init__(self):
        # Thresholds from config
        self.ear_threshold = config.blink.ear_threshold
        self.min_blink_frames = config.blink.min_blink_frames
        self.long_closure_threshold = config.blink.long_closure_threshold
        self.smoothing_alpha = config.blink.smoothing_alpha

        # Adaptive calibration
        self._calibration_duration = config.blink.calibration_duration
        self._calibration_ratio = config.blink.calibration_ratio
        self._calibration_start_time = None
        self._calibration_ear_samples = []
        self._is_calibrated = False
        self._baseline_ear = None

        # EAR smoothing
        self._ear_smoothed = 0.0
        self._smoothing_initialized = False

        # State tracking
        self._is_eye_closed = False
        self._closed_frame_count = 0
        self._closure_start_time = None

        # Blink statistics
        self._blink_count = 0
        self._blink_durations = []  # Duration of each blink in seconds

        # Long closure: persistent flag with timestamp
        self._long_closure_detected = False
        self._long_closure_timestamp = None
        self._long_closure_persist_seconds = 3.0  # Flag stays true for 3s after eyes reopen

        # Rolling window for blink rate (timestamps of each blink)
        self._blink_timestamps = deque()
        self._rate_window_seconds = 60  # 1-minute rolling window

        # Blink interval tracking (for variance calculation)
        self._blink_intervals = deque(maxlen=30)  # Last 30 inter-blink intervals

        # Current frame EAR values (updated each call)
        self._ear_left = 0.0
        self._ear_right = 0.0
        self._ear_avg = 0.0

        logger.info("Blink Estimator initialized.")

    def _landmark_to_point(self, landmark, width, height):
        """Convert a MediaPipe landmark to pixel coordinates."""
        return np.array([landmark.x * width, landmark.y * height])

    def _compute_ear(self, landmarks, eye_indices, width, height):
        """
        Compute the Eye Aspect Ratio for one eye.

        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

        Where:
            p1 = outer corner, p4 = inner corner
            p2, p3 = upper lid landmarks
            p6, p5 = lower lid landmarks (paired vertically with p2, p3)
        """
        # Corners
        p1 = self._landmark_to_point(landmarks[eye_indices["corner_outer"]], width, height)
        p4 = self._landmark_to_point(landmarks[eye_indices["corner_inner"]], width, height)

        # Upper lid points
        p2 = self._landmark_to_point(landmarks[eye_indices["upper"][0]], width, height)
        p3 = self._landmark_to_point(landmarks[eye_indices["upper"][1]], width, height)

        # Lower lid points (paired with upper)
        p6 = self._landmark_to_point(landmarks[eye_indices["lower"][0]], width, height)
        p5 = self._landmark_to_point(landmarks[eye_indices["lower"][1]], width, height)

        # Vertical distances
        v1 = np.linalg.norm(p2 - p6)
        v2 = np.linalg.norm(p3 - p5)

        # Horizontal distance
        h = np.linalg.norm(p1 - p4)

        if h == 0:
            return 0.0

        ear = (v1 + v2) / (2.0 * h)
        return ear

    def _smooth_ear(self, raw_ear):
        """
        Apply exponential smoothing to EAR to reduce landmark jitter.
        Prevents false blink detection from lighting changes and head pose.
        """
        if not self._smoothing_initialized:
            self._ear_smoothed = raw_ear
            self._smoothing_initialized = True
        else:
            alpha = self.smoothing_alpha
            self._ear_smoothed = alpha * raw_ear + (1 - alpha) * self._ear_smoothed
        return self._ear_smoothed

    def _update_calibration(self, ear_avg):
        """
        Adaptive threshold calibration during first N seconds.
        Collects open-eye EAR samples and sets threshold = baseline * ratio.
        """
        if self._is_calibrated:
            return

        now = time.time()

        if self._calibration_start_time is None:
            self._calibration_start_time = now
            logger.info(f"Blink calibration started ({self._calibration_duration}s)...")

        elapsed = now - self._calibration_start_time

        if elapsed < self._calibration_duration:
            # Only collect samples where eyes are likely open (above default threshold)
            if ear_avg > self.ear_threshold * 0.8:
                self._calibration_ear_samples.append(ear_avg)
        else:
            # Finish calibration
            if len(self._calibration_ear_samples) > 10:
                self._baseline_ear = np.mean(self._calibration_ear_samples)
                old_threshold = self.ear_threshold
                self.ear_threshold = self._baseline_ear * self._calibration_ratio
                logger.info(
                    f"Blink calibrated: baseline EAR={self._baseline_ear:.4f}, "
                    f"threshold {old_threshold:.3f} → {self.ear_threshold:.3f}"
                )
            else:
                logger.warning("Blink calibration: too few samples, keeping default threshold.")

            self._is_calibrated = True

    def _update_long_closure_flag(self):
        """
        Manage persistent long closure flag.
        Flag stays true for _long_closure_persist_seconds after eyes reopen,
        so downstream analyzers don't miss the event.
        """
        if self._long_closure_detected and self._long_closure_timestamp is not None:
            if not self._is_eye_closed:
                # Eyes reopened — check if persist window expired
                elapsed = time.time() - self._long_closure_timestamp
                if elapsed >= self._long_closure_persist_seconds:
                    self._long_closure_detected = False
                    self._long_closure_timestamp = None

    def estimate_blink(self, landmarks, width, height):
        """
        Process one frame of landmarks to update blink state.

        Args:
            landmarks: MediaPipe face landmarks for one face.
            width: Frame width in pixels.
            height: Frame height in pixels.

        Returns:
            tuple: (is_blinking: bool, ear_left: float, ear_right: float, ear_avg: float)
        """
        try:
            # Compute EAR for both eyes
            self._ear_left = self._compute_ear(landmarks, self.LEFT_EYE, width, height)
            self._ear_right = self._compute_ear(landmarks, self.RIGHT_EYE, width, height)
            self._ear_avg = (self._ear_left + self._ear_right) / 2.0

            # Smooth the EAR
            smoothed_ear = self._smooth_ear(self._ear_avg)

            # Adaptive calibration (runs during first N seconds)
            self._update_calibration(self._ear_avg)

            # Update persistent long closure flag
            self._update_long_closure_flag()

            now = time.time()

            # State machine for blink detection (uses smoothed EAR)
            if smoothed_ear < self.ear_threshold:
                # Eyes are closed this frame
                if not self._is_eye_closed:
                    # Transition: open -> closed
                    self._is_eye_closed = True
                    self._closed_frame_count = 1
                    self._closure_start_time = now
                else:
                    self._closed_frame_count += 1

                # Check for prolonged closure
                if self._closure_start_time is not None:
                    closure_duration = now - self._closure_start_time
                    if closure_duration >= self.long_closure_threshold:
                        self._long_closure_detected = True
                        self._long_closure_timestamp = now  # Keep updating while closed
            else:
                # Eyes are open this frame
                if self._is_eye_closed:
                    # Transition: closed -> open = potential blink
                    if self._closed_frame_count >= self.min_blink_frames:
                        # Register a blink
                        self._blink_count += 1

                        # Record inter-blink interval
                        if self._blink_timestamps:
                            interval = now - self._blink_timestamps[-1]
                            self._blink_intervals.append(interval)

                        self._blink_timestamps.append(now)

                        # Record duration
                        if self._closure_start_time is not None:
                            duration = now - self._closure_start_time
                            self._blink_durations.append(duration)

                    # Reset closure state (but NOT long_closure — it persists)
                    self._is_eye_closed = False
                    self._closed_frame_count = 0
                    self._closure_start_time = None

            return self._is_eye_closed, self._ear_left, self._ear_right, self._ear_avg

        except Exception as e:
            logger.error(f"Blink estimation error: {e}")
            return False, 0.0, 0.0, 0.0

    def _get_blink_rate(self):
        """
        Calculate blinks per minute using a rolling window.

        Returns:
            float: Blinks per minute.
        """
        now = time.time()
        cutoff = now - self._rate_window_seconds

        # Evict old timestamps
        while self._blink_timestamps and self._blink_timestamps[0] < cutoff:
            self._blink_timestamps.popleft()

        count = len(self._blink_timestamps)

        # If we have blinks, calculate rate based on actual elapsed time
        if count > 0 and self._blink_timestamps:
            elapsed = now - self._blink_timestamps[0]
            if elapsed > 0:
                return (count / elapsed) * 60.0

        return 0.0

    def _get_avg_blink_duration(self):
        """
        Calculate average blink duration.

        Returns:
            float: Average duration in seconds, 0.0 if no blinks recorded.
        """
        if not self._blink_durations:
            return 0.0
        return sum(self._blink_durations) / len(self._blink_durations)

    def _get_blink_interval_variance(self):
        """
        Compute variance of inter-blink intervals.

        Behavioral insight:
            - Normal: semi-regular intervals → low variance
            - Stress: increased blink rate → low intervals
            - Reading: decreased blink rate → high intervals
            - Cheating: irregular bursts → high variance

        Returns:
            float: Variance of last 30 inter-blink intervals, 0.0 if < 2 samples.
        """
        if len(self._blink_intervals) < 2:
            return 0.0
        return float(np.var(self._blink_intervals))

    def get_features(self):
        """
        Get current blink-related features for model training.

        Returns:
            dict: Feature dictionary with keys:
                - ear_left (float): Left eye EAR
                - ear_right (float): Right eye EAR
                - ear_avg (float): Average EAR of both eyes
                - ear_smoothed (float): Exponentially smoothed EAR
                - is_blinking (bool): Whether eyes are currently closed
                - blink_count (int): Total blinks this session
                - blink_rate (float): Blinks per minute (rolling 60s window)
                - avg_blink_duration (float): Average blink duration in seconds
                - blink_interval_variance (float): Variance of inter-blink intervals
                - long_closure_detected (bool): Eyes closed > threshold (persists after reopen)
                - long_closure_timestamp (float|None): When the long closure was detected
                - baseline_ear (float|None): Calibrated baseline EAR (None if not yet calibrated)
                - ear_threshold (float): Current active EAR threshold
        """
        return {
            "ear_left": round(self._ear_left, 4),
            "ear_right": round(self._ear_right, 4),
            "ear_avg": round(self._ear_avg, 4),
            "ear_smoothed": round(self._ear_smoothed, 4),
            "is_blinking": self._is_eye_closed,
            "blink_count": self._blink_count,
            "blink_rate": round(self._get_blink_rate(), 2),
            "avg_blink_duration": round(self._get_avg_blink_duration(), 4),
            "blink_interval_variance": round(self._get_blink_interval_variance(), 4),
            "long_closure_detected": self._long_closure_detected,
            "long_closure_timestamp": self._long_closure_timestamp,
            "baseline_ear": round(self._baseline_ear, 4) if self._baseline_ear else None,
            "ear_threshold": round(self.ear_threshold, 4),
        }

    def reset(self):
        """Reset all blink tracking state."""
        self._is_eye_closed = False
        self._closed_frame_count = 0
        self._closure_start_time = None
        self._blink_count = 0
        self._blink_durations = []
        self._long_closure_detected = False
        self._long_closure_timestamp = None
        self._blink_timestamps.clear()
        self._blink_intervals.clear()
        self._ear_left = 0.0
        self._ear_right = 0.0
        self._ear_avg = 0.0
        self._ear_smoothed = 0.0
        self._smoothing_initialized = False
        # Keep calibration — it's user-specific and shouldn't be recalculated on reset
        logger.info("Blink Estimator reset.")
