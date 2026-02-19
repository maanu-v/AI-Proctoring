import sys
import os
import unittest
import time
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engine.face.blink_estimation import BlinkEstimator


class MockLandmark:
    """Mock MediaPipe landmark with x, y coordinates (normalized 0-1)."""
    def __init__(self, x, y):
        self.x = x
        self.y = y


class TestBlinkEstimator(unittest.TestCase):
    def setUp(self):
        self.estimator = BlinkEstimator()
        # Force calibrated state for tests (skip adaptive calibration)
        self.estimator._is_calibrated = True
        # Use raw EAR for tests (alpha=1.0 = no smoothing lag)
        self.estimator.smoothing_alpha = 1.0
        self.width = 640
        self.height = 480

    def _create_landmarks(self, left_ear_open=True, right_ear_open=True):
        """
        Create mock landmarks with eyes open or closed.

        Open eye: upper/lower lids far apart -> high EAR (~0.45)
        Closed eye: upper/lower lids close together -> low EAR (~0.05)
        """
        landmarks = [MockLandmark(0.0, 0.0)] * 500

        # Left eye corners
        landmarks[33] = MockLandmark(0.15, 0.50)   # outer corner
        landmarks[133] = MockLandmark(0.30, 0.50)  # inner corner

        if left_ear_open:
            landmarks[159] = MockLandmark(0.19, 0.46)
            landmarks[160] = MockLandmark(0.22, 0.45)
            landmarks[161] = MockLandmark(0.25, 0.46)
            landmarks[144] = MockLandmark(0.19, 0.54)
            landmarks[145] = MockLandmark(0.22, 0.55)
            landmarks[153] = MockLandmark(0.25, 0.54)
        else:
            landmarks[159] = MockLandmark(0.19, 0.495)
            landmarks[160] = MockLandmark(0.22, 0.495)
            landmarks[161] = MockLandmark(0.25, 0.495)
            landmarks[144] = MockLandmark(0.19, 0.505)
            landmarks[145] = MockLandmark(0.22, 0.505)
            landmarks[153] = MockLandmark(0.25, 0.505)

        # Right eye corners
        landmarks[362] = MockLandmark(0.70, 0.50)
        landmarks[263] = MockLandmark(0.85, 0.50)

        if right_ear_open:
            landmarks[386] = MockLandmark(0.74, 0.46)
            landmarks[385] = MockLandmark(0.77, 0.45)
            landmarks[384] = MockLandmark(0.80, 0.46)
            landmarks[380] = MockLandmark(0.74, 0.54)
            landmarks[374] = MockLandmark(0.77, 0.55)
            landmarks[381] = MockLandmark(0.80, 0.54)
        else:
            landmarks[386] = MockLandmark(0.74, 0.495)
            landmarks[385] = MockLandmark(0.77, 0.495)
            landmarks[384] = MockLandmark(0.80, 0.495)
            landmarks[380] = MockLandmark(0.74, 0.505)
            landmarks[374] = MockLandmark(0.77, 0.505)
            landmarks[381] = MockLandmark(0.80, 0.505)

        return landmarks

    def test_ear_open_eyes(self):
        """EAR should be above threshold when eyes are open."""
        landmarks = self._create_landmarks(True, True)
        is_blinking, ear_l, ear_r, ear_avg = self.estimator.estimate_blink(
            landmarks, self.width, self.height
        )
        self.assertFalse(is_blinking)
        self.assertGreater(ear_avg, self.estimator.ear_threshold)
        print(f"Open eyes EAR: L={ear_l:.4f}, R={ear_r:.4f}, Avg={ear_avg:.4f}")

    def test_ear_closed_eyes(self):
        """EAR should be below threshold when eyes are closed."""
        landmarks = self._create_landmarks(False, False)
        is_blinking, ear_l, ear_r, ear_avg = self.estimator.estimate_blink(
            landmarks, self.width, self.height
        )
        self.assertTrue(is_blinking)
        self.assertLess(ear_avg, self.estimator.ear_threshold)
        print(f"Closed eyes EAR: L={ear_l:.4f}, R={ear_r:.4f}, Avg={ear_avg:.4f}")

    def test_blink_detection(self):
        """A blink (close then open) should increment blink_count."""
        open_lm = self._create_landmarks(True, True)
        closed_lm = self._create_landmarks(False, False)

        self.estimator.estimate_blink(open_lm, self.width, self.height)
        self.assertEqual(self.estimator._blink_count, 0)

        for _ in range(self.estimator.min_blink_frames):
            self.estimator.estimate_blink(closed_lm, self.width, self.height)
        self.assertEqual(self.estimator._blink_count, 0)  # Still closed

        self.estimator.estimate_blink(open_lm, self.width, self.height)
        self.assertEqual(self.estimator._blink_count, 1)
        print("Blink detected correctly after close->open transition")

    def test_short_closure_not_counted(self):
        """A closure shorter than min_blink_frames should NOT count."""
        open_lm = self._create_landmarks(True, True)
        closed_lm = self._create_landmarks(False, False)

        self.estimator.estimate_blink(open_lm, self.width, self.height)
        self.estimator.estimate_blink(closed_lm, self.width, self.height)
        self.estimator.estimate_blink(open_lm, self.width, self.height)

        if self.estimator.min_blink_frames > 1:
            self.assertEqual(self.estimator._blink_count, 0)
            print("Short closure correctly ignored")

    def test_long_closure_detected(self):
        """Eyes closed longer than threshold should flag."""
        closed_lm = self._create_landmarks(False, False)

        self.estimator.estimate_blink(closed_lm, self.width, self.height)
        self.estimator._closure_start_time = time.time() - (self.estimator.long_closure_threshold + 0.1)
        self.estimator.estimate_blink(closed_lm, self.width, self.height)

        self.assertTrue(self.estimator._long_closure_detected)
        print("Long closure detected correctly")

    def test_long_closure_persists_after_reopen(self):
        """Long closure flag should persist for a few seconds after eyes reopen."""
        open_lm = self._create_landmarks(True, True)
        closed_lm = self._create_landmarks(False, False)

        # Trigger long closure
        self.estimator.estimate_blink(closed_lm, self.width, self.height)
        self.estimator._closure_start_time = time.time() - (self.estimator.long_closure_threshold + 0.1)
        self.estimator.estimate_blink(closed_lm, self.width, self.height)
        self.assertTrue(self.estimator._long_closure_detected)

        # Reopen eyes — flag should persist
        self.estimator.estimate_blink(open_lm, self.width, self.height)
        self.assertTrue(self.estimator._long_closure_detected)
        print("Long closure flag persists after reopen")

    def test_long_closure_expires(self):
        """Long closure flag should expire after persist window."""
        open_lm = self._create_landmarks(True, True)

        # Set up expired long closure
        self.estimator._long_closure_detected = True
        self.estimator._long_closure_timestamp = time.time() - 10.0  # Well past persist window
        self.estimator._is_eye_closed = False

        self.estimator.estimate_blink(open_lm, self.width, self.height)
        self.assertFalse(self.estimator._long_closure_detected)
        print("Long closure flag expired correctly")

    def test_ear_smoothing(self):
        """Smoothed EAR should lag behind raw EAR changes."""
        self.estimator.smoothing_alpha = 0.4  # Enable smoothing for this test
        open_lm = self._create_landmarks(True, True)
        closed_lm = self._create_landmarks(False, False)

        # Initialize with open eyes
        self.estimator.estimate_blink(open_lm, self.width, self.height)
        open_smoothed = self.estimator._ear_smoothed

        # One frame of closed eyes — smoothed should NOT drop as fast as raw
        self.estimator.estimate_blink(closed_lm, self.width, self.height)
        self.assertGreater(self.estimator._ear_smoothed, self.estimator._ear_avg)
        print(f"Smoothing: raw={self.estimator._ear_avg:.4f}, smoothed={self.estimator._ear_smoothed:.4f}")

    def test_blink_interval_variance(self):
        """Blink interval variance should be > 0 with irregular blinks."""
        open_lm = self._create_landmarks(True, True)
        closed_lm = self._create_landmarks(False, False)

        # Simulate 3 blinks with varying intervals
        for i in range(3):
            self.estimator.estimate_blink(open_lm, self.width, self.height)
            for _ in range(self.estimator.min_blink_frames):
                self.estimator.estimate_blink(closed_lm, self.width, self.height)
            # Backdate closure for measurable duration
            self.estimator._closure_start_time = time.time() - 0.1
            # Fake different blink timestamps for interval variance
            if self.estimator._blink_timestamps:
                self.estimator._blink_timestamps[-1] -= (i + 1) * 0.5
            self.estimator.estimate_blink(open_lm, self.width, self.height)

        variance = self.estimator._get_blink_interval_variance()
        # With manipulated intervals, variance should be > 0
        if len(self.estimator._blink_intervals) >= 2:
            self.assertGreater(variance, 0.0)
            print(f"Blink interval variance: {variance:.4f}")
        else:
            print(f"Only {len(self.estimator._blink_intervals)} intervals, skipping variance check")

    def test_get_features_structure(self):
        """get_features() should return dict with all expected keys."""
        features = self.estimator.get_features()

        expected_keys = [
            "ear_left", "ear_right", "ear_avg", "ear_smoothed",
            "is_blinking", "blink_count", "blink_rate",
            "avg_blink_duration", "blink_interval_variance",
            "long_closure_detected", "long_closure_timestamp",
            "baseline_ear", "ear_threshold",
        ]
        for key in expected_keys:
            self.assertIn(key, features, f"Missing key: {key}")
        print(f"Feature dict has all {len(expected_keys)} expected keys")

    def test_get_features_after_blink(self):
        """Features should update after a blink."""
        open_lm = self._create_landmarks(True, True)
        closed_lm = self._create_landmarks(False, False)

        self.estimator.estimate_blink(open_lm, self.width, self.height)
        for _ in range(self.estimator.min_blink_frames):
            self.estimator.estimate_blink(closed_lm, self.width, self.height)
        self.estimator._closure_start_time = time.time() - 0.1
        self.estimator.estimate_blink(open_lm, self.width, self.height)

        features = self.estimator.get_features()
        self.assertEqual(features["blink_count"], 1)
        self.assertGreater(features["avg_blink_duration"], 0.0)
        self.assertFalse(features["is_blinking"])
        print(f"Features after blink: count={features['blink_count']}, dur={features['avg_blink_duration']}")

    def test_reset(self):
        """Reset should clear tracking state but keep calibration."""
        open_lm = self._create_landmarks(True, True)
        closed_lm = self._create_landmarks(False, False)

        self.estimator._baseline_ear = 0.35
        self.estimator.estimate_blink(open_lm, self.width, self.height)
        for _ in range(self.estimator.min_blink_frames):
            self.estimator.estimate_blink(closed_lm, self.width, self.height)
        self.estimator.estimate_blink(open_lm, self.width, self.height)

        self.estimator.reset()
        features = self.estimator.get_features()
        self.assertEqual(features["blink_count"], 0)
        self.assertEqual(features["ear_avg"], 0.0)
        self.assertFalse(features["is_blinking"])
        # Calibration should be preserved
        self.assertEqual(self.estimator._baseline_ear, 0.35)
        print("Reset clears state but keeps calibration")


if __name__ == '__main__':
    unittest.main()
