
import sys
import os
import unittest
from unittest.mock import MagicMock
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engine.face.gaze_estimation import GazeEstimator
from src.utils.config import config

class MockLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class TestGazeEstimator(unittest.TestCase):
    def setUp(self):
        self.estimator = GazeEstimator()
        self.width = 1000
        self.height = 1000
        
        # Reset Smoothing to instant for testing (alpha=1.0)
        config.gaze.smoothing_factor = 1.0

    def create_mock_landmarks(self, left_iris_pos, right_iris_pos, 
                            left_outer=(0.1, 0.5), left_inner=(0.3, 0.5),
                            right_inner=(0.7, 0.5), right_outer=(0.9, 0.5),
                            left_top=(0.2, 0.4), left_bottom=(0.2, 0.6),
                            right_top=(0.8, 0.4), right_bottom=(0.8, 0.6)):
        
        landmarks = [MockLandmark(0, 0)] * 500 # Initialize with dummy
        
        # Left Eye (Patient Left)
        # Outer 33, Inner 133
        landmarks[33] = MockLandmark(*left_outer)
        landmarks[133] = MockLandmark(*left_inner)
        # Top 159, Bottom 145
        landmarks[159] = MockLandmark(*left_top)
        landmarks[145] = MockLandmark(*left_bottom)
        
        # Iris
        for idx in self.estimator.LEFT_EYE_IRIS:
            landmarks[idx] = MockLandmark(*left_iris_pos)
            
        # Right Eye (Patient Right)
        # Outer 362, Inner 263
        landmarks[362] = MockLandmark(*right_outer)
        landmarks[263] = MockLandmark(*right_inner)
        # Top 386, Bottom 374
        landmarks[386] = MockLandmark(*right_top)
        landmarks[374] = MockLandmark(*right_bottom)
        
        # Iris
        for idx in self.estimator.RIGHT_EYE_IRIS:
            landmarks[idx] = MockLandmark(*right_iris_pos)
            
        return landmarks

    def test_center_gaze(self):
        # Iris exactly in middle of eyes
        # Left Eye: 0.1 to 0.3. Center 0.2.
        # Right Eye: 0.7 to 0.9. Center 0.8.
        landmarks = self.create_mock_landmarks(
            left_iris_pos=(0.2, 0.5),
            right_iris_pos=(0.8, 0.5)
        )
        
        direction, h_ratio, v_ratio = self.estimator.estimate_gaze(landmarks, self.width, self.height)
        print(f"Center Test: Dir={direction}, H={h_ratio}, V={v_ratio}")
        self.assertEqual(direction, "Center")
        # Ratio should be around 0.5
        self.assertAlmostEqual(h_ratio, 0.5, delta=0.05)
        
    def test_look_left(self):
        # User looks to their left (Image Left)
        # Left Eye: moves to Outer (0.1). Pos = 0.12
        # Right Eye: moves to Inner (0.7). Pos = 0.72
        landmarks = self.create_mock_landmarks(
            left_iris_pos=(0.12, 0.5),
            right_iris_pos=(0.72, 0.5)
        )
        
        direction, h_ratio, v_ratio = self.estimator.estimate_gaze(landmarks, self.width, self.height)
        print(f"Left Test: Dir={direction}, H={h_ratio}, V={v_ratio}")
        self.assertEqual(direction, "Left")
        self.assertLess(h_ratio, 0.35)

    def test_look_right(self):
        # User looks to their right (Image Right)
        # Left Eye: moves to Inner (0.3). Pos = 0.28
        # Right Eye: moves to Outer (0.9). Pos = 0.88
        landmarks = self.create_mock_landmarks(
            left_iris_pos=(0.28, 0.5),
            right_iris_pos=(0.88, 0.5)
        )
        
        direction, h_ratio, v_ratio = self.estimator.estimate_gaze(landmarks, self.width, self.height)
        print(f"Right Test: Dir={direction}, H={h_ratio}, V={v_ratio}")
        self.assertEqual(direction, "Right")
        self.assertGreater(h_ratio, 0.65)

    def test_look_up(self):
        # Iris moves up (smaller y)
        # Left Eye Y: Top 0.4, Bottom 0.6. Center 0.5.
        # Move to 0.42
        landmarks = self.create_mock_landmarks(
            left_iris_pos=(0.2, 0.42),
            right_iris_pos=(0.8, 0.42)
        )
        
        direction, h_ratio, v_ratio = self.estimator.estimate_gaze(landmarks, self.width, self.height)
        print(f"Up Test: Dir={direction}, H={h_ratio}, V={v_ratio}")
        # Vertical Ratio should be small (close to top)
        self.assertIn("Up", direction)
        self.assertLess(v_ratio, 0.35)

if __name__ == '__main__':
    unittest.main()
