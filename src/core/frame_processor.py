import cv2
import numpy as np
from src.core.video_stream import VideoStream
from src.models.face.face_landmarks import FaceLandmarkDetector
from src.models.gaze.gaze_estimator import GazeEstimator
from src.utils.logger import logger

class FrameProcessor:
    def __init__(self, camera_id=None, enable_face_mesh=True, enable_gaze_tracking=True):
        """
        Initialize FrameProcessor with optional face mesh and gaze tracking.
        
        Args:
            camera_id: Camera source ID
            enable_face_mesh: Whether to enable face mesh detection and visualization
            enable_gaze_tracking: Whether to enable gaze tracking
        """
        # Initialize VideoStream (Singleton)
        self.video_stream = VideoStream(source=camera_id)
        self.video_stream.start()
        
        # Initialize face landmark detector if enabled
        self.enable_face_mesh = enable_face_mesh
        self.face_detector = None
        
        if self.enable_face_mesh:
            try:
                self.face_detector = FaceLandmarkDetector()
                logger.info("Face mesh visualization enabled")
            except Exception as e:
                logger.error(f"Failed to initialize face detector: {e}")
                self.enable_face_mesh = False
        
        # Initialize gaze estimator if enabled
        self.enable_gaze_tracking = enable_gaze_tracking
        self.gaze_estimator = None
        
        if self.enable_gaze_tracking:
            try:
                self.gaze_estimator = GazeEstimator()
                logger.info("Gaze tracking enabled")
            except Exception as e:
                logger.error(f"Failed to initialize gaze estimator: {e}")
                self.enable_gaze_tracking = False

    def get_frame(self, flip_horizontal=True, show_face_mesh=True, show_gaze_info=True):
        """
        Get processed frame with optional face mesh and gaze visualization.
        
        Args:
            flip_horizontal: Whether to flip the frame horizontally (fix mirror effect)
            show_face_mesh: Whether to draw face mesh on the frame
            show_gaze_info: Whether to show gaze tracking information
        
        Returns:
            JPEG encoded frame bytes
        """
        frame = self.video_stream.read()
        if frame is None:
            return None
        
        # Flip frame horizontally to fix mirror effect
        if flip_horizontal:
            frame = cv2.flip(frame, 1)
        
        gaze_metrics = None
        
        # Process face mesh and gaze if enabled
        if self.enable_face_mesh and self.face_detector:
            try:
                face_landmarks_list = self.face_detector.detect(frame)
                
                for face_landmarks in face_landmarks_list:
                    # Determine what to draw based on enabled features
                    if show_face_mesh:
                        # Full face mesh when face mesh is enabled
                        frame = self.face_detector.draw_landmarks(
                            frame,
                            face_landmarks,
                            draw_connections=True,
                            landmark_color=(0, 255, 0),
                            connection_color=(80, 110, 255),
                            thickness=1,
                            circle_radius=1
                        )
                    elif show_gaze_info and self.enable_gaze_tracking:
                        # Only draw eyes when only gaze tracking is enabled
                        frame = self._draw_eyes_only(frame, face_landmarks)
                    
                    # Estimate gaze if enabled
                    if self.enable_gaze_tracking and self.gaze_estimator and show_gaze_info:
                        gaze_metrics = self.gaze_estimator.estimate(face_landmarks)
                        
                        if gaze_metrics:
                            # Draw gaze indicator
                            frame = self._draw_gaze_indicator(frame, gaze_metrics, face_landmarks)
                            
            except Exception as e:
                logger.warning(f"Face processing error: {e}")
        
        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            return None
            
        return buffer.tobytes()
    
    def _draw_eyes_only(self, frame, face_landmarks):
        """Draw only eye landmarks when only gaze tracking is enabled."""
        h, w = face_landmarks.image_height, face_landmarks.image_width
        
        # Get eye landmarks
        eyes = self.face_detector.get_eye_landmarks(face_landmarks)
        
        # Draw left eye
        left_eye_coords = (eyes['left_eye'][:, :2] * [w, h]).astype(int)
        for i in range(len(left_eye_coords)):
            cv2.line(frame, tuple(left_eye_coords[i]), 
                    tuple(left_eye_coords[(i+1) % len(left_eye_coords)]), 
                    (0, 255, 255), 2)  # Yellow for eyes
        
        # Draw right eye
        right_eye_coords = (eyes['right_eye'][:, :2] * [w, h]).astype(int)
        for i in range(len(right_eye_coords)):
            cv2.line(frame, tuple(right_eye_coords[i]), 
                    tuple(right_eye_coords[(i+1) % len(right_eye_coords)]), 
                    (0, 255, 255), 2)  # Yellow for eyes
        
        # Draw iris if available
        if eyes['left_iris'] is not None:
            left_iris_coords = (eyes['left_iris'][:, :2] * [w, h]).astype(int)
            left_iris_center = left_iris_coords.mean(axis=0).astype(int)
            cv2.circle(frame, tuple(left_iris_center), 3, (255, 0, 255), -1)  # Magenta for iris
        
        if eyes['right_iris'] is not None:
            right_iris_coords = (eyes['right_iris'][:, :2] * [w, h]).astype(int)
            right_iris_center = right_iris_coords.mean(axis=0).astype(int)
            cv2.circle(frame, tuple(right_iris_center), 3, (255, 0, 255), -1)  # Magenta for iris
        
        return frame
    
    def _draw_gaze_indicator(self, frame, gaze_metrics, face_landmarks):
        """Draw gaze direction indicator on frame."""
        h, w = frame.shape[:2]
        
        # Get face center
        pixel_coords = face_landmarks.to_pixel_coords(normalized=False)
        face_center_x = int(pixel_coords[:, 0].mean())
        face_center_y = int(pixel_coords[:, 1].mean())
        
        # Draw gaze direction arrow
        # Map gaze ratio to arrow direction
        arrow_length = 60
        dx = (gaze_metrics.horizontal_ratio - 0.5) * arrow_length * 2
        dy = (gaze_metrics.vertical_ratio - 0.5) * arrow_length * 2
        
        end_x = int(face_center_x + dx)
        end_y = int(face_center_y + dy)
        
        # Color based on whether looking at screen
        color = (0, 255, 0) if gaze_metrics.is_looking_at_screen else (0, 0, 255)
        
        # Draw arrow
        cv2.arrowedLine(frame, (face_center_x, face_center_y), (end_x, end_y), color, 3, tipLength=0.3)
        
        # Draw status text
        status_text = "Looking at screen" if gaze_metrics.is_looking_at_screen else "Looking away"
        cv2.putText(
            frame,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )
        
        # Show warning if looking away too long
        if self.gaze_estimator.is_warning_state():
            cv2.putText(
                frame,
                "WARNING: Please look at the screen!",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )
        
        return frame
    
    def get_gaze_statistics(self):
        """Get current gaze tracking statistics."""
        if self.gaze_estimator:
            return self.gaze_estimator.get_statistics()
        return None
    
    def reset_gaze_statistics(self):
        """Reset gaze tracking statistics."""
        if self.gaze_estimator:
            self.gaze_estimator.reset_statistics()

    def __del__(self):
        # VideoStream handles its own cleanup via its own __del__ or explicit stop
        pass
