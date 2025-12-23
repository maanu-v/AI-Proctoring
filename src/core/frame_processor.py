import cv2
from src.core.video_stream import VideoStream
from src.models.face.face_landmarks import FaceLandmarkDetector
from src.utils.logger import logger

class FrameProcessor:
    def __init__(self, camera_id=None, enable_face_mesh=True):
        """
        Initialize FrameProcessor with optional face mesh visualization.
        
        Args:
            camera_id: Camera source ID
            enable_face_mesh: Whether to enable face mesh detection and visualization
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

    def get_frame(self, flip_horizontal=True, show_face_mesh=True):
        """
        Get processed frame with optional face mesh visualization.
        
        Args:
            flip_horizontal: Whether to flip the frame horizontally (fix mirror effect)
            show_face_mesh: Whether to draw face mesh on the frame
        
        Returns:
            JPEG encoded frame bytes
        """
        frame = self.video_stream.read()
        if frame is None:
            return None
        
        # Flip frame horizontally to fix mirror effect
        if flip_horizontal:
            frame = cv2.flip(frame, 1)
        
        # Draw face mesh if enabled
        if show_face_mesh and self.enable_face_mesh and self.face_detector:
            try:
                face_landmarks_list = self.face_detector.detect(frame)
                
                for face_landmarks in face_landmarks_list:
                    # Draw landmarks with connections
                    frame = self.face_detector.draw_landmarks(
                        frame,
                        face_landmarks,
                        draw_connections=True,
                        landmark_color=(0, 255, 0),
                        connection_color=(80, 110, 255),
                        thickness=1,
                        circle_radius=1
                    )
            except Exception as e:
                logger.warning(f"Face mesh detection error: {e}")
        
        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            return None
            
        return buffer.tobytes()

    def __del__(self):
        # VideoStream handles its own cleanup via its own __del__ or explicit stop
        pass
