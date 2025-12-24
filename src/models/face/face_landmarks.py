import cv2
import numpy as np
import os
import warnings

# Import warning suppression utility first
from src.utils.suppress_warnings import SuppressStderr

# Suppress MediaPipe warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
warnings.filterwarnings('ignore', category=UserWarning, module='mediapipe')

# Import MediaPipe with stderr suppression
with SuppressStderr():
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from src.utils.logger import logger

@dataclass
class FaceLandmarks:
    """Container for face landmark data."""
    landmarks: np.ndarray  # Nx3 array of (x, y, z) coordinates
    image_width: int
    image_height: int
    
    def get_landmark(self, index: int) -> Optional[Tuple[float, float, float]]:
        """Get a specific landmark by index."""
        if 0 <= index < len(self.landmarks):
            return tuple(self.landmarks[index])
        return None
    
    def get_landmarks_subset(self, indices: List[int]) -> np.ndarray:
        """Get a subset of landmarks by indices."""
        return self.landmarks[indices]
    
    def to_pixel_coords(self, normalized: bool = False) -> np.ndarray:
        """
        Convert landmarks to pixel coordinates.
        
        Args:
            normalized: If True, keep normalized coords [0-1], else convert to pixels
        
        Returns:
            Nx2 array of (x, y) pixel coordinates
        """
        if normalized:
            return self.landmarks[:, :2]
        
        pixel_coords = self.landmarks[:, :2].copy()
        pixel_coords[:, 0] *= self.image_width
        pixel_coords[:, 1] *= self.image_height
        return pixel_coords.astype(int)


class FaceLandmarkDetector:
    """
    Face landmark detection using MediaPipe Face Landmarker.
    Provides 478 3D face landmarks for detailed facial analysis.
    """
    
    # Key landmark indices for common facial features
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    
    LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
    RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]
    
    MOUTH_OUTER_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88]
    MOUTH_INNER_INDICES = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
    
    NOSE_TIP_INDEX = 1
    NOSE_BRIDGE_INDICES = [6, 197, 195, 5]
    
    FACE_OVAL_INDICES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    
    LEFT_EYEBROW_INDICES = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
    RIGHT_EYEBROW_INDICES = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        num_faces: int = 1,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        output_face_blendshapes: bool = False,
        output_facial_transformation_matrixes: bool = False
    ):
        """
        Initialize the Face Landmark Detector.
        
        Args:
            model_path: Path to the face landmarker model file. If None, downloads default model.
            num_faces: Maximum number of faces to detect.
            min_face_detection_confidence: Minimum confidence for face detection.
            min_face_presence_confidence: Minimum confidence for face presence.
            min_tracking_confidence: Minimum confidence for landmark tracking.
            output_face_blendshapes: Whether to output face blendshapes.
            output_facial_transformation_matrixes: Whether to output facial transformation matrices.
        """
        # Download model if not provided
        if model_path is None:
            model_path = self._download_model()
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Create face landmarker options
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=num_faces,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=output_face_blendshapes,
            output_facial_transformation_matrixes=output_facial_transformation_matrixes
        )
        
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.num_faces = num_faces
        self.frame_timestamp = 0
        
        logger.info(f"FaceLandmarkDetector initialized: max_faces={num_faces}")
    
    def _download_model(self) -> str:
        """Download the default face landmarker model."""
        import urllib.request
        
        model_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, "face_landmarker.task")
        
        if not os.path.exists(model_path):
            logger.info("Downloading face landmarker model...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            logger.info(f"Model downloaded to {model_path}")
        
        return model_path
    
    def detect(self, image: np.ndarray) -> List[FaceLandmarks]:
        """
        Detect face landmarks in an image.
        
        Args:
            image: Input image in BGR format (OpenCV format)
        
        Returns:
            List of FaceLandmarks objects, one per detected face
        """
        if image is None or image.size == 0:
            logger.warning("Empty image provided to face landmark detector")
            return []
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Detect landmarks
        self.frame_timestamp += 1
        detection_result = self.detector.detect_for_video(mp_image, self.frame_timestamp)
        
        if not detection_result.face_landmarks:
            return []
        
        # Extract landmarks for each detected face
        face_landmarks_list = []
        for face_landmarks in detection_result.face_landmarks:
            # Convert landmarks to numpy array
            landmarks = np.array([
                [lm.x, lm.y, lm.z]
                for lm in face_landmarks
            ])
            
            face_landmarks_list.append(FaceLandmarks(
                landmarks=landmarks,
                image_width=w,
                image_height=h
            ))
        
        return face_landmarks_list
    
    def detect_single(self, image: np.ndarray) -> Optional[FaceLandmarks]:
        """
        Detect landmarks for a single face (returns first detected face).
        
        Args:
            image: Input image in BGR format
        
        Returns:
            FaceLandmarks object or None if no face detected
        """
        results = self.detect(image)
        return results[0] if results else None
    
    def get_eye_landmarks(self, face_landmarks: FaceLandmarks) -> Dict[str, np.ndarray]:
        """Extract eye landmarks."""
        left_iris = None
        right_iris = None
        
        # Check if iris landmarks are available (indices 468-477)
        if len(face_landmarks.landmarks) > 477:
            left_iris = face_landmarks.get_landmarks_subset(self.LEFT_IRIS_INDICES)
            right_iris = face_landmarks.get_landmarks_subset(self.RIGHT_IRIS_INDICES)
        
        return {
            'left_eye': face_landmarks.get_landmarks_subset(self.LEFT_EYE_INDICES),
            'right_eye': face_landmarks.get_landmarks_subset(self.RIGHT_EYE_INDICES),
            'left_iris': left_iris,
            'right_iris': right_iris,
        }
    
    def get_mouth_landmarks(self, face_landmarks: FaceLandmarks) -> Dict[str, np.ndarray]:
        """Extract mouth landmarks."""
        return {
            'outer': face_landmarks.get_landmarks_subset(self.MOUTH_OUTER_INDICES),
            'inner': face_landmarks.get_landmarks_subset(self.MOUTH_INNER_INDICES),
        }
    
    def get_nose_landmarks(self, face_landmarks: FaceLandmarks) -> Dict[str, np.ndarray]:
        """Extract nose landmarks."""
        return {
            'tip': face_landmarks.get_landmark(self.NOSE_TIP_INDEX),
            'bridge': face_landmarks.get_landmarks_subset(self.NOSE_BRIDGE_INDICES),
        }
    
    def get_face_oval(self, face_landmarks: FaceLandmarks) -> np.ndarray:
        """Extract face oval/contour landmarks."""
        return face_landmarks.get_landmarks_subset(self.FACE_OVAL_INDICES)
    
    def get_eyebrow_landmarks(self, face_landmarks: FaceLandmarks) -> Dict[str, np.ndarray]:
        """Extract eyebrow landmarks."""
        return {
            'left': face_landmarks.get_landmarks_subset(self.LEFT_EYEBROW_INDICES),
            'right': face_landmarks.get_landmarks_subset(self.RIGHT_EYEBROW_INDICES),
        }
    
    def draw_landmarks(
        self,
        image: np.ndarray,
        face_landmarks: FaceLandmarks,
        draw_connections: bool = False,
        landmark_color: Tuple[int, int, int] = (0, 255, 0),
        connection_color: Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 1,
        circle_radius: int = 1
    ) -> np.ndarray:
        """
        Draw face landmarks on an image.
        
        Args:
            image: Input image to draw on
            face_landmarks: FaceLandmarks object
            draw_connections: Whether to draw connections between landmarks
            landmark_color: Color for landmark points (BGR)
            connection_color: Color for connections (BGR)
            thickness: Line thickness
            circle_radius: Radius of landmark circles
        
        Returns:
            Image with landmarks drawn
        """
        annotated_image = image.copy()
        
        # Convert landmarks to pixel coordinates
        pixel_coords = face_landmarks.to_pixel_coords(normalized=False)
        
        # Draw landmark points
        for x, y in pixel_coords:
            cv2.circle(annotated_image, (x, y), circle_radius, landmark_color, -1)
        
        # Optionally draw connections (simplified version)
        if draw_connections:
            # Draw face oval
            oval_coords = face_landmarks.get_landmarks_subset(self.FACE_OVAL_INDICES)
            oval_pixels = (oval_coords[:, :2] * [face_landmarks.image_width, face_landmarks.image_height]).astype(int)
            for i in range(len(oval_pixels) - 1):
                cv2.line(annotated_image, tuple(oval_pixels[i]), tuple(oval_pixels[i+1]), connection_color, thickness)
            
            # Draw eyes
            for eye_indices in [self.LEFT_EYE_INDICES, self.RIGHT_EYE_INDICES]:
                eye_coords = face_landmarks.get_landmarks_subset(eye_indices)
                eye_pixels = (eye_coords[:, :2] * [face_landmarks.image_width, face_landmarks.image_height]).astype(int)
                for i in range(len(eye_pixels)):
                    cv2.line(annotated_image, tuple(eye_pixels[i]), tuple(eye_pixels[(i+1) % len(eye_pixels)]), connection_color, thickness)
            
            # Draw mouth
            mouth_coords = face_landmarks.get_landmarks_subset(self.MOUTH_OUTER_INDICES)
            mouth_pixels = (mouth_coords[:, :2] * [face_landmarks.image_width, face_landmarks.image_height]).astype(int)
            for i in range(len(mouth_pixels)):
                cv2.line(annotated_image, tuple(mouth_pixels[i]), tuple(mouth_pixels[(i+1) % len(mouth_pixels)]), connection_color, thickness)
        
        return annotated_image
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'detector'):
            self.detector.close()
