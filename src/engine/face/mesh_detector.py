import mediapipe as mp
import cv2
import numpy as np
import time
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class MeshDetector:
    def __init__(self, model_path: str = "src/models/face_landmarker.task"):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=config.mediapipe.num_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.VIDEO
        )
        try:
            self.landmarker = vision.FaceLandmarker.create_from_options(options)
            logger.info(f"Face Landmarker initialized successfully with num_faces={config.mediapipe.num_faces}")
        except Exception as e:
            logger.error(f"Failed to initialize Face Landmarker: {e}")
            raise

    def process(self, image: cv2.Mat, timestamp_ms: int):
        # MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Detect
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        return result

    def draw_landmarks(self, image: cv2.Mat, result) -> cv2.Mat:
        if not result.face_landmarks:
            return image
            
        annotated_image = image.copy()
        
        for face_landmarks in result.face_landmarks:
            # Draw face mesh tesselation
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style())
            
            # Draw face mesh contours
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style())
            
            # Draw irises
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())
                
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())
                
        return annotated_image

    def close(self):
        self.landmarker.close()

if __name__ == "__main__":
    # Test the detector
    try:
        detector = MeshDetector()
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            timestamp_ms = int(time.time() * 1000)
            result = detector.process(frame, timestamp_ms)
            annotated_frame = detector.draw_landmarks(frame, result)
            
            cv2.imshow("Face Mesh", annotated_frame)
            
            if cv2.waitKey(5) & 0xFF == 27:
                break
                
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
    except Exception as e:
        print(f"Error: {e}")
