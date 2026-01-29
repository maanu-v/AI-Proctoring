import cv2
import mediapipe as mp
import numpy as np
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ..utils.logger import logger

class FaceMeshDetector:
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path='src/face/face_landmarker.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=5,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame):
        """
        Process the frame to detect face mesh.
        Returns:
            results: The FaceLandmarkerResult object
        """
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Run inference
        detection_result = self.detector.detect(mp_image)
        return detection_result

    def draw_landmarks(self, image, results, analyzed_mode=False):
        """
        Draw landmarks and face count on the image.
        If analyzed_mode is True, draws the mesh (points).
        Always draws the face count.
        """
        face_count = 0
        if results.face_landmarks:
            face_count = len(results.face_landmarks)
            
            if analyzed_mode:
                for face_landmarks in results.face_landmarks:
                    for landmark in face_landmarks:
                        x = int(landmark.x * image.shape[1])
                        y = int(landmark.y * image.shape[0])
                        cv2.circle(image, (x, y), 1, (0, 255, 255), -1)

        # Draw face count (Always visible)
        cv2.putText(
            image,
            f"Faces: {face_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0) if face_count == 1 else (0, 0, 255),
            2,
            cv2.LINE_AA
        )
        
        return image
