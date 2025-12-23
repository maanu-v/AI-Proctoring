"""
Test script for Face Landmark Detection
Demonstrates usage of the FaceLandmarkDetector class
"""

import cv2
import numpy as np
from src.models.face.face_landmarks import FaceLandmarkDetector
from src.core.video_stream import VideoStream
from src.utils.logger import logger

def main():
    logger.info("Starting Face Landmark Detection Test")
    
    # Initialize video stream
    video_stream = VideoStream()
    video_stream.start()
    
    # Initialize face landmark detector
    detector = FaceLandmarkDetector(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    logger.info("Press 'q' to quit, 's' to save screenshot")
    
    try:
        while True:
            # Get frame from video stream
            frame = video_stream.read()
            if frame is None:
                continue
            
            # Detect face landmarks
            face_landmarks_list = detector.detect(frame)
            
            # Process each detected face
            for face_landmarks in face_landmarks_list:
                # Draw landmarks on frame
                frame = detector.draw_landmarks(
                    frame,
                    face_landmarks,
                    draw_connections=True,
                    landmark_color=(0, 255, 0),
                    connection_color=(255, 255, 255),
                    thickness=1
                )
                
                # Get specific facial features
                eyes = detector.get_eye_landmarks(face_landmarks)
                mouth = detector.get_mouth_landmarks(face_landmarks)
                nose = detector.get_nose_landmarks(face_landmarks)
                
                # Draw eye centers
                if eyes['left_iris'] is not None:
                    left_iris_center = eyes['left_iris'].mean(axis=0)
                    right_iris_center = eyes['right_iris'].mean(axis=0)
                    
                    # Convert to pixel coords
                    left_x = int(left_iris_center[0] * face_landmarks.image_width)
                    left_y = int(left_iris_center[1] * face_landmarks.image_height)
                    right_x = int(right_iris_center[0] * face_landmarks.image_width)
                    right_y = int(right_iris_center[1] * face_landmarks.image_height)
                    
                    cv2.circle(frame, (left_x, left_y), 3, (0, 0, 255), -1)
                    cv2.circle(frame, (right_x, right_y), 3, (0, 0, 255), -1)
                
                # Display landmark count
                cv2.putText(
                    frame,
                    f"Landmarks: {len(face_landmarks.landmarks)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
            
            # Display status
            status_text = f"Faces: {len(face_landmarks_list)}"
            cv2.putText(
                frame,
                status_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Show frame
            cv2.imshow('Face Landmark Detection Test', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Quit requested")
                break
            elif key == ord('s'):
                filename = f"face_landmarks_screenshot_{np.random.randint(1000, 9999)}.jpg"
                cv2.imwrite(filename, frame)
                logger.info(f"Screenshot saved: {filename}")
    
    finally:
        # Cleanup
        video_stream.stop()
        cv2.destroyAllWindows()
        logger.info("Test completed")

if __name__ == "__main__":
    main()
