import cv2
import time

class FrameProcessor:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.camera = cv2.VideoCapture(self.camera_id)
        if not self.camera.isOpened():
            raise RuntimeError(f"Could not start camera with id {self.camera_id}")

    def get_frame(self):
        success, frame = self.camera.read()
        if not success:
            return None
        
        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            return None
            
        return buffer.tobytes()

    def __del__(self):
        if self.camera.isOpened():
            self.camera.release()
