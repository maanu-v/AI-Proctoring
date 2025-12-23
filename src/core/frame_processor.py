import cv2
from src.core.video_stream import VideoStream

class FrameProcessor:
    def __init__(self, camera_id=None):
        # Initialize VideoStream (Singleton)
        self.video_stream = VideoStream(source=camera_id)
        self.video_stream.start()

    def get_frame(self):
        frame = self.video_stream.read()
        if frame is None:
            return None
        
        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            return None
            
        return buffer.tobytes()

    def __del__(self):
        # VideoStream handles its own cleanup via its own __del__ or explicit stop
        pass
