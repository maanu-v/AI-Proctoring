import cv2
import time
from typing import Optional, Tuple, Union
from src.utils.logger import get_logger

logger = get_logger(__name__)

class VideoStream:
    def __init__(self, source: Union[int, str] = 0):
        self.source = source
        self.cap: Optional[cv2.VideoCapture] = None
        self.width: int = 640
        self.height: int = 480
        self.fps: int = 30
        self._is_running: bool = False

    def start(self):
        if self._is_running:
            return
        
        logger.info(f"Starting video stream from source {self.source}")
        self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            logger.error(f"Failed to open video source {self.source}")
            raise RuntimeError(f"Could not open video source {self.source}")
            
        self._is_running = True
        
        # Set resolution if possible
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

    def read(self) -> Tuple[bool, Optional[cv2.Mat]]:
        if not self._is_running or self.cap is None:
            return False, None
            
        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Failed to read frame")
            return False, None
            
        return True, frame

    def stop(self):
        if self._is_running and self.cap is not None:
            logger.info("Stopping video stream")
            self.cap.release()
            
        self._is_running = False
        self.cap = None

    def is_opened(self) -> bool:
        return self.cap.isOpened() if self.cap else False
