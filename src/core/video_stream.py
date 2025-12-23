import cv2
import time
import threading
from typing import Optional, Union
import numpy as np
from src.utils.logger import logger

class VideoStream:
    """
    Handles raw video capture from a camera source using a single-producer architecture.
    Manages camera initialization, frame capture, FPS regulation, and resource cleanup.
    """
    _instance: Optional['VideoStream'] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(VideoStream, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, source: Optional[Union[int, str]] = None, width: Optional[int] = None, height: Optional[int] = None, fps: Optional[int] = None):
        if self._initialized:
            return
            
        from src.utils.config import config
        
        self.source = source if source is not None else config.CAMERA_SOURCE
        self.width = width if width is not None else config.CAMERA_WIDTH
        self.height = height if height is not None else config.CAMERA_HEIGHT
        self.target_fps = fps if fps is not None else config.CAMERA_FPS
        self.frame_duration = 1.0 / self.target_fps
        
        self.camera: Optional[cv2.VideoCapture] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self._current_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        
        self._initialized = True
        logger.info(f"VideoStream initialized configuration: source={self.source}, resolution={self.width}x{self.height}, fps={self.target_fps}")

    def start(self):
        """Start the video stream capture thread."""
        with self._lock:
            if self.running:
                logger.warning("VideoStream is already running")
                return

            logger.info(f"Starting VideoStream from source {self.source}...")
            self.camera = cv2.VideoCapture(self.source)
            
            if not self.camera.isOpened():
                logger.error(f"Failed to open camera source: {self.source}")
                raise RuntimeError(f"Could not start camera with source {self.source}")

            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.camera.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Camera started. Actual properties: {actual_width}x{actual_height} @ {actual_fps} FPS")

            self.running = True
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()

    def _update(self):
        """Internal loop to continuously capture frames and regulate FPS."""
        while self.running:
            start_time = time.time()
            
            if self.camera and self.camera.isOpened():
                ret, frame = self.camera.read()
                if ret:
                    with self._frame_lock:
                        self._current_frame = frame
                else:
                    logger.warning("Failed to read frame from camera")
            
            # FPS regulation
            elapsed = time.time() - start_time
            wait_time = self.frame_duration - elapsed
            if wait_time > 0:
                time.sleep(wait_time)

    def read(self) -> Optional[np.ndarray]:
        """
        Get the latest captured frame.
        Returns None if no frame is available yet.
        """
        with self._frame_lock:
            if self._current_frame is not None:
                return self._current_frame.copy()
        return None

    def stop(self):
        """Stop the video stream and release resources."""
        logger.info("Stopping VideoStream...")
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        
        if self.camera and self.camera.isOpened():
            self.camera.release()
        
        self.camera = None
        logger.info("VideoStream stopped and resources released")

    def __del__(self):
        self.stop()
