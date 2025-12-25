import cv2
import threading
from typing import Optional, Tuple
import numpy as np
from ..utils.logger import logger
from ..utils.config import config


class VideoStream:
    """
    Singleton class for managing video stream from camera.
    Ensures only one instance of the video capture is used across the application.
    Thread-safe implementation.
    """
    _instance: Optional['VideoStream'] = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False

    def __new__(cls) -> 'VideoStream':
        """Thread-safe singleton implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(VideoStream, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the video stream (only once)."""
        if not VideoStream._initialized:
            with VideoStream._lock:
                if not VideoStream._initialized:
                    self.camera_source = config.CAMERA_SOURCE
                    self.width = config.CAMERA_WIDTH
                    self.height = config.CAMERA_HEIGHT
                    self.fps = config.CAMERA_FPS
                    
                    self.cap: Optional[cv2.VideoCapture] = None
                    self.frame: Optional[np.ndarray] = None
                    self.is_running: bool = False
                    self.frame_lock: threading.Lock = threading.Lock()
                    self.capture_thread: Optional[threading.Thread] = None
                    
                    VideoStream._initialized = True
                    logger.info("VideoStream singleton initialized")

    def start(self) -> bool:
        """
        Start the video capture stream.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        if self.is_running:
            logger.warning("VideoStream is already running")
            return True

        try:
            self.cap = cv2.VideoCapture(self.camera_source)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera source: {self.camera_source}")
                return False

            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            # Read first frame to verify
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to read initial frame from camera")
                self.cap.release()
                return False

            with self.frame_lock:
                self.frame = frame

            self.is_running = True
            
            # Start capture thread
            self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
            self.capture_thread.start()
            
            logger.info(f"VideoStream started successfully (Source: {self.camera_source}, "
                       f"Resolution: {self.width}x{self.height}, FPS: {self.fps})")
            return True

        except Exception as e:
            logger.error(f"Error starting VideoStream: {e}")
            if self.cap:
                self.cap.release()
            return False

    def _capture_frames(self) -> None:
        """Internal method to continuously capture frames in a separate thread."""
        logger.info("Frame capture thread started")
        
        while self.is_running:
            try:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    
                    if ret:
                        with self.frame_lock:
                            self.frame = frame
                    else:
                        logger.warning("Failed to read frame from camera")
                        # Try to reconnect
                        self.cap.release()
                        self.cap = cv2.VideoCapture(self.camera_source)
                        if not self.cap.isOpened():
                            logger.error("Failed to reconnect to camera")
                            break
                else:
                    logger.error("Camera is not opened")
                    break
                    
            except Exception as e:
                logger.error(f"Error in frame capture thread: {e}")
                break

        logger.info("Frame capture thread stopped")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the current frame from the video stream.
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame)
        """
        if not self.is_running:
            return False, None

        with self.frame_lock:
            if self.frame is not None:
                return True, self.frame.copy()
            return False, None

    def get_frame_jpeg(self, quality: int = 90) -> Optional[bytes]:
        """
        Get the current frame as JPEG bytes.
        
        Args:
            quality: JPEG quality (0-100)
            
        Returns:
            Optional[bytes]: JPEG encoded frame or None
        """
        ret, frame = self.read()
        
        if not ret or frame is None:
            return None

        try:
            # Encode frame as JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            ret, buffer = cv2.imencode('.jpg', frame, encode_param)
            
            if ret:
                return buffer.tobytes()
            return None
            
        except Exception as e:
            logger.error(f"Error encoding frame to JPEG: {e}")
            return None

    def stop(self) -> None:
        """Stop the video capture stream."""
        if not self.is_running:
            logger.warning("VideoStream is not running")
            return

        logger.info("Stopping VideoStream...")
        self.is_running = False

        # Wait for capture thread to finish
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)

        # Release camera
        if self.cap:
            self.cap.release()
            self.cap = None

        with self.frame_lock:
            self.frame = None

        logger.info("VideoStream stopped successfully")

    def is_active(self) -> bool:
        """Check if the video stream is active."""
        return self.is_running and self.cap is not None and self.cap.isOpened()

    def get_properties(self) -> dict:
        """Get current camera properties."""
        if not self.cap or not self.cap.isOpened():
            return {}

        return {
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": int(self.cap.get(cv2.CAP_PROP_FPS)),
            "is_running": self.is_running,
            "source": self.camera_source
        }

    @classmethod
    def get_instance(cls) -> 'VideoStream':
        """Get the singleton instance of VideoStream."""
        return cls()

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.stop()


# Global singleton instance
video_stream = VideoStream.get_instance()
