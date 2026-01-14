import cv2
import threading
import queue
import time
from typing import List, Optional
from ..utils.logger import logger
from ..utils.config import config

class VideoStream:
    _instance: Optional['VideoStream'] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self):
        if VideoStream._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            VideoStream._instance = self
            
        self._capture_thread: Optional[threading.Thread] = None
        self._stop_event: threading.Event = threading.Event()
        self._subscribers: List[queue.Queue] = []
        self._subscribers_lock: threading.Lock = threading.Lock()
        self._camera_active: bool = False

    @staticmethod
    def get_instance() -> 'VideoStream':
        if VideoStream._instance is None:
            with VideoStream._lock:
                if VideoStream._instance is None:
                    VideoStream()
        return VideoStream._instance

    def _start_camera(self):
        """Start the camera thread if it's not already running."""
        with self._lock:
            if self._camera_active:
                return

            self._stop_event.clear()
            self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._capture_thread.start()
            self._camera_active = True
            logger.info("Camera started.")

    def _stop_camera(self):
        """Stop the camera thread."""
        with self._lock:
            if not self._camera_active:
                return

            self._stop_event.set()
            if self._capture_thread:
                self._capture_thread.join(timeout=2.0)
            self._camera_active = False
            logger.info("Camera stopped.")

    def subscribe(self) -> queue.Queue:
        """
        Subscribe to the video stream.
        Returns a Queue that will receive frames.
        Starts the camera if it's not already running.
        """
        if not self._camera_active:
            self._start_camera()

        q = queue.Queue(maxsize=10) # Drop old frames if consumer is slow
        with self._subscribers_lock:
            self._subscribers.append(q)
            logger.debug(f"New subscriber. Total: {len(self._subscribers)}")
        return q

    def unsubscribe(self, q: queue.Queue):
        """
        Unsubscribe from the video stream.
        Stops the camera if there are no more subscribers.
        """
        with self._subscribers_lock:
            if q in self._subscribers:
                self._subscribers.remove(q)
                logger.debug(f"Subscriber removed. Total: {len(self._subscribers)}")
            
            should_stop = len(self._subscribers) == 0
        
        if should_stop:
            logger.info("No more subscribers. Stopping camera.")
            self._stop_camera()

    def _capture_loop(self):
        """Main loop for capturing frames and distributing them to subscribers."""
        cap = cv2.VideoCapture(config.VIDEO_SOURCE)
        if not cap.isOpened():
            logger.error(f"Could not open video source {config.VIDEO_SOURCE}")
            self._camera_active = False
            return

        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.VIDEO_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.VIDEO_HEIGHT)

        fps_delay = 1.0 / config.VIDEO_SOURCE if isinstance(config.VIDEO_SOURCE, int) and config.VIDEO_SOURCE > 10 else 0.03 # approximate 30fps if not specified

        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to receive frame from camera.")
                time.sleep(0.1)
                continue

            # Identify dead subscribers
            dead_queues = []
            
            with self._subscribers_lock:
                current_subscribers = list(self._subscribers) # Copy list to iterate safely

            if not current_subscribers:
                # Should have been stopped by unsubscribe, but double check
                time.sleep(0.1)
                continue

            for q in current_subscribers:
                try:
                    # Non-blocking put, remove old frame if full to ensure real-time
                    if q.full():
                        try:
                            q.get_nowait()
                        except queue.Empty:
                            pass
                    q.put_nowait(frame.copy())
                except queue.Full:
                    pass # Should not happen with get_nowait above
                except Exception as e:
                    logger.error(f"Error putting frame to queue: {e}")
                    # Consider removing subscriber if it errors repeatedly? For now, we rely on manual unsubscribe.
            
            # Simple rate limiting if needed, but cv2.read blocks usually
            # time.sleep(0.001)

        cap.release()
        logger.info("Camera released.")
