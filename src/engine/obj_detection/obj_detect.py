import cv2
import time
import logging
from ultralytics import YOLO, settings as yolo_settings

# Suppress Ultralytics startup banners in multiprocessing workers
yolo_settings.update({"sync": False})
import ultralytics.utils as _ult_utils
_ult_utils.LOGGER.setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.5):
        """
        Initialize the ObjectDetector with a YOLO model.
        """
        self.conf_threshold = conf_threshold
        try:
            self.model = YOLO(model_path)
            # Warmup
            # self.model.predict(source=np.zeros((640, 640, 3), dtype=np.uint8), verbose=False) 
            logger.info(f"YOLO model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model = None

    def detect(self, frame):
        """
        Run detection on a frame.
        Returns a dictionary with detection results:
        {
            'person_count': int,
            'phone_detected': bool,
            'detections': list of (box, class_name, conf)
        }
        """
        if self.model is None:
            return {'person_count': 0, 'phone_detected': False, 'detections': []}

        results = self.model(frame, verbose=False, conf=self.conf_threshold)
        
        person_count = 0
        phone_detected = False
        detections = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = self.model.names[cls_id]
                
                # Filter for person (0) and cell phone (67)
                # COCO indices: person=0, cell phone=67
                if cls_id == 0:
                    person_count += 1
                    detections.append((box.xyxy[0].tolist(), class_name, conf))
                elif cls_id == 67:
                    phone_detected = True
                    detections.append((box.xyxy[0].tolist(), class_name, conf))

        return {
            'person_count': person_count,
            'phone_detected': phone_detected,
            'detections': detections
        }
