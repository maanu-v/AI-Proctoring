"""
Analyzer initialization and management
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.engine.face.mesh_detector import MeshDetector
from src.engine.face.head_pose import HeadPoseEstimator
from src.engine.face.gaze_estimation import GazeEstimator
from src.engine.face.blink_estimation import BlinkEstimator
from src.engine.obj_detection.obj_detect import ObjectDetector
from src.engine.face.face_embedding import FaceEmbedder
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AnalyzerManager:
    """Manages all analysis components (singleton pattern)"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern implementation"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize analyzers (only once)"""
        if self._initialized:
            return
        
        self.mesh_detector = None
        self.head_pose_estimator = None
        self.gaze_estimator = None
        self.blink_estimator = None
        self.object_detector = None
        self.face_embedder = None
        
        self._initialized = True
    
    def initialize(self) -> None:
        """Lazy initialization of all analyzers"""
        if self.mesh_detector is None:
            logger.info("Initializing MeshDetector...")
            self.mesh_detector = MeshDetector()
        
        if self.head_pose_estimator is None:
            logger.info("Initializing HeadPoseEstimator...")
            self.head_pose_estimator = HeadPoseEstimator(
                yaw_threshold=config.head_pose.yaw_threshold,
                pitch_threshold=config.head_pose.pitch_threshold,
                roll_threshold=config.head_pose.roll_threshold
            )
        
        if self.gaze_estimator is None:
            logger.info("Initializing GazeEstimator...")
            self.gaze_estimator = GazeEstimator()
        
        if self.blink_estimator is None:
            logger.info("Initializing BlinkEstimator...")
            self.blink_estimator = BlinkEstimator()
        
        if self.object_detector is None:
            logger.info("Initializing ObjectDetector...")
            self.object_detector = ObjectDetector()
        
        if self.face_embedder is None:
            logger.info("Initializing FaceEmbedder...")
            self.face_embedder = FaceEmbedder()
        
        logger.info("All analyzers initialized successfully")
    
    def get_analyzers(self) -> dict:
        """
        Get all analyzers, initializing if necessary
        
        Returns:
            Dictionary with all analyzer instances
        """
        self.initialize()
        
        return {
            "mesh": self.mesh_detector,
            "head_pose": self.head_pose_estimator,
            "gaze": self.gaze_estimator,
            "blink": self.blink_estimator,
            "object": self.object_detector,
            "embedder": self.face_embedder
        }
    
    def is_initialized(self) -> bool:
        """
        Check if analyzers are initialized
        
        Returns:
            True if all analyzers are initialized
        """
        return all([
            self.mesh_detector is not None,
            self.head_pose_estimator is not None,
            self.gaze_estimator is not None,
            self.blink_estimator is not None,
            self.object_detector is not None,
            self.face_embedder is not None
        ])


# Global analyzer manager instance
analyzer_manager = AnalyzerManager()


def get_analyzer_manager() -> AnalyzerManager:
    """
    Dependency injection function for FastAPI
    
    Returns:
        Global analyzer manager instance
    """
    return analyzer_manager
