import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class CameraConfig:
    index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30

@dataclass
class MediaPipeConfig:
    num_faces: int = 1

@dataclass
class HeadPoseConfig:
    yaw_threshold: float = 30.0
    pitch_threshold: float = 25.0
    roll_threshold: float = 25.0
    auto_calibration: bool = False
    calibration_time: int = 3

@dataclass
class ThresholdsConfig:
    max_num_faces: int = 1
    enable_no_face_warning: bool = True
    multi_face_persistence_time: float = 3.0
    no_face_persistence_time: float = 3.0
    head_pose_persistence_time: float = 3.0
    gaze_persistence_time: float = 2.0
    identity_check_interval_frames: int = 30
    identity_persistence_time: float = 2.0

@dataclass
class BlinkConfig:
    ear_threshold: float = 0.21
    min_blink_frames: int = 2
    long_closure_threshold: float = 1.0
    smoothing_alpha: float = 0.4
    calibration_duration: float = 5.0
    calibration_ratio: float = 0.75

@dataclass
class GazeConfig:
    horizontal_threshold_left: float = 0.42
    horizontal_threshold_right: float = 0.58
    vertical_threshold_up: float = 0.40
    vertical_threshold_down: float = 0.60
    smoothing_factor: float = 0.5

@dataclass
class ModelConfig:
    # Paths
    dataset_path: str = "data/raw/database"
    processed_path: str = "data/processed"
    model_path: str = "data/models"
    reports_path: str = "data/reports"
    # Data pipeline
    sequence_length: int = 10
    overlap: float = 0.5
    test_size: float = 0.2
    # Training hyperparameters
    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 1e-4
    patience: int = 10
    lr_reduce_factor: float = 0.5
    lr_patience: int = 5
    min_lr: float = 1e-7
    anomaly_threshold: float = 0.3

@dataclass
class BatchConfig:
    database_dir: str = "data/raw/database"
    output_dir: str = "data/model_processing/results"
    max_workers: int = 2
    sample_rate: int = 3

class Config:
    def __init__(self, config_path: str = "src/configs/app.yaml"):
        self.config_path = config_path
        self._config: Dict[str, Any] = self._load_config()
        
        self.camera = CameraConfig(**self._config.get("camera", {}).get("params", {}))
        self.mediapipe = MediaPipeConfig(**self._config.get("mediapipe", {}))
        self.head_pose = HeadPoseConfig(**self._config.get("head_pose", {}))
        self.gaze = GazeConfig(**self._config.get("gaze", {}))
        self.blink = BlinkConfig(**self._config.get("blink", {}))
        self.thresholds = ThresholdsConfig(**self._config.get("thresholds", {}))
        self.model = ModelConfig(**self._config.get("model", {}))
        self.batch = BatchConfig(**self._config.get("batch", {}))

    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            print(f"Warning: Config file not found at {self.config_path}. Using defaults.")
            return {}

config = Config()
