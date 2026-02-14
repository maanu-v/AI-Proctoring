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

@dataclass
class ThresholdsConfig:
    max_num_faces: int = 1
    enable_no_face_warning: bool = True
    multi_face_persistence_time: float = 3.0
    no_face_persistence_time: float = 3.0
    head_pose_persistence_time: float = 3.0

class Config:
    def __init__(self, config_path: str = "src/configs/app.yaml"):
        self.config_path = config_path
        self._config: Dict[str, Any] = self._load_config()
        
        self.camera = CameraConfig(**self._config.get("camera", {}).get("params", {}))
        self.mediapipe = MediaPipeConfig(**self._config.get("mediapipe", {}))
        self.head_pose = HeadPoseConfig(**self._config.get("head_pose", {}))
        self.thresholds = ThresholdsConfig(**self._config.get("thresholds", {}))

    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            print(f"Warning: Config file not found at {self.config_path}. Using defaults.")
            return {}

config = Config()
