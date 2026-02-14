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
class ThresholdsConfig:
    max_num_faces: int = 1

class Config:
    def __init__(self, config_path: str = "src/configs/app.yaml"):
        self.config_path = config_path
        self._config: Dict[str, Any] = self._load_config()
        
        self.camera = CameraConfig(**self._config.get("camera", {}).get("params", {}))
        self.mediapipe = MediaPipeConfig(**self._config.get("mediapipe", {}))
        self.thresholds = ThresholdsConfig(**self._config.get("thresholds", {}))

    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            print(f"Warning: Config file not found at {self.config_path}. Using defaults.")
            return {}

config = Config()
