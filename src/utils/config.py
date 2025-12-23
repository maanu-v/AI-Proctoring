import os
import yaml
from typing import Any, Dict

class Config:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_path = os.path.join(self.base_dir, "configs", "app.yaml")
        self._config: Dict[str, Any] = self._load_config()

        # App settings
        self.APP_NAME = self._get_nested("app.name", "AI Proctor")
        self.APP_VERSION = self._get_nested("app.version", "1.0.0")
        self.DEBUG = self._get_nested("app.debug", True)
        self.HOST = self._get_nested("app.host", "0.0.0.0")
        self.PORT = self._get_nested("app.port", 8000)

        # Camera settings
        self.CAMERA_SOURCE = self._get_nested("camera.source", 0)
        self.CAMERA_WIDTH = self._get_nested("camera.width", 1280)
        self.CAMERA_HEIGHT = self._get_nested("camera.height", 720)
        self.CAMERA_FPS = self._get_nested("camera.fps", 30)

        # Logging settings
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", self._get_nested("logging.level", "INFO"))
        self.USE_RICH_LOGGING = self._get_nested("logging.use_rich", True)
        self.RICH_LOG_FORMAT = self._get_nested("logging.rich_format", "%(message)s")
        self.LOG_FORMAT = self._get_nested("logging.standard_format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    def _load_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            print(f"Warning: Config file not found at {self.config_path}. Using defaults.")
            return {}
        
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Error loading config file: {e}")
            return {}

    def _get_nested(self, path: str, default: Any = None) -> Any:
        keys = path.split(".")
        value = self._config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
        return value if value is not None else default

config = Config()
