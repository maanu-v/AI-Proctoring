import os
import yaml
from typing import Any, Dict
from pathlib import Path

class Config:
    def __init__(self, config_path: str = "src/configs/app.yaml"):
        self.config_path = config_path
        self._config: Dict[str, Any] = {}
        self.load_config()

    def load_config(self):
        """Load configuration from YAML file."""
        # Adjust path relative to project root if needed
        # Assuming run from project root
        path = Path(self.config_path)
        if not path.exists():
            # Try absolute path or relative to this file
            base_dir = Path(__file__).parent.parent.parent
            path = base_dir / self.config_path
        
        if path.exists():
            with open(path, 'r') as f:
                self._config = yaml.safe_load(f) or {}
        else:
            print(f"Warning: Config file not found at {path}. Using defaults.")
            self._config = {}

    @property
    def USE_RICH_LOGGING(self) -> bool:
        return self._config.get('logging', {}).get('use_rich', True)

    @property
    def LOG_LEVEL(self) -> str:
        return self._config.get('logging', {}).get('level', 'INFO')

    @property
    def LOG_FORMAT(self) -> str:
        return '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    @property
    def RICH_LOG_FORMAT(self) -> str:
        return '%(message)s'
    
    @property
    def VIDEO_SOURCE(self):
        return self._config.get('video', {}).get('source', 0)

    @property
    def VIDEO_WIDTH(self) -> int:
        return self._config.get('video', {}).get('width', 640)

    @property
    def VIDEO_HEIGHT(self) -> int:
        return self._config.get('video', {}).get('height', 480)
        
    @property
    def FLASK_HOST(self) -> str:
        return self._config.get('app', {}).get('host', '0.0.0.0')

    @property
    def FLASK_PORT(self) -> int:
        return self._config.get('app', {}).get('port', 5000)

config = Config()
