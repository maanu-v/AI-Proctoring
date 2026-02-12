import time
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class ViolationEvent:
    timestamp: float
    type: str
    duration: float
    message: str

class ViolationTracker:
    def __init__(self, away_threshold_seconds: float = 5.0):
        self.away_threshold = away_threshold_seconds
        self.current_away_start: Optional[float] = None
        self.violations: List[ViolationEvent] = []
        self.is_warning_active: bool = False
        
    def check_head_pose(self, direction_label: str) -> bool:
        """
        Checks head pose direction.
        Returns True if a warning should be active (looking away > threshold).
        """
        is_looking_away = direction_label != "Forward"
        now = time.time()
        
        if is_looking_away:
            if self.current_away_start is None:
                self.current_away_start = now
            
            elapsed = now - self.current_away_start
            
            if elapsed > self.away_threshold:
                if not self.is_warning_active:
                    # New violation triggered
                    self.is_warning_active = True
                    logger.warning(f"Violation triggered: Looking {direction_label} for {elapsed:.1f}s")
                return True
        else:
            # User looked back
            if self.current_away_start is not None:
                elapsed = now - self.current_away_start
                if elapsed > self.away_threshold:
                    # Log the completed violation
                    self.violations.append(ViolationEvent(
                        timestamp=now,
                        type="HEAD_POSE",
                        duration=elapsed,
                        message=f"Looked away for {elapsed:.1f}s"
                    ))
                    logger.info(f"Violation ended. Duration: {elapsed:.1f}s")
                
                self.current_away_start = None
                self.is_warning_active = False
                
        return self.is_warning_active

    def get_violation_count(self) -> int:
        return len(self.violations)
        
    def get_logs(self) -> List[Dict]:
        return [asdict(v) for v in self.violations]
    
    def reset(self):
        self.current_away_start = None
        self.violations = []
        self.is_warning_active = False
