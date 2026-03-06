"""
Configuration management routes
"""

from fastapi import APIRouter, HTTPException
import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from src.web.api.models import ConfigUpdateRequest, ConfigResponse
from src.utils.config import config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/config", tags=["Configuration"])


@router.get("", response_model=ConfigResponse)
async def get_config():
    """Get current global configuration"""
    return {
        "camera": {
            "index": config.camera.index,
            "width": config.camera.width,
            "height": config.camera.height,
            "fps": config.camera.fps
        },
        "mediapipe": {
            "num_faces": config.mediapipe.num_faces
        },
        "head_pose": {
            "yaw_threshold": config.head_pose.yaw_threshold,
            "pitch_threshold": config.head_pose.pitch_threshold,
            "roll_threshold": config.head_pose.roll_threshold,
            "auto_calibration": config.head_pose.auto_calibration,
            "calibration_time": config.head_pose.calibration_time
        },
        "gaze": {
            "horizontal_threshold_left": config.gaze.horizontal_threshold_left,
            "horizontal_threshold_right": config.gaze.horizontal_threshold_right,
            "vertical_threshold_up": config.gaze.vertical_threshold_up,
            "vertical_threshold_down": config.gaze.vertical_threshold_down,
            "smoothing_factor": config.gaze.smoothing_factor
        },
        "blink": {
            "ear_threshold": config.blink.ear_threshold,
            "min_blink_frames": config.blink.min_blink_frames,
            "long_closure_threshold": config.blink.long_closure_threshold,
            "smoothing_alpha": config.blink.smoothing_alpha
        },
        "thresholds": {
            "max_num_faces": config.thresholds.max_num_faces,
            "enable_no_face_warning": config.thresholds.enable_no_face_warning,
            "multi_face_persistence_time": config.thresholds.multi_face_persistence_time,
            "no_face_persistence_time": config.thresholds.no_face_persistence_time,
            "head_pose_persistence_time": config.thresholds.head_pose_persistence_time,
            "gaze_persistence_time": config.thresholds.gaze_persistence_time,
            "identity_check_interval_frames": config.thresholds.identity_check_interval_frames,
            "identity_persistence_time": config.thresholds.identity_persistence_time
        }
    }


@router.put("")
async def update_config(config_update: ConfigUpdateRequest):
    """
    Update global configuration (affects all new sessions)
    
    Note: This updates in-memory config, not the YAML file.
    For persistent changes, modify src/configs/app.yaml
    """
    try:
        update_data = config_update.dict(exclude_unset=True)
        
        # Update camera config
        if "camera" in update_data:
            for key, value in update_data["camera"].items():
                if hasattr(config.camera, key):
                    setattr(config.camera, key, value)
                    logger.info(f"Updated camera.{key} = {value}")
        
        # Update head pose config
        if "head_pose" in update_data:
            for key, value in update_data["head_pose"].items():
                if hasattr(config.head_pose, key):
                    setattr(config.head_pose, key, value)
                    logger.info(f"Updated head_pose.{key} = {value}")
        
        # Update gaze config
        if "gaze" in update_data:
            for key, value in update_data["gaze"].items():
                if hasattr(config.gaze, key):
                    setattr(config.gaze, key, value)
                    logger.info(f"Updated gaze.{key} = {value}")
        
        # Update blink config
        if "blink" in update_data:
            for key, value in update_data["blink"].items():
                if hasattr(config.blink, key):
                    setattr(config.blink, key, value)
                    logger.info(f"Updated blink.{key} = {value}")
        
        # Update thresholds config
        if "thresholds" in update_data:
            for key, value in update_data["thresholds"].items():
                if hasattr(config.thresholds, key):
                    setattr(config.thresholds, key, value)
                    logger.info(f"Updated thresholds.{key} = {value}")
        
        logger.info("Global configuration updated successfully")
        
        return {
            "message": "Configuration updated successfully",
            "note": "Changes apply to new sessions and analyzers. Restart server for full effect.",
            "updated_fields": list(update_data.keys())
        }
        
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        raise HTTPException(status_code=500, detail=str(e))
