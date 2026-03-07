"""
Pydantic models for API requests and responses
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any


class SessionCreateRequest(BaseModel):
    """Request model for creating a new session"""
    student_id: str = Field(..., description="Unique student identifier")
    quiz_id: str = Field(..., description="Unique quiz identifier")
    profile_image_base64: Optional[str] = Field(
        None, 
        description="Base64 encoded profile image for identity verification"
    )


class SessionResponse(BaseModel):
    """Response model for session creation"""
    session_id: str
    student_id: str
    quiz_id: str
    created_at: str
    message: str


class SessionInfo(BaseModel):
    """Model for session information"""
    student_id: str
    quiz_id: str
    created_at: str
    frame_count: int
    violation_count: int
    settings: Dict[str, bool]


class AnalysisResult(BaseModel):
    """Response model for frame analysis"""
    session_id: str
    frame_count: int
    timestamp: str
    face_detected: bool
    face_count: int
    violations: List[Dict[str, Any]]
    analysis: Dict[str, Any]
    warnings: List[str]


class FrameAnalysisRequest(BaseModel):
    """Request model for frame analysis"""
    frame_base64: str = Field(..., description="Base64 encoded image frame")


class SettingsUpdateRequest(BaseModel):
    """Request model for updating session settings"""
    enable_face_detection: Optional[bool] = None
    enable_head_pose: Optional[bool] = None
    enable_gaze: Optional[bool] = None
    enable_blink: Optional[bool] = None
    enable_object_detection: Optional[bool] = None
    enable_identity_verification: Optional[bool] = None
    enable_no_face_warning: Optional[bool] = None
    enable_no_frame_warning: Optional[bool] = None


class ConfigUpdateRequest(BaseModel):
    """Request model for updating global configuration"""
    camera: Optional[Dict[str, Any]] = None
    mediapipe: Optional[Dict[str, Any]] = None
    head_pose: Optional[Dict[str, Any]] = None
    gaze: Optional[Dict[str, Any]] = None
    blink: Optional[Dict[str, Any]] = None
    thresholds: Optional[Dict[str, Any]] = None


class ViolationsResponse(BaseModel):
    """Response model for violations"""
    session_id: str
    total_violations: int
    violations: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    """Response model for health check"""
    service: str
    status: str
    version: str
    active_sessions: int


class MessageResponse(BaseModel):
    """Generic message response"""
    message: str


class ConfigResponse(BaseModel):
    """Response model for configuration"""
    camera: Dict[str, Any]
    mediapipe: Dict[str, Any]
    head_pose: Dict[str, Any]
    gaze: Dict[str, Any]
    blink: Dict[str, Any]
    thresholds: Dict[str, Any]
