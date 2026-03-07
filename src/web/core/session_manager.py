"""
Session management and state tracking
"""

import time
from datetime import datetime
from typing import Optional, Dict
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.engine.proctor import ViolationTracker
from src.utils.config import config


class QuizSession:
    """Manages state for an active proctoring session"""
    
    def __init__(
        self, 
        student_id: str, 
        quiz_id: str, 
        profile_image: Optional[np.ndarray] = None
    ):
        """
        Initialize a new quiz session
        
        Args:
            student_id: Unique student identifier
            quiz_id: Unique quiz/exam identifier
            profile_image: Optional profile image for identity verification
        """
        self.student_id = student_id
        self.quiz_id = quiz_id
        self.profile_image = profile_image
        self.reference_embedding = None
        self.violation_tracker = ViolationTracker()
        self.created_at = datetime.now()
        self.frame_count = 0
        
        # Default settings (can be modified per session)
        self.settings = {
            "enable_face_detection": True,
            "enable_head_pose": True,
            "enable_gaze": True,
            "enable_blink": True,
            "enable_object_detection": True,
            "enable_identity_verification": True,
            "enable_no_face_warning": config.thresholds.enable_no_face_warning,
            "enable_no_frame_warning": True,
        }
    
    def to_dict(self) -> Dict:
        """
        Convert session to dictionary representation
        
        Returns:
            Dictionary with session information
        """
        return {
            "student_id": self.student_id,
            "quiz_id": self.quiz_id,
            "created_at": self.created_at.isoformat(),
            "frame_count": self.frame_count,
            "violation_count": self.violation_tracker.get_violation_count(),
            "settings": self.settings
        }
    
    def update_settings(self, settings: Dict) -> None:
        """
        Update session settings and clear violation states for disabled features
        
        Args:
            settings: Dictionary of settings to update
        """
        # Track which features are being disabled
        features_to_clear = []
        
        # Map setting keys to feature names for violation tracker
        setting_to_feature = {
            'enable_head_pose': 'head_pose',
            'enable_gaze': 'gaze',
            'enable_identity_verification': 'identity',
            'enable_object_detection': 'object',
            'enable_face_detection': 'face',
            'enable_no_face_warning': 'face'
        }
        
        # Check which features are being disabled
        for setting_key, feature_name in setting_to_feature.items():
            if setting_key in settings:
                # If the setting was previously enabled and is now being disabled
                if self.settings.get(setting_key, True) and not settings[setting_key]:
                    features_to_clear.append(feature_name)
        
        # Update settings
        self.settings.update(settings)
        
        # Clear violation states for disabled features
        for feature in features_to_clear:
            self.violation_tracker.clear_feature_state(feature)
    
    def increment_frame_count(self) -> int:
        """
        Increment and return current frame count
        
        Returns:
            Updated frame count
        """
        self.frame_count += 1
        return self.frame_count
    
    def set_reference_embedding(self, embedding) -> None:
        """
        Set the reference face embedding for identity verification
        
        Args:
            embedding: Face embedding vector
        """
        self.reference_embedding = embedding
    
    def get_violations(self) -> list:
        """
        Get all violations for this session
        
        Returns:
            List of violation dictionaries
        """
        return self.violation_tracker.get_logs()
    
    def clear_violations(self) -> None:
        """Clear all violations"""
        self.violation_tracker.reset()


class SessionManager:
    """Manages multiple active sessions"""
    
    def __init__(self):
        """Initialize the session manager"""
        self._sessions: Dict[str, QuizSession] = {}
    
    def create_session(
        self, 
        student_id: str, 
        quiz_id: str,
        profile_image: Optional[np.ndarray] = None
    ) -> tuple[str, QuizSession]:
        """
        Create a new session
        
        Args:
            student_id: Student identifier
            quiz_id: Quiz identifier
            profile_image: Optional profile image
            
        Returns:
            Tuple of (session_id, QuizSession)
            
        Raises:
            ValueError: If session already exists
        """
        session_id = self._generate_session_id(student_id, quiz_id)
        
        if session_id in self._sessions:
            raise ValueError("Session already exists")
        
        session = QuizSession(student_id, quiz_id, profile_image)
        self._sessions[session_id] = session
        
        return session_id, session
    
    def get_session(self, session_id: str) -> Optional[QuizSession]:
        """
        Get a session by ID
        
        Args:
            session_id: Session identifier
            
        Returns:
            QuizSession if found, None otherwise
        """
        return self._sessions.get(session_id)
    
    def end_session(self, session_id: str) -> Optional[QuizSession]:
        """
        End and remove a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            The removed session if found, None otherwise
        """
        return self._sessions.pop(session_id, None)
    
    def list_sessions(self) -> list:
        """
        Get list of all active sessions
        
        Returns:
            List of session dictionaries
        """
        return [session.to_dict() for session in self._sessions.values()]
    
    def get_session_count(self) -> int:
        """
        Get number of active sessions
        
        Returns:
            Number of active sessions
        """
        return len(self._sessions)
    
    def session_exists(self, session_id: str) -> bool:
        """
        Check if a session exists
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session exists, False otherwise
        """
        return session_id in self._sessions
    
    @staticmethod
    def _generate_session_id(student_id: str, quiz_id: str) -> str:
        """
        Generate unique session ID
        
        Args:
            student_id: Student identifier
            quiz_id: Quiz identifier
            
        Returns:
            Unique session identifier
        """
        timestamp = int(time.time())
        return f"{student_id}_{quiz_id}_{timestamp}"


# Global session manager instance
session_manager = SessionManager()
