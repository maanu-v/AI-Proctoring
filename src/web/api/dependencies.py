"""
FastAPI dependencies
"""

from fastapi import HTTPException, Depends
from typing import Optional

from src.web.core.session_manager import session_manager, QuizSession
from src.web.core.analyzers import analyzer_manager, AnalyzerManager


def get_session_manager():
    """Dependency for session manager"""
    return session_manager


def get_analyzer_manager():
    """Dependency for analyzer manager"""
    return analyzer_manager


def get_session(session_id: str) -> QuizSession:
    """
    Dependency to get and validate a session
    
    Args:
        session_id: Session identifier
        
    Returns:
        QuizSession instance
        
    Raises:
        HTTPException: If session not found
    """
    session = session_manager.get_session(session_id)
    
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session


def validate_session_exists(session_id: str) -> str:
    """
    Dependency to validate session exists
    
    Args:
        session_id: Session identifier
        
    Returns:
        session_id if valid
        
    Raises:
        HTTPException: If session not found
    """
    if not session_manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session_id
