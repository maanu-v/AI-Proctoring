"""
Violations management routes
"""

from fastapi import APIRouter, HTTPException, Depends
import logging

from src.web.api.models import ViolationsResponse, MessageResponse
from src.web.api.dependencies import get_session
from src.web.core.session_manager import QuizSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/session", tags=["Violations"])


@router.get("/{session_id}/violations", response_model=ViolationsResponse)
async def get_violations(
    session_id: str,
    session: QuizSession = Depends(get_session)
):
    """Get all violations for a session"""
    return ViolationsResponse(
        session_id=session_id,
        total_violations=session.violation_tracker.get_violation_count(),
        violations=session.get_violations()
    )


@router.delete("/{session_id}/violations")
async def clear_violations(
    session_id: str,
    session: QuizSession = Depends(get_session)
):
    """
    Clear all violations for a session
    
    Useful for testing or resetting violation state
    """
    try:
        session.clear_violations()
        logger.info(f"Violations cleared for session {session_id}")
        
        return MessageResponse(message="Violations cleared successfully")
        
    except Exception as e:
        logger.error(f"Error clearing violations: {e}")
        raise HTTPException(status_code=500, detail=str(e))
