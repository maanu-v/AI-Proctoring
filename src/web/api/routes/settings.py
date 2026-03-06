"""
Settings management routes
"""

from fastapi import APIRouter, HTTPException, Depends
import logging

from src.web.api.models import SettingsUpdateRequest, MessageResponse
from src.web.api.dependencies import get_session
from src.web.core.session_manager import QuizSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/session", tags=["Settings"])


@router.get("/{session_id}/settings")
async def get_session_settings(
    session_id: str,
    session: QuizSession = Depends(get_session)
):
    """Get current settings for a session"""
    return session.settings


@router.put("/{session_id}/settings")
async def update_session_settings(
    session_id: str,
    settings: SettingsUpdateRequest,
    session: QuizSession = Depends(get_session)
):
    """
    Update settings for an active session
    
    Allows enabling/disabling features on-the-fly during an exam
    """
    try:
        # Get only the fields that were actually set
        update_data = settings.dict(exclude_unset=True)
        
        # Update session settings
        session.update_settings(update_data)
        
        logger.info(f"Updated settings for session {session_id}: {update_data}")
        
        return {
            "message": "Settings updated successfully",
            "settings": session.settings
        }
        
    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))
