"""
Session management routes
"""

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends
from datetime import datetime
import numpy as np
import logging

from src.web.api.models import SessionResponse
from src.web.api.dependencies import get_session_manager, get_analyzer_manager
from src.web.core.session_manager import SessionManager
from src.web.core.analyzers import AnalyzerManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/session", tags=["Session Management"])


@router.post("/start", response_model=SessionResponse)
async def start_session(
    student_id: str = Form(...),
    quiz_id: str = Form(...),
    profile_image: UploadFile = File(None),
    session_mgr: SessionManager = Depends(get_session_manager),
    analyzer_mgr: AnalyzerManager = Depends(get_analyzer_manager)
):
    """
    Start a new proctoring session
    
    - **student_id**: Unique identifier for the student
    - **quiz_id**: Unique identifier for the quiz/exam
    - **profile_image**: Optional profile image for identity verification
    """
    try:
        # Process profile image if provided
        profile_img = None
        ref_embedding = None
        
        if profile_image:
            contents = await profile_image.read()
            nparr = np.frombuffer(contents, np.uint8)
            
            import cv2
            profile_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if profile_img is not None:
                # Generate reference embedding
                analyzers = analyzer_mgr.get_analyzers()
                ref_embedding = analyzers["embedder"].get_embedding(profile_img)
                
                if ref_embedding is None:
                    logger.warning("Could not generate embedding from profile image")
        
        # Create session (or resume if already exists)
        try:
            session_id, session, is_resumed = session_mgr.create_session(
                student_id, 
                quiz_id, 
                profile_img,
                allow_resume=True
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Set reference embedding if available (only for new sessions)
        if ref_embedding is not None and not is_resumed:
            session.set_reference_embedding(ref_embedding)
        
        if is_resumed:
            logger.warning(f"Resumed session {session_id} for student {student_id}, quiz {quiz_id} (Resume #{session.resume_count})")
            message = f"Session resumed successfully (Resume #{session.resume_count})"
        else:
            logger.info(f"Started session {session_id} for student {student_id}, quiz {quiz_id}")
            message = "Session started successfully"
        
        return SessionResponse(
            session_id=session_id,
            student_id=student_id,
            quiz_id=quiz_id,
            created_at=session.created_at.isoformat(),
            message=message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/end/{session_id}")
async def end_session(
    session_id: str,
    session_mgr: SessionManager = Depends(get_session_manager)
):
    """
    End a proctoring session and return final report
    """
    session = session_mgr.get_session(session_id)
    
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Generate final report
    duration_seconds = (datetime.now() - session.created_at).total_seconds()
    
    report = {
        "session_id": session_id,
        "student_id": session.student_id,
        "quiz_id": session.quiz_id,
        "started_at": session.created_at.isoformat(),
        "ended_at": datetime.now().isoformat(),
        "duration_seconds": int(duration_seconds),
        "total_frames": session.frame_count,
        "total_violations": session.violation_tracker.get_violation_count(),
        "resume_count": session.resume_count,
        "resume_events": session.resume_events,
        "violations": session.get_violations(),
        "settings": session.settings
    }
    
    # Remove session
    session_mgr.end_session(session_id)
    logger.info(f"Ended session {session_id}")
    
    return report


@router.get("/{session_id}")
async def get_session_info(
    session_id: str,
    session_mgr: SessionManager = Depends(get_session_manager)
):
    """Get information about an active session"""
    session = session_mgr.get_session(session_id)
    
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session.to_dict()


@router.get("s")  # This becomes /api/sessions
async def list_sessions(
    session_mgr: SessionManager = Depends(get_session_manager)
):
    """List all active sessions"""
    return {
        "active_sessions": session_mgr.get_session_count(),
        "sessions": session_mgr.list_sessions()
    }


@router.get("/s/inactive")  # This becomes /api/sessions/inactive
async def list_inactive_sessions(
    inactivity_minutes: int = 30,
    session_mgr: SessionManager = Depends(get_session_manager)
):
    """
    List sessions that have been inactive for a specified duration
    
    - **inactivity_minutes**: Minimum inactivity duration in minutes (default: 30)
    """
    inactive = session_mgr.get_inactive_sessions(inactivity_minutes)
    
    return {
        "inactive_sessions": len(inactive),
        "inactivity_threshold_minutes": inactivity_minutes,
        "sessions": inactive
    }


@router.post("/s/cleanup")  # This becomes /api/sessions/cleanup
async def cleanup_sessions(
    max_inactivity_minutes: int = 60,
    session_mgr: SessionManager = Depends(get_session_manager)
):
    """
    Manually trigger cleanup of inactive sessions
    
    - **max_inactivity_minutes**: Remove sessions inactive for more than this duration (default: 60)
    """
    cleaned = session_mgr.cleanup_inactive_sessions(max_inactivity_minutes)
    
    logger.info(f"Manual cleanup: Removed {cleaned} inactive sessions")
    
    return {
        "message": f"Cleaned up {cleaned} inactive sessions",
        "sessions_removed": cleaned,
        "inactivity_threshold_minutes": max_inactivity_minutes,
        "remaining_sessions": session_mgr.get_session_count()
    }
