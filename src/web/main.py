"""
FastAPI Backend for AI Proctoring System
Main application entry point with modular architecture
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
import asyncio

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.web.api.routes import session, analysis, settings, config, violations, websocket
from src.web.api.models import HealthResponse
from src.web.core.session_manager import session_manager
from src.utils.logger import get_logger
from src.utils.config import config as app_config

logger = get_logger(__name__)

# ============================================================================
# Background Tasks
# ============================================================================

cleanup_task = None  # Global reference to cleanup task

async def cleanup_sessions_periodically():
    """
    Background task that periodically cleans up inactive sessions
    Runs at configured interval and removes sessions inactive for configured duration
    """
    while True:
        try:
            # Use config values
            await asyncio.sleep(app_config.session.cleanup_interval_seconds)
            
            # Cleanup sessions inactive for more than configured minutes
            cleaned = session_manager.cleanup_inactive_sessions(
                max_inactivity_minutes=app_config.session.max_inactivity_minutes
            )
            
            if cleaned > 0:
                logger.info(f"Automatic cleanup: Removed {cleaned} inactive sessions")
            
            # Log inactive sessions (warning for configured threshold)
            inactive = session_manager.get_inactive_sessions(
                inactivity_minutes=app_config.session.inactivity_warning_minutes
            )
            if inactive:
                logger.warning(f"Warning: {len(inactive)} sessions inactive for {app_config.session.inactivity_warning_minutes}+ minutes")
                
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")

# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="AI Proctoring System",
    description="Real-time exam proctoring with face detection, gaze tracking, and object detection",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============================================================================
# CORS Middleware
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Include Routers
# ============================================================================

# Session management routes
app.include_router(session.router)

# Frame analysis routes
app.include_router(analysis.router)

# Settings management routes
app.include_router(settings.router)

# Configuration routes
app.include_router(config.router)

# Violations routes
app.include_router(violations.router)

# WebSocket routes
app.include_router(websocket.router)

# ============================================================================
# Root Endpoint
# ============================================================================

@app.get("/", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint
    
    Returns service status and active session count
    """
    return HealthResponse(
        service="AI Proctoring System",
        status="running",
        version="1.0.0",
        active_sessions=session_manager.get_session_count()
    )

# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    global cleanup_task
    
    logger.info("="*60)
    logger.info("AI Proctoring System - FastAPI Backend")
    logger.info("="*60)
    logger.info("Starting server...")
    logger.info("API Documentation: http://localhost:8000/docs")
    logger.info("Alternative Docs: http://localhost:8000/redoc")
    logger.info("WebSocket Endpoint: ws://localhost:8000/ws/{session_id}")
    logger.info("="*60)
    
    # Start background cleanup task
    cleanup_task = asyncio.create_task(cleanup_sessions_periodically())
    logger.info(f"Started automatic session cleanup task (runs every {app_config.session.cleanup_interval_seconds//60} minutes)")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global cleanup_task
    
    logger.info("Shutting down server...")
    logger.info(f"Active sessions at shutdown: {session_manager.get_session_count()}")
    
    # Cancel cleanup task
    if cleanup_task:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            logger.info("Cleanup task cancelled")
    
    logger.info("Server shutdown complete")

# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
