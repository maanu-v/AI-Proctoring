"""
FastAPI Backend for AI Proctoring System
Main application entry point with modular architecture
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.web.api.routes import session, analysis, settings, config, violations, websocket
from src.web.api.models import HealthResponse
from src.web.core.session_manager import session_manager
from src.utils.logger import get_logger

logger = get_logger(__name__)

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
    logger.info("="*60)
    logger.info("AI Proctoring System - FastAPI Backend")
    logger.info("="*60)
    logger.info("Starting server...")
    logger.info("API Documentation: http://localhost:8000/docs")
    logger.info("Alternative Docs: http://localhost:8000/redoc")
    logger.info("WebSocket Endpoint: ws://localhost:8000/ws/{session_id}")
    logger.info("="*60)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down server...")
    logger.info(f"Active sessions at shutdown: {session_manager.get_session_count()}")

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
