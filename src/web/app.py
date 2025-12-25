from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path
from typing import List
import asyncio
import json

from ..utils.logger import logger
from ..utils.config import config
from ..core.video_stream import video_stream

# Initialize FastAPI app
app = FastAPI(
    title=config.APP_NAME,
    version=config.APP_VERSION,
    debug=config.DEBUG
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Store active WebSocket connections
active_connections: List[WebSocket] = []


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info(f"Starting {config.APP_NAME} v{config.APP_VERSION}")
    
    # Start video stream
    if video_stream.start():
        logger.info("Video stream started successfully")
    else:
        logger.error("Failed to start video stream")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down application...")
    video_stream.stop()
    logger.info("Application shutdown complete")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page."""
    html_file = static_dir / "index.html"
    
    if not html_file.exists():
        return HTMLResponse(
            content="<h1>Error: index.html not found</h1>",
            status_code=500
        )
    
    with open(html_file, "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app_name": config.APP_NAME,
        "version": config.APP_VERSION,
        "video_stream_active": video_stream.is_active()
    }


@app.get("/api/camera/properties")
async def get_camera_properties():
    """Get camera properties."""
    return video_stream.get_properties()


def generate_video_stream():
    """Generator function for video streaming."""
    while True:
        frame_bytes = video_stream.get_frame_jpeg(quality=85)
        
        if frame_bytes is None:
            # If no frame available, wait a bit and continue
            import time
            time.sleep(0.033)  # ~30 FPS
            continue
        
        # Yield frame in multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.get("/api/video/feed")
async def video_feed():
    """
    Video streaming endpoint.
    Returns a multipart stream of JPEG frames.
    """
    return StreamingResponse(
        generate_video_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.websocket("/ws/video")
async def websocket_video_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time video streaming.
    Provides better performance than HTTP streaming.
    """
    await websocket.accept()
    active_connections.append(websocket)
    logger.info(f"WebSocket client connected. Total connections: {len(active_connections)}")
    
    try:
        while True:
            frame_bytes = video_stream.get_frame_jpeg(quality=85)
            
            if frame_bytes is not None:
                # Send frame as binary data
                await websocket.send_bytes(frame_bytes)
            
            # Control frame rate
            await asyncio.sleep(0.033)  # ~30 FPS
            
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total connections: {len(active_connections)}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)


@app.websocket("/ws/control")
async def websocket_control_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for control messages and status updates.
    """
    await websocket.accept()
    logger.info("Control WebSocket client connected")
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            logger.info(f"Received control message: {message}")
            
            # Handle different control messages
            action = message.get("action")
            
            if action == "get_status":
                status = {
                    "type": "status",
                    "video_active": video_stream.is_active(),
                    "properties": video_stream.get_properties()
                }
                await websocket.send_json(status)
            
            elif action == "toggle_feature":
                feature = message.get("feature")
                enabled = message.get("enabled")
                logger.info(f"Toggle {feature}: {enabled}")
                
                # Send acknowledgment
                await websocket.send_json({
                    "type": "ack",
                    "action": "toggle_feature",
                    "feature": feature,
                    "enabled": enabled
                })
            
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown action: {action}"
                })
                
    except WebSocketDisconnect:
        logger.info("Control WebSocket client disconnected")
    except Exception as e:
        logger.error(f"Control WebSocket error: {e}")


@app.get("/api/features")
async def get_features():
    """Get available AI proctoring features."""
    return {
        "features": [
            {
                "id": "face_detection",
                "name": "Face Detection",
                "description": "Detect faces in the video stream",
                "enabled": False
            },
            {
                "id": "gaze_tracking",
                "name": "Gaze Tracking",
                "description": "Track eye gaze direction",
                "enabled": False
            },
            {
                "id": "head_pose",
                "name": "Head Pose Estimation",
                "description": "Estimate head orientation",
                "enabled": False
            },
            {
                "id": "mouth_detection",
                "name": "Mouth Activity Detection",
                "description": "Detect mouth movements and speech",
                "enabled": False
            },
            {
                "id": "person_count",
                "name": "Person Counter",
                "description": "Count number of people in frame",
                "enabled": False,
                "max_allowed": 1
            },
            {
                "id": "object_detection",
                "name": "Object Detection",
                "description": "Detect prohibited objects",
                "enabled": False
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)
