from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from src.core.frame_processor import FrameProcessor
import os

from contextlib import asynccontextmanager
from src.core.video_stream import VideoStream
from src.utils.logger import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("FastAPI application starting up...")
    logger.info("Video stream ready for initialization")
    yield
    # Shutdown
    logger.info("FastAPI application shutting down...")
    VideoStream().stop()
    logger.info("Application shutdown complete")

app = FastAPI(lifespan=lifespan)

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")

frame_processor = None
face_mesh_enabled = True  # Global state for face mesh toggle

def get_frame_processor():
    global frame_processor
    if frame_processor is None:
        try:
            frame_processor = FrameProcessor(enable_face_mesh=True)
        except RuntimeError as e:
            logger.error(f"Error initializing camera: {e}")
            return None
    return frame_processor

def generate_frames():
    global face_mesh_enabled
    processor = get_frame_processor()
    if not processor:
        yield b''
        return

    while True:
        frame_bytes = processor.get_frame(
            flip_horizontal=True,
            show_face_mesh=face_mesh_enabled
        )
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            break

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/toggle_face_mesh")
async def toggle_face_mesh(enabled: bool):
    """Toggle face mesh visualization on/off."""
    global face_mesh_enabled
    face_mesh_enabled = enabled
    logger.info(f"Face mesh visualization {'enabled' if enabled else 'disabled'}")
    return JSONResponse({"face_mesh_enabled": face_mesh_enabled})

@app.get("/face_mesh_status")
async def face_mesh_status():
    """Get current face mesh status."""
    return JSONResponse({"face_mesh_enabled": face_mesh_enabled})

@app.get("/")
async def index():
    # Redirect to the static index.html or serve it directly
    # Since we mounted static with html=True, /static/ should serve index.html
    # But user might hit root /. Let's redirect or just read the file.
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(static_dir, "index.html"))
