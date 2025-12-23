from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
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

def get_frame_processor():
    global frame_processor
    if frame_processor is None:
        try:
            frame_processor = FrameProcessor()
        except RuntimeError as e:
            print(f"Error initializing camera: {e}")
            return None
    return frame_processor

def generate_frames():
    processor = get_frame_processor()
    if not processor:
        yield b''
        return

    while True:
        frame_bytes = processor.get_frame()
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            break

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/")
async def index():
    # Redirect to the static index.html or serve it directly
    # Since we mounted static with html=True, /static/ should serve index.html
    # But user might hit root /. Let's redirect or just read the file.
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(static_dir, "index.html"))
