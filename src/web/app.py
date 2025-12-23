from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from src.core.frame_processor import FrameProcessor
import os

app = FastAPI()

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")

# Initialize FrameProcessor (lazy initialization or global)
# For simplicity in this demo, we'll create it when needed or keep a global instance.
# A global instance is better to avoid opening/closing camera on every request if we want a continuous stream,
# but for a simple MJPEG stream, usually the generator handles the loop.
# However, multiple clients might need a more robust broadcasting mechanism.
# For this basic task, we'll instantiate it in the generator loop or use a simple global.
# Let's use a global variable but handle re-initialization if needed.

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
            # If we can't get a frame, maybe wait a bit or break
            # For now, just yield empty or break
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
