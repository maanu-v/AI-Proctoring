from flask import Flask, Response, render_template, send_from_directory
import cv2
import time
from ..core.video_stream import VideoStream
from ..utils.config import config
from ..utils.logger import logger

app = Flask(__name__, static_folder='static', template_folder='static')

def generate_frames():
    """Generator that yields frames from the VideoStream."""
    video_stream = VideoStream.get_instance()
    # Subscribe to get a queue
    frame_queue = video_stream.subscribe()
    
    try:
        while True:
            # Get frame from queue (blocking with timeout to allow checking for disconnect)
            try:
                frame = frame_queue.get(timeout=2.0)
                
                # Encode frame to JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            except Exception as e:
                # Timeout or other error, check if we should stop? 
                # Actually for generator, if client disconnects, next yield might raise error
                # or just loop. Flask handles disconnect mostly by stopping generator iteration?
                # We'll rely on `finally` block.
                pass
                
    except GeneratorExit:
        logger.info("Client disconnected from video stream.")
    except Exception as e:
        logger.error(f"Error in video feed generator: {e}")
    finally:
        video_stream.unsubscribe(frame_queue)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def main():
    logger.info(f"Starting generic app on {config.FLASK_HOST}:{config.FLASK_PORT}")
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=True, threaded=True)

if __name__ == '__main__':
    main()
