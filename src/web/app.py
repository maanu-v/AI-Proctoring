from flask import Flask, Response, render_template, request
import cv2
import time
from ..core.video_stream import VideoStream
from ..face.landmarks import FaceMeshDetector
from ..utils.config import config
from ..utils.logger import logger

app = Flask(__name__, static_folder='static', template_folder='static')

# Initialize detector
face_detector = FaceMeshDetector()

def generate_frames(analyzed_mode: bool = False):
    """
    Generator that yields frames from the VideoStream.
    Args:
        analyzed_mode: If True, draws landmarks on the frame.
    """
    video_stream = VideoStream.get_instance()
    # Subscribe to get a queue
    frame_queue = video_stream.subscribe()
    
    try:
        while True:
            # Get frame from queue (blocking with timeout to allow checking for disconnect)
            try:
                frame = frame_queue.get(timeout=2.0)
                
                # Run inference
                results = face_detector.detect(frame)
                
                # Draw visualization (always face count, landmarks if analyzed_mode)
                # Note: detect() returns results, we call draw_landmarks on the frame
                # to modify it in-place or return a new one. 
                # Our draw_landmarks modifies in-place.
                PROCESSED_FRAME = face_detector.draw_landmarks(frame, results, analyzed_mode=analyzed_mode)
                
                # Encode frame to JPEG
                ret, buffer = cv2.imencode('.jpg', PROCESSED_FRAME)
                if not ret:
                    continue
                
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            except Exception as e:
                # Timeout or other error
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
    mode = request.args.get('mode', 'raw')
    analyzed_mode = (mode == 'analyzed')
    return Response(generate_frames(analyzed_mode=analyzed_mode),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def main():
    logger.info(f"Starting generic app on {config.FLASK_HOST}:{config.FLASK_PORT}")
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=True, threaded=True)

if __name__ == '__main__':
    main()
