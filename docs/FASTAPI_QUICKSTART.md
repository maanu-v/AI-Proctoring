# FastAPI Backend - Quick Start Guide

## Installation & Setup

### 1. Install Dependencies

```bash
# Install all dependencies including FastAPI
pip install -e .

# Or install FastAPI separately if needed
pip install fastapi uvicorn[standard] python-multipart pydantic
```

### 2. Start the Server

```bash
# Method 1: Direct execution
python src/web/fastapi_app.py

# Method 2: Using uvicorn
uvicorn src.web.fastapi_app:app --reload --host 0.0.0.0 --port 8000

# Method 3: With custom port
uvicorn src.web.fastapi_app:app --reload --port 8080
```

The API will be running at:
- **API**: http://localhost:8000
- **Interactive Docs (Swagger)**: http://localhost:8000/docs
- **Alternative Docs (ReDoc)**: http://localhost:8000/redoc

### 3. Test the API

#### Option A: Use the Example Client

```bash
python examples/client_example.py
```

#### Option B: Use cURL

```bash
# Health check
curl http://localhost:8000/

# Start a session
curl -X POST http://localhost:8000/api/session/start \
  -F "student_id=S12345" \
  -F "quiz_id=QUIZ_001"
```

#### Option C: Use the Interactive Docs

Open http://localhost:8000/docs in your browser and try the endpoints interactively.

## Basic Usage

### Python Client Example

```python
import requests
import cv2
import base64

# 1. Start session
response = requests.post(
    "http://localhost:8000/api/session/start",
    data={"student_id": "S12345", "quiz_id": "QUIZ_001"}
)
session_id = response.json()["session_id"]
print(f"Session started: {session_id}")

# 2. Capture and analyze frame
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

_, buffer = cv2.imencode('.jpg', frame)
frame_base64 = base64.b64encode(buffer).decode('utf-8')

response = requests.post(
    f"http://localhost:8000/api/analyze/frame/{session_id}",
    json={"frame_base64": frame_base64}
)

result = response.json()
print(f"Faces detected: {result['face_count']}")
print(f"Violations: {len(result['violations'])}")

# 3. End session
response = requests.post(f"http://localhost:8000/api/session/end/{session_id}")
report = response.json()
print(f"Total violations: {report['total_violations']}")

cap.release()
```

### JavaScript/Frontend Example

```javascript
// Start session
async function startSession(studentId, quizId) {
  const formData = new FormData();
  formData.append('student_id', studentId);
  formData.append('quiz_id', quizId);
  
  const response = await fetch('http://localhost:8000/api/session/start', {
    method: 'POST',
    body: formData
  });
  
  const data = await response.json();
  return data.session_id;
}

// Analyze frame from webcam
async function analyzeFrame(sessionId, videoElement) {
  const canvas = document.createElement('canvas');
  canvas.width = videoElement.videoWidth;
  canvas.height = videoElement.videoHeight;
  
  const ctx = canvas.getContext('2d');
  ctx.drawImage(videoElement, 0, 0);
  
  const frameBase64 = canvas.toDataURL('image/jpeg').split(',')[1];
  
  const response = await fetch(
    `http://localhost:8000/api/analyze/frame/${sessionId}`,
    {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({frame_base64: frameBase64})
    }
  );
  
  return await response.json();
}

// Usage
const sessionId = await startSession('S12345', 'QUIZ_001');

// Analyze every 2 seconds
setInterval(async () => {
  const result = await analyzeFrame(sessionId, videoElement);
  
  if (result.warnings.length > 0) {
    console.warn('Warnings:', result.warnings);
  }
  
  if (result.violations.length > 0) {
    console.error('Violations:', result.violations);
  }
}, 2000);
```

## Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/api/session/start` | Start proctoring session |
| POST | `/api/session/end/{session_id}` | End session and get report |
| POST | `/api/analyze/frame/{session_id}` | Analyze a video frame |
| GET | `/api/session/{session_id}` | Get session info |
| GET | `/api/sessions` | List all active sessions |
| GET | `/api/session/{session_id}/settings` | Get session settings |
| PUT | `/api/session/{session_id}/settings` | Update session settings |
| GET | `/api/config` | Get global configuration |
| PUT | `/api/config` | Update global configuration |
| GET | `/api/session/{session_id}/violations` | Get all violations |

## Settings Configuration

### Per-Session Settings

Enable/disable features for a specific session:

```python
import requests

# Update settings
requests.put(
    f"http://localhost:8000/api/session/{session_id}/settings",
    json={
        "enable_face_detection": True,
        "enable_head_pose": True,
        "enable_gaze": False,  # Disable gaze tracking
        "enable_blink": True,
        "enable_object_detection": True,
        "enable_identity_verification": True,
        "enable_no_face_warning": True
    }
)
```

### Global Configuration

Update default thresholds:

```python
import requests

requests.put(
    "http://localhost:8000/api/config",
    json={
        "thresholds": {
            "no_face_persistence_time": 5.0,  # Wait 5 seconds before violation
            "head_pose_persistence_time": 4.0,
            "gaze_persistence_time": 3.0
        },
        "head_pose": {
            "yaw_threshold": 25,  # More lenient
            "pitch_threshold": 20
        }
    }
)
```

## Features

### ✅ Face Detection
- Detects number of faces in frame
- Tracks multiple faces simultaneously
- Violations: No face, multiple faces

### ✅ Head Pose Analysis
- Estimates head orientation (yaw, pitch, roll)
- Detects looking away from screen
- Directions: Forward, Left, Right, Up, Down

### ✅ Gaze Tracking
- Tracks eye/iris movements
- Detects looking away from screen center
- Directions: Center, Left, Right, Up, Down

### ✅ Blink Detection
- Monitors eye aspect ratio (EAR)
- Counts blinks
- Detects prolonged eye closure

### ✅ Object Detection
- Detects prohibited objects (phones, etc.)
- Counts people in frame

### ✅ Identity Verification
- Face recognition against profile image
- Periodic verification during session

### ✅ Violation Tracking
- Persistent tracking with timestamps
- Consolidates repeated violations
- Configurable persistence times

## Configuration Files

### Main Config: `src/configs/app.yaml`

All thresholds and settings are defined in this YAML file:

```yaml
camera:
  params:
    index: 0
    width: 640
    height: 480
    fps: 30

thresholds:
  max_num_faces: 1
  enable_no_face_warning: true
  multi_face_persistence_time: 3.0
  no_face_persistence_time: 2.0
  head_pose_persistence_time: 3.0
  gaze_persistence_time: 2.0
  identity_check_interval_frames: 30
  identity_persistence_time: 2.0

head_pose:
  yaw_threshold: 20
  pitch_threshold: 15
  roll_threshold: 20

gaze:
  horizontal_threshold_left: 0.42
  horizontal_threshold_right: 0.58
  vertical_threshold_up: 0.40
  vertical_threshold_down: 0.60
  smoothing_factor: 0.5

blink:
  ear_threshold: 0.21
  min_blink_frames: 2
  long_closure_threshold: 1.0
  smoothing_alpha: 0.4
```

## Architecture

```
┌─────────────────┐
│   Client App    │ (Web/Mobile Frontend)
│  (React/Vue/JS) │
└────────┬────────┘
         │ HTTP/REST
         ▼
┌─────────────────┐
│  FastAPI Server │
│   (Port 8000)   │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌─────────┐ ┌─────────────┐
│ Session │ │  Analyzers  │
│ Manager │ │  (Cached)   │
└─────────┘ └──────┬──────┘
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
    ┌───────┐ ┌────────┐ ┌──────────┐
    │ Face  │ │ Gaze   │ │ Object   │
    │ Mesh  │ │ Track  │ │ Detect   │
    └───────┘ └────────┘ └──────────┘
```

## Performance Tips

1. **Frame Rate**: Don't analyze every frame. Use 1-2 second intervals:
   ```python
   if frame_count % 30 == 0:  # Every 1 second at 30 FPS
       analyze_frame(frame)
   ```

2. **Resolution**: Use 640x480 or smaller for analysis

3. **Features**: Disable unused features to save CPU:
   ```python
   # Disable heavy features if not needed
   settings = {
       "enable_identity_verification": False,  # Most CPU intensive
       "enable_object_detection": False
   }
   ```

4. **Identity Check Interval**: Increase for better performance:
   ```yaml
   identity_check_interval_frames: 60  # Check every 2 seconds
   ```

## Troubleshooting

### Server won't start

```bash
# Check if port is in use
lsof -i :8000

# Use different port
uvicorn src.web.fastapi_app:app --port 8080
```

### Module not found

```bash
# Reinstall dependencies
pip install -e .

# Check Python path
python -c "import sys; print(sys.path)"
```

### High CPU usage

- Reduce frame analysis frequency
- Lower camera resolution
- Disable identity verification
- Use GPU if available (configure TensorFlow/PyTorch)

### Models not loading

First run downloads models (DeepFace, YOLO). Wait for completion:

```bash
# Pre-download models
python -c "from deepface import DeepFace; DeepFace.build_model('ArcFace')"
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

## Documentation

- **Full API Documentation**: [docs/API_DOCUMENTATION.md](../../docs/API_DOCUMENTATION.md)
- **Interactive Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Development

### Running in Development Mode

```bash
# Auto-reload on code changes
uvicorn src.web.fastapi_app:app --reload
```

### Adding New Features

1. Modify analyzer classes in `src/engine/`
2. Update analysis logic in `fastapi_app.py`
3. Update settings models if needed
4. Test with example client

### Testing

```bash
# Run example client
python examples/client_example.py

# Manual testing
curl http://localhost:8000/
```

## Production Deployment

### Using Gunicorn (Recommended)

```bash
pip install gunicorn

gunicorn src.web.fastapi_app:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

### Using Docker

```dockerfile
FROM python:3.13-slim

WORKDIR /app
COPY . /app

RUN pip install -e .

EXPOSE 8000

CMD ["uvicorn", "src.web.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables

```bash
# Set in production
export API_ENV=production
export API_HOST=0.0.0.0
export API_PORT=8000
export LOG_LEVEL=info
```

## Next Steps

1. ✅ Start the server
2. ✅ Test with example client
3. ✅ Integrate with your frontend
4. Configure settings for your use case
5. Deploy to production

For detailed API documentation, see [docs/API_DOCUMENTATION.md](../../docs/API_DOCUMENTATION.md)
