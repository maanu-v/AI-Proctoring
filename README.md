# 🎥 AI Proctoring System

An advanced AI-powered proctoring system for online exams with real-time monitoring, violation detection, and comprehensive analysis.

## Features

### Core Capabilities

- **Face Detection & Tracking**: Multi-face detection using MediaPipe
- **Head Pose Analysis**: Real-time head orientation tracking (yaw, pitch, roll)
- **Gaze Estimation**: Eye movement and direction tracking
- **Blink Detection**: Eye aspect ratio monitoring and blink counting
- **Object Detection**: Prohibited item detection (phones, etc.) using YOLO
- **Identity Verification**: Face recognition for student verification using DeepFace
- **Violation Tracking**: Comprehensive logging with timestamps and persistence

### Two Deployment Options

1. **Streamlit App**: Interactive dashboard for testing and demonstration
2. **FastAPI Backend**: Production-ready REST API for integration with any frontend

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd ai-proctor

# Install dependencies
pip install -e .
```

### Option 1: Run Streamlit Dashboard

```bash
streamlit run src/web/app.py
```

Open your browser to `http://localhost:8501`

### Option 2: Run FastAPI Backend (Modular Architecture)

**New modular structure for better maintainability!**

```bash
# Method 1: Using start script
./start_api.sh

# Method 2: Direct execution
python src/web/main.py

# Method 3: Using uvicorn
uvicorn src.web.main:app --reload --port 8000
```

API available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## FastAPI Backend

### Quick Example

```python
import requests
import cv2
import base64

# Start session
response = requests.post(
    "http://localhost:8000/api/session/start",
    data={"student_id": "S12345", "quiz_id": "QUIZ_001"}
)
session_id = response.json()["session_id"]

# Analyze frame
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

_, buffer = cv2.imencode('.jpg', frame)
frame_base64 = base64.b64encode(buffer).decode('utf-8')

response = requests.post(
    f"http://localhost:8000/api/analyze/frame/{session_id}",
    json={"frame_base64": frame_base64}
)

result = response.json()
print(f"Faces: {result['face_count']}")
print(f"Violations: {len(result['violations'])}")

# End session
response = requests.post(f"http://localhost:8000/api/session/end/{session_id}")
report = response.json()
```

### Run Example Client

```bash
python examples/client_example.py
```

### Test the API

```bash
python tests/test_fastapi.py
```

## Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/session/start` | Start proctoring session |
| POST | `/api/analyze/frame/{session_id}` | Analyze video frame |
| POST | `/api/session/end/{session_id}` | End session, get report |
| GET | `/api/session/{session_id}` | Get session info |
| PUT | `/api/session/{session_id}/settings` | Update settings |
| GET | `/api/config` | Get global configuration |

## Documentation

- **[Architecture Guide](docs/ARCHITECTURE.md)**: Modular structure and design patterns
- **[Modular Structure Quick Reference](docs/MODULAR_STRUCTURE.md)**: Quick guide to new structure
- **[API Documentation](docs/API_DOCUMENTATION.md)**: Complete API reference
- **[Quick Start Guide](docs/FASTAPI_QUICKSTART.md)**: Step-by-step setup
- **[Head Pose Details](docs/head_pose.md)**: Head pose analysis details
- **[MediaPipe Landmarks](docs/mediapipe_facemesh_landmark.md)**: Face landmark information

## Configuration

All settings are in `src/configs/app.yaml`:

```yaml
thresholds:
  max_num_faces: 1
  no_face_persistence_time: 2.0
  head_pose_persistence_time: 3.0
  gaze_persistence_time: 2.0

head_pose:
  yaw_threshold: 20    # Left/Right
  pitch_threshold: 15  # Up/Down
  roll_threshold: 20   # Tilt

gaze:
  horizontal_threshold_left: 0.42
  horizontal_threshold_right: 0.58
  vertical_threshold_up: 0.40
  vertical_threshold_down: 0.60
```

### Runtime Configuration

Update settings via API:

```python
# Update session settings
requests.put(
    f"http://localhost:8000/api/session/{session_id}/settings",
    json={
        "enable_head_pose": True,
        "enable_gaze": False,
        "enable_identity_verification": True
    }
)

# Update global config
requests.put(
    "http://localhost:8000/api/config",
    json={
        "thresholds": {
            "no_face_persistence_time": 5.0
        }
    }
)
```

## Architecture

```
Client (Web/Mobile)
        │
        ├──> FastAPI Server (Port 8000)
        │         │
        │         ├──> Session Manager
        │         │
        │         └──> Analysis Engine
        │                   │
        │                   ├──> MediaPipe (Face Mesh)
        │                   ├──> Head Pose Estimator
        │                   ├──> Gaze Estimator
        │                   ├──> Blink Detector
        │                   ├──> YOLO (Object Detection)
        │                   └──> DeepFace (Identity)
        │
        └──> Streamlit App (Port 8501)
```

## Violation Types

| Type | Description | Persistence Time |
|------|-------------|------------------|
| `no_face` | No face detected | 2.0s |
| `multiple_faces` | More than 1 face | 3.0s |
| `head_pose_*` | Looking away from screen | 3.0s |
| `gaze_*` | Eyes looking away | 2.0s |
| `object_phone` | Mobile phone detected | 2.0s |
| `object_multiple_people` | Multiple people (body) | 2.0s |
| `identity_mismatch` | Face doesn't match profile | 2.0s |

## Performance Optimization

### Client-Side

```python
# Analyze every 1-2 seconds, not every frame
FRAME_SKIP = 30  # For 30 FPS video

if frame_count % FRAME_SKIP == 0:
    analyze_frame(frame)
```

### Server-Side

```bash
# Use multiple workers for production
gunicorn src.web.fastapi_app:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

## Dependencies

- **FastAPI**: REST API framework
- **OpenCV**: Computer vision
- **MediaPipe**: Face mesh detection
- **DeepFace**: Face recognition
- **YOLO (Ultralytics)**: Object detection
- **Streamlit**: Web dashboard
- **Uvicorn**: ASGI server

## Project Structure

```
ai-proctor/
├── src/
│   ├── configs/
│   │   └── app.yaml              # Configuration
│   ├── core/
│   │   └── video_stream.py       # Video capture
│   ├── engine/
│   │   ├── proctor.py            # Violation tracker
│   │   ├── face/                 # Face analysis modules
│   │   │   ├── mesh_detector.py
│   │   │   ├── head_pose.py
│   │   │   ├── gaze_estimation.py
│   │   │   ├── blink_estimation.py
│   │   │   └── face_embedding.py
│   │   └── obj_detection/        # Object detection
│   │       └── obj_detect.py
│   ├── utils/
│   │   ├── config.py             # Config loader
│   │   └── logger.py             # Logging
│   └── ├── main.py               # FastAPI app (modular) ⭐ NEW
│       ├── fastapi_app.py        # FastAPI app (legacy)
│       │
│       ├── api/                  # API layer ⭐ NEW
│       │   ├── models.py         # Pydantic models
│       │   ├── dependencies.py   # Dependency injection
│       │   └── routes/           # Route handlers
│       │       ├── session.py
│       │       ├── analysis.py
│       │       ├── settings.py
│       │       ├── config.py
│       │       └── violations.py
│       │
│       ├── core/                 # Business logic ⭐ NEW
│       │   ├── session_manager.py
│       │   └── analyzers.py
│       │
│       └── utils/                # Web utilities ⭐ NEW
│           └── image_utils.py
│
├── examples/
│   └── client_example.py         # Example client
├── tests/
│   └── test_fastapi.py           # API tests
├── docs/
│   ├── ARCHITECTURE.md           # Architecture guide ⭐ NEW
│   ├── MODULAR_STRUCTURE.md      # Quick reference ⭐ NEW
│   ├── API_DOCUMENTATION.md      # Full API docs
│   └── FASTAPI_QUICKSTART.md     # Quick start
├── start_api.sh                  # Start script ⭐ NEW
│   ├── API_DOCUMENTATION.md      # Full API docs
│   └── FASTAPI_QUICKSTART.md     # Quick start
└── pyproject.toml                # Dependencies
```

## Development

### Run in Development Mode

```bash
# FastAPI with auto-reload
uvicorn src.web.fastapi_app:app --reload

# Streamlit with auto-reload (default)
streamlit run src/web/app.py
```

### Run Tests

```bash
# Test API endpoints
python tests/test_fastapi.py

# Test specific modules
python tests/test_blink.py
python tests/test_gaze.py
```

## Use Cases

1. **Online Exams**: Monitor students during remote exams
2. **Interview Proctoring**: Automated interview monitoring
3. **Training Assessment**: Track engagement during training
4. **Study Monitoring**: Focus tracking for students
5. **Security Monitoring**: Detect unauthorized access

## Integration Examples

### React/Vue Frontend

```javascript
async function startProctoring(studentId, quizId) {
  const formData = new FormData();
  formData.append('student_id', studentId);
  formData.append('quiz_id', quizId);
  
  const response = await fetch('http://localhost:8000/api/session/start', {
    method: 'POST',
    body: formData
  });
  
  const { session_id } = await response.json();
  
  // Start analyzing frames
  setInterval(async () => {
    const frame = captureFrame(); // Your webcam capture
    const result = await analyzeFrame(session_id, frame);
    
    if (result.violations.length > 0) {
      showWarning(result.violations);
    }
  }, 2000);
}
```

### Python Desktop App

```python
import tkinter as tk
from PIL import Image, ImageTk
import cv2
from proctor_client import ProctorClient

class ProctorApp:
    def __init__(self):
        self.client = ProctorClient()
        self.session_id = self.client.start_session("S123", "QUIZ_01")
        
    def analyze_loop(self):
        ret, frame = self.cap.read()
        result = self.client.analyze_frame(frame)
        
        if result['warnings']:
            self.show_warning(result['warnings'])
```

## Troubleshooting

### Server won't start

```bash
# Check if port is in use
lsof -i :8000

# Use different port
uvicorn src.web.fastapi_app:app --port 8080
```

### High CPU usage

- Reduce frame analysis frequency
- Disable unused features
- Use lower resolution

### Models not loading

```bash
# Pre-download models
python -c "from deepface import DeepFace; DeepFace.build_model('ArcFace')"
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

[Add your license here]

## Support

For issues or questions:
- Check [API Documentation](docs/API_DOCUMENTATION.md)
- Check [Quick Start Guide](docs/FASTAPI_QUICKSTART.md)
- Open an issue on GitHub

## Acknowledgments

- MediaPipe for face mesh detection
- DeepFace for face recognition
- Ultralytics YOLO for object detection
- FastAPI for the excellent framework

---

**Ready to get started?**

```bash
# Install and run
pip install -e .
python src/web/fastapi_app.py

# Or test it out
python examples/client_example.py
```

Open http://localhost:8000/docs for interactive API documentation!
