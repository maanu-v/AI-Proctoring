# AI Proctoring System - FastAPI Backend Documentation

## Overview

The AI Proctoring FastAPI backend provides a comprehensive REST API for real-time exam proctoring with features including:

- **Face Detection & Tracking**: Multiple face detection with MediaPipe
- **Head Pose Analysis**: Detect if student is looking away
- **Gaze Tracking**: Monitor eye movements and direction
- **Blink Detection**: Detect eye blinks and prolonged closures
- **Object Detection**: Identify prohibited items (phones, etc.)
- **Identity Verification**: Face recognition to verify student identity
- **Violation Tracking**: Persistent tracking of all violations with timestamps
- **Configurable Settings**: Per-session and global configuration management

## Quick Start

### Installation

```bash
# Install dependencies
pip install -e .

# Or install individually
pip install fastapi uvicorn python-multipart pydantic opencv-python mediapipe numpy deepface
```

### Running the Server

```bash
# Method 1: Using uvicorn directly
uvicorn src.web.fastapi_app:app --reload --host 0.0.0.0 --port 8000

# Method 2: Running the module
python -m src.web.fastapi_app

# Method 3: From the web directory
cd src/web
python fastapi_app.py
```

The API will be available at `http://localhost:8000`

- **API Docs (Swagger UI)**: http://localhost:8000/docs
- **Alternative API Docs (ReDoc)**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## API Endpoints

### 1. Health Check

**GET** `/`

Check if the API is running and get basic information.

**Response:**
```json
{
  "service": "AI Proctoring System",
  "status": "running",
  "version": "1.0.0",
  "active_sessions": 2
}
```

---

### 2. Session Management

#### Start Session

**POST** `/api/session/start`

Start a new proctoring session for a student.

**Request (multipart/form-data):**
- `student_id` (string, required): Unique student identifier
- `quiz_id` (string, required): Unique quiz/exam identifier
- `profile_image` (file, optional): Student's profile photo for identity verification

**Example (cURL):**
```bash
curl -X POST "http://localhost:8000/api/session/start" \
  -F "student_id=S12345" \
  -F "quiz_id=QUIZ_2024_01" \
  -F "profile_image=@/path/to/photo.jpg"
```

**Response:**
```json
{
  "session_id": "S12345_QUIZ_2024_01_1709737200",
  "student_id": "S12345",
  "quiz_id": "QUIZ_2024_01",
  "created_at": "2024-03-06T10:00:00",
  "message": "Session started successfully"
}
```

#### Get Session Info

**GET** `/api/session/{session_id}`

Get information about an active session.

**Response:**
```json
{
  "student_id": "S12345",
  "quiz_id": "QUIZ_2024_01",
  "created_at": "2024-03-06T10:00:00",
  "frame_count": 450,
  "violation_count": 3,
  "settings": {
    "enable_face_detection": true,
    "enable_head_pose": true,
    "enable_gaze": true,
    "enable_blink": true,
    "enable_object_detection": true,
    "enable_identity_verification": true,
    "enable_no_face_warning": true
  }
}
```

#### List All Sessions

**GET** `/api/sessions`

Get a list of all active sessions.

**Response:**
```json
{
  "active_sessions": 2,
  "sessions": [
    {
      "student_id": "S12345",
      "quiz_id": "QUIZ_2024_01",
      "created_at": "2024-03-06T10:00:00",
      "frame_count": 450,
      "violation_count": 3
    }
  ]
}
```

#### End Session

**POST** `/api/session/end/{session_id}`

End a proctoring session and get final report.

**Response:**
```json
{
  "session_id": "S12345_QUIZ_2024_01_1709737200",
  "student_id": "S12345",
  "quiz_id": "QUIZ_2024_01",
  "started_at": "2024-03-06T10:00:00",
  "ended_at": "2024-03-06T11:00:00",
  "total_frames": 1800,
  "total_violations": 5,
  "violations": [
    {
      "timestamp": 1709740800.123,
      "message": "Multiple faces detected: 2",
      "type": "multiple_faces"
    }
  ],
  "settings": { /* session settings */ }
}
```

---

### 3. Frame Analysis

#### Analyze Frame

**POST** `/api/analyze/frame/{session_id}`

Analyze a single video frame from the student's webcam.

**Request Body (JSON):**
```json
{
  "frame_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

**Example (Python):**
```python
import cv2
import base64
import requests

# Capture frame
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# Encode to base64
_, buffer = cv2.imencode('.jpg', frame)
frame_base64 = base64.b64encode(buffer).decode('utf-8')

# Send to API
response = requests.post(
    f"http://localhost:8000/api/analyze/frame/{session_id}",
    json={"frame_base64": frame_base64}
)

result = response.json()
```

**Response:**
```json
{
  "session_id": "S12345_QUIZ_2024_01_1709737200",
  "frame_count": 451,
  "timestamp": "2024-03-06T10:15:30",
  "face_detected": true,
  "face_count": 1,
  "violations": [
    {
      "type": "head_pose",
      "message": "User looking Right for 5 seconds.",
      "timestamp": "2024-03-06T10:15:30"
    }
  ],
  "analysis": {
    "face_detection": {
      "face_count": 1,
      "faces_detected": true
    },
    "head_pose": {
      "direction": "Right",
      "yaw": 35.2,
      "pitch": -5.1,
      "roll": 2.3
    },
    "gaze": {
      "direction": "Center",
      "left_iris": [0.48, 0.52],
      "right_iris": [0.49, 0.51]
    },
    "blink": {
      "ear": 0.28,
      "blink_detected": false,
      "eyes_closed": false,
      "blink_count": 15
    },
    "object_detection": {
      "phone_detected": false,
      "person_count": 1,
      "objects": []
    },
    "identity_verification": {
      "verified": true,
      "checked_at_frame": 450
    }
  },
  "warnings": [
    "User looking Right for 5 seconds."
  ]
}
```

**Analysis Fields Explained:**

- **face_detection**: Number of faces detected
- **head_pose**: Direction user is looking (Forward, Left, Right, Up, Down) with angles
  - `yaw`: Left/Right rotation (-90 to 90 degrees)
  - `pitch`: Up/Down tilt (-90 to 90 degrees)
  - `roll`: Head tilt (-90 to 90 degrees)
- **gaze**: Eye gaze direction (Center, Left, Right, Up, Down) with iris positions
- **blink**: Eye Aspect Ratio (EAR) and blink detection
- **object_detection**: Prohibited items and person count
- **identity_verification**: Face recognition result (compared to profile image)

---

### 4. Settings Management

#### Get Session Settings

**GET** `/api/session/{session_id}/settings`

Get current settings for a session.

**Response:**
```json
{
  "enable_face_detection": true,
  "enable_head_pose": true,
  "enable_gaze": true,
  "enable_blink": true,
  "enable_object_detection": true,
  "enable_identity_verification": true,
  "enable_no_face_warning": true
}
```

#### Update Session Settings

**PUT** `/api/session/{session_id}/settings`

Update settings for an active session (enable/disable features on-the-fly).

**Request Body:**
```json
{
  "enable_head_pose": false,
  "enable_gaze": false
}
```

**Response:**
```json
{
  "message": "Settings updated successfully",
  "settings": {
    "enable_face_detection": true,
    "enable_head_pose": false,
    "enable_gaze": false,
    "enable_blink": true,
    "enable_object_detection": true,
    "enable_identity_verification": true,
    "enable_no_face_warning": true
  }
}
```

---

### 5. Configuration Management

#### Get Global Configuration

**GET** `/api/config`

Get current global configuration (affects all new sessions).

**Response:**
```json
{
  "camera": {
    "index": 0,
    "width": 640,
    "height": 480,
    "fps": 30
  },
  "mediapipe": {
    "num_faces": 5
  },
  "head_pose": {
    "yaw_threshold": 20,
    "pitch_threshold": 15,
    "roll_threshold": 20,
    "auto_calibration": true,
    "calibration_time": 3
  },
  "gaze": {
    "horizontal_threshold_left": 0.42,
    "horizontal_threshold_right": 0.58,
    "vertical_threshold_up": 0.4,
    "vertical_threshold_down": 0.6,
    "smoothing_factor": 0.5
  },
  "blink": {
    "ear_threshold": 0.21,
    "min_blink_frames": 2,
    "long_closure_threshold": 1.0,
    "smoothing_alpha": 0.4
  },
  "thresholds": {
    "max_num_faces": 1,
    "enable_no_face_warning": true,
    "multi_face_persistence_time": 3.0,
    "no_face_persistence_time": 2.0,
    "head_pose_persistence_time": 3.0,
    "gaze_persistence_time": 2.0,
    "identity_check_interval_frames": 30,
    "identity_persistence_time": 2.0
  }
}
```

#### Update Global Configuration

**PUT** `/api/config`

Update global configuration parameters.

**Request Body:**
```json
{
  "thresholds": {
    "no_face_persistence_time": 5.0,
    "head_pose_persistence_time": 4.0
  },
  "head_pose": {
    "yaw_threshold": 25
  }
}
```

**Response:**
```json
{
  "message": "Configuration updated successfully",
  "note": "Changes apply to new sessions only. Restart analyzers for full effect."
}
```

---

### 6. Violations Management

#### Get Violations

**GET** `/api/session/{session_id}/violations`

Get all violations for a session.

**Response:**
```json
{
  "session_id": "S12345_QUIZ_2024_01_1709737200",
  "total_violations": 5,
  "violations": [
    {
      "timestamp": 1709740800.123,
      "message": "Multiple faces detected: 2",
      "type": "multiple_faces"
    },
    {
      "timestamp": 1709741000.456,
      "message": "User looking Right for 5 seconds.",
      "type": "head_pose_Right"
    },
    {
      "timestamp": 1709741200.789,
      "message": "Mobile Phone Detected!",
      "type": "object_phone"
    }
  ]
}
```

**Violation Types:**
- `no_face`: No face detected for extended period
- `multiple_faces`: More than one face detected
- `head_pose_{direction}`: Looking away (Left, Right, Up, Down)
- `gaze_{direction}`: Eyes looking away from screen
- `object_phone`: Mobile phone detected
- `object_multiple_people`: Multiple people detected (by body)
- `identity_mismatch`: Face does not match profile image

#### Clear Violations

**DELETE** `/api/session/{session_id}/violations`

Clear all violations for a session (for testing/reset purposes).

**Response:**
```json
{
  "message": "Violations cleared successfully"
}
```

---

## Usage Flow

### Typical Workflow

1. **Student Starts Quiz**
   - Frontend captures student's profile photo
   - Sends `POST /api/session/start` with student_id, quiz_id, and profile_image
   - Receives session_id

2. **During Quiz (Real-time Monitoring)**
   - Frontend captures webcam frames (e.g., every 1-2 seconds)
   - For each frame, sends `POST /api/analyze/frame/{session_id}`
   - Receives analysis results with violations and warnings
   - Display warnings to student in real-time

3. **Settings Adjustment (Optional)**
   - Admin/proctor can adjust settings mid-quiz
   - Send `PUT /api/session/{session_id}/settings`

4. **Quiz End**
   - Send `POST /api/session/end/{session_id}`
   - Receive comprehensive final report with all violations
   - Store report for review

### Frame Analysis Best Practices

**Frame Rate Considerations:**
- **High Frequency (Every frame)**: Most accurate but resource-intensive
- **Medium Frequency (Every 1-2 seconds)**: Balanced approach (recommended)
- **Low Frequency (Every 5 seconds)**: Lower resource usage, may miss quick violations

**Recommended Settings:**
```python
# For 30 FPS webcam
FRAME_SKIP = 30  # Analyze every 30 frames (1 second at 30 FPS)
FRAME_SKIP = 60  # Analyze every 60 frames (2 seconds at 30 FPS)
```

---

## Configuration Guide

### Persistence Times

All violation types have a "persistence time" - the duration a condition must be present before triggering a violation.

**Default Values:**
- `no_face_persistence_time`: 2.0 seconds
- `multi_face_persistence_time`: 3.0 seconds
- `head_pose_persistence_time`: 3.0 seconds
- `gaze_persistence_time`: 2.0 seconds
- `identity_persistence_time`: 2.0 seconds

**Adjustment Guidelines:**
- **Stricter**: Lower values (1-2s) - More sensitive, more violations
- **Lenient**: Higher values (5-10s) - Less sensitive, fewer false positives

### Head Pose Thresholds

Control when head rotation is considered "looking away":

```yaml
head_pose:
  yaw_threshold: 20      # Left/Right (degrees)
  pitch_threshold: 15    # Up/Down (degrees)
  roll_threshold: 20     # Tilt (degrees)
```

**Guidelines:**
- **Strict**: 15-20 degrees - Detects slight head movements
- **Moderate**: 25-30 degrees - Allows natural movement
- **Lenient**: 35-45 degrees - Only detects obvious looking away

### Gaze Thresholds

Control when eye movement is considered "looking away":

```yaml
gaze:
  horizontal_threshold_left: 0.42   # Looking left
  horizontal_threshold_right: 0.58  # Looking right
  vertical_threshold_up: 0.40       # Looking up
  vertical_threshold_down: 0.60     # Looking down
```

### Identity Verification

```yaml
thresholds:
  identity_check_interval_frames: 30  # Check every 30 frames (1s at 30 FPS)
  identity_persistence_time: 2.0      # Confirm mismatch for 2 seconds
```

**Performance vs Accuracy:**
- **Frequent checks** (every 10-30 frames): More accurate, higher CPU usage
- **Infrequent checks** (every 60-120 frames): Lower CPU, may miss quick swaps

---

## Error Handling

### Common Error Codes

**400 Bad Request**
- Invalid image data
- Session already exists
- Malformed request

**404 Not Found**
- Session not found
- Invalid session_id

**500 Internal Server Error**
- Analysis engine error
- Model loading failure
- Unexpected exception

### Error Response Format

```json
{
  "detail": "Session not found"
}
```

---

## Performance Optimization

### Server-Side

1. **Analyzers are Singleton**: Initialized once and reused across requests
2. **Thread Pool**: Async operations use ThreadPoolExecutor
3. **Lazy Loading**: Models loaded on first use

### Client-Side

1. **Frame Rate**: Don't send every frame, use 1-2 second intervals
2. **Resolution**: Scale down frames before encoding (e.g., 640x480)
3. **Compression**: Use JPEG encoding with moderate quality
4. **Batch Requests**: If possible, batch multiple frames (future enhancement)

### Recommended Client Code

```python
import cv2
import base64
import time

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0
ANALYZE_INTERVAL = 30  # Analyze every 30 frames

while True:
    ret, frame = cap.read()
    frame_count += 1
    
    if frame_count % ANALYZE_INTERVAL == 0:
        # Encode
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Send to API
        response = requests.post(
            f"http://localhost:8000/api/analyze/frame/{session_id}",
            json={"frame_base64": frame_base64}
        )
        
        result = response.json()
        # Handle result...
```

---

## Security Considerations

### Production Deployment

1. **CORS Configuration**: Update `allow_origins` to specific domains
   ```python
   allow_origins=["https://your-frontend-domain.com"]
   ```

2. **Authentication**: Add JWT or API key authentication
3. **Rate Limiting**: Implement rate limiting per session/IP
4. **HTTPS**: Use HTTPS in production
5. **Session Cleanup**: Implement automatic session cleanup for abandoned sessions
6. **Data Privacy**: Consider encrypting stored violation data

### Example: Adding API Key Authentication

```python
from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader

API_KEY = "your-secret-key"
api_key_header = APIKeyHeader(name="X-API-Key")

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

# Add to routes
@app.post("/api/session/start", dependencies=[Security(get_api_key)])
async def start_session(...):
    ...
```

---

## Testing

### Manual Testing

Use the included client example:

```bash
python examples/client_example.py
```

### Using cURL

```bash
# Health check
curl http://localhost:8000/

# Start session
curl -X POST http://localhost:8000/api/session/start \
  -F "student_id=TEST_STUDENT" \
  -F "quiz_id=TEST_QUIZ"

# Get config
curl http://localhost:8000/api/config
```

### Using Postman/Insomnia

Import the OpenAPI schema from `http://localhost:8000/openapi.json`

---

## Troubleshooting

### Issue: "ModuleNotFoundError"

**Solution**: Ensure all dependencies are installed
```bash
pip install -e .
```

### Issue: "Could not load model"

**Solution**: First run might download models. Wait for completion.
```bash
# Pre-download DeepFace models
python -c "from deepface import DeepFace; DeepFace.build_model('ArcFace')"
```

### Issue: "Session not found"

**Solution**: Sessions are in-memory. Restarting server clears all sessions.

### Issue: High CPU usage

**Solutions**:
- Reduce frame analysis frequency
- Lower camera resolution
- Disable heavy features (identity verification) if not needed

---

## Future Enhancements

- [ ] Database persistence for sessions and violations
- [ ] WebSocket support for real-time bidirectional communication
- [ ] Batch frame analysis
- [ ] Redis caching for sessions
- [ ] Admin dashboard
- [ ] Automated report generation (PDF)
- [ ] Multi-language support for violation messages
- [ ] Custom violation rules engine

---

## Support

For issues, questions, or contributions, please refer to the main project README.md.

---

## License

[Add your license information here]
