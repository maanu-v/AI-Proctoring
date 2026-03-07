# AI Proctor API Reference

**Base URL:** `http://localhost:8000`

**API Documentation:** `http://localhost:8000/docs` (Interactive Swagger UI)

---

## Table of Contents

1. [Session Management](#1-session-management)
2. [Frame Analysis](#2-frame-analysis)
3. [WebSocket Frame Streaming](#3-websocket-frame-streaming)
4. [Settings Management](#4-settings-management)
5. [Violations Management](#5-violations-management)
6. [Configuration Management](#6-configuration-management)
7. [Health Check](#7-health-check)
8. [Integration Examples](#8-integration-examples)
9. [Violation Types](#9-violation-types)

---

## 1. Session Management

### Create Session
**POST** `/api/session/start`

Start a new proctoring session.

**Request:** `multipart/form-data`
```
student_id: string (required)
quiz_id: string (required)
profile_image: file (optional) - Image file for identity verification
```

**Response:**
```json
{
  "session_id": "STU123_QUIZ001_1234567890",
  "student_id": "STU123",
  "quiz_id": "QUIZ001", 
  "created_at": "2026-03-06T12:30:45.123456",
  "message": "Session started successfully"
}
```

**If Session Already Exists (Student returns after leaving):**
```json
{
  "session_id": "STU123_QUIZ001_1234567890",
  "student_id": "STU123",
  "quiz_id": "QUIZ001", 
  "created_at": "2026-03-06T12:30:45.123456",
  "message": "Session resumed successfully (Resume #1)"
}
```

**Behavior:**
- If session doesn't exist → Creates new session
- If session exists → **Resumes existing session** (no error)
- Resume tracking: Logs inactivity duration and increments resume counter
- If inactivity > 10 seconds → Logged as `session_resume` violation
- Helps detect students leaving exam to consult notes/others

**Status Codes:**
- `200 OK` - Session created or resumed successfully
- `400 Bad Request` - Invalid input
- `500 Internal Server Error` - Server error

---

### Get Session Info
**GET** `/api/session/{session_id}`

Get information about an active session.

**Parameters:**
- `session_id` (path) - Session identifier

**Response:**
```json
{
  "student_id": "STU123",
  "quiz_id": "QUIZ001",
  "created_at": "2026-03-06T12:30:45.123456",
  "last_activity": "2026-03-06T12:45:30.789012",
  "frame_count": 1250,
  "violation_count": 3,
  "resume_count": 1,
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

**Status Codes:**
- `200 OK` - Session info retrieved
- `404 Not Found` - Session not found

---

### List All Sessions
**GET** `/api/sessions`

Get all active sessions.

**Response:**
```json
{
  "active_sessions": 2,
  "sessions": [
    {
      "session_id": "STU123_QUIZ001_1234567890",
      "student_id": "STU123",
      "quiz_id": "QUIZ001"
    },
    {
      "session_id": "STU456_QUIZ002_1234567891",
      "student_id": "STU456",
      "quiz_id": "QUIZ002"
    }
  ]
}
```

**Status Codes:**
- `200 OK` - Sessions list retrieved

---

### End Session
**POST** `/api/session/end/{session_id}`

End a proctoring session and get final report.

**Parameters:**
- `session_id` (path) - Session identifier

**Response:**
```json
{
  "session_id": "STU123_QUIZ001_1234567890",
  "student_id": "STU123",
  "quiz_id": "QUIZ001",
  "started_at": "2026-03-06T12:30:45.123456",
  "ended_at": "2026-03-06T13:45:30.654321",
  "total_frames": 4500,
  "total_violations": 12,
  "violations": [
    {
      "type": "head_pose",
      "message": "Looking away detected",
      "timestamp": "2026-03-06T12:35:12.123456"
    },
    {
      "type": "multiple_faces",
      "message": "Multiple faces detected: 2",
      "timestamp": "2026-03-06T12:38:45.654321"
    }
  ],
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

**Status Codes:**
- `200 OK` - Session ended successfully
- `404 Not Found` - Session not found

---

## 2. Frame Analysis

### Analyze Frame
**POST** `/api/analyze/frame/{session_id}`

Analyze a single webcam frame. This is the main endpoint for real-time proctoring.

**Parameters:**
- `session_id` (path) - Session identifier

**Request:**
```json
{
  "frame_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

**Response:**
```json
{
  "session_id": "STU123_QUIZ001_1234567890",
  "frame_count": 1251,
  "timestamp": "2026-03-06T12:31:15.234567",
  "face_detected": true,
  "face_count": 1,
  "violations": [
    {
      "type": "head_pose",
      "message": "Looking away detected",
      "timestamp": "2026-03-06T12:31:15.234567"
    }
  ],
  "analysis": {
    "face_detection": {
      "face_count": 1,
      "faces_detected": true
    },
    "head_pose": {
      "direction": "left",
      "yaw": -32.5,
      "pitch": 5.2,
      "roll": -2.1,
      "is_looking_forward": false
    },
    "gaze": {
      "direction": "Left",
      "horizontal_ratio": 0.35,
      "vertical_ratio": 0.48,
      "is_looking_at_screen": false
    },
    "blink": {
      "ear": 0.28,
      "ear_left": 0.27,
      "ear_right": 0.29,
      "eyes_closed": false,
      "total_blinks": 45
    },
    "object_detection": {
      "phone_detected": false,
      "person_count": 1,
      "objects": [
        {
          "class": "person",
          "confidence": 0.95,
          "box": [120, 50, 480, 640]
        }
      ]
    },
    "identity_verification": {
      "match": true,
      "distance": 0.42,
      "checked_at_frame": 1200
    }
  },
  "warnings": ["Looking away detected"]
}
```

**Analysis Fields:**
- `face_detection` - Face detection results (always populated if enabled)
- `head_pose` - Head orientation (empty if face not detected or disabled)
- `gaze` - Eye gaze direction (empty if face not detected or disabled)
- `blink` - Blink/eye closure detection (empty if face not detected or disabled)
- `object_detection` - Phone and person detection (empty if disabled)
- `identity_verification` - Face matching (empty if no profile image or disabled)

**Status Codes:**
- `200 OK` - Frame analyzed successfully
- `404 Not Found` - Session not found
- `500 Internal Server Error` - Analysis error

**Notes:**
- Only enabled features are analyzed (see Settings Management)
- Identity verification runs every N frames (configurable)
- Violations are accumulated in the session tracker

---

## 3. WebSocket Frame Streaming

### WebSocket Connection
**WebSocket** `/ws/{session_id}`

Real-time bidirectional communication for frame streaming. More efficient than HTTP POST for continuous frame analysis.

**Parameters:**
- `session_id` (path) - Session identifier

**Connection:**
```javascript
const ws = new WebSocket(`ws://localhost:8000/ws/${sessionId}`);
```

**Client → Server Messages:**

**Frame Analysis:**
```json
{
  "type": "frame",
  "frame_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

**Ping (Keep-Alive):**
```json
{
  "type": "ping"
}
```

**Server → Client Messages:**

**Analysis Result:**
```json
{
  "type": "analysis",
  "data": {
    "session_id": "STU123_QUIZ001_1234567890",
    "frame_count": 1251,
    "timestamp": "2026-03-06T12:31:15.234567",
    "face_detected": true,
    "face_count": 1,
    "violations": [
      {
        "type": "head_pose",
        "message": "Looking away detected",
        "timestamp": "2026-03-06T12:31:15.234567"
      }
    ],
    "analysis": {
      "face_detection": { ... },
      "head_pose": { ... },
      "gaze": { ... },
      "blink": { ... },
      "object_detection": { ... },
      "identity_verification": { ... }
    },
    "warnings": ["Looking away detected"]
  }
}
```

**Pong Response:**
```json
{
  "type": "pong"
}
```

**Error Response:**
```json
{
  "type": "error",
  "message": "Error description"
}
```

**WebSocket Close Codes:**
- `1000` - Normal closure
- `1008` - Policy violation (session not found)
- `1011` - Internal server error

**Benefits over HTTP:**
- Lower latency (~10-50ms vs ~50-200ms)
- Reduced overhead (no HTTP headers per frame)
- Persistent connection
- Real-time bidirectional communication
- More efficient for high frame rates (20-30 FPS)

**Example Usage:**
```javascript
const ws = new WebSocket(`ws://localhost:8000/ws/${sessionId}`);

ws.onopen = () => {
  console.log('WebSocket connected');
  
  // Start sending frames
  setInterval(() => {
    const frameBase64 = captureFrame(); // Your frame capture logic
    
    ws.send(JSON.stringify({
      type: 'frame',
      frame_base64: frameBase64
    }));
  }, 100); // 10 FPS
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  
  if (message.type === 'analysis') {
    console.log('Analysis result:', message.data);
    
    // Handle violations
    if (message.data.violations.length > 0) {
      console.warn('Violations detected:', message.data.violations);
    }
  } else if (message.type === 'error') {
    console.error('Error:', message.message);
  }
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = (event) => {
  console.log(`WebSocket closed: ${event.code} - ${event.reason}`);
};

// Send ping every 30 seconds to keep connection alive
setInterval(() => {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'ping' }));
  }
}, 30000);
```

**Important Notes:**
- Create session first using REST API before connecting WebSocket
- WebSocket validates session exists on connection
- All analysis settings and config apply the same as HTTP endpoint
- Connection automatically closes if session doesn't exist
- Use either WebSocket OR HTTP POST for frames, not both simultaneously

---

## 4. Settings Management

### Get Session Settings
**GET** `/api/session/{session_id}/settings`

Get current settings for a session.

**Parameters:**
- `session_id` (path) - Session identifier

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

**Status Codes:**
- `200 OK` - Settings retrieved
- `404 Not Found` - Session not found

---

### Update Session Settings
**PUT** `/api/session/{session_id}/settings`

Update settings for an active session. Changes take effect immediately on the next frame.

**Parameters:**
- `session_id` (path) - Session identifier

**Request:**
```json
{
  "enable_face_detection": true,
  "enable_head_pose": false,
  "enable_gaze": true,
  "enable_blink": false,
  "enable_object_detection": true,
  "enable_identity_verification": true,
  "enable_no_face_warning": true
}
```

**Note:** All fields are optional. Only send the fields you want to update.

**Response:**
```json
{
  "message": "Settings updated successfully",
  "settings": {
    "enable_face_detection": true,
    "enable_head_pose": false,
    "enable_gaze": true,
    "enable_blink": false,
    "enable_object_detection": true,
    "enable_identity_verification": true,
    "enable_no_face_warning": true
  }
}
```

**Status Codes:**
- `200 OK` - Settings updated
- `404 Not Found` - Session not found
- `500 Internal Server Error` - Update error

**Settings Description:**
- `enable_face_detection` - Enable/disable face detection (required for other features)
- `enable_head_pose` - Enable/disable head pose analysis
- `enable_gaze` - Enable/disable gaze tracking
- `enable_blink` - Enable/disable blink detection
- `enable_object_detection` - Enable/disable phone/person detection
- `enable_identity_verification` - Enable/disable face matching with profile
- `enable_no_face_warning` - Enable/disable "no face detected" warnings

---

## 5. Violations Management

### Get Violations
**GET** `/api/session/{session_id}/violations`

Get all violations for a session.

**Parameters:**
- `session_id` (path) - Session identifier

**Response:**
```json
{
  "session_id": "STU123_QUIZ001_1234567890",
  "total_violations": 12,
  "violations": [
    {
      "type": "head_pose",
      "message": "Looking away detected",
      "timestamp": "2026-03-06T12:35:12.123456"
    },
    {
      "type": "multiple_faces",
      "message": "Multiple faces detected: 2",
      "timestamp": "2026-03-06T12:38:45.654321"
    },
    {
      "type": "object_detection",
      "message": "Phone detected in frame",
      "timestamp": "2026-03-06T12:42:20.987654"
    },
    {
      "type": "gaze",
      "message": "Looking away from screen",
      "timestamp": "2026-03-06T12:45:10.123456"
    },
    {
      "type": "identity_mismatch",
      "message": "Face does not match profile image",
      "timestamp": "2026-03-06T12:50:30.456789"
    },
    {
      "type": "no_face",
      "message": "No face detected in frame",
      "timestamp": "2026-03-06T12:55:45.789012"
    }
  ]
}
```

**Status Codes:**
- `200 OK` - Violations retrieved
- `404 Not Found` - Session not found

---

### Clear Violations
**DELETE** `/api/session/{session_id}/violations`

Clear all violations for a session. Useful for testing or resetting violation state.

**Parameters:**
- `session_id` (path) - Session identifier

**Response:**
```json
{
  "message": "Violations cleared successfully"
}
```

**Status Codes:**
- `200 OK` - Violations cleared
- `404 Not Found` - Session not found
- `500 Internal Server Error` - Clear error

---

## 6. Configuration Management

### Get Global Config
**GET** `/api/config`

Get current global configuration. These settings affect all sessions and define detection thresholds.

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
    "num_faces": 2
  },
  "head_pose": {
    "yaw_threshold": 25,
    "pitch_threshold": 20,
    "roll_threshold": 30,
    "auto_calibration": true,
    "calibration_time": 3
  },
  "gaze": {
    "horizontal_threshold_left": 0.4,
    "horizontal_threshold_right": 0.6,
    "vertical_threshold_up": 0.4,
    "vertical_threshold_down": 0.6,
    "smoothing_factor": 0.5
  },
  "blink": {
    "ear_threshold": 0.25,
    "min_blink_frames": 2,
    "long_closure_threshold": 2.0,
    "smoothing_alpha": 0.3
  },
  "thresholds": {
    "max_num_faces": 1,
    "enable_no_face_warning": true,
    "multi_face_persistence_time": 2.0,
    "no_face_persistence_time": 3.0,
    "head_pose_persistence_time": 2.0,
    "gaze_persistence_time": 2.0,
    "identity_check_interval_frames": 30,
    "identity_persistence_time": 3.0
  }
}
```

**Configuration Sections:**

**Camera:**
- `index` - Camera device index
- `width`, `height` - Video resolution
- `fps` - Frames per second

**MediaPipe:**
- `num_faces` - Maximum number of faces to detect

**Head Pose:**
- `yaw_threshold` - Maximum degrees left/right (default: 25°)
- `pitch_threshold` - Maximum degrees up/down (default: 20°)
- `roll_threshold` - Maximum degrees tilt (default: 30°)
- `auto_calibration` - Enable auto-calibration
- `calibration_time` - Calibration duration in seconds

**Gaze:**
- `horizontal_threshold_left`, `horizontal_threshold_right` - Horizontal thresholds (0-1)
- `vertical_threshold_up`, `vertical_threshold_down` - Vertical thresholds (0-1)
- `smoothing_factor` - Gaze smoothing factor

**Blink:**
- `ear_threshold` - Eye Aspect Ratio threshold (default: 0.25)
- `min_blink_frames` - Minimum frames for blink detection
- `long_closure_threshold` - Prolonged closure threshold in seconds
- `smoothing_alpha` - EAR smoothing factor

**Thresholds:**
- `max_num_faces` - Maximum allowed faces in frame
- `enable_no_face_warning` - Enable no-face warnings
- `multi_face_persistence_time` - Time before triggering multiple face violation
- `no_face_persistence_time` - Time before triggering no face violation
- `head_pose_persistence_time` - Time before triggering head pose violation
- `gaze_persistence_time` - Time before triggering gaze violation
- `identity_check_interval_frames` - Frames between identity checks
- `identity_persistence_time` - Time before triggering identity violation

**Status Codes:**
- `200 OK` - Config retrieved

---

### Update Global Config
**PUT** `/api/config`

Update global configuration. Changes take effect immediately for all active sessions.

**Request:**
```json
{
  "head_pose": {
    "yaw_threshold": 30,
    "pitch_threshold": 25
  },
  "blink": {
    "ear_threshold": 0.22
  },
  "thresholds": {
    "identity_check_interval_frames": 60,
    "head_pose_persistence_time": 3.0
  }
}
```

**Note:** All fields are optional. Only include the nested objects you want to update.

**Response:**
```json
{
  "message": "Configuration updated successfully",
  "note": "Changes apply to new sessions and analyzers. Restart server for full effect.",
  "updated_fields": ["head_pose", "blink", "thresholds"]
}
```

**Status Codes:**
- `200 OK` - Config updated
- `500 Internal Server Error` - Update error

**Important:**
- Changes do NOT modify the `app.yaml` file (default config)
- In-memory config is updated immediately
- Some changes may require server restart for full effect

---

## 7. Health Check

### Health Status
**GET** `/`

Check API health status and get active session count.

**Response:**
```json
{
  "service": "AI Proctor API",
  "status": "healthy",
  "version": "1.0.0",
  "active_sessions": 2
}
```

**Status Codes:**
- `200 OK` - Service is healthy

---

## 8. Integration Examples

### JavaScript/TypeScript

```javascript
const API_BASE = 'http://localhost:8000';

class ProctorAPI {
  // Create session with profile image
  async createSession(studentId, quizId, profileImageFile) {
    const formData = new FormData();
    formData.append('student_id', studentId);
    formData.append('quiz_id', quizId);
    if (profileImageFile) {
      formData.append('profile_image', profileImageFile);
    }
    
    const response = await fetch(`${API_BASE}/api/session/start`, {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      throw new Error(`Failed to create session: ${response.statusText}`);
    }
    
    return await response.json();
  }

  // Analyze frame
  async analyzeFrame(sessionId, frameBase64) {
    const response = await fetch(`${API_BASE}/api/analyze/frame/${sessionId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frame_base64: frameBase64 })
    });
    
    if (!response.ok) {
      throw new Error(`Frame analysis failed: ${response.statusText}`);
    }
    
    return await response.json();
  }

  // Get session info
  async getSessionInfo(sessionId) {
    const response = await fetch(`${API_BASE}/api/session/${sessionId}`);
    
    if (!response.ok) {
      throw new Error(`Failed to get session: ${response.statusText}`);
    }
    
    return await response.json();
  }

  // Update settings
  async updateSettings(sessionId, settings) {
    const response = await fetch(`${API_BASE}/api/session/${sessionId}/settings`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings)
    });
    
    if (!response.ok) {
      throw new Error(`Failed to update settings: ${response.statusText}`);
    }
    
    return await response.json();
  }

  // Get violations
  async getViolations(sessionId) {
    const response = await fetch(`${API_BASE}/api/session/${sessionId}/violations`);
    
    if (!response.ok) {
      throw new Error(`Failed to get violations: ${response.statusText}`);
    }
    
    return await response.json();
  }

  // Clear violations
  async clearViolations(sessionId) {
    const response = await fetch(`${API_BASE}/api/session/${sessionId}/violations`, {
      method: 'DELETE'
    });
    
    if (!response.ok) {
      throw new Error(`Failed to clear violations: ${response.statusText}`);
    }
    
    return await response.json();
  }

  // Get config
  async getConfig() {
    const response = await fetch(`${API_BASE}/api/config`);
    
    if (!response.ok) {
      throw new Error(`Failed to get config: ${response.statusText}`);
    }
    
    return await response.json();
  }

  // Update config
  async updateConfig(config) {
    const response = await fetch(`${API_BASE}/api/config`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    });
    
    if (!response.ok) {
      throw new Error(`Failed to update config: ${response.statusText}`);
    }
    
    return await response.json();
  }

  // End session
  async endSession(sessionId) {
    const response = await fetch(`${API_BASE}/api/session/end/${sessionId}`, {
      method: 'POST'
    });
    
    if (!response.ok) {
      throw new Error(`Failed to end session: ${response.statusText}`);
    }
    
    return await response.json();
  }

  // List all sessions
  async listSessions() {
    const response = await fetch(`${API_BASE}/api/sessions`);
    
    if (!response.ok) {
      throw new Error(`Failed to list sessions: ${response.statusText}`);
    }
    
    return await response.json();
  }

  // Health check
  async healthCheck() {
    const response = await fetch(`${API_BASE}/`);
    
    if (!response.ok) {
      throw new Error(`Health check failed: ${response.statusText}`);
    }
    
    return await response.json();
  }
}

// Usage Example
async function startProctoring() {
  const api = new ProctorAPI();
  
  try {
    // 1. Create session
    const session = await api.createSession('STU123', 'QUIZ001', profileImageFile);
    console.log('Session created:', session.session_id);
    
    // 2. Set up video capture
    const video = document.getElementById('video');
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    // 3. Start frame analysis loop
    setInterval(async () => {
      // Capture frame
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0);
      const frameBase64 = canvas.toDataURL('image/jpeg', 0.8);
      
      // Analyze frame
      const result = await api.analyzeFrame(session.session_id, frameBase64);
      
      // Display warnings
      if (result.warnings.length > 0) {
        console.warn('Warnings:', result.warnings);
      }
      
      // Display violations
      if (result.violations.length > 0) {
        console.error('New violations:', result.violations);
      }
    }, 100); // 10 FPS
    
    // 4. Periodically fetch violations
    setInterval(async () => {
      const violations = await api.getViolations(session.session_id);
      console.log('Total violations:', violations.total_violations);
    }, 2000);
    
  } catch (error) {
    console.error('Proctoring error:', error);
  }
}
```

**WebSocket Client Class:**

```javascript
class ProctorWebSocket {
  constructor(sessionId, baseUrl = 'ws://localhost:8000') {
    this.sessionId = sessionId;
    this.baseUrl = baseUrl;
    this.ws = null;
    this.isConnected = false;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.onAnalysisCallback = null;
    this.onErrorCallback = null;
    this.onConnectCallback = null;
    this.onDisconnectCallback = null;
  }

  connect() {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(`${this.baseUrl}/ws/${this.sessionId}`);
      
      this.ws.onopen = () => {
        console.log('WebSocket connected');
        this.isConnected = true;
        this.reconnectAttempts = 0;
        
        if (this.onConnectCallback) {
          this.onConnectCallback();
        }
        
        resolve();
      };
      
      this.ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        
        if (message.type === 'analysis') {
          if (this.onAnalysisCallback) {
            this.onAnalysisCallback(message.data);
          }
        } else if (message.type === 'error') {
          console.error('Server error:', message.message);
          if (this.onErrorCallback) {
            this.onErrorCallback(message.message);
          }
        } else if (message.type === 'pong') {
          // Heartbeat response received
          console.log('Pong received');
        }
      };
      
      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        if (this.onErrorCallback) {
          this.onErrorCallback(error);
        }
      };
      
      this.ws.onclose = (event) => {
        console.log(`WebSocket closed: ${event.code}`);
        this.isConnected = false;
        
        if (this.onDisconnectCallback) {
          this.onDisconnectCallback(event);
        }
        
        // Auto-reconnect
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
          this.reconnectAttempts++;
          console.log(`Reconnecting... Attempt ${this.reconnectAttempts}`);
          setTimeout(() => this.connect(), 2000 * this.reconnectAttempts);
        }
      };
      
      // Reject on connection timeout
      setTimeout(() => {
        if (!this.isConnected) {
          reject(new Error('Connection timeout'));
        }
      }, 10000);
    });
  }

  sendFrame(frameBase64) {
    if (!this.isConnected || this.ws.readyState !== WebSocket.OPEN) {
      console.warn('WebSocket not connected');
      return false;
    }
    
    this.ws.send(JSON.stringify({
      type: 'frame',
      frame_base64: frameBase64
    }));
    
    return true;
  }

  ping() {
    if (this.isConnected && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'ping' }));
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.isConnected = false;
    }
  }

  onAnalysis(callback) {
    this.onAnalysisCallback = callback;
  }

  onError(callback) {
    this.onErrorCallback = callback;
  }

  onConnect(callback) {
    this.onConnectCallback = callback;
  }

  onDisconnect(callback) {
    this.onDisconnectCallback = callback;
  }
}

// Usage Example with WebSocket
async function startProctoringWithWebSocket() {
  const api = new ProctorAPI();
  
  try {
    // 1. Create session first (REST API)
    const session = await api.createSession('STU123', 'QUIZ001', profileImageFile);
    console.log('Session created:', session.session_id);
    
    // 2. Connect WebSocket
    const wsClient = new ProctorWebSocket(session.session_id);
    
    // Set up callbacks
    wsClient.onConnect(() => {
      console.log('Ready to stream frames');
    });
    
    wsClient.onAnalysis((data) => {
      console.log('Frame analysis:', data.frame_count);
      
      // Display violations in real-time
      if (data.violations.length > 0) {
        console.warn('Violations:', data.violations);
        // Update UI with violations
      }
      
      // Display warnings
      if (data.warnings.length > 0) {
        console.warn('Warnings:', data.warnings);
      }
      
      // Update analysis display
      updateAnalysisDisplay(data.analysis);
    });
    
    wsClient.onError((error) => {
      console.error('WebSocket error:', error);
    });
    
    wsClient.onDisconnect((event) => {
      console.log('Disconnected:', event.code);
    });
    
    // Connect
    await wsClient.connect();
    
    // 3. Set up video capture
    const video = document.getElementById('video');
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    // 4. Start streaming frames via WebSocket
    const frameInterval = setInterval(() => {
      if (!wsClient.isConnected) {
        console.warn('WebSocket not connected, skipping frame');
        return;
      }
      
      // Capture frame
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0);
      const frameBase64 = canvas.toDataURL('image/jpeg', 0.8);
      
      // Send via WebSocket
      wsClient.sendFrame(frameBase64);
    }, 100); // 10 FPS
    
    // 5. Keep connection alive with ping
    const pingInterval = setInterval(() => {
      wsClient.ping();
    }, 30000); // Ping every 30 seconds
    
    // 6. Cleanup on page unload
    window.addEventListener('beforeunload', () => {
      clearInterval(frameInterval);
      clearInterval(pingInterval);
      wsClient.disconnect();
    });
    
  } catch (error) {
    console.error('Proctoring error:', error);
  }
}

function updateAnalysisDisplay(analysis) {
  // Update your UI with analysis results
  document.getElementById('face-count').textContent = analysis.face_detection.face_count;
  
  if (analysis.head_pose.direction) {
    document.getElementById('head-direction').textContent = analysis.head_pose.direction;
  }
  
  if (analysis.gaze.direction) {
    document.getElementById('gaze-direction').textContent = analysis.gaze.direction;
  }
  
  // ... update other UI elements
}
```

### Python

```python
import requests
import base64
from typing import Optional, Dict, Any

class ProctorAPI:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def create_session(
        self, 
        student_id: str, 
        quiz_id: str, 
        profile_image_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new proctoring session"""
        files = {}
        data = {
            'student_id': student_id,
            'quiz_id': quiz_id
        }
        
        if profile_image_path:
            files['profile_image'] = open(profile_image_path, 'rb')
        
        response = requests.post(
            f"{self.base_url}/api/session/start",
            data=data,
            files=files
        )
        response.raise_for_status()
        return response.json()
    
    def analyze_frame(self, session_id: str, frame_base64: str) -> Dict[str, Any]:
        """Analyze a video frame"""
        response = requests.post(
            f"{self.base_url}/api/analyze/frame/{session_id}",
            json={'frame_base64': frame_base64}
        )
        response.raise_for_status()
        return response.json()
    
    def get_violations(self, session_id: str) -> Dict[str, Any]:
        """Get all violations for a session"""
        response = requests.get(
            f"{self.base_url}/api/session/{session_id}/violations"
        )
        response.raise_for_status()
        return response.json()
    
    def update_settings(self, session_id: str, settings: Dict[str, bool]) -> Dict[str, Any]:
        """Update session settings"""
        response = requests.put(
            f"{self.base_url}/api/session/{session_id}/settings",
            json=settings
        )
        response.raise_for_status()
        return response.json()
    
    def update_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Update global configuration"""
        response = requests.put(
            f"{self.base_url}/api/config",
            json=config
        )
        response.raise_for_status()
        return response.json()
    
    def end_session(self, session_id: str) -> Dict[str, Any]:
        """End a proctoring session"""
        response = requests.post(
            f"{self.base_url}/api/session/end/{session_id}"
        )
        response.raise_for_status()
        return response.json()

# Usage
api = ProctorAPI()

# Create session
session = api.create_session('STU123', 'QUIZ001', 'profile.jpg')
print(f"Session created: {session['session_id']}")

# Analyze frame
with open('frame.jpg', 'rb') as f:
    frame_data = base64.b64encode(f.read()).decode()
    frame_base64 = f"data:image/jpeg;base64,{frame_data}"

result = api.analyze_frame(session['session_id'], frame_base64)
print(f"Violations: {result['violations']}")

# Get all violations
violations = api.get_violations(session['session_id'])
print(f"Total violations: {violations['total_violations']}")

# End session
report = api.end_session(session['session_id'])
print(f"Final report: {report}")
```

---

## 9. Violation Types

| Type | Description | Trigger Condition |
|------|-------------|-------------------|
| `no_face` | No face detected in frame | Face not detected for persistence time (default: 3s) |
| `multiple_faces` | More than allowed faces | More faces than `max_num_faces` for persistence time (default: 2s) |
| `head_pose` | Head turned away | Yaw/pitch exceeds thresholds for persistence time (default: 2s) |
| `gaze` | Eyes looking away | Gaze not centered for persistence time (default: 2s) |
| `object_detection` | Phone or unauthorized person | Phone detected or person_count > 1 for persistence time (default: 2s) |
| `identity_mismatch` | Face doesn't match profile | Face embedding distance > threshold for persistence time (default: 3s) |
| `session_resume` | Session resumed after inactivity | Student left and returned to exam (inactivity > 10s) |
| `no_frames` | No frames received | No analysis frames received for configured time |

**Persistence Time:** All violations require the condition to persist for a configured duration before being triggered. This prevents false positives from momentary movements.

### Session Resume Detection

When a student restarts the session (closes browser and comes back), the system:
- **Automatically resumes** the existing session
- **Tracks inactivity duration** (time between last activity and resume)
- **Logs as violation** if inactivity > 10 seconds
- **Increments resume counter** to track total number of resumes

This helps detect if students are leaving the exam to look up answers or consult with others.

---

## Error Responses

All endpoints may return the following error responses:

**404 Not Found:**
```json
{
  "detail": "Session not found: STU123_QUIZ001_1234567890. Please create a new session first."
}
```

**400 Bad Request:**
```json
{
  "detail": "Invalid input or duplicate session"
}
```

**500 Internal Server Error:**
```json
{
  "detail": "Analysis error: [error details]"
}
```

---

## Rate Limiting

Currently, no rate limiting is implemented. For production use, consider:
- Limiting frame analysis requests per session (e.g., 10-30 FPS)
- Throttling violation queries
- Implementing session timeouts

---

## Best Practices

1. **Frame Rate:** Send frames at 10-30 FPS for optimal performance
2. **Transport Method:**
   - Use **WebSocket** for real-time streaming (recommended for production)
   - Use **HTTP POST** for testing or lower frame rates (< 5 FPS)
3. **Image Quality:** Use JPEG with 70-80% quality for balance between size and accuracy
4. **Session Management:** Always end sessions to free resources
5. **Error Handling:** Implement retry logic with exponential backoff (especially for WebSocket reconnection)
6. **Profile Images:** Use clear, front-facing photos for best identity verification
7. **Settings:** Disable unused features to improve performance
8. **WebSocket Keep-Alive:** Send ping messages every 30 seconds to maintain connection
9. **Connection Management:** Don't mix HTTP POST and WebSocket for the same session

---

## CORS

The API has CORS enabled for all origins (`*`). For production, configure specific allowed origins in `src/web/main.py`.

---

## WebSocket Support

✅ **WebSocket support is now implemented!**

**Endpoint:** `ws://localhost:8000/ws/{session_id}`

Use WebSocket for:
- Real-time frame streaming with lower latency
- Reduced network overhead
- Bidirectional communication
- High frame rate applications (20-30 FPS)

See [WebSocket Frame Streaming](#3-websocket-frame-streaming) section for full details and examples.

---

## Changelog

**v1.1.0** (March 2026)
- ✅ Added WebSocket support for real-time frame streaming
- WebSocket endpoint: `/ws/{session_id}`
- Lower latency and reduced overhead compared to HTTP POST
- Bidirectional communication with real-time analysis results
- Auto-reconnect and keep-alive support
- Complete JavaScript WebSocket client class example

**v1.0.0** (March 2026)
- Initial API release
- Session management, frame analysis, settings, violations, config endpoints
- Support for face detection, head pose, gaze, blink, object detection, identity verification
- Real-time violation tracking with persistence

---

For more information, visit the interactive API documentation at `http://localhost:8000/docs` when the server is running.
