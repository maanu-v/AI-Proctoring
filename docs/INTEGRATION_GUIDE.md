# AI Proctor Integration Guide for Exam Platforms

## Overview

This guide will help you integrate the AI Proctor system into your exam platform. The system provides real-time proctoring capabilities using computer vision to monitor students during online exams.

**Key Features:**
- ✅ Face detection and tracking
- ✅ Head pose monitoring (looking away detection)
- ✅ Gaze tracking (eye movement)
- ✅ Blink detection (signs of life)
- ✅ Object detection (phones, unauthorized persons)
- ✅ Identity verification (face matching)
- ✅ Real-time violation alerts
- ✅ WebSocket support for low-latency streaming

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Integration Steps](#integration-steps)
4. [API Endpoints](#api-endpoints)
5. [WebSocket Integration](#websocket-integration)
6. [Complete Integration Example](#complete-integration-example)
7. [Handling Violations](#handling-violations)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites

- AI Proctor API running on `http://localhost:8000`
- Student's webcam access permission
- Modern browser with WebSocket and Canvas support

### Installation

```bash
# Start the AI Proctor API
cd /path/to/ai-proctor
./start_api.sh

# API will be available at http://localhost:8000
# Interactive documentation at http://localhost:8000/docs
```

### Basic Flow

```
1. Student starts exam → Create proctoring session
2. Capture webcam frames → Send to API via WebSocket
3. Receive analysis results → Display warnings/violations
4. Student submits exam → End session and get report
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Your Exam Platform                       │
│  ┌────────────────────────────────────────────────────┐    │
│  │           Frontend (React/Vue/Angular)             │    │
│  │                                                     │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────┐    │    │
│  │  │ Quiz UI  │  │ Webcam   │  │  Violations  │    │    │
│  │  │          │  │ Capture  │  │  Display     │    │    │
│  │  └──────────┘  └─────┬────┘  └──────────────┘    │    │
│  │                      │                             │    │
│  │                      ▼                             │    │
│  │            ┌─────────────────┐                    │    │
│  │            │  Proctor Client │                    │    │
│  │            │   (WebSocket)   │                    │    │
│  │            └────────┬────────┘                    │    │
│  └─────────────────────┼─────────────────────────────┘    │
└────────────────────────┼──────────────────────────────────┘
                         │
                         │ WebSocket: ws://localhost:8000/ws/{session_id}
                         │ HTTP: http://localhost:8000/api/*
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  AI Proctor API Server                      │
│  ┌────────────────────────────────────────────────────┐    │
│  │              FastAPI Backend                       │    │
│  │                                                     │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────┐    │    │
│  │  │ Session  │  │ Analysis │  │  Violation   │    │    │
│  │  │ Manager  │  │ Engine   │  │  Tracker     │    │    │
│  │  └──────────┘  └──────────┘  └──────────────┘    │    │
│  │                                                     │    │
│  │  ┌──────────────────────────────────────────┐    │    │
│  │  │      AI Models (MediaPipe, YOLO)         │    │    │
│  │  └──────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## Integration Steps

### Step 1: Create a Proctoring Session

When a student starts an exam, create a proctoring session:

```javascript
// Create session with identity verification
async function startProctoring(studentId, quizId, profileImageFile) {
    const formData = new FormData();
    formData.append('student_id', studentId);
    formData.append('quiz_id', quizId);
    
    // Optional: Add profile image for identity verification
    if (profileImageFile) {
        formData.append('profile_image', profileImageFile);
    }
    
    const response = await fetch('http://localhost:8000/api/session/start', {
        method: 'POST',
        body: formData
    });
    
    const data = await response.json();
    
    // Check if session was resumed
    if (data.message.includes('resumed')) {
        console.warn('Session resumed - student may have left exam');
        // Optionally notify proctor or log this event
    }
    
    return data.session_id;
}
```

**Response:**
```json
{
  "session_id": "STU123_QUIZ001_1710675045",
  "student_id": "STU123",
  "quiz_id": "QUIZ001",
  "created_at": "2026-03-07T10:30:45.123456",
  "message": "Session started successfully"
}
```

### Step 2: Capture Webcam Frames

Set up webcam capture in your frontend:

```javascript
class WebcamCapture {
    constructor(videoElement) {
        this.video = videoElement;
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');
    }
    
    async start() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    frameRate: { ideal: 30, max: 30 }
                }
            });
            this.video.srcObject = stream;
            await this.video.play();
            return true;
        } catch (error) {
            console.error('Webcam access denied:', error);
            return false;
        }
    }
    
    captureFrame() {
        // Set canvas size to video size
        this.canvas.width = this.video.videoWidth;
        this.canvas.height = this.video.videoHeight;
        
        // Draw current frame
        this.ctx.drawImage(this.video, 0, 0);
        
        // Convert to base64 (data URL format)
        return this.canvas.toDataURL('image/jpeg', 0.8);
    }
    
    stop() {
        if (this.video.srcObject) {
            this.video.srcObject.getTracks().forEach(track => track.stop());
        }
    }
}
```

### Step 3: Connect via WebSocket

Establish WebSocket connection for real-time frame streaming:

```javascript
class ProctorWebSocket {
    constructor(sessionId, onAnalysis, onViolation) {
        this.sessionId = sessionId;
        this.ws = null;
        this.onAnalysis = onAnalysis;
        this.onViolation = onViolation;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
    }
    
    connect() {
        const wsUrl = `ws://localhost:8000/ws/${this.sessionId}`;
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.reconnectAttempts = 0;
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.type === 'analysis') {
                this.onAnalysis(data);
                
                // Check for violations
                if (data.violations && data.violations.length > 0) {
                    this.onViolation(data.violations);
                }
            } else if (data.type === 'error') {
                console.error('Analysis error:', data.message);
            }
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            this.attemptReconnect();
        };
    }
    
    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 10000);
            console.log(`Reconnecting in ${delay}ms...`);
            setTimeout(() => this.connect(), delay);
        }
    }
    
    sendFrame(frameBase64) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'frame',
                frame_base64: frameBase64
            }));
        }
    }
    
    disconnect() {
        if (this.ws) {
            this.ws.close();
        }
    }
}
```

### Step 4: Start Frame Streaming

Stream frames to the API at regular intervals:

```javascript
class ProctorManager {
    constructor(sessionId, videoElement) {
        this.sessionId = sessionId;
        this.webcam = new WebcamCapture(videoElement);
        this.violations = [];
        this.isActive = false;
        this.frameInterval = null;
        
        // Initialize WebSocket
        this.ws = new ProctorWebSocket(
            sessionId,
            this.handleAnalysis.bind(this),
            this.handleViolation.bind(this)
        );
    }
    
    async start() {
        // Start webcam
        const started = await this.webcam.start();
        if (!started) {
            throw new Error('Failed to start webcam');
        }
        
        // Connect WebSocket
        this.ws.connect();
        
        // Start sending frames (20 FPS)
        this.isActive = true;
        this.frameInterval = setInterval(() => {
            if (this.isActive) {
                const frame = this.webcam.captureFrame();
                this.ws.sendFrame(frame);
            }
        }, 50); // 20 FPS (1000ms / 20 = 50ms)
    }
    
    handleAnalysis(data) {
        // Update UI with analysis results
        console.log('Analysis:', data);
        
        // Display face count
        if (data.face_detection) {
            this.updateFaceCount(data.face_detection.face_count);
        }
        
        // Display head pose
        if (data.head_pose) {
            this.updateHeadPose(data.head_pose);
        }
        
        // Display gaze
        if (data.gaze) {
            this.updateGaze(data.gaze);
        }
        
        // Display warnings
        if (data.warnings && data.warnings.length > 0) {
            this.showWarnings(data.warnings);
        }
    }
    
    handleViolation(violations) {
        // Store violations
        this.violations.push(...violations);
        
        // Alert user
        violations.forEach(violation => {
            this.showAlert(violation.type, violation.message);
        });
        
        // Notify your backend
        this.reportViolations(violations);
    }
    
    async reportViolations(violations) {
        // Send to your exam platform backend
        await fetch(`/your-api/exam/violations`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: this.sessionId,
                violations: violations
            })
        });
    }
    
    stop() {
        this.isActive = false;
        
        if (this.frameInterval) {
            clearInterval(this.frameInterval);
        }
        
        this.ws.disconnect();
        this.webcam.stop();
    }
    
    async getViolationReport() {
        const response = await fetch(
            `http://localhost:8000/api/session/${this.sessionId}/violations`
        );
        return await response.json();
    }
}
```

### Step 5: End Session

When exam is completed or submitted:

```javascript
async function endProctoring(sessionId) {
    // Get final report
    const response = await fetch(
        `http://localhost:8000/api/session/${sessionId}/end`,
        { method: 'POST' }
    );
    
    const report = await response.json();
    return report;
}
```

**Response:**
```json
{
  "session_id": "STU123_QUIZ001_1710675045",
  "student_id": "STU123",
  "quiz_id": "QUIZ001",
  "duration_seconds": 3600,
  "total_violations": 5,
  "violation_summary": {
    "no_face": 1,
    "head_pose": 2,
    "object_detection": 2
  },
  "violations": [
    {
      "type": "head_pose",
      "message": "Head turned away",
      "timestamp": "2026-03-07T10:35:12.123456"
    }
  ],
  "message": "Session ended successfully"
}
```

---

## API Endpoints

### Session Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/session/start` | Create new session |
| GET | `/api/session/{session_id}` | Get session info |
| POST | `/api/session/{session_id}/end` | End session |
| GET | `/api/sessions` | List all sessions |

### Frame Analysis

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/session/{session_id}/analyze` | Analyze single frame (HTTP) |
| WS | `/ws/{session_id}` | Stream frames (WebSocket) |

### Settings

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/session/{session_id}/settings` | Get settings |
| PUT | `/api/session/{session_id}/settings` | Update settings |

### Violations

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/session/{session_id}/violations` | Get all violations |
| DELETE | `/api/session/{session_id}/violations` | Clear violations |

### Configuration

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/config` | Get global config |
| PUT | `/api/config` | Update config |

---

## WebSocket Integration

### Connection

```javascript
const ws = new WebSocket(`ws://localhost:8000/ws/${sessionId}`);
```

### Send Frame

```javascript
ws.send(JSON.stringify({
    type: 'frame',
    frame_base64: 'data:image/jpeg;base64,/9j/4AAQ...'
}));
```

### Receive Analysis

```javascript
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    // Analysis result
    if (data.type === 'analysis') {
        console.log('Face count:', data.face_detection.face_count);
        console.log('Head pose:', data.head_pose);
        console.log('Gaze:', data.gaze);
        console.log('Violations:', data.violations);
        console.log('Warnings:', data.warnings);
    }
    
    // Error
    if (data.type === 'error') {
        console.error('Error:', data.message);
    }
};
```

---

## Complete Integration Example

Here's a complete React component example:

```jsx
import React, { useEffect, useRef, useState } from 'react';

const ExamProctor = ({ studentId, quizId, profileImage }) => {
    const videoRef = useRef(null);
    const [sessionId, setSessionId] = useState(null);
    const [proctorManager, setProctorManager] = useState(null);
    const [violations, setViolations] = useState([]);
    const [warnings, setWarnings] = useState([]);
    const [faceCount, setFaceCount] = useState(0);
    const [isLookingForward, setIsLookingForward] = useState(true);
    
    // Start proctoring when component mounts
    useEffect(() => {
        startProctoring();
        return () => stopProctoring();
    }, []);
    
    const startProctoring = async () => {
        try {
            // Create session
            const formData = new FormData();
            formData.append('student_id', studentId);
            formData.append('quiz_id', quizId);
            if (profileImage) {
                formData.append('profile_image', profileImage);
            }
            
            const response = await fetch('http://localhost:8000/api/session/start', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            const newSessionId = data.session_id;
            setSessionId(newSessionId);
            
            // Initialize proctor manager
            const manager = new ProctorManager(newSessionId, videoRef.current);
            manager.onAnalysis = handleAnalysis;
            manager.onViolation = handleViolation;
            
            await manager.start();
            setProctorManager(manager);
            
        } catch (error) {
            console.error('Failed to start proctoring:', error);
            alert('Failed to start proctoring. Please check camera permissions.');
        }
    };
    
    const stopProctoring = async () => {
        if (proctorManager) {
            proctorManager.stop();
            
            // Get final report
            const report = await fetch(
                `http://localhost:8000/api/session/${sessionId}/end`,
                { method: 'POST' }
            );
            const data = await report.json();
            console.log('Final report:', data);
        }
    };
    
    const handleAnalysis = (data) => {
        // Update UI with real-time analysis
        if (data.face_detection) {
            setFaceCount(data.face_detection.face_count);
        }
        
        if (data.head_pose) {
            setIsLookingForward(data.head_pose.is_looking_forward);
        }
        
        if (data.warnings) {
            setWarnings(data.warnings);
        }
    };
    
    const handleViolation = (newViolations) => {
        setViolations(prev => [...prev, ...newViolations]);
        
        // Show toast notification
        newViolations.forEach(v => {
            showToast(v.message, 'warning');
        });
    };
    
    return (
        <div className="exam-proctor">
            <div className="video-container">
                <video ref={videoRef} autoPlay muted />
                
                {/* Status indicators */}
                <div className="status-overlay">
                    <div className={`face-status ${faceCount === 1 ? 'ok' : 'warning'}`}>
                        Faces: {faceCount}
                    </div>
                    <div className={`pose-status ${isLookingForward ? 'ok' : 'warning'}`}>
                        {isLookingForward ? '✓ Looking Forward' : '⚠ Looking Away'}
                    </div>
                </div>
            </div>
            
            {/* Warnings */}
            {warnings.length > 0 && (
                <div className="warnings-panel">
                    <h3>⚠️ Warnings</h3>
                    <ul>
                        {warnings.map((warning, i) => (
                            <li key={i}>{warning}</li>
                        ))}
                    </ul>
                </div>
            )}
            
            {/* Violation count */}
            <div className="violation-count">
                Violations: {violations.length}
            </div>
        </div>
    );
};

export default ExamProctor;
```

For Vue.js or Angular implementations, see the examples in the `examples/` directory.

---

## Handling Violations

### Violation Types

| Type | Description | Typical Response |
|------|-------------|------------------|
| `no_face` | No face detected | Warn student to stay in frame |
| `multiple_faces` | Multiple faces detected | Alert proctor, may indicate cheating |
| `head_pose` | Head turned away | Warn student to look at screen |
| `gaze` | Eyes looking away | Warn student (may be reading notes) |
| `object_phone` | Phone detected | Alert proctor, serious violation |
| `object_multiple_people` | Extra person detected | Alert proctor, serious violation |
| `identity_mismatch` | Face doesn't match profile | Alert proctor, possible impersonation |
| `session_resume` | Student left and returned | Track pattern, alert if frequent |
| `no_frames` | No video frames received | Warn student about connection |

### Violation Response Strategy

```javascript
function handleViolation(violation) {
    switch (violation.type) {
        case 'no_face':
        case 'head_pose':
        case 'gaze':
        case 'no_frames':
            // Minor violations - warn student
            showWarning(violation.message);
            break;
        
        case 'session_resume':
            // Track leaving pattern
            if (violation.message.includes('Resume #')) {
                const count = parseInt(violation.message.match(/Resume #(\d+)/)[1]);
                if (count > 2) {
                    alertProctor({
                        type: 'suspicious_pattern',
                        message: `Student has left exam ${count} times`,
                        violation: violation
                    });
                }
            }
            break;
            
        case 'multiple_faces':
        case 'object_phone':
        case 'object_multiple_people':
        case 'identity_mismatch':
            // Serious violations - alert proctor
            alertProctor(violation);
            // May auto-submit exam or lock screen
            if (violation.type === 'identity_mismatch') {
                considerAutoSubmit();
            }
            break;
    }
    
    // Log to your backend
    logViolation(violation);
}
```

---

## Best Practices

### 1. Frame Rate Optimization

```javascript
// Recommended frame rates
const FRAME_RATES = {
    HIGH_QUALITY: 30,    // 30 FPS - best accuracy, high bandwidth
    BALANCED: 20,        // 20 FPS - good accuracy, moderate bandwidth
    LOW_BANDWIDTH: 10    // 10 FPS - acceptable accuracy, low bandwidth
};

// Adjust based on connection quality
function getOptimalFrameRate(connectionSpeed) {
    if (connectionSpeed > 5) return FRAME_RATES.HIGH_QUALITY;
    if (connectionSpeed > 2) return FRAME_RATES.BALANCED;
    return FRAME_RATES.LOW_BANDWIDTH;
}
```

### 2. Image Quality

```javascript
// JPEG quality balances size and accuracy
canvas.toDataURL('image/jpeg', 0.8); // 80% quality recommended
```

### 3. Error Handling

```javascript
// Implement retry logic
async function sendFrameWithRetry(frame, maxRetries = 3) {
    for (let i = 0; i < maxRetries; i++) {
        try {
            await sendFrame(frame);
            return;
        } catch (error) {
            if (i === maxRetries - 1) throw error;
            await sleep(1000 * Math.pow(2, i)); // Exponential backoff
        }
    }
}
```

### 4. Resource Management

```javascript
// Clean up when exam ends
function cleanup() {
    // Stop webcam
    webcam.stop();
    
    // Close WebSocket
    ws.disconnect();
    
    // Clear intervals
    clearInterval(frameInterval);
    
    // End session
    endSession(sessionId);
}
```

### 5. Privacy Considerations

```javascript
// Inform user about monitoring
function showPrivacyNotice() {
    return confirm(
        'This exam is proctored. Your webcam will be monitored for:\n' +
        '- Face detection\n' +
        '- Head movement\n' +
        '- Eye gaze\n' +
        '- Unauthorized objects\n\n' +
        'Do you consent to monitoring?'
    );
}
```

### 6. Settings Configuration

```javascript
// Adjust settings based on exam type
const examSettings = {
    strict: {
        enable_face_detection: true,
        enable_head_pose: true,
        enable_gaze: true,
        enable_blink: true,
        enable_object_detection: true,
        enable_identity_verification: true,
        enable_no_face_warning: true
    },
    moderate: {
        enable_face_detection: true,
        enable_head_pose: true,
        enable_gaze: false,  // Allow some eye movement
        enable_blink: true,
        enable_object_detection: true,
        enable_identity_verification: true,
        enable_no_face_warning: true
    },
    lenient: {
        enable_face_detection: true,
        enable_head_pose: false,
        enable_gaze: false,
        enable_blink: true,
        enable_object_detection: false,
        enable_identity_verification: true,
        enable_no_face_warning: true
    }
};

// Update settings during exam
await updateSettings(sessionId, examSettings.moderate);
```

### 7. Session Resumption Handling

The system automatically handles students leaving and returning to exams:

```javascript
// When student returns to exam page
async function resumeExam(studentId, quizId) {
    // Calling start again will resume the existing session
    const formData = new FormData();
    formData.append('student_id', studentId);
    formData.append('quiz_id', quizId);
    
    const response = await fetch('http://localhost:8000/api/session/start', {
        method: 'POST',
        body: formData
    });
    
    const data = await response.json();
    
    // Check if session was resumed
    if (data.message.includes('resumed')) {
        const resumeMatch = data.message.match(/Resume #(\d+)/);
        const resumeCount = resumeMatch ? parseInt(resumeMatch[1]) : 0;
        
        console.warn(`Session resumed - attempt #${resumeCount}`);
        
        // Alert proctor if resume count is suspicious
        if (resumeCount > 3) {
            alertProctor({
                student_id: studentId,
                message: `Student has left and returned ${resumeCount} times`,
                severity: 'high'
            });
        }
        
        // Show warning to student
        showStudentWarning(
            `Your session has been resumed. ` +
            `Please stay on this page until exam completion.`
        );
    }
    
    return data.session_id;
}

// Get session info to check resume count
async function checkSessionStatus(sessionId) {
    const response = await fetch(`http://localhost:8000/api/session/${sessionId}`);
    const data = await response.json();
    
    return {
        resumeCount: data.resume_count,
        lastActivity: data.last_activity,
        violations: data.violation_count
    };
}
```

**Resume Detection Features:**
- Automatically resumes existing sessions (no 400 error)
- Tracks inactivity duration between last activity and resume
- Logs `session_resume` violation if inactivity > 10 seconds
- Increments resume counter for suspicious pattern detection
- Includes resume events in final report with timestamps and durations

**Use Cases:**
- Student accidentally closes browser
- Network disconnection and reconnection
- Intentionally leaving to consult notes (violation)
- Switching tabs/windows (tracked via inactivity)

---

## Troubleshooting

### Common Issues

#### 1. WebSocket Connection Failed

```
Error: WebSocket connection failed
```

**Solutions:**
- Verify API server is running: `curl http://localhost:8000/health`
- Check session exists before connecting
- Ensure CORS is configured correctly
- Check firewall/network settings

#### 2. Webcam Access Denied

```
Error: NotAllowedError: Permission denied
```

**Solutions:**
- Request camera permission early in user flow
- Guide user to enable camera in browser settings
- Provide clear instructions
- Test in HTTPS context (required by many browsers)

#### 3. High Latency

**Solutions:**
- Reduce frame rate
- Lower image quality
- Use WebSocket instead of HTTP POST
- Check network bandwidth
- Optimize image resolution (640x480 recommended)

#### 4. False Positives

**Solutions:**
- Adjust thresholds in config
- Increase persistence time
- Disable overly sensitive features
- Consider lighting conditions

#### 5. Identity Verification Failing

**Solutions:**
- Use clear, well-lit profile photo
- Ensure consistent lighting during exam
- Use front-facing camera
- Avoid glasses glare

---

## Production Deployment

### Security Considerations

1. **HTTPS Required:** Use TLS for production
2. **Authentication:** Add JWT/OAuth to API
3. **Rate Limiting:** Prevent API abuse
4. **CORS Configuration:** Restrict allowed origins
5. **Data Privacy:** Comply with privacy regulations

### Infrastructure

```
┌─────────────────────────────────────────────┐
│          Load Balancer (HTTPS)             │
└─────────────────┬───────────────────────────┘
                  │
      ┌───────────┼───────────┐
      ▼           ▼           ▼
┌─────────┐ ┌─────────┐ ┌─────────┐
│ API     │ │ API     │ │ API     │
│ Server  │ │ Server  │ │ Server  │
│ Instance│ │ Instance│ │ Instance│
└─────────┘ └─────────┘ └─────────┘
```

### Scaling Recommendations

- **Single Exam:** 1 server (4 CPU, 8GB RAM)
- **10-50 concurrent:** 2-3 servers behind load balancer
- **50-200 concurrent:** 5-10 servers + CDN
- **200+ concurrent:** Kubernetes cluster with auto-scaling

### Monitoring

Monitor these metrics:
- Active sessions
- Frames processed per second
- Average latency
- Violation rate
- Error rate
- CPU/Memory usage

---

## Support & Resources

- **API Documentation:** http://localhost:8000/docs
- **GitHub Repository:** [Your repo URL]
- **Technical Support:** [Your support email]

---

## Example Code Repository

Complete working examples available at:
- React Integration: `/examples/react-integration/`
- Vue.js Integration: `/examples/vue-integration/`
- Vanilla JS: `/examples/vanilla-js/`
- Python Client: `/examples/python-client/`

---

## License

[Your License]

---

**Last Updated:** March 7, 2026
**API Version:** 1.1.0
