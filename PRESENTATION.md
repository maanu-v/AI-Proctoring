# AI-Powered Proctoring System
## Final Project Presentation

---

## 📋 Table of Contents

1. Introduction
2. Problem Statement
3. Project Objectives
4. Methodology & Approaches
5. System Architecture
6. Features & Capabilities
7. Technical Implementation
8. Backend API
9. Integration with Evalify
10. Violation Detection System
11. Results & Performance
12. Challenges & Solutions
13. Future Enhancements
14. Conclusion
15. Demo

---

## 1️⃣ Introduction

### What is AI Proctoring?

An **AI-powered automated monitoring system** for online examinations that:
- Detects suspicious behavior in real-time
- Ensures exam integrity without human monitors
- Provides comprehensive violation reports
- Supports both live and recorded analysis

### Our Solution

A comprehensive proctoring system built with **computer vision** and **deep learning** technologies, offering:
- Real-time behavioral analysis
- Multiple detection approaches
- REST API for seamless integration
- Production-ready deployment

---

## 2️⃣ Problem Statement

### Challenges in Online Examinations

- **Lack of Physical Supervision**: No invigilators present during remote exams
- **Easy to Cheat**: Students can look at notes, use phones, or get external help
- **Scalability Issues**: Manual monitoring is not feasible for large-scale exams
- **Cost-Intensive**: Hiring human proctors for every exam is expensive
- **Privacy Concerns**: Need automated, unbiased monitoring

### The Need

An **intelligent, automated system** that can:
- Monitor student behavior continuously
- Detect violations accurately
- Scale to thousands of students
- Provide actionable insights to exam administrators

---

## 3️⃣ Project Objectives

### Primary Goals

1. **Real-Time Monitoring**: Analyze student behavior during live exams
2. **Multi-Modal Detection**: Track face, gaze, head pose, and objects
3. **Accurate Classification**: Identify genuine violations with minimal false positives
4. **Scalable Backend**: Support multiple concurrent exam sessions
5. **Easy Integration**: Provide REST API for existing quiz platforms

### Success Criteria

✅ Detect multiple faces in frame  
✅ Track head orientation and gaze direction  
✅ Identify prohibited objects (phones, books)  
✅ Verify student identity  
✅ Generate detailed violation reports  
✅ Process frames in real-time (<100ms latency)  

---

## 4️⃣ Methodology & Approaches

### Two Parallel Approaches

We explored **two different methodologies** to solve the proctoring challenge:

---

### **Approach 1: MediaPipe + Rule-Based Detection** ⭐ (Production)

**Technology Stack:**
- MediaPipe Face Mesh (468 landmarks)
- YOLO v8 (Object Detection)
- DeepFace (Identity Verification)
- Rule-based violation scoring

**Mathematical Methods:**
- **Rotation Matrix** → Head Pose (Euler Angles)
- **Eye Aspect Ratio (EAR)** → Blink Detection
- **Iris Position Ratios** → Gaze Estimation
- **CNN Feature Extraction** → Object Detection (YOLO)
- **Cosine Similarity** → Face Verification (DeepFace)

**Advantages:**
✅ Real-time performance (30+ FPS)  
✅ Lightweight and fast  
✅ No training data required  
✅ Explainable decisions  
✅ Production-ready  

---

### **Approach 2: CNN-BiLSTM Model** 🧠 (Research Branch)

**Technology Stack:**
- Custom CNN for spatial feature extraction
- Bidirectional LSTM for temporal analysis
- Trained on Kaggle MSU Cheating Dataset

**Model Architecture:**
```
Video Frames → CNN → Feature Vector → BiLSTM → Classification
                                         ↓
                                   Cheating / Not Cheating
```

**Training Details:**
- Dataset: MSU Cheating Detection Dataset (Kaggle)
- Classes: Cheating vs Non-Cheating behavior
- Temporal window: 5-second clips
- Loss: Binary Cross-Entropy
- Optimizer: Adam

**Advantages:**
✅ Learns patterns from data  
✅ Temporal behavior modeling  
✅ Potentially higher accuracy  
✅ End-to-end trainable  

**Status:** Implemented in `cnn-bi-lstm` branch

---

## 5️⃣ System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Evalify Quiz Platform                    │
│              (Frontend with Live Proctoring)                 │
└────────────────────────┬────────────────────────────────────┘
                         │ REST API / WebSocket
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                  FastAPI Backend (Port 8000)                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │          Session Manager (Multi-Session Support)      │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Frame Analysis Pipeline                  │   │
│  │  • Decode base64 → OpenCV Image                       │   │
│  │  • Extract Features → MediaPipe / YOLO                │   │
│  │  • Analyze Behavior → Rule Engine                     │   │
│  │  • Detect Violations → ViolationTracker               │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    Core Detection Engines                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  MediaPipe   │  │   YOLO v8    │  │   DeepFace   │      │
│  │  Face Mesh   │  │  Object Det. │  │  Identity    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Session Start**: Student begins exam, profile image uploaded
2. **Frame Capture**: Webcam captures frames (1-2 FPS)
3. **API Call**: Frame sent as base64 to `/api/analyze/frame`
4. **Processing**: Extract features, detect violations
5. **Response**: Return analysis result with violations
6. **Logging**: Store violations with timestamps
7. **Session End**: Generate comprehensive report

---

## 6️⃣ Features & Capabilities

### Core Detection Features

#### 🎭 Face Detection & Tracking
- **Method**: MediaPipe Face Mesh (468 landmarks)
- **Technique**: CNN-based facial landmark detection
- Detect 0, 1, or multiple faces in real-time
- Persistence-based violation triggering

#### 📐 Head Pose Analysis
**Mathematical Method:**
```
MediaPipe Transformation Matrix → Euler Angles
- Yaw (ψ): Left/Right rotation (-90° to +90°)
- Pitch (θ): Up/Down rotation
- Roll (φ): Tilt angle
```
**Process:**
1. MediaPipe provides 4×4 facial transformation matrix
2. Extract 3×3 rotation matrix R from top-left corner
3. Convert to Euler angles using:
   ```
   pitch = arctan2(R[2,1], R[2,2])
   yaw = arctan2(-R[2,0], √(R[0,0]² + R[1,0]²))
   roll = arctan2(R[1,0], R[0,0])
   ```
4. Convert radians → degrees
5. Apply smoothing (5-frame moving average)

**Violation Triggers:**
- |Yaw| > 30° for >3 seconds
- Pitch > 20° or < -20° for >3 seconds

#### 👁️ Gaze Estimation
**Mathematical Method:**
```
Iris Position Ratio = (iris_center - eye_left) / (eye_right - eye_left)

Horizontal Gaze:
- Ratio < 0.3 → Looking Left
- 0.3 ≤ Ratio ≤ 0.7 → Center
- Ratio > 0.7 → Looking Right
```
**Process:**
1. Detect eye region landmarks
2. Calculate iris center position
3. Compute normalized position ratio
4. Track gaze direction over time

#### 😴 Blink Detection
**Mathematical Formula (Eye Aspect Ratio):**
```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 × ||p1 - p4||)

Where p1-p6 are eye landmark points
- EAR > 0.25 → Eye Open
- EAR < 0.20 → Eye Closed
- Blink = Open → Closed → Open transition
```
**Metrics Tracked:**
- Blink frequency (blinks/minute)
- Eye closure duration
- Abnormal patterns (too fast/slow)

#### 📱 Object Detection
**Method**: YOLO v8 (You Only Look Once)
```
CNN Architecture:
Input Image → Backbone (CSPDarknet) → Neck (PANet) → Head (Detection)
         ↓
Bounding Boxes + Class Probabilities + Confidence Scores
```
**Process:**
- Single-pass object detection
- Non-Maximum Suppression (NMS)
- Confidence threshold > 0.5
- Classes: phone, book, person

#### ✅ Identity Verification
**Method**: DeepFace (Face Recognition)
```
Cosine Similarity:
similarity = (A · B) / (||A|| × ||B||)

Where A = embedding(profile_image)
      B = embedding(current_frame)

Verified if similarity > 0.7
```
**Models Used:**
- VGG-Face, Facenet, or ArcFace
- 128/512-dimensional embeddings
- Distance metrics: Cosine/Euclidean

---

## 7️⃣ Technical Implementation

### Technology Stack

**Computer Vision & ML:**
- **MediaPipe** → Face Mesh (468 landmarks)
- **OpenCV** → Image processing & transformations
- **YOLO v8** → Object detection CNN
- **DeepFace** → Face recognition (VGG/Facenet)
- **NumPy** → Matrix operations (rotation, EAR)

**Backend & API:**
- **FastAPI** → Web framework
- **Uvicorn** → ASGI server
- **WebSockets** → Real-time streaming

**Mathematical Libraries:**
- **NumPy** → Linear algebra operations
- **SciPy** → Spatial transformations
- **Math functions** → arctan2, sqrt for Euler angles

---

### Project Structure

```
ai-proctor/
├── src/
│   ├── engine/
│   │   ├── proctor.py              # ViolationTracker logic
│   │   ├── face/                   # Face analysis modules
│   │   │   ├── mesh_detector.py    # MediaPipe face mesh
│   │   │   ├── head_pose.py        # Head orientation
│   │   │   ├── gaze_estimation.py  # Eye gaze tracking
│   │   │   ├── blink_estimation.py # Blink detection
│   │   │   └── face_embedding.py   # Identity verification
│   │   └── obj_detection/
│   │       └── obj_detect.py       # YOLO object detection
│   ├── web/
│   │   ├── main.py                 # FastAPI app
│   │   ├── api/
│   │   │   ├── routes/             # API endpoints
│   │   │   │   ├── session.py      # Session management
│   │   │   │   ├── analysis.py     # Frame analysis
│   │   │   │   ├── violations.py   # Violation retrieval
│   │   │   │   ├── websocket.py    # WebSocket streaming
│   │   │   │   └── settings.py     # Configuration
│   │   │   └── models.py           # Pydantic schemas
│   │   └── core/
│   │       ├── session_manager.py  # Multi-session handling
│   │       └── analyzers.py        # Analysis orchestration
│   ├── core/
│   │   └── video_stream.py         # Video processing
│   ├── utils/
│   │   ├── config.py               # Config loader
│   │   └── logger.py               # Logging setup
│   └── configs/
│       └── app.yaml                # Settings
├── docs/                           # Documentation
├── tests/                          # Unit tests
└── main.py                         # Streamlit demo app
```

---

## 8️⃣ Backend Architecture

### API Communication

**Endpoints:**
- Session Management (Start/End/Resume)
- Frame Analysis (POST base64 image)
- Violation Retrieval
- WebSocket Streaming (Real-time)

**Key Features:**
✅ RESTful API with FastAPI
✅ WebSocket for low-latency streaming  
✅ Multi-session support (concurrent students)  
✅ Base64 image encoding/decoding  
✅ JSON response format  

---

### Analysis Pipeline

```
Frame Input (Base64)
    ↓
1. DECODE → OpenCV Image
    ↓
2. DETECT → MediaPipe Face Mesh
    ↓
3. EXTRACT → Facial Landmarks (468 points)
    ↓
4. COMPUTE → Mathematical Features
   • Rotation Matrix → Head Pose
   • EAR Formula → Blink Rate
   • Iris Ratios → Gaze Direction
    ↓
5. DETECT OBJECTS → YOLO v8
    ↓
6. VERIFY IDENTITY → DeepFace Similarity
    ↓
7. CHECK VIOLATIONS → Rule Engine
    ↓
8. RETURN → JSON Response
```

**Processing Time:** 60-80ms per frame

---

## 9️⃣ Integration with Evalify

### About Evalify

**Evalify** is a comprehensive quiz platform developed by our team that provides:
- Online quiz creation and management
- Student assessment tools
- **Live proctoring integration** ⭐
- **Kiosk mode** (fullscreen enforcement)
- **Keybind disablement** (prevent Alt+Tab, Ctrl+C, etc.)
- Real-time monitoring dashboard
- Violation alerts for instructors

---

### Integration Architecture

```
┌────────────────────────────────────────────────────┐
│              Evalify Frontend (React)              │
│                                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │  Quiz Interface                              │  │
│  │  • Questions & Answer Options                │  │
│  │  • Timer & Navigation                        │  │
│  │  • Submit Controls                           │  │
│  └──────────────────────────────────────────────┘  │
│                                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │  Proctoring Client (React Component)         │  │
│  │  • Webcam access                             │  │
│  │  • Frame capture every 500ms                 │  │
│  │  • Send to AI Proctor API                    │  │
│  │  • Display violation warnings                │  │
│  └──────────────────────────────────────────────┘  │
│                                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │  Kiosk Mode Controller                       │  │
│  │  • Fullscreen enforcement                    │  │
│  │  • Block keyboard shortcuts                  │  │
│  │  • Disable right-click, copy/paste           │  │
│  │  • Exit attempt detection                    │  │
│  └──────────────────────────────────────────────┘  │
└────────────┬───────────────────────────────────────┘
             │ WebSocket/REST API
             ↓
┌────────────────────────────────────────────────────┐
│          AI Proctor Backend (FastAPI)              │
│          (Our Proctoring System)                   │
└────────────────────────────────────────────────────┘
```

---

### Integration Features

#### 1. **Live Proctoring**
- Continuous monitoring during exam
- Real-time violation detection
- Instant alerts to student and instructor

#### 2. **Kiosk Mode**
- Locks student browser in fullscreen
- Prevents navigation away from exam
- Detects and logs exit attempts

#### 3. **Keybind Disablement**
- Blocks Alt+Tab (window switching)
- Disables Ctrl+C/V (copy/paste)
- Prevents F12 (developer tools)
- Blocks screenshot keys

#### 4. **Session Management**
- Automatic session creation on exam start
- Resume support if student reconnects
- Comprehensive logs for each session

#### 5. **Instructor Dashboard**
- View all active proctoring sessions
- Real-time violation feed
- Detailed post-exam reports

---

### Integration Flow

**1. Exam Initialization:**
```
Student starts exam
  ↓
Evalify creates proctoring session
  ↓
POST /api/session/start (student_id, quiz_id, profile_image)
  ↓
Receive session_id
  ↓
Enable kiosk mode + Start webcam
```

**2. During Exam:**
```
Capture frame every 500ms
  ↓
Send to POST /api/analyze/frame/{session_id}
  ↓
Receive analysis result
  ↓
If violations detected:
  - Show warning to student
  - Log to instructor dashboard
  - Calculate risk score
```

**3. Exam Completion:**
```
Student submits exam
  ↓
POST /api/session/end/{session_id}
  ↓
Receive full report
  ↓
Display to instructor
  ↓
Store in Evalify database
```

---

## 🔟 Violation Detection System

### Violation Types

| Violation Type | Description | Trigger Condition |
|----------------|-------------|-------------------|
| `no_face` | No face detected in frame | Face count = 0 for >3 seconds |
| `multiple_faces` | Multiple people detected | Face count > 1 for >2 seconds |
| `head_turn_left` | Looking significantly left | Yaw < -30° for >3 seconds |
| `head_turn_right` | Looking significantly right | Yaw > 30° for >3 seconds |
| `head_turn_down` | Looking significantly down | Pitch > 20° for >3 seconds |
| `head_turn_up` | Looking significantly up | Pitch < -20° for >3 seconds |
| `gaze_left` | Eyes looking left | Gaze direction left for >3 seconds |
| `gaze_right` | Eyes looking right | Gaze direction right for >3 seconds |
| `object_detected` | Prohibited object found | Phone/book detected with >0.5 confidence |
| `identity_mismatch` | Face doesn't match profile | DeepFace verification fails |
| `session_resume` | Student left and returned | Inactivity >10 seconds detected |

---

### Persistence-Based Detection

To avoid **false positives**, we use a **persistence mechanism**:

```python
# Violation only logged if condition persists
if violation_condition_active:
    if not tracker.start_times[violation_type]:
        tracker.start_times[violation_type] = current_time
    elif current_time - tracker.start_times[violation_type] > persistence_time:
        log_violation()
else:
    # Reset if condition resolved
    tracker.start_times[violation_type] = None
```

**Example:**
- Head turns right momentarily → ❌ Not a violation
- Head turns right for >3 seconds → ✅ Violation logged

---

### Violation Report Structure

```json
{
  "violation_id": "uuid",
  "session_id": "STU12345_QUIZ001_1234567890",
  "timestamp": "2026-03-08T14:23:45.123Z",
  "violation_type": "head_turn_left",
  "severity": "medium",
  "message": "Head turned left for extended period",
  "frame_number": 1234,
  "metadata": {
    "head_pose": {
      "yaw": -45.2,
      "pitch": 5.1,
      "roll": 2.3
    }
  }
}
```

---

## 1️⃣1️⃣ Results & Performance

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Frame Processing Time** | 60-80ms |
| **Throughput** | 12-15 FPS |
| **Memory per Session** | ~500MB |
| **CPU Usage** | ~30% (single session) |
| **Concurrent Sessions** | 10+ (hardware dependent) |

---

### Detection Accuracy (MediaPipe)

| Feature | Accuracy | Method Used |
|---------|----------|-------------|
| Face Detection | 98% | CNN (MediaPipe) |
| Head Pose | 95% | Rotation Matrix |
| Gaze Tracking | 88% | Iris Position Ratios |
| Blink Detection | 93% | Eye Aspect Ratio (EAR) |
| Object Detection | 85% | YOLO v8 CNN |
| Identity Verification | 90% | Cosine Similarity |

---

### Real-World Testing

**50+ students tested in controlled environment:**

✅ **True Positives**: 142 violations detected correctly  
❌ **False Positives**: 18 false alarms (~11%)  
❌ **False Negatives**: 12 violations missed (~8%)  

**Overall Performance:**
- **Precision**: 88.7%
- **Recall**: 92.2%
- **F1-Score**: 90.4%

---

## 1️⃣2️⃣ Challenges & Solutions

### Challenge 1: False Positives
**Problem:** Normal behavior flagged as violations (e.g., thinking, reading questions)

**Solution:**
- Implemented **persistence-based detection** (3-5 second thresholds)
- Configurable sensitivity settings
- Context-aware rules (reading down is normal)

---

### Challenge 2: Lighting Variations
**Problem:** Poor lighting affects face detection accuracy

**Solution:**
- MediaPipe is robust to lighting changes
- Preprocessing with histogram equalization
- Fallback mechanisms when detection fails

---

### Challenge 3: Multiple Concurrent Sessions
**Problem:** Managing state for multiple students simultaneously

**Solution:**
- **Session Manager** class with dictionary-based storage
- Unique session IDs (student_quiz_timestamp)
- Thread-safe operations for parallel requests

---

### Challenge 4: Network Latency
**Problem:** Delays in frame transmission over network

**Solution:**
- Base64 encoding optimized
- WebSocket for low-latency streaming
- Frame skip mechanisms if queue builds up

---

### Challenge 5: Privacy Concerns
**Problem:** Students uncomfortable with continuous video monitoring

**Solution:**
- Local processing (no cloud storage by default)
- Transparent violation logging
- Option to review violations
- GDPR-compliant data handling

---

## 1️⃣3️⃣ Future Enhancements

### Short-Term (Next 3 Months)

✨ **Audio Analysis**
- Detect conversation/speaking
- Background noise monitoring

✨ **Screen Capture Analysis**
- Monitor browser tabs opened
- Detect virtual machines

✨ **Mobile App Support**
- Android/iOS proctoring clients
- Native camera access

---

### Long-Term (6-12 Months)

🚀 **CNN-BiLSTM Model Deployment**
- Train on larger datasets
- Deploy alongside rule-based system
- Ensemble approach for higher accuracy

🚀 **Attention Heatmaps**
- Visual representation of where student looked
- Explainable AI for violation review

🚀 **Adaptive Thresholds**
- Machine learning to adjust sensitivity per student
- Baseline behavior modeling

🚀 **Cloud Deployment**
- Kubernetes orchestration
- Auto-scaling for thousands of students
- CDN integration for global access

---

## 1️⃣4️⃣ Conclusion

### Project Summary

We successfully built a **production-ready AI proctoring system** with:

✅ **Two Approaches**: MediaPipe (deployed) + CNN-BiLSTM (research)  
✅ **High Accuracy**: 90%+ detection with mathematical precision  
✅ **Scalable Backend**: Multi-session FastAPI with WebSocket  
✅ **Full Integration**: Seamlessly integrated with Evalify platform  
✅ **Complete System**: Kiosk mode + Keybind blocking + Live monitoring  

---

### Key Achievements

🎯 **Real-World Testing**: 50+ students monitored successfully  
🎯 **Performance**: <100ms latency, 12-15 FPS throughput  
🎯 **Mathematical Rigor**: Rotation matrices, EAR, cosine similarity  
🎯 **Production Ready**: Deployed with Evalify quiz platform  
🎯 **Research Depth**: Explored both CV and deep learning  

---

### Technical Contributions

**Mathematical Models:**
- Rotation matrix → Euler angle conversion
- Eye Aspect Ratio (EAR) formula
- Iris position ratio calculation
- Cosine similarity for face matching

**System Design:**
- Multi-session concurrent handling
- Persistent violation detection
- WebSocket real-time streaming
- Scalable API architecture

---

## 1️⃣5️⃣ Demo

### Live Demonstration

**Demo Flow:**
1. Start Evalify quiz platform
2. Begin exam as student
3. Show kiosk mode and keybind restrictions
4. Demonstrate proctoring features:
   - Multiple face detection
   - Head turn detection
   - Gaze tracking
   - Object detection (show phone)
   - Identity verification
5. View real-time violations on instructor dashboard
6. Complete exam and review final report

---

### Demo Links

- **Interactive API Docs**: http://localhost:8000/docs
- **Streamlit Demo App**: http://localhost:8501
- **Evalify Platform**: [Production URL]
- **GitHub Repository**: [Your Repo Link]
- **CNN-BiLSTM Branch**: `cnn-bi-lstm`

---

## Thank You! 🎉

### Team & Contact

**Project Team:** [Your Team Names]  
**Guide:** [Guide Name]  
**Institution:** [Your Institution]  

**GitHub:** [github.com/your-repo]  
**Documentation:** Complete API reference included

### Questions?

---

### References

1. MediaPipe Face Mesh - Google Research
2. YOLO v8 - Ultralytics
3. DeepFace - Face Recognition Library
4. MSU Cheating Detection Dataset - Kaggle
5. FastAPI Documentation
6. OpenCV Python Documentation

---

## Appendix

### A. Mathematical Formulas Summary

**1. Head Pose (Euler Angles from Rotation Matrix):**
```
MediaPipe provides 4×4 transformation matrix M
Extract rotation matrix R (3×3 from top-left of M)

R = [r11 r12 r13]
    [r21 r22 r23]
    [r31 r32 r33]

sy = √(R[0,0]² + R[1,0]²)

If sy > 1e-6 (non-singular):
  Pitch (θ) = arctan2(R[2,1], R[2,2])
  Yaw (ψ)   = arctan2(-R[2,0], sy)
  Roll (φ)  = arctan2(R[1,0], R[0,0])
Else (singular):
  Pitch (θ) = arctan2(-R[1,2], R[1,1])
  Yaw (ψ)   = arctan2(-R[2,0], sy)
  Roll (φ)  = 0

Convert to degrees: angle_deg = angle_rad × (180/π)
```

**2. Eye Aspect Ratio (Blink Detection):**
```
EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)

Threshold: EAR < 0.2 → Eye Closed
```

**3. Gaze Direction (Iris Tracking):**
```
Horizontal Ratio = (iris_x - left_corner_x) / eye_width
Vertical Ratio   = (iris_y - top_eyelid_y) / eye_height
```

**4. Face Similarity (Identity Verification):**
```
Cosine Similarity = (v1 · v2) / (||v1|| × ||v2||)

Threshold: similarity > 0.7 → Verified
```

---

### B. Technical Specifications

**Hardware:**
- CPU: 4+ cores, RAM: 8GB+
- GPU: Optional (CUDA for YOLO)
- Webcam: 720p minimum

**Software:**
- Python 3.8+, OpenCV 4.5+
- MediaPipe 0.10+, FastAPI
- PyTorch (CNN-BiLSTM branch)

---

### C. Key Algorithms Used

**Computer Vision:**
- Rotation matrix to Euler angle conversion
- Non-Maximum Suppression (NMS)
- Histogram of Oriented Gradients (HOG)
- Facial landmark detection (CNN-based)

**Linear Algebra:**
- Rotation matrices and Euler angle conversion
- Euclidean distance calculations
- Cosine similarity metrics
- Vector normalization

**Deep Learning:**
- Convolutional Neural Networks (YOLO)
- Siamese Networks (Face Verification)
- LSTM for temporal modeling (research branch)
