# AI Proctor - Intelligent Exam Monitoring System

<div align="center">

![AI Proctor](https://img.shields.io/badge/AI-Proctor-667eea?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.12+-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.127+-009688?style=for-the-badge&logo=fastapi)
![OpenCV](https://img.shields.io/badge/OpenCV-4.12+-5C3EE8?style=for-the-badge&logo=opencv)

**Advanced AI-powered proctoring system with real-time monitoring capabilities**

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Architecture](#architecture) • [API](#api-documentation)

</div>

---

## 🎯 Overview

AI Proctor is a modern, real-time exam monitoring system that leverages computer vision and AI to ensure exam integrity. Built with a singleton video stream architecture, it provides efficient, thread-safe video processing with a beautiful, responsive web interface.

## ✨ Features

### Current Implementation

- ✅ **Singleton Video Stream** - Thread-safe, efficient camera management
- ✅ **Real-time Video Streaming** - HTTP multipart streaming for live feed
- ✅ **WebSocket Communication** - Low-latency control messages
- ✅ **Modern Web UI** - Premium dark theme with glassmorphism
- ✅ **Session Tracking** - Automatic session timer and monitoring
- ✅ **Responsive Design** - Works on desktop, tablet, and mobile
- ✅ **Health Monitoring** - Real-time system status checks
- ✅ **Toast Notifications** - User-friendly feedback system

### AI Features (Ready for Integration)

- 🔄 **Face Detection** - Detect and track faces in video stream
- 🔄 **Gaze Tracking** - Monitor eye movement and attention
- 🔄 **Head Pose Estimation** - Track head orientation
- 🔄 **Mouth Activity Detection** - Detect speech and mouth movements
- 🔄 **Person Counter** - Count number of people in frame
- 🔄 **Object Detection** - Identify prohibited objects

## 🚀 Installation

### Prerequisites

- Python 3.12 or higher
- Webcam or video input device
- UV package manager (recommended) or pip

### Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd ai-proctor

# Install dependencies with UV (recommended)
uv sync

# Or with pip
pip install -r requirements.txt

# Run the application
uv run python main.py

# Or with python directly
python main.py
```

The application will start on `http://localhost:8000`

## 📖 Usage

### Starting the Server

```bash
# Development mode (with auto-reload)
uv run python main.py

# Or using uvicorn directly
uv run uvicorn src.web.app:app --host 0.0.0.0 --port 8000 --reload
```

### Accessing the Web Interface

1. Open your browser and navigate to `http://localhost:8000`
2. The live video feed will start automatically
3. Use the sidebar to toggle AI features (UI ready, backend integration pending)
4. Monitor session statistics in real-time

### Taking Snapshots

Click the camera icon in the video controls to capture and download a snapshot.

### Fullscreen Mode

Click the fullscreen icon in the header to enter fullscreen mode.

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────┐
│         Frontend (Browser)              │
│  ┌─────────────────────────────────┐   │
│  │  HTML5 + CSS3 + JavaScript      │   │
│  │  - Video Display                │   │
│  │  - Feature Controls             │   │
│  │  - WebSocket Client             │   │
│  └─────────────────────────────────┘   │
└──────────────┬──────────────────────────┘
               │ HTTP/WebSocket
┌──────────────┴──────────────────────────┐
│         Backend (FastAPI)               │
│  ┌─────────────────────────────────┐   │
│  │  FastAPI Application            │   │
│  │  - Video Streaming Endpoint     │   │
│  │  - WebSocket Control            │   │
│  │  - REST API                     │   │
│  └─────────────────────────────────┘   │
└──────────────┬──────────────────────────┘
               │
┌──────────────┴──────────────────────────┐
│      Core (VideoStream Singleton)       │
│  ┌─────────────────────────────────┐   │
│  │  Thread-Safe Video Capture      │   │
│  │  - OpenCV Integration           │   │
│  │  - Frame Buffer                 │   │
│  │  - JPEG Encoding                │   │
│  └─────────────────────────────────┘   │
└──────────────┬──────────────────────────┘
               │
         ┌─────┴─────┐
         │  Camera   │
         └───────────┘
```

### Project Structure

```
ai-proctor/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   └── video_stream.py      # Singleton video stream
│   ├── web/
│   │   ├── __init__.py
│   │   ├── app.py                # FastAPI application
│   │   └── static/
│   │       ├── index.html        # Web UI
│   │       ├── styles.css        # Styling
│   │       └── app.js            # Frontend logic
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py             # Logging utilities
│   │   └── config.py             # Configuration management
│   └── configs/
│       └── app.yaml              # Application configuration
├── main.py                       # Entry point
├── pyproject.toml                # Dependencies
├── IMPLEMENTATION.md             # Implementation details
└── README.md                     # This file
```

## 🔧 Configuration

Edit `src/configs/app.yaml` to customize settings:

```yaml
app:
  name: "AI Proctor"
  version: "1.0.0"
  debug: true
  host: "0.0.0.0"
  port: 8000

camera:
  source: 0              # 0 for default webcam, or path to video file
  width: 1280
  height: 720
  fps: 30

logging:
  level: "INFO"
  use_rich: true
```

## 📡 API Documentation

### HTTP Endpoints

#### `GET /`
Serve the main web interface.

#### `GET /api/health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "app_name": "AI Proctor",
  "version": "1.0.0",
  "video_stream_active": true
}
```

#### `GET /api/camera/properties`
Get camera properties.

**Response:**
```json
{
  "width": 1280,
  "height": 720,
  "fps": 30,
  "is_running": true,
  "source": 0
}
```

#### `GET /api/video/feed`
Multipart JPEG video stream.

**Response:** `multipart/x-mixed-replace` stream

#### `GET /api/features`
List available AI proctoring features.

**Response:**
```json
{
  "features": [
    {
      "id": "face_detection",
      "name": "Face Detection",
      "description": "Detect faces in the video stream",
      "enabled": false
    },
    ...
  ]
}
```

### WebSocket Endpoints

#### `WS /ws/control`
Control messages and status updates.

**Client → Server:**
```json
{
  "action": "toggle_feature",
  "feature": "face_detection",
  "enabled": true
}
```

**Server → Client:**
```json
{
  "type": "ack",
  "action": "toggle_feature",
  "feature": "face_detection",
  "enabled": true
}
```

#### `WS /ws/video`
Alternative WebSocket video streaming (not currently used by frontend).

## 🎨 UI Features

### Design Highlights

- **Premium Dark Theme** - Easy on the eyes during long monitoring sessions
- **Glassmorphism Effects** - Modern, translucent UI elements
- **Smooth Animations** - Micro-interactions for better UX
- **Responsive Layout** - Adapts to different screen sizes
- **Custom Scrollbars** - Styled to match the theme
- **Gradient Accents** - Beautiful color transitions

### Color Palette

- Primary: `#667eea` (Purple-blue)
- Secondary: `#764ba2` (Deep purple)
- Background: `#0a0e27` (Dark blue-black)
- Text: `#e2e8f0` (Light gray)

## 🔐 Security Considerations

- [ ] Add authentication and authorization
- [ ] Implement HTTPS/WSS for production
- [ ] Add CSRF protection
- [ ] Implement rate limiting
- [ ] Add input validation and sanitization
- [ ] Secure WebSocket connections

## 🚧 Roadmap

### Phase 1: Core Features (Current)
- ✅ Video streaming infrastructure
- ✅ Web interface
- ✅ WebSocket communication

### Phase 2: AI Integration (Next)
- [ ] MediaPipe face detection
- [ ] Gaze tracking implementation
- [ ] Head pose estimation
- [ ] Mouth activity detection

### Phase 3: Advanced Features
- [ ] Recording and playback
- [ ] Alert system with notifications
- [ ] Violation logging and reporting
- [ ] Multi-session support
- [ ] Admin dashboard

### Phase 4: Production Ready
- [ ] Authentication system
- [ ] Database integration
- [ ] Cloud deployment
- [ ] Performance optimization
- [ ] Comprehensive testing

## 🛠️ Development

### Running Tests

```bash
# Coming soon
pytest tests/
```

### Code Style

```bash
# Format code
black src/

# Lint code
ruff check src/
```

## 📝 License

[Add your license here]

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

[Add your contact information]

## 🙏 Acknowledgments

- FastAPI for the excellent web framework
- OpenCV for video processing capabilities
- MediaPipe for AI/ML models (ready to integrate)
- Rich for beautiful terminal logging

---

<div align="center">

**Built with ❤️ for secure and fair examinations**

</div>
