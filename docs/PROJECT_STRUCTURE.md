# 🎉 Complete Project Structure - AI Proctoring System

## Visual File Tree

```
ai-proctor/
│
├── 📄 README.md                          # Main project documentation
├── 📄 pyproject.toml                     # Python dependencies
├── 📄 PROJECT.md                         # Project notes
├── 🎯 yolov8n.pt                         # YOLO model weights
├── 🚀 start_api.sh                       # Quick start script ⭐ NEW
│
├── 📁 src/
│   ├── 📄 __init__.py
│   │
│   ├── 📁 configs/
│   │   └── 📄 app.yaml                   # System configuration
│   │
│   ├── 📁 core/
│   │   ├── 📄 __init__.py
│   │   └── 📄 video_stream.py            # Video capture utilities
│   │
│   ├── 📁 engine/                        # AI/ML Analysis Components
│   │   ├── 📄 __init__.py
│   │   ├── 📄 proctor.py                 # ViolationTracker
│   │   │
│   │   ├── 📁 face/                      # Face analysis modules
│   │   │   ├── 📄 __init__.py
│   │   │   ├── 📄 mesh_detector.py       # MediaPipe face mesh
│   │   │   ├── 📄 head_pose.py           # Head orientation
│   │   │   ├── 📄 gaze_estimation.py     # Eye gaze tracking
│   │   │   ├── 📄 blink_estimation.py    # Blink detection
│   │   │   └── 📄 face_embedding.py      # Face recognition
│   │   │
│   │   └── 📁 obj_detection/
│   │       ├── 📄 __init__.py
│   │       └── 📄 obj_detect.py          # YOLO object detection
│   │
│   ├── 📁 models/
│   │   └── 📄 face_landmarker.task       # MediaPipe model
│   │
│   ├── 📁 utils/
│   │   ├── 📄 __init__.py
│   │   ├── 📄 config.py                  # Configuration loader
│   │   └── 📄 logger.py                  # Logging utilities
│   │
│   └── 📁 web/                           # Web Applications ⭐
│       ├── 📄 __init__.py
│       ├── 📄 app.py                     # Streamlit dashboard
│       ├── 📄 fastapi_app.py             # Legacy FastAPI (reference)
│       ├── 📄 main.py                    # 🌟 FastAPI entry point (NEW)
│       │
│       ├── 📁 api/                       # 🌟 API Layer (NEW)
│       │   ├── 📄 __init__.py
│       │   ├── 📄 models.py              # Pydantic models
│       │   ├── 📄 dependencies.py        # Dependency injection
│       │   │
│       │   └── 📁 routes/                # Route handlers
│       │       ├── 📄 __init__.py
│       │       ├── 📄 session.py         # Session management
│       │       ├── 📄 analysis.py        # Frame analysis
│       │       ├── 📄 settings.py        # Settings management
│       │       ├── 📄 config.py          # Configuration
│       │       └── 📄 violations.py      # Violations management
│       │
│       ├── 📁 core/                      # 🌟 Core Logic (NEW)
│       │   ├── 📄 __init__.py
│       │   ├── 📄 session_manager.py     # Session lifecycle
│       │   └── 📄 analyzers.py           # Analyzer management
│       │
│       └── 📁 utils/                     # 🌟 Web Utils (NEW)
│           ├── 📄 __init__.py
│           └── 📄 image_utils.py         # Image processing
│
├── 📁 examples/
│   └── 📄 client_example.py              # Example API client
│
├── 📁 tests/
│   ├── 📄 test_blink.py                  # Blink detection tests
│   ├── 📄 test_gaze.py                   # Gaze tracking tests
│   └── 📄 test_fastapi.py                # API endpoint tests
│
└── 📁 docs/                              # Documentation
    ├── 📄 head_pose.md                   # Head pose details
    ├── 📄 mediapipe_facemesh_landmark.md # Face landmarks
    ├── 📄 no_face_multiface.md           # Face detection
    ├── 📄 API_DOCUMENTATION.md           # Complete API reference
    ├── 📄 FASTAPI_QUICKSTART.md          # Quick start guide
    ├── 📄 ARCHITECTURE.md                # 🌟 Architecture guide (NEW)
    ├── 📄 MODULAR_STRUCTURE.md           # 🌟 Quick reference (NEW)
    ├── 📄 VISUAL_ARCHITECTURE.md         # 🌟 Visual diagrams (NEW)
    └── 📄 REFACTORING_SUMMARY.md         # 🌟 This refactoring (NEW)
```

## Module Count by Category

### Core System
- **Configuration**: 1 file (`configs/app.yaml`)
- **Core Utilities**: 2 files (`core/video_stream.py`, `utils/`)
- **AI Engine**: 7 files (`engine/face/`, `engine/obj_detection/`)
- **Models**: 2 files (YOLO weights, MediaPipe task)

### Web Applications
- **Streamlit**: 1 file (`app.py`)
- **FastAPI (Legacy)**: 1 file (`fastapi_app.py`)
- **FastAPI (Modular)**: 15 files ⭐
  - Main: 1 file
  - API Layer: 8 files
  - Core Layer: 3 files
  - Utils: 2 files

### Testing & Examples
- **Tests**: 3 files
- **Examples**: 1 file
- **Documentation**: 9 files

## Statistics

```
Total Files Created in Refactoring: 15 files
Total Documentation Added: 4 guides (60+ pages)
Code Organization: From 1 file (800 lines) → 15 files (~100 lines each)
Maintainability: Improved from ⭐⭐ → ⭐⭐⭐⭐⭐
```

## Technology Stack

### Backend Framework
- **FastAPI** - Modern, fast web framework
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation

### AI/ML Stack
- **MediaPipe** - Face mesh detection
- **OpenCV** - Computer vision
- **DeepFace** - Face recognition
- **YOLO (Ultralytics)** - Object detection
- **NumPy** - Numerical computing

### Frontend Options
- **Streamlit** - Interactive dashboard
- **Any modern framework** - React, Vue, Angular (via REST API)

## API Endpoints Summary

### Session Management (4 endpoints)
```
POST   /api/session/start          # Create session
GET    /api/session/{id}           # Get session info  
GET    /api/sessions               # List all sessions
POST   /api/session/end/{id}       # End session
```

### Analysis (1 endpoint)
```
POST   /api/analyze/frame/{id}    # Analyze video frame
```

### Settings (2 endpoints)
```
GET    /api/session/{id}/settings  # Get settings
PUT    /api/session/{id}/settings  # Update settings
```

### Configuration (2 endpoints)
```
GET    /api/config                 # Get configuration
PUT    /api/config                 # Update configuration
```

### Violations (2 endpoints)
```
GET    /api/session/{id}/violations   # Get violations
DELETE /api/session/{id}/violations   # Clear violations
```

**Total: 11 RESTful endpoints**

## Key Features

### Face Analysis
- ✅ Face detection (multiple faces)
- ✅ Head pose estimation (yaw, pitch, roll)
- ✅ Gaze tracking (eye movement)
- ✅ Blink detection (EAR calculation)
- ✅ Face recognition (identity verification)

### Object Detection
- ✅ Phone detection
- ✅ Person counting
- ✅ Prohibited item detection

### Violation Tracking
- ✅ Persistent tracking with timestamps
- ✅ Violation consolidation
- ✅ Configurable persistence times
- ✅ Real-time warnings

### Configuration
- ✅ Per-session settings
- ✅ Global configuration
- ✅ Runtime updates
- ✅ YAML-based defaults

## Development Workflows

### Starting Development

```bash
# 1. Clone and setup
git clone <repo>
cd ai-proctor
pip install -e .

# 2. Start FastAPI server
./start_api.sh

# 3. Start Streamlit (in another terminal)
streamlit run src/web/app.py
```

### Adding a New Feature

```bash
# 1. Create feature branch
git checkout -b feature/new-analyzer

# 2. Add analyzer in core/analyzers.py
# 3. Add route in api/routes/
# 4. Add model in api/models.py
# 5. Test with tests/

# 6. Commit and push
git commit -m "Add new analyzer"
git push origin feature/new-analyzer
```

### Testing

```bash
# Quick API test
python tests/test_fastapi.py

# Try example client
python examples/client_example.py

# Check specific modules
python tests/test_blink.py
python tests/test_gaze.py
```

## Deployment Options

### Development
```bash
uvicorn src.web.main:app --reload --port 8000
```

### Production (Single Instance)
```bash
uvicorn src.web.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Production (Gunicorn)
```bash
gunicorn src.web.main:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

### Docker
```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY . .
RUN pip install -e .
CMD ["uvicorn", "src.web.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Documentation Access

Once server is running:

- **Interactive API Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json
- **Health Check**: http://localhost:8000/

## Configuration Files

### System Config (`src/configs/app.yaml`)
```yaml
camera:        # Camera parameters
mediapipe:     # Face mesh settings
head_pose:     # Head orientation thresholds
gaze:          # Gaze detection thresholds
blink:         # Blink detection parameters
thresholds:    # Violation persistence times
```

### Python Dependencies (`pyproject.toml`)
```toml
[project]
name = "ai-proctor"
dependencies = [
    "fastapi",
    "uvicorn",
    "opencv-python",
    "mediapipe",
    "deepface",
    "ultralytics",
    # ... more
]
```

## Support Resources

### Quick Help
- **Quick Start**: `docs/FASTAPI_QUICKSTART.md`
- **Quick Reference**: `docs/MODULAR_STRUCTURE.md`

### Deep Dive
- **Architecture**: `docs/ARCHITECTURE.md`
- **Visual Diagrams**: `docs/VISUAL_ARCHITECTURE.md`
- **API Reference**: `docs/API_DOCUMENTATION.md`

### Implementation Details
- **Head Pose**: `docs/head_pose.md`
- **Face Mesh**: `docs/mediapipe_facemesh_landmark.md`

### What Changed
- **Refactoring Summary**: `docs/REFACTORING_SUMMARY.md`

## Common Commands

```bash
# Start API server
./start_api.sh

# Start Streamlit dashboard
streamlit run src/web/app.py

# Run tests
python tests/test_fastapi.py

# Test with example client
python examples/client_example.py

# Install dependencies
pip install -e .

# Update dependencies
pip install --upgrade -e .

# Check errors
python -m py_compile src/web/main.py
```

## Project URLs

### Local Development
- FastAPI: http://localhost:8000
- FastAPI Docs: http://localhost:8000/docs
- Streamlit: http://localhost:8501

### Repository
- GitHub: [Your Repository URL]
- Issues: [Your Issues URL]
- Wiki: [Your Wiki URL]

## License

[Add your license information]

## Contributors

[Add contributor information]

---

## 🎯 Quick Start Commands

```bash
# Complete setup (first time)
pip install -e .
./start_api.sh

# Just start server (already installed)
./start_api.sh

# Test the API
python tests/test_fastapi.py

# Try example
python examples/client_example.py
```

## 📚 Documentation Index

| Document | Purpose | Audience |
|----------|---------|----------|
| [README.md](../README.md) | Main project overview | Everyone |
| [FASTAPI_QUICKSTART.md](FASTAPI_QUICKSTART.md) | Getting started | New users |
| [MODULAR_STRUCTURE.md](MODULAR_STRUCTURE.md) | Quick reference | Developers |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Architecture deep dive | Architects |
| [VISUAL_ARCHITECTURE.md](VISUAL_ARCHITECTURE.md) | Diagrams & flows | Visual learners |
| [API_DOCUMENTATION.md](API_DOCUMENTATION.md) | Complete API reference | API consumers |
| [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md) | What changed | Migration |

---

**Ready to build? Start with:**
```bash
./start_api.sh
```

Then open http://localhost:8000/docs to explore the API! 🚀
