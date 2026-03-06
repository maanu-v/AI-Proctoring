# FastAPI Backend - Visual Architecture

## System Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                     CLIENT APPLICATION                        │
│              (Web Browser / Mobile App / Desktop)             │
└────────────────────────┬─────────────────────────────────────┘
                         │ HTTP/REST API
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                      FASTAPI SERVER                           │
│                      (main.py)                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              CORS Middleware                         │   │
│  └──────────────────────────────────────────────────────┘   │
│                         │                                     │
│  ┌──────────────────────┴──────────────────────────────┐   │
│  │              API ROUTERS                             │   │
│  │  ┌─────────┬──────────┬──────────┬────────────┐    │   │
│  │  │Session  │Analysis  │Settings  │Config      │    │   │
│  │  │Routes   │Routes    │Routes    │Routes      │    │   │
│  │  │         │          │          │Violations  │    │   │
│  │  └────┬────┴────┬─────┴────┬─────┴─────┬──────┘    │   │
│  └───────┼──────────┼──────────┼───────────┼───────────┘   │
│          │          │          │           │                │
│  ┌───────▼──────────▼──────────▼───────────▼───────────┐   │
│  │           DEPENDENCY INJECTION                       │   │
│  │     (SessionManager, AnalyzerManager)                │   │
│  └──────────────────────┬───────────────────────────────┘   │
└─────────────────────────┼───────────────────────────────────┘
                          │
         ┌────────────────┴────────────────┐
         │                                  │
         ▼                                  ▼
┌─────────────────────┐        ┌─────────────────────┐
│  SESSION MANAGER    │        │  ANALYZER MANAGER   │
│  (Core)             │        │  (Core)             │
├─────────────────────┤        ├─────────────────────┤
│ • QuizSession       │        │ • MeshDetector      │
│ • Session CRUD      │        │ • HeadPoseEst.      │
│ • State Management  │        │ • GazeEstimator     │
│ • Violation Track   │        │ • BlinkEstimator    │
└─────────────────────┘        │ • ObjectDetector    │
                               │ • FaceEmbedder      │
                               └─────────────────────┘
                                         │
                     ┌──────────────────┼──────────────────┐
                     │                  │                   │
                     ▼                  ▼                   ▼
            ┌────────────────┐  ┌────────────┐   ┌──────────────┐
            │   MediaPipe    │  │   YOLO     │   │  DeepFace    │
            │  (Face Mesh)   │  │ (Objects)  │   │ (Identity)   │
            └────────────────┘  └────────────┘   └──────────────┘
```

## Request Flow

### 1. Session Creation Flow

```
Client                 API                 Core              Analyzer
  │                     │                   │                   │
  ├─POST /session/start─►                  │                   │
  │                     │                   │                   │
  │                     ├─create_session()─►│                   │
  │                     │                   │                   │
  │  (with profile)     │                   ├──get_embedding()─►│
  │                     │                   │                   │
  │                     │                   │◄─────embedding────┤
  │                     │                   │                   │
  │                     │◄─session_id──────┤                   │
  │                     │                   │                   │
  │◄──session_id───────┤                   │                   │
  │                     │                   │                   │
```

### 2. Frame Analysis Flow

```
Client              API           Core            Analyzer      Engine
  │                  │              │                │            │
  ├─POST /analyze──►│              │                │            │
  │ {frame_base64}   │              │                │            │
  │                  │              │                │            │
  │                  ├─get_session()►                │            │
  │                  │              │                │            │
  │                  ├──decode_frame()               │            │
  │                  │              │                │            │
  │                  │              │─get_analyzers()►            │
  │                  │              │                │            │
  │                  │              │                │            │
  │                  ├──────────────┴────Face Mesh──►│            │
  │                  │◄───────────────landmarks──────┤            │
  │                  │                                │            │
  │                  ├────────────Head Pose─────────►│            │
  │                  │◄──────────direction───────────┤            │
  │                  │                                │            │
  │                  ├──────────────Gaze────────────►│            │
  │                  │◄────────gaze direction────────┤            │
  │                  │                                │            │
  │                  ├──────────────Blink───────────►│            │
  │                  │◄──────────blink data──────────┤            │
  │                  │                                │            │
  │                  ├────────Object Detection──────►│            │
  │                  │◄─────phone/person count──────┤            │
  │                  │                                │            │
  │                  ├─────check_violations()────────┼───────────►│
  │                  │◄──────violations list─────────┼────────────┤
  │                  │                                │            │
  │◄─AnalysisResult─┤                                │            │
  │                  │                                │            │
```

## Module Dependency Graph

```
main.py
  │
  ├─► api/routes/*.py
  │     │
  │     ├─► api/models.py (Pydantic)
  │     │
  │     ├─► api/dependencies.py
  │     │     │
  │     │     ├─► core/session_manager.py
  │     │     │     │
  │     │     │     └─► engine/proctor.py (ViolationTracker)
  │     │     │
  │     │     └─► core/analyzers.py
  │     │           │
  │     │           ├─► engine/face/*.py
  │     │           └─► engine/obj_detection/*.py
  │     │
  │     └─► utils/image_utils.py
  │
  └─► utils/config.py
        └─► configs/app.yaml
```

## Layer Responsibilities

### 1. API Layer (`api/`)

**Responsibility**: HTTP request/response handling

```
┌─────────────────────────────────────┐
│         API Layer                   │
├─────────────────────────────────────┤
│ • Route definitions                 │
│ • Request validation (Pydantic)     │
│ • Response formatting               │
│ • HTTP error handling               │
│ • Dependency injection              │
└─────────────────────────────────────┘
         │ calls ↓
```

### 2. Core Layer (`core/`)

**Responsibility**: Business logic and state management

```
┌─────────────────────────────────────┐
│         Core Layer                  │
├─────────────────────────────────────┤
│ • Session lifecycle management      │
│ • Analyzer initialization           │
│ • Business rules enforcement        │
│ • State persistence                 │
│ • Coordination between analyzers    │
└─────────────────────────────────────┘
         │ calls ↓
```

### 3. Engine Layer (`engine/`)

**Responsibility**: AI/ML analysis

```
┌─────────────────────────────────────┐
│         Engine Layer                │
├─────────────────────────────────────┤
│ • Face detection                    │
│ • Pose estimation                   │
│ • Gaze tracking                     │
│ • Blink detection                   │
│ • Object detection                  │
│ • Face recognition                  │
│ • Violation tracking                │
└─────────────────────────────────────┘
```

## Data Flow

### Session State

```
QuizSession
├── student_id: str
├── quiz_id: str
├── created_at: datetime
├── frame_count: int
├── profile_image: np.ndarray
├── reference_embedding: list
├── violation_tracker: ViolationTracker
└── settings: dict
      ├── enable_face_detection: bool
      ├── enable_head_pose: bool
      ├── enable_gaze: bool
      ├── enable_blink: bool
      ├── enable_object_detection: bool
      └── enable_identity_verification: bool
```

### Analysis Pipeline

```
Frame (base64)
    │
    ├─► Decode ────► OpenCV Image (BGR)
    │                    │
    │                    ├─► Face Mesh ───► Landmarks
    │                    │                      │
    │                    │                      ├─► Head Pose
    │                    │                      ├─► Gaze
    │                    │                      └─► Blink
    │                    │
    │                    ├─► Object Detection ► Objects
    │                    │
    │                    └─► Identity Check ──► Verified
    │
    └─► Check Violations ────► ViolationTracker
                                    │
                                    └─► Violations List
```

## Scalability Patterns

### Horizontal Scaling

```
Load Balancer
      │
      ├──► FastAPI Instance 1 ──► Shared Redis (Sessions)
      ├──► FastAPI Instance 2 ──► Shared Redis (Sessions)
      └──► FastAPI Instance 3 ──► Shared Redis (Sessions)
                                        │
                                        └──► PostgreSQL (Reports)
```

### Async Processing

```
Client ──POST /analyze──► FastAPI
                            │
                            ├─► Quick validation
                            │
                            ├─► Task Queue (Celery/RQ)
                            │         │
                            │         ├─► Worker 1
                            │         ├─► Worker 2
                            │         └─► Worker 3
                            │
                            └─► Return task_id
                            
Client ──GET /result/{id}─► FastAPI ──► Check result
```

## Security Layers

```
┌─────────────────────────────────────┐
│         HTTPS/TLS                   │
└─────────────────────────────────────┘
            ▼
┌─────────────────────────────────────┐
│    API Gateway (Rate Limiting)      │
└─────────────────────────────────────┘
            ▼
┌─────────────────────────────────────┐
│    Authentication (JWT)             │
└─────────────────────────────────────┘
            ▼
┌─────────────────────────────────────┐
│    Authorization (RBAC)             │
└─────────────────────────────────────┘
            ▼
┌─────────────────────────────────────┐
│    Input Validation (Pydantic)      │
└─────────────────────────────────────┘
            ▼
┌─────────────────────────────────────┐
│         Business Logic              │
└─────────────────────────────────────┘
```

## Testing Architecture

```
Unit Tests
├── test_api/
│   ├── test_models.py (Pydantic validation)
│   └── test_dependencies.py (DI logic)
│
├── test_core/
│   ├── test_session_manager.py (Session CRUD)
│   └── test_analyzers.py (Initialization)
│
└── test_utils/
    └── test_image_utils.py (Image processing)

Integration Tests
└── test_routes/
    ├── test_session_routes.py (API endpoints)
    ├── test_analysis_routes.py (End-to-end analysis)
    └── test_settings_routes.py (Configuration)

End-to-End Tests
└── test_scenarios/
    ├── test_full_quiz_flow.py (Complete workflow)
    └── test_violation_scenarios.py (Edge cases)
```

## Monitoring & Observability

```
Application Logs
      │
      ├─► Structured Logging (JSON)
      │         │
      │         └─► ELK Stack / CloudWatch
      │
      ├─► Metrics (Prometheus)
      │         │
      │         ├─► Request count
      │         ├─► Response time
      │         ├─► Error rate
      │         └─► Active sessions
      │
      └─► Traces (OpenTelemetry)
                │
                └─► Jaeger / Zipkin
```

## Deployment Architecture

```
┌──────────────────────────────────────────┐
│           Docker Container               │
│  ┌────────────────────────────────────┐ │
│  │  FastAPI App                       │ │
│  │  + Uvicorn/Gunicorn Workers        │ │
│  │  + ML Models (cached)              │ │
│  └────────────────────────────────────┘ │
└──────────────────────────────────────────┘
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
   ┌──────────┐      ┌──────────┐
   │  Redis   │      │ Database │
   │ (Cache)  │      │(Reports) │
   └──────────┘      └──────────┘
```

---

This architecture provides:
- ✅ Clear separation of concerns
- ✅ Easy to test and maintain
- ✅ Scalable and extensible
- ✅ Production-ready patterns
