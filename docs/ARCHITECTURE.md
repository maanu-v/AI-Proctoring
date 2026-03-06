# FastAPI Backend - Modular Architecture

## Directory Structure

```
src/web/
├── main.py                      # Main FastAPI application entry point
├── fastapi_app.py              # Legacy monolithic version (kept for reference)
├── app.py                       # Streamlit application
│
├── api/                         # API layer
│   ├── __init__.py
│   ├── models.py               # Pydantic models for requests/responses
│   ├── dependencies.py         # Dependency injection functions
│   │
│   └── routes/                 # API route handlers
│       ├── __init__.py
│       ├── session.py          # Session management endpoints
│       ├── analysis.py         # Frame analysis endpoints
│       ├── settings.py         # Settings management endpoints
│       ├── config.py           # Configuration endpoints
│       └── violations.py       # Violations endpoints
│
├── core/                        # Core business logic
│   ├── __init__.py
│   ├── session_manager.py      # Session state management
│   └── analyzers.py            # Analyzer initialization and management
│
└── utils/                       # Utility functions
    ├── __init__.py
    └── image_utils.py          # Image encoding/decoding utilities
```

## File Descriptions

### Main Application

**`main.py`** - FastAPI application entry point
- Initializes FastAPI app with metadata
- Configures CORS middleware
- Includes all route modules
- Defines root health check endpoint
- Handles startup/shutdown events

### API Layer

**`api/models.py`** - Pydantic Models
- `SessionCreateRequest` - Request to create session
- `SessionResponse` - Session creation response
- `AnalysisResult` - Frame analysis response
- `SettingsUpdateRequest` - Settings update request
- `ConfigUpdateRequest` - Configuration update request
- `ViolationsResponse` - Violations listing response
- `HealthResponse` - Health check response

**`api/dependencies.py`** - Dependency Injection
- `get_session_manager()` - Provides session manager instance
- `get_analyzer_manager()` - Provides analyzer manager instance
- `get_session()` - Validates and retrieves session
- `validate_session_exists()` - Validates session existence

**`api/routes/session.py`** - Session Management
- `POST /api/session/start` - Start proctoring session
- `POST /api/session/end/{session_id}` - End session
- `GET /api/session/{session_id}` - Get session info
- `GET /api/sessions` - List all sessions

**`api/routes/analysis.py`** - Frame Analysis
- `POST /api/analyze/frame/{session_id}` - Analyze video frame

**`api/routes/settings.py`** - Settings Management
- `GET /api/session/{session_id}/settings` - Get settings
- `PUT /api/session/{session_id}/settings` - Update settings

**`api/routes/config.py`** - Configuration Management
- `GET /api/config` - Get global configuration
- `PUT /api/config` - Update global configuration

**`api/routes/violations.py`** - Violations Management
- `GET /api/session/{session_id}/violations` - Get violations
- `DELETE /api/session/{session_id}/violations` - Clear violations

### Core Layer

**`core/session_manager.py`** - Session Management
- `QuizSession` - Individual session state class
  - Stores student ID, quiz ID, settings
  - Tracks frame count and violations
  - Manages reference embeddings
- `SessionManager` - Session lifecycle manager
  - Creates and tracks active sessions
  - Provides session lookup and removal
  - Generates unique session IDs

**`core/analyzers.py`** - Analyzer Management
- `AnalyzerManager` - Singleton analyzer manager
  - Lazy initialization of all analyzers
  - Provides unified access to all analysis components
  - Manages: MeshDetector, HeadPoseEstimator, GazeEstimator, BlinkEstimator, ObjectDetector, FaceEmbedder

### Utilities

**`utils/image_utils.py`** - Image Processing
- `decode_base64_image()` - Decode base64 to OpenCV image
- `encode_image_to_base64()` - Encode image to base64
- `validate_image()` - Validate image format
- `resize_image()` - Resize with aspect ratio preservation

## Running the Application

### Method 1: Direct Execution

```bash
python src/web/main.py
```

### Method 2: Using Uvicorn

```bash
# From project root
uvicorn src.web.main:app --reload --port 8000

# With custom host and port
uvicorn src.web.main:app --host 0.0.0.0 --port 8080 --reload
```

### Method 3: Production with Gunicorn

```bash
gunicorn src.web.main:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

## Advantages of Modular Architecture

### 1. **Separation of Concerns**
- Routes handle HTTP logic
- Core handles business logic
- Utils provide reusable functions
- Models define data structures

### 2. **Maintainability**
- Easy to locate specific functionality
- Changes isolated to relevant modules
- Clear dependencies between components

### 3. **Testability**
- Each module can be tested independently
- Mock dependencies easily
- Clear test organization

### 4. **Scalability**
- Add new routes without modifying existing code
- Easy to extend functionality
- Multiple developers can work simultaneously

### 5. **Reusability**
- Core logic reusable across different interfaces
- Utilities shared across modules
- Models ensure consistency

## Development Workflow

### Adding a New Route

1. Define models in `api/models.py`:
```python
class NewFeatureRequest(BaseModel):
    param: str
```

2. Create route in `api/routes/new_feature.py`:
```python
from fastapi import APIRouter
router = APIRouter(prefix="/api/new", tags=["New Feature"])

@router.post("/endpoint")
async def new_endpoint(request: NewFeatureRequest):
    return {"result": "success"}
```

3. Include in `main.py`:
```python
from src.web.api.routes import new_feature
app.include_router(new_feature.router)
```

### Adding Business Logic

Add to appropriate core module or create new one:

```python
# core/new_logic.py
class NewLogicManager:
    def process(self, data):
        # Business logic here
        return result
```

### Adding Utilities

Add to `utils/` directory:

```python
# utils/new_util.py
def helper_function(input):
    # Utility logic
    return output
```

## Testing Structure

Recommended test organization:

```
tests/
├── test_api/
│   ├── test_session_routes.py
│   ├── test_analysis_routes.py
│   ├── test_settings_routes.py
│   └── test_config_routes.py
├── test_core/
│   ├── test_session_manager.py
│   └── test_analyzers.py
└── test_utils/
    └── test_image_utils.py
```

## Migration from Monolithic

The original `fastapi_app.py` has been split into:

| Original Section | New Location |
|-----------------|--------------|
| Pydantic models | `api/models.py` |
| Global state | `core/session_manager.py`, `core/analyzers.py` |
| Image utilities | `utils/image_utils.py` |
| Session routes | `api/routes/session.py` |
| Analysis logic | `api/routes/analysis.py` |
| Settings routes | `api/routes/settings.py` |
| Config routes | `api/routes/config.py` |
| Violations routes | `api/routes/violations.py` |
| Main app setup | `main.py` |

## API Documentation

Once running, access:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Environment Variables

Set these for production:

```bash
export API_HOST=0.0.0.0
export API_PORT=8000
export LOG_LEVEL=info
export CORS_ORIGINS=https://your-frontend.com
```

## Best Practices

1. **Keep routes thin** - Move logic to core modules
2. **Use dependency injection** - Leverage FastAPI's DI system
3. **Type everything** - Use Pydantic models and type hints
4. **Handle errors properly** - Use appropriate HTTP exceptions
5. **Log important events** - Use structured logging
6. **Document endpoints** - Add docstrings to all routes
7. **Validate inputs** - Use Pydantic model validation

## Common Patterns

### Adding a New Analyzer

```python
# core/analyzers.py
class AnalyzerManager:
    def __init__(self):
        self.new_analyzer = None
    
    def initialize(self):
        if self.new_analyzer is None:
            self.new_analyzer = NewAnalyzer()
    
    def get_analyzers(self):
        return {
            # ... existing
            "new": self.new_analyzer
        }
```

### Creating a Dependency

```python
# api/dependencies.py
def get_current_user(token: str = Header(...)):
    user = verify_token(token)
    if not user:
        raise HTTPException(401, "Invalid token")
    return user
```

### Using the Dependency

```python
# api/routes/secure.py
@router.get("/secure")
async def secure_endpoint(user = Depends(get_current_user)):
    return {"user": user.id}
```

## Troubleshooting

### Import Errors

Ensure Python path includes project root:
```python
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
```

### Module Not Found

Check `__init__.py` files have proper exports:
```python
__all__ = ["module1", "module2"]
```

### Circular Imports

Use dependency injection or move shared code to separate module.

## Future Enhancements

- [ ] Add authentication and authorization
- [ ] Implement rate limiting
- [ ] Add database persistence
- [ ] WebSocket support for real-time updates
- [ ] Caching layer with Redis
- [ ] Background task queue
- [ ] Metrics and monitoring
- [ ] API versioning

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
