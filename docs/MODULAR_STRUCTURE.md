# FastAPI Backend - Quick Reference

## Starting the Server

### Option 1: Using the start script
```bash
./start_api.sh
```

### Option 2: Direct Python execution
```bash
python src/web/main.py
```

### Option 3: Using uvicorn
```bash
uvicorn src.web.main:app --reload --port 8000
```

## New Modular Structure

### Key Files

| File | Purpose |
|------|---------|
| `src/web/main.py` | Main application entry point |
| `src/web/api/models.py` | Request/response models |
| `src/web/api/dependencies.py` | Dependency injection |
| `src/web/core/session_manager.py` | Session state management |
| `src/web/core/analyzers.py` | Analyzer initialization |
| `src/web/utils/image_utils.py` | Image utilities |

### Route Modules

| Module | Endpoints |
|--------|-----------|
| `api/routes/session.py` | Session create, end, info, list |
| `api/routes/analysis.py` | Frame analysis |
| `api/routes/settings.py` | Get/update settings |
| `api/routes/config.py` | Get/update config |
| `api/routes/violations.py` | Get/clear violations |

## Directory Tree

```
src/web/
├── main.py                      ← MAIN ENTRY POINT
├── fastapi_app.py              ← Legacy (kept for reference)
│
├── api/                         ← API Layer
│   ├── models.py               ← Pydantic models
│   ├── dependencies.py         ← Dependency injection
│   └── routes/                 ← Route handlers
│       ├── session.py          ← Session endpoints
│       ├── analysis.py         ← Analysis endpoints
│       ├── settings.py         ← Settings endpoints
│       ├── config.py           ← Config endpoints
│       └── violations.py       ← Violations endpoints
│
├── core/                        ← Business Logic
│   ├── session_manager.py      ← Session management
│   └── analyzers.py            ← Analyzer management
│
└── utils/                       ← Utilities
    └── image_utils.py          ← Image processing
```

## Quick API Reference

### Session Management
```python
# Start session
POST /api/session/start
Form: student_id, quiz_id, profile_image (optional)

# Get session info
GET /api/session/{session_id}

# List all sessions
GET /api/sessions

# End session
POST /api/session/end/{session_id}
```

### Frame Analysis
```python
# Analyze frame
POST /api/analyze/frame/{session_id}
JSON: {"frame_base64": "..."}
```

### Settings
```python
# Get settings
GET /api/session/{session_id}/settings

# Update settings
PUT /api/session/{session_id}/settings
JSON: {"enable_head_pose": true, ...}
```

### Configuration
```python
# Get config
GET /api/config

# Update config
PUT /api/config
JSON: {"thresholds": {"no_face_persistence_time": 5.0}}
```

### Violations
```python
# Get violations
GET /api/session/{session_id}/violations

# Clear violations
DELETE /api/session/{session_id}/violations
```

## Code Organization

### Where to Add New Features

**New Route?**
→ Create file in `api/routes/`
→ Include in `main.py`

**New Business Logic?**
→ Add to `core/` modules

**New Utility?**
→ Add to `utils/`

**New Model?**
→ Add to `api/models.py`

**New Dependency?**
→ Add to `api/dependencies.py`

## Testing

```bash
# Test the API
python tests/test_fastapi.py

# Run example client
python examples/client_example.py
```

## Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Architecture Guide**: [docs/ARCHITECTURE.md](ARCHITECTURE.md)
- **Full API Docs**: [docs/API_DOCUMENTATION.md](API_DOCUMENTATION.md)

## Common Tasks

### Add a New Analyzer

1. Initialize in `core/analyzers.py`:
```python
def initialize(self):
    if self.new_analyzer is None:
        self.new_analyzer = NewAnalyzer()
```

2. Add to `get_analyzers()` return dict:
```python
return {
    "new": self.new_analyzer,
    # ... existing
}
```

### Add Session Settings

1. Add to `QuizSession.__init__()` in `core/session_manager.py`:
```python
self.settings = {
    "new_setting": True,
    # ... existing
}
```

2. Add to `SettingsUpdateRequest` in `api/models.py`:
```python
new_setting: Optional[bool] = None
```

### Add Configuration Parameter

1. Add to config schema in `src/configs/app.yaml`

2. Add to `Config` class in `src/utils/config.py`

3. Add to `get_config()` in `api/routes/config.py`

## Migration from Monolithic

Old `fastapi_app.py` → New modular structure:

- ✅ Single 800+ line file → Multiple focused modules
- ✅ Mixed concerns → Separated layers (API/Core/Utils)
- ✅ Hard to test → Easy to mock and test
- ✅ Difficult to extend → Simple to add features
- ✅ One developer at a time → Team-friendly

## Benefits

1. **Maintainability**: Easy to find and modify code
2. **Testability**: Test modules independently
3. **Scalability**: Add features without conflicts
4. **Readability**: Clear structure and responsibility
5. **Collaboration**: Multiple developers can work together

## Next Steps

1. Start the server: `./start_api.sh`
2. Open docs: http://localhost:8000/docs
3. Try the example: `python examples/client_example.py`
4. Read architecture: [docs/ARCHITECTURE.md](ARCHITECTURE.md)

---

**Questions?** Check the full documentation in `docs/` directory.
