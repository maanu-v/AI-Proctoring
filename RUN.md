# Running AI Proctor

## Quick Start

To run the AI Proctor application with integrated logging:

```bash
uv run python main.py
```

This will:
- Start the FastAPI application with uvicorn
- Use the centralized logger for all components (uvicorn, FastAPI, VideoStream, etc.)
- Load configuration from `src/configs/app.yaml`
- Run on the configured host and port (default: http://0.0.0.0:8000)

## Features

- **Unified Logging**: All components (uvicorn, FastAPI, VideoStream) use the same Rich logger
- **Centralized Configuration**: All settings in `src/configs/app.yaml`
- **Auto-reload**: Enabled in debug mode for development
- **Modular Architecture**: Clean separation of concerns

## Configuration

Edit `src/configs/app.yaml` to customize:

```yaml
app:
  host: "0.0.0.0"
  port: 8000
  debug: true

camera:
  source: 0  # 0 for default webcam
  width: 1280
  height: 720
  fps: 30

logging:
  level: "INFO"
  use_rich: true
```

## Accessing the Application

Once running, open your browser to:
- **Web Interface**: http://localhost:8000
- **Video Feed**: http://localhost:8000/video_feed

## Stopping the Application

Press `Ctrl+C` to gracefully shutdown the application. The logger will show:
- Application shutdown messages
- VideoStream cleanup
- Resource release confirmation
