# AI Proctoring System Documentation

This document serves as the central index for the AI Proctoring System's documentation. It outlines the core architecture and provides links to detailed feature documentation.

## Feature Documentation

For detailed logic, configuration, and implementation details of specific features, please refer to:

- **[No Face & Multi-Face Detection](no_face_multiface.md)**: Covers persistence logic, configuration, and violation triggers for face count checks.
- **[Head Pose Estimation](head_pose.md)**: Covers yaw/pitch/roll calculation, axis visualization, and "looking away" violation logic.

## Core System Overview

### 1. Configuration Management (`src/configs/app.yaml`)
Centralized configuration controls all aspects of the system.
- **Camera**: Resolution, FPS.
- **MediaPipe**: Max tracked faces.
- **Thresholds**: Separate persistence times for different violation types.

### 2. Video Pipeline
- **FPS Throttling**: Implemented to prevent UI crashes during high-frequency updates.
- **Base64 Rendering**: Uses HTML `<img>` tags for video display to bypass Streamlit storage limits.
- **Live Logs**: Real-time logging of violations in the sidebar.

### 3. Violation Tracking (`src/engine/proctor.py`)
- **Persistence**: All violations use a time-based persistence check to ignore transient glitches.
- **Consolidation**: Continuous violations (e.g., "No face detected") update a single log entry in-place to keep logs clean.

## Model Assets
- **`src/models/face_landmarker.task`**: The MediaPipe Task Bundle containing the TFLite models for face detection and landmark regression.

## How to Run
```bash
streamlit run src/web/app.py
```
