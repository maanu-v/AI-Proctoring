# MediaPipe Face Mesh & Proctoring System Updates

This document summarizes the recent implementation details, feature additions, and fixes applied to the AI Proctoring System based on `MediaPipe Face Mesh`.

## 1. Core Logic Simplification
Refactured the core engine to focus on essential proctoring violations:
- **Consolidated**: The system now primarily tracks:
    - **No Face Detected**: Configuring alerts when the user leaves the frame.
    - **Multiple Faces Detected**: Configuring alerts when unauthorized individuals enter the frame. (Default limit: 1 face)

## 2. Configuration Management (`src/configs/app.yaml`)
Centralized configuration with detailed comments for better maintainability.
- **Camera Settings**: Index, resolution (640x480), FPS (30).
- **MediaPipe**: `num_faces` set to 5 to allow multi-face detection.
- **Thresholds**:
    - `max_num_faces`: Limit for allowed faces (Default: 1).
    - `enable_no_face_warning`: Toggle for the "No Face Detected" warning overlay.
    - `violation_persistence_time`: Time (in seconds) a violation must persist to trigger an alert. This applies to both "No Face" and "Multiple Faces" scenarios to overlapping transient glitches.

## 3. Video Upload Handling
Addressed critical issues with video file processing:
- **File Extension Fix**: Added logic to preserve original file extensions for temporary files, resolving `MediaFileStorageError` and `cv2` reading failures.
- **FPS Throttling**: Implemented a global FPS cap (approx. 30 FPS) for both webcam and video file playback. This prevents video files from playing at hyper-speed and crashing the UI.

## 4. UI Enhancements (`src/web/app.py`)
- **Live Proctoring Logs**: Moved log rendering to the main processing loop. Violations are now displayed in the sidebar *in real-time* as they occur.
- **Analysis Panel**: Updated to show "Faces Detected: 0" and "Violations: N" even when no face is detected, ensuring a consistent UI state instead of a generic "No face detected" info message.
- **Rendering Fix**: Reverted `st.image` usage to `width="stretch"` to resolve Streamlit deprecation warnings and ensure proper video scaling.

## 5. Violation Persistence Logic (`src/engine/proctor.py`)
Implemented a robust `ViolationTracker` with time-based persistence:
- **`check_face_count`**: Validates if `face_count > max_faces`. Triggers only if the condition persists for > `violation_persistence_time`.
- **`check_no_face`**: Validates if `face_count == 0`. Triggers only if the condition persists for > `violation_persistence_time`.
- **Debouncing**: Added logic to prevent log spamming for identical consecutive violation messages.

## 6. Model Assets (`src/models/face_landmarker.task`)
- **Purpose**: This file is a MediaPipe Task Bundle. It acts as the "brain" of the face mesh system.
- **Contents**: It packages the TFLite models for face detection and landmark regression, along with necessary metadata and tensors.
- **Requirement**: The `MeshDetector` class requires this file to initialize the `FaceLandmarker` API.

## 7. How to Run
```bash
streamlit run src/web/app.py
```
