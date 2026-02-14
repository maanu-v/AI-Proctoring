# No Face & Multi-Face Detection

This document details the implementation, logic, and configuration for the Face Count violation features in the AI Proctoring System.

## 1. Multi-Face Detection

The system detects when more than one face is present in the video feed, which is a common violation in proctoring scenarios.

### Logic
- **Detection**: Uses MediaPipe Face Mesh to count the number of detected faces in each frame.
- **Threshold**: Compares the count against `max_num_faces` (default: 1).
- **Persistence**: A violation is only triggered if the condition (`faces > max_num_faces`) persists for longer than `multi_face_persistence_time`.
    - Transient detections (e.g., a poster in the background momentarily detected as a face) are ignored if they last less than the persistence threshold.
- **Feedback**:
    - **Overlay**: A red warning "WARNING: MULTIPLE FACES!" appears on the video feed.
    - **Toast**: A notification pops up.
    - **Logs**: An entry is added to the sidebar logs. Continuous violations update a single log entry (e.g., "Multiple faces detected: 2" updates its timestamp).

### Configuration (`src/configs/app.yaml`)
```yaml
mediapipe:
  num_faces: 5          # Max faces MediaPipe attempts to track (set higher than 1 to enable detection)

thresholds:
  max_num_faces: 1      # Limit for allowed faces. >1 triggers violation.
  multi_face_persistence_time: 3  # Seconds the condition must persist to trigger warning.
```

## 2. No Face Detection

The system detects when the user leaves the frame or obscures their face.

### Logic
- **Detection**: Checks if the number of detected faces is 0.
- **Persistence**: A violation is only triggered if the condition (`faces == 0`) persists for longer than `no_face_persistence_time`.
    - Brief occlusions (e.g., sneezing, rubbing eyes) are ignored.
- **Dynamic Feedback**:
    - The system provides real-time feedback on the duration of the violation.
    - **Log Message**: "No face detected for X seconds". This message updates in-place in the logs, incrementing the seconds as the violation continues.
- **Feedback**:
    - **Overlay**: "WARNING: NO FACE DETECTED!".
    - **Toast**: Notification with the duration message.
    - **Logs**: Updating log entry.

### Configuration (`src/configs/app.yaml`)
```yaml
thresholds:
  enable_no_face_warning: true    # Toggle this feature on/off.
  no_face_persistence_time: 3     # Seconds the condition must persist to trigger warning.
```

## implementation Details (`src/engine/proctor.py`)

The `ViolationTracker` class manages the state of these checks.
- `check_face_count`: Handles multi-face logic.
- `check_no_face`: Handles no-face logic.
- `log_violation`: New logic consolidates continuous violations of the same type into a single log entry to prevent spam.
