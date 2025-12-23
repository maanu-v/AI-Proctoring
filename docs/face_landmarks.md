# Face Landmark Detection Module

## Overview

The Face Landmark Detection module provides comprehensive facial landmark detection using MediaPipe Face Mesh. It detects **468 3D facial landmarks** including refined iris landmarks, enabling detailed facial analysis for various AI proctoring tasks.

## Features

- **468 3D Landmarks**: Full face mesh with x, y, z coordinates
- **Iris Tracking**: Refined iris landmarks for gaze detection (5 points per eye)
- **Pre-defined Landmark Groups**: Easy access to eyes, mouth, nose, eyebrows, face oval
- **Utility Functions**: EAR, MAR, head pose estimation, iris position
- **Visualization**: Draw landmarks and connections on images
- **Optimized for Video**: Tracking mode for real-time performance

## Quick Start

```python
from src.models.face.face_landmarks import FaceLandmarkDetector
import cv2

# Initialize detector
detector = FaceLandmarkDetector(
    static_image_mode=False,  # Use tracking for video
    max_num_faces=1,
    refine_landmarks=True,    # Enable iris landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Detect landmarks
image = cv2.imread('face.jpg')
face_landmarks = detector.detect_single(image)

if face_landmarks:
    # Get specific features
    eyes = detector.get_eye_landmarks(face_landmarks)
    mouth = detector.get_mouth_landmarks(face_landmarks)
    nose = detector.get_nose_landmarks(face_landmarks)
    
    # Draw landmarks
    annotated = detector.draw_landmarks(image, face_landmarks)
    cv2.imshow('Landmarks', annotated)
```

## Landmark Groups

### Eyes (12 points per eye)
- **Left Eye**: Indices 33, 160, 158, 133, 153, 144
- **Right Eye**: Indices 362, 385, 387, 263, 373, 380
- **Left Iris**: Indices 468-472 (requires `refine_landmarks=True`)
- **Right Iris**: Indices 473-477 (requires `refine_landmarks=True`)

### Mouth (20 points)
- **Outer Lips**: 20 points around outer mouth contour
- **Inner Lips**: 20 points around inner mouth contour

### Nose (5 points)
- **Tip**: Index 1
- **Bridge**: Indices 6, 197, 195, 5

### Face Oval (36 points)
- Contour points around the face perimeter

### Eyebrows (10 points per eyebrow)
- **Left Eyebrow**: 10 points
- **Right Eyebrow**: 10 points

## Utility Functions

### Eye Aspect Ratio (EAR)
Used for blink detection:

```python
from src.utils.landmark_utils import calculate_eye_aspect_ratio

eyes = detector.get_eye_landmarks(face_landmarks)
left_ear = calculate_eye_aspect_ratio(eyes['left_eye'])

# Typical values:
# Open eye: ~0.25-0.35
# Closed eye: <0.2
```

### Mouth Aspect Ratio (MAR)
Used for mouth open detection:

```python
from src.utils.landmark_utils import calculate_mouth_aspect_ratio

mouth = detector.get_mouth_landmarks(face_landmarks)
mar = calculate_mouth_aspect_ratio(mouth['outer'])

# Typical values:
# Closed mouth: <0.5
# Open mouth: >0.6
```

### Head Pose Estimation
Get pitch, yaw, roll angles:

```python
from src.utils.landmark_utils import estimate_head_pose

pitch, yaw, roll = estimate_head_pose(face_landmarks)

# Angles in degrees:
# pitch: up/down rotation
# yaw: left/right rotation
# roll: tilt rotation
```

### Iris Position
Get normalized iris position for gaze tracking:

```python
from src.utils.landmark_utils import get_iris_position

eyes = detector.get_eye_landmarks(face_landmarks)
h_ratio, v_ratio = get_iris_position(eyes['left_iris'], eyes['left_eye'])

# Ratios from 0.0 to 1.0:
# 0.5, 0.5 = center
# <0.5, _ = looking left
# >0.5, _ = looking right
```

## Integration with Other Models

The face landmarks can be used by various AI proctoring models:

### Gaze Detection
```python
from src.models.gaze.gaze_detector import GazeDetector

gaze_detector = GazeDetector()
gaze_direction = gaze_detector.detect(face_landmarks)
```

### Head Pose Analysis
```python
from src.models.head_pose.head_pose_estimator import HeadPoseEstimator

pose_estimator = HeadPoseEstimator()
is_looking_away = pose_estimator.is_looking_away(face_landmarks)
```

### Mouth Activity Detection
```python
from src.models.mouth.mouth_detector import MouthDetector

mouth_detector = MouthDetector()
is_speaking = mouth_detector.detect_speaking(face_landmarks)
```

## Performance Tips

1. **Use Tracking Mode**: Set `static_image_mode=False` for video streams
2. **Adjust Confidence**: Lower `min_tracking_confidence` for smoother tracking
3. **Limit Faces**: Set `max_num_faces=1` if only one person is expected
4. **Disable Iris**: Set `refine_landmarks=False` if iris tracking is not needed

## Testing

Run the test script to see landmarks in action:

```bash
uv run python tests/test_face_landmarks.py
```

Controls:
- Press 'q' to quit
- Press 's' to save screenshot

## Configuration

Edit `src/configs/model.yaml`:

```yaml
face_mesh:
  static_image_mode: false
  max_num_faces: 1
  refine_landmarks: true
  min_detection_confidence: 0.5
  min_tracking_confidence: 0.5
```

## API Reference

### FaceLandmarks Class

**Attributes:**
- `landmarks`: Nx3 numpy array of (x, y, z) normalized coordinates
- `image_width`: Width of the source image
- `image_height`: Height of the source image

**Methods:**
- `get_landmark(index)`: Get specific landmark by index
- `get_landmarks_subset(indices)`: Get multiple landmarks
- `to_pixel_coords(normalized)`: Convert to pixel coordinates

### FaceLandmarkDetector Class

**Methods:**
- `detect(image)`: Detect all faces in image
- `detect_single(image)`: Detect first face only
- `get_eye_landmarks(face_landmarks)`: Extract eye landmarks
- `get_mouth_landmarks(face_landmarks)`: Extract mouth landmarks
- `get_nose_landmarks(face_landmarks)`: Extract nose landmarks
- `get_face_oval(face_landmarks)`: Extract face contour
- `get_eyebrow_landmarks(face_landmarks)`: Extract eyebrow landmarks
- `draw_landmarks(image, face_landmarks, ...)`: Visualize landmarks

## References

- [MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh.html)
- [Landmark Visualization](https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png)
