# Head Pose Estimation and Axis Visualization

This document details the implementation, logic, and configuration for the Head Pose Estimation feature in the AI Proctoring System.

## 1. Head Pose Estimation Logic

The system estimates the user's head orientation (Pitch, Yaw, Roll) based on facial landmarks detected by MediaPipe Face Mesh.

### Logic (`src/engine/face/head_pose.py`)
- **Pitch (Looking Up/Down)**: Rotation around the X-axis.
    - `> threshold`: Looking Up (Note: Swapped from previous convention to align with visual overlay).
    - `< -threshold`: Looking Down.
- **Yaw (Looking Left/Right)**: Rotation around the Y-axis.
    - `> threshold`: Looking Right.
    - `< -threshold`: Looking Left.
- **Roll (Head Tilt)**: Rotation around the Z-axis.
    - Currently used for visual feedback but not as a strict violation trigger (though it contributes to overall pose).

### Calculations
- The system extracts the `facial_transformation_matrixes` from MediaPipe.
- It decomposes this 4x4 matrix into Euler angles (Pitch, Yaw, Roll) using standard rotation matrix decomposition.
- Angles are converted to degrees for readability and threshold comparison.

## 2. Axis Visualization

When enabled, the system overlays a 3D axis on the user's nose to visualize their head orientation.
- **Red (X-axis)**: Pitch (Up/Down)
- **Green (Y-axis)**: Yaw (Left/Right)
- **Blue (Z-axis)**: Roll / Forward Vector

### Implementation
- **Origin**: The tip of the nose (Landmark 1).
- **Projection**: The system calculates the endpoint of each axis vector (X, Y, Z) based on the current Pitch, Yaw, and Roll.
- **Drawing**: It uses OpenCV (`cv2.line`) to draw these vectors from the nose tip to the projected endpoint on the video frame.

## 3. Head Pose Violations

The system detects if the user is looking away from the screen for an extended period.

### Logic
- **"Looking Away"**: Defined as any direction other than "Forward". This includes Looking Left, Right, Up, or Down.
- **Persistence**: A violation is only triggered if the user looks away continuously for longer than `head_pose_persistence_time`.
    - Brief glances away are ignored.
    - Switching directions (e.g., looking Left then Right) resets the timer for the specific direction, tracking sustained attention deviation.
- **Feedback**:
    - **Overlay**: "WARNING: LOOKING [DIRECTION]!" appears on the video feed.
    - **Toast**: A notification pops up with the duration message.
    - **Logs**: An entry is added to the sidebar logs. Continuous violations update a single log entry (e.g., "Looking Right for 3s" -> "Looking Right for 4s").

### Configuration (`src/configs/app.yaml`)
```yaml
head_pose:
  # Thresholds in degrees. Adjust based on sensitivity needs.
  yaw_threshold: 30
  pitch_threshold: 20
  roll_threshold: 25

thresholds:
  # Duration in seconds looking away must persist to trigger warning.
  head_pose_persistence_time: 3
```

## How to Test
1.  **Start App**: `streamlit run src/web/app.py`
2.  **Enable Feature**: Check "Enable Head Pose Analysis" in the sidebar.
3.  **Visualization**: Observe the RGB axes on your nose moving with your head.
4.  **Trigger Violation**: Look to the side (Left/Right) or Up/Down for > `head_pose_persistence_time`.
5.  **Verify**: Check for the red warning overlay and the toast/log notification.
