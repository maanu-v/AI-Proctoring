import streamlit as st
import cv2
import tempfile
import time
import sys
import os
import base64
from typing import Union

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.video_stream import VideoStream
from src.engine.face.mesh_detector import MeshDetector
from src.engine.face.head_pose import HeadPoseEstimator
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

st.set_page_config(page_title="AI Proctoring Dashboard", layout="wide")

st.title("🎥 AI Proctoring System")
st.sidebar.title("Settings")

# Sidebar Configuration
source_type = st.sidebar.radio("Video Source", ["Webcam", "Video File"])
show_mesh = st.sidebar.checkbox("Show Face Mesh", value=True)
enable_head_pose = st.sidebar.checkbox("Enable Head Pose Analysis", value=False)

st.sidebar.caption(f"System: Max Detectable Faces={config.mediapipe.num_faces}")
st.sidebar.caption(f"Violation Threshold: > {config.thresholds.max_num_faces} faces")

# Global variables for caching
if "video_stream" not in st.session_state:
    st.session_state.video_stream = None

# Calibration State (Removed Head Pose logic)


if "violation_tracker" not in st.session_state:
    from src.engine.proctor import ViolationTracker
    st.session_state.violation_tracker = ViolationTracker()

@st.cache_resource
def get_mesh_detector(num_faces):
    return MeshDetector()

def frame_to_base64(frame):
    """Convert a CV2 frame to a base64 string for HTML display."""
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

@st.cache_resource
def get_head_pose_estimator():
    return HeadPoseEstimator(
        yaw_threshold=config.head_pose.yaw_threshold,
        pitch_threshold=config.head_pose.pitch_threshold,
        roll_threshold=config.head_pose.roll_threshold
    )

def main():
    detector = get_mesh_detector(config.mediapipe.num_faces)
    pose_estimator = get_head_pose_estimator()
    
    # Source Selection Logic
    source: Union[int, str] = config.camera.index
    
    if source_type == "Video File":
        video_file = st.sidebar.file_uploader("Upload Video", type=['mp4', 'mov', 'avi'])
        if video_file is not None:
            # Save to temp file because cv2.VideoCapture needs a path
            suffix = os.path.splitext(video_file.name)[1]
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix) 
            tfile.write(video_file.read())
            source = tfile.name
        else:
            st.info("Please upload a video file.")
            return

    # Start/Stop Controls
    col_btn1, col_btn2 = st.sidebar.columns(2)
    start_button = col_btn1.button("Start")
    stop_button = col_btn2.button("Stop")
    

    
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
        
    if start_button:
        st.session_state.is_running = True
        
        # Reset Violation Tracker
        st.session_state.violation_tracker.reset()
        

        # Initialize/Re-initialize stream
        if st.session_state.video_stream is not None:
             st.session_state.video_stream.stop()
        st.session_state.video_stream = VideoStream(source=source)
        st.session_state.video_stream.start()

    if stop_button:
        st.session_state.is_running = False
        if st.session_state.video_stream is not None:
            st.session_state.video_stream.stop()

    # Layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Video Feed")
        frame_placeholder = st.empty()
        
    with col2:
        st.subheader("Analysis")
        stats_placeholder = st.empty()

    # Sidebar Log Placeholder
    with st.sidebar.expander("Proctoring Logs", expanded=True):
        log_placeholder = st.empty()

    # Processing Loop
    if st.session_state.is_running and st.session_state.video_stream:
        stream = st.session_state.video_stream
        
        while stream.is_opened():
            ret, frame = stream.read()
            if not ret:
                st.warning("End of video or failed to read frame.")
                st.session_state.is_running = False
                stream.stop()
                break
                
            # Global FPS Throttling to prevent Streamlit MediaFileStorageError
            # Cap at ~30 FPS
            time.sleep(0.03)
                
            # Mirror the frame by default
            frame = cv2.flip(frame, 1)
            
            # Timestamp
            timestamp_ms = int(time.time() * 1000)
            
            # 1. Face Mesh Detection
            # Only run if we need mesh or checks, but user wants check so run always
            try:
                results = detector.process(frame, timestamp_ms)
                
                if results.face_landmarks:
                    # Draw Mesh if enabled
                    if show_mesh:
                        frame = detector.draw_landmarks(frame, results)
                    
                    # Check Violations
                    # Multiple Faces Only
                    face_count = len(results.face_landmarks)
                    fc_active, fc_triggered = st.session_state.violation_tracker.check_face_count(
                        face_count, 
                        config.thresholds.max_num_faces,
                        config.thresholds.violation_persistence_time
                    )

                    # Also reset no face timer since we found a face
                    st.session_state.violation_tracker.check_no_face(face_count, True, 0)
                    
                    if fc_triggered:
                        # Assuming ViolationTracker returns triggered status
                        # Or if we just rely on logs.
                        # The simplified tracker logs and returns active/triggered.
                        # Wait, simplified tracker returns (active, triggered) tuple? No, I defined `check_face_count` to return `(is_active, is_triggered)`.
                        # But in my plan above I wrote: returns True, True.
                        # Wait, I wrote `return is_active, is_triggered` in the file.
                        # And `is_active` is True if face_count > max, `is_triggered` is True if newly logged.
                        # So this logic holds.
                        last_msg = st.session_state.violation_tracker.violations[-1]['message']
                        st.toast(last_msg, icon="⚠️")
                    
                    if fc_active:
                        # Overlay Warning
                        cv2.putText(frame, "WARNING: MULTIPLE FACES!", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
                    
                    # Update Stats
                    stats_markdown = f"""
**Violations**: {st.session_state.violation_tracker.get_violation_count()}

**Face Analysis**  
- Faces Detected: {face_count} (Max: {config.thresholds.max_num_faces})
"""

                    if enable_head_pose:
                        poses = pose_estimator.extract_pose(results)
                        if poses:
                             stats_markdown += "\n**Head Pose Analysis**\n"
                             for i, pose in enumerate(poses):
                                 direction = pose_estimator.classify_direction(pose)
                                 stats_markdown += f"""
**Face {i+1}**
- Yaw: {pose['yaw']:.1f}°
- Pitch: {pose['pitch']:.1f}°
- Roll: {pose['roll']:.1f}°
- Direction: {direction}
"""
                                 # Draw Axes
                                 if results.face_landmarks and i < len(results.face_landmarks):
                                     landmarks = results.face_landmarks[i]
                                     nose = landmarks[1] # Tip of nose
                                     h, w, _ = frame.shape
                                     nx, ny = int(nose.x * w), int(nose.y * h)
                                     frame = pose_estimator.draw_axes(frame, pose['pitch'], pose['yaw'], pose['roll'], nx, ny)
                                
                                 # Check Head Pose Violations (Per Face)
                                 # For simplicity, if ANY face is looking away, trigger violation.
                                 # Or maybe just the "main" face? Let's assume all faces count.
                                 hp_active, hp_triggered, hp_msg = st.session_state.violation_tracker.check_head_pose(
                                     direction, config.thresholds.violation_persistence_time
                                 )
                                 
                                 if hp_triggered:
                                     st.toast(hp_msg, icon="⚠️")
                                     
                                 if hp_active:
                                     cv2.putText(frame, f"WARNING: {direction.upper()}!", (50, 100 + i*50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                                     cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
                    
                    stats_placeholder.markdown(stats_markdown)

                else:
                    # No face detected
                    # Check No Face Violation with Persistence
                    nf_active, nf_triggered, nf_msg = st.session_state.violation_tracker.check_no_face(
                        0, 
                        config.thresholds.enable_no_face_warning, 
                        config.thresholds.violation_persistence_time
                    )
                    
                    if nf_triggered:
                        st.toast(nf_msg, icon="⚠️")
                        
                    if nf_active:
                         # Overlay Warning
                        cv2.putText(frame, "WARNING: NO FACE DETECTED!", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
                        
                    # Update Stats (Show 0 faces)
                    stats_placeholder.markdown(f"""
                    **Face Analysis**:
                    - **Faces Detected**: 0 (Max: {config.thresholds.max_num_faces})
                    
                    **Violations**: {st.session_state.violation_tracker.get_violation_count()}
                    """)
                    
            except Exception as e:
                logger.error(f"Processing error: {e}")
            
            # Display
            # Convert to HTML to avoid Streamlit MediaFileStorageError
            b64_frame = frame_to_base64(frame)
            frame_placeholder.markdown(
                f'<img src="data:image/jpeg;base64,{b64_frame}" style="width: 100%;" />',
                unsafe_allow_html=True
            )

            # Update Logs Live
            with log_placeholder.container():
                if st.session_state.violation_tracker.get_violation_count() > 0:
                     for v in reversed(st.session_state.violation_tracker.get_logs()):
                        st.warning(f"{time.strftime('%H:%M:%S', time.localtime(v['timestamp']))}: {v['message']}")
                else:
                    st.info("No violations detected.")

    elif not st.session_state.is_running:
         frame_placeholder.info("Click 'Start' to begin.")

if __name__ == "__main__":
    main()
