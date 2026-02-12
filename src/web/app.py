import streamlit as st
import cv2
import tempfile
import time
import sys
import os
from typing import Union

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.video_stream import VideoStream
from src.engine.face.mesh_detector import MeshDetector
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

st.set_page_config(page_title="AI Proctoring Dashboard", layout="wide")

st.title("🎥 AI Proctoring System")
st.sidebar.title("Settings")

# Sidebar Configuration
source_type = st.sidebar.radio("Video Source", ["Webcam", "Video File"])
show_mesh = st.sidebar.checkbox("Show Face Mesh", value=True)
show_head_pose = st.sidebar.checkbox("Show Head Pose & Axis", value=True)
confidence = st.sidebar.slider("Min Detection Confidence", 0.0, 1.0, 0.5)

# Global variables for caching
if "video_stream" not in st.session_state:
    st.session_state.video_stream = None

# Calibration State
if "calibration_active" not in st.session_state:
    st.session_state.calibration_active = False
    st.session_state.calibration_start_time = 0
    st.session_state.calibration_data = []
    st.session_state.pose_offsets = {"pitch": 0, "yaw": 0, "roll": 0}

if "violation_tracker" not in st.session_state:
    from src.engine.proctor import ViolationTracker
    st.session_state.violation_tracker = ViolationTracker()

@st.cache_resource
def get_mesh_detector():
    return MeshDetector()

def main():
    detector = get_mesh_detector()
    
    # Source Selection Logic
    source: Union[int, str] = config.camera.index
    
    if source_type == "Video File":
        video_file = st.sidebar.file_uploader("Upload Video", type=['mp4', 'mov', 'avi'])
        if video_file is not None:
            # Save to temp file because cv2.VideoCapture needs a path
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(video_file.read())
            source = tfile.name
        else:
            st.info("Please upload a video file.")
            return

    # Start/Stop Controls
    col_btn1, col_btn2 = st.sidebar.columns(2)
    start_button = col_btn1.button("Start")
    stop_button = col_btn2.button("Stop")
    
    # Calibration Button (Only if Head Pose is enabled)
    if show_head_pose:
        if st.sidebar.button("Calibrate Head Pose"):
            st.session_state.calibration_active = True
            st.session_state.calibration_start_time = time.time()
            st.session_state.calibration_data = []
            st.sidebar.info("Calibration started. Look straight at the screen for 5 seconds.")
    
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
        
    if start_button:
        st.session_state.is_running = True
        
        # Reset Violation Tracker
        st.session_state.violation_tracker.reset()
        
        # Auto-Calibrate on Start if Head Pose is enabled
        if show_head_pose:
            st.session_state.calibration_active = True
            st.session_state.calibration_start_time = time.time()
            st.session_state.calibration_data = []
            st.session_state.pose_offsets = {"pitch": 0, "yaw": 0, "roll": 0}
            st.toast("Starting... Calibrating Head Pose (5s)", icon="⏳")
            
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

    # Processing Loop
    if st.session_state.is_running and st.session_state.video_stream:
        stream = st.session_state.video_stream
        
        # Initialize Estimators
        from src.engine.face.head_pose import HeadPoseEstimator
        pose_estimator = HeadPoseEstimator()
        
        while stream.is_opened():
            ret, frame = stream.read()
            if not ret:
                st.warning("End of video or failed to read frame.")
                st.session_state.is_running = False
                stream.stop()
                break
                
            # Process
            # Mirror the frame by default
            frame = cv2.flip(frame, 1)
            
            # Timestamp
            timestamp_ms = int(time.time() * 1000)
            
            # 1. Face Mesh Detection
            if show_mesh or show_head_pose:
                try:
                    results = detector.process(frame, timestamp_ms)
                    
                    if results.face_landmarks:
                        # Draw Mesh if enabled
                        if show_mesh:
                            frame = detector.draw_landmarks(frame, results)
                        
                        # Head Pose Estimation if enabled
                        if show_head_pose:
                            face_landmarks = results.face_landmarks[0]
                            (pitch, yaw, roll), rvec, tvec, cam_matrix = pose_estimator.estimate(face_landmarks, frame.shape)
                            
                            if pitch is not None:
                                # Apply Calibration Offsets
                                pitch -= st.session_state.pose_offsets["pitch"]
                                yaw -= st.session_state.pose_offsets["yaw"]
                                roll -= st.session_state.pose_offsets["roll"]
                                
                                # Calibration Logic
                                if st.session_state.calibration_active:
                                    elapsed = time.time() - st.session_state.calibration_start_time
                                    if elapsed < 5.0:
                                        st.session_state.calibration_data.append((pitch, yaw, roll))
                                        cv2.putText(frame, f"CALIBRATING... {5-elapsed:.1f}s", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                    else:
                                        # Finish Calibration
                                        st.session_state.calibration_active = False
                                        data = st.session_state.calibration_data
                                        if data:
                                            avg_pitch = sum(p for p, y, r in data) / len(data)
                                            avg_yaw = sum(y for p, y, r in data) / len(data)
                                            avg_roll = sum(r for p, y, r in data) / len(data)
                                            
                                            # Update Offsets (Accumulate)
                                            st.session_state.pose_offsets["pitch"] += avg_pitch
                                            st.session_state.pose_offsets["yaw"] += avg_yaw
                                            st.session_state.pose_offsets["roll"] += avg_roll
                                            
                                            st.sidebar.success(f"Calibration Complete! Offsets: P={avg_pitch:.1f}, Y={avg_yaw:.1f}, R={avg_roll:.1f}")

                                # Draw Axis
                                frame = pose_estimator.draw_axis(frame, rvec, tvec, cam_matrix)
                                
                                # Get Label (using corrected values)
                                label = pose_estimator.get_orientation_label(pitch, yaw, roll)
                                
                                # Check Violations
                                warning_active = st.session_state.violation_tracker.check_head_pose(label)
                                
                                if warning_active:
                                    # Overlay Warning
                                    cv2.putText(frame, "WARNING: LOOKING AWAY!", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                                    cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
                                
                                # Update Stats
                                stats_placeholder.markdown(f"""
                                **Head Pose Analysis**:
                                - **Direction**: {label}
                                - **Yaw**: {yaw:.1f}°
                                - **Pitch**: {pitch:.1f}°
                                - **Roll**: {roll:.1f}°
                                - **Offsets**: P={st.session_state.pose_offsets['pitch']:.1f}, Y={st.session_state.pose_offsets['yaw']:.1f}, R={st.session_state.pose_offsets['roll']:.1f}
                                
                                **Violations**: {st.session_state.violation_tracker.get_violation_count()}
                                """)
                                
                                if warning_active:
                                    st.error("⚠️ WARNING: FOCUS ON THE SCREEN!")
                                    
                    else:
                        stats_placeholder.info("No face detected.")
                        
                except Exception as e:
                    logger.error(f"Processing error: {e}")
            
            # Display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", width="stretch")

        with st.sidebar.expander("Proctoring Logs", expanded=True):
            if st.session_state.violation_tracker.get_violation_count() > 0:
                for v in reversed(st.session_state.violation_tracker.get_logs()):
                    st.warning(f"{time.strftime('%H:%M:%S', time.localtime(v['timestamp']))}: {v['message']}")
            else:
                st.info("No violations detected.")

    elif not st.session_state.is_running:
         frame_placeholder.info("Click 'Start' to begin.")

if __name__ == "__main__":
    main()
