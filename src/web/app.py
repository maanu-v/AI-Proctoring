import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import sys
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import base64
from concurrent.futures import ThreadPoolExecutor
from typing import Union

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.video_stream import VideoStream
from src.engine.face.mesh_detector import MeshDetector
from src.engine.face.head_pose import HeadPoseEstimator
from src.engine.obj_detection.obj_detect import ObjectDetector
from src.engine.face.face_embedding import FaceEmbedder
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
if "violation_tracker" not in st.session_state:
    from src.engine.proctor import ViolationTracker
    st.session_state.violation_tracker = ViolationTracker()
    
# Background Thread Executor
if "executor" not in st.session_state:
    st.session_state.executor = ThreadPoolExecutor(max_workers=1)
if "identity_future" not in st.session_state:
    st.session_state.identity_future = None
    
if "identity_violation_active" not in st.session_state:
    st.session_state.identity_violation_active = False
    st.session_state.identity_violation_msg = ""
    st.session_state.identity_violation_msg = ""
    st.session_state.identity_violation_triggered = False
    st.session_state.identity_violation_start_time = 0
if "identity_score" not in st.session_state:
    st.session_state.identity_score = None

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

@st.cache_resource
def get_object_detector():
    return ObjectDetector()

@st.cache_resource
def get_face_embedder():
    return FaceEmbedder(model_name="ArcFace")

def main():
    detector = get_mesh_detector(config.mediapipe.num_faces)
    pose_estimator = get_head_pose_estimator()
    obj_detector = get_object_detector()
    face_embedder = get_face_embedder()
    
    # Session State for Reference Embedding
    if "reference_embedding" not in st.session_state:
        st.session_state.reference_embedding = None
        
    # Reference Image Uploader (Sidebar)
    st.sidebar.markdown("### Identity Verification")
    ref_image_file = st.sidebar.file_uploader("Upload Profile Image", type=['jpg', 'jpeg', 'png'])
    
    if ref_image_file:
        # Convert to CV2 image
        file_bytes = np.asarray(bytearray(ref_image_file.read()), dtype=np.uint8)
        ref_image = cv2.imdecode(file_bytes, 1)
        
        if ref_image is not None:
             # Compute embedding if not already set or changed
             # Simple check: reuse embedding if file name same? 
             # Streamlit re-runs, so we need to check if we already computed it for this file.
             # We can't easily check file identity, but we can check if st.session_state.reference_embedding is None.
             # Better: Compute once per upload. 
             # Only recompute if filename changes? 
             # Or just recompute now, it's one-time per run.
             # BUT deepface is slow, so we should cache it.
             
             if "last_ref_image_name" not in st.session_state or st.session_state.last_ref_image_name != ref_image_file.name:
                 with st.spinner("Computing Reference Embedding..."):
                     st.session_state.reference_embedding = face_embedder.get_embedding(ref_image)
                     st.session_state.last_ref_image_name = ref_image_file.name
                     if st.session_state.reference_embedding:
                         st.sidebar.success("Reference Embedding Set!")
                     else:
                         st.sidebar.error("No face detected in reference image!")
        
    # Frame Counter for skipping object detection frames

    if "frame_count" not in st.session_state:
        st.session_state.frame_count = 0
    if "last_obj_data" not in st.session_state:
        st.session_state.last_obj_data = {}
    
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
                st.session_state.is_running = False
                stream.stop()
                break
            
            st.session_state.frame_count += 1

                
            # Global FPS Throttling to prevent Streamlit MediaFileStorageError
            # Cap at ~30 FPS
            time.sleep(0.03)
                
            # Mirror the frame by default
            frame = cv2.flip(frame, 1)
            
            # Timestamp
            timestamp_ms = int(time.time() * 1000)
            
            # Calibration Logic
            if config.head_pose.auto_calibration and "calibration_done" not in st.session_state:
                if "calibration_start" not in st.session_state:
                    st.session_state.calibration_start = time.time()
                    st.session_state.calibration_data = {0: {'pitch': [], 'yaw': [], 'roll': []}} # Buffer for face 0
                    st.toast("Calibrating... Please look forward.", icon="🎯")
                
                elapsed = time.time() - st.session_state.calibration_start
                remaining = config.head_pose.calibration_time - elapsed
                
                if remaining > 0:
                    # Show Calibration Overlay
                    cv2.putText(frame, f"CALIBRATING... {int(remaining)+1}s", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                    cv2.putText(frame, "LOOK FORWARD", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                    
                    # Collect Data
                    results = detector.process(frame, timestamp_ms)
                    if results.face_landmarks:
                        # Use raw pose extraction (we haven't set offsets yet)
                        # We need valid poses to average.
                        # Note: pose_estimator.extract_pose ALREADY applies offsets if set.
                        # Initial offsets are empty, so it returns raw-ish (smoothed) values.
                        poses = pose_estimator.extract_pose(results)
                        if poses:
                            # Assume main face is index 0 for calibration
                            pose = poses[0]
                            st.session_state.calibration_data[0]['pitch'].append(pose['pitch'])
                            st.session_state.calibration_data[0]['yaw'].append(pose['yaw'])
                            st.session_state.calibration_data[0]['roll'].append(pose['roll'])
                            
                            # Draw mesh/axes during calibration too?
                            if show_mesh:
                                frame = detector.draw_landmarks(frame, results)
                            # Draw axes for visual feedback
                            # ... (reuse drawing logic or just show overlay)
                else:
                    # Finish Calibration
                    data = st.session_state.calibration_data[0]
                    if data['pitch']:
                        avg_pitch = sum(data['pitch']) / len(data['pitch'])
                        avg_yaw = sum(data['yaw']) / len(data['yaw'])
                        avg_roll = sum(data['roll']) / len(data['roll'])
                        
                        pose_estimator.set_calibration_offsets(0, avg_pitch, avg_yaw, avg_roll)
                        logger.info(f"Calibration Complete. Offsets: P={avg_pitch:.1f}, Y={avg_yaw:.1f}, R={avg_roll:.1f}")
                        st.toast("Calibration Complete!", icon="✅")
                    else:
                        st.warning("Calibration failed: No face detected.")
                    
                    st.session_state.calibration_done = True
                    # Clean up
                    del st.session_state.calibration_start
                    del st.session_state.calibration_data
                    
                # Skip normal processing loop during calibration?
                # Yes, to avoid triggering violations while calibrating.
                
                # Display Frame
                b64_frame = frame_to_base64(frame)
                frame_placeholder.markdown(
                    f'<img src="data:image/jpeg;base64,{b64_frame}" style="width: 100%;" />',
                    unsafe_allow_html=True
                )
                continue 

            
            # 1. Face Mesh Detection
            # Only run if we need mesh or checks, but user wants check so run always
            try:
                results = detector.process(frame, timestamp_ms)
                
                # 2. Object Detection (Every 30 frames ~ 1 sec)
                if st.session_state.frame_count % 30 == 0:
                    st.session_state.last_obj_data = obj_detector.detect(frame)
                    
                    # 3. Identity Verification (Background Thread)
                    # Check if previous task completed
                    if st.session_state.identity_future is not None:
                        if st.session_state.identity_future.done():
                            try:
                                is_match, score = st.session_state.identity_future.result()
                                st.session_state.identity_future = None # Reset
                                st.session_state.identity_score = score
                                
                                # Update Violation Tracker with result
                                id_active, id_triggered, id_msg = st.session_state.violation_tracker.check_identity(
                                    is_match, 
                                    persistence_time=config.thresholds.identity_persistence_time
                                )
                                
                                # Store result for continuous overlay (since check is infrequent)
                                st.session_state.identity_violation_active = id_active
                                st.session_state.identity_violation_msg = id_msg
                                st.session_state.identity_violation_triggered = id_triggered
                                
                                if id_active:
                                    st.session_state.identity_violation_start_time = time.time()
                                
                            except Exception as e:
                                logger.error(f"Identity check future failed: {e}")
                                st.session_state.identity_future = None

                    # Trigger new check periodically
                    if st.session_state.frame_count % config.thresholds.identity_check_interval_frames == 0:
                        if st.session_state.reference_embedding is not None and st.session_state.identity_future is None:
                             # Submit task to background thread
                             # We copy frame to keep it safe? Actually deepface reads it. 
                             # Frame is numpy array, passing it is fine (copy might happen or ref).
                             # Use .copy() to be safe if frame is modified elsewhere.
                             frame_copy = frame.copy()
                             ref_emb = st.session_state.reference_embedding
                             
                             def run_verification(img, ref):
                                 try:
                                     # DeepFace might be slow, so this runs in thread
                                     curr = face_embedder.get_embedding(img)
                                     if curr is not None:
                                         return face_embedder.compare_embeddings(ref, curr)
                                     return True, 0.0 # Assume match if face not found to avoid false positive
                                 except Exception as e:
                                     logger.error(f"Error in verification thread: {e}")
                                     return True, 0.0 # Fail safe

                             st.session_state.identity_future = st.session_state.executor.submit(run_verification, frame_copy, ref_emb)
                    
                    # Display Violation Overlay (Persisted until next check clears it)
                    if getattr(st.session_state, 'identity_violation_triggered', False):
                        st.toast(st.session_state.identity_violation_msg, icon="🚫")
                        st.session_state.identity_violation_triggered = False # Show toast once per trigger
                        
                    if getattr(st.session_state, 'identity_violation_active', False):
                         # Show only for 10 seconds after detection
                         if time.time() - getattr(st.session_state, 'identity_violation_start_time', 0) < 10:
                             cv2.putText(frame, "IDENTITY VERIFICATION FAILED!", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                             cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
                         else:
                             # Expire the active state visually (stats will still show Mismatch until next check? 
                             # User said "make the warning as overlay for 10 secs". 
                             # We can keep the state "active" in background but hide overlay.
                             pass

                
                obj_data = st.session_state.last_obj_data
                
                # Check Object Violations
                obj_active, obj_triggered, obj_msg = st.session_state.violation_tracker.check_object_violation(
                    obj_data, 
                    persistence_time=2.0
                )
                
                if obj_triggered:
                    st.toast(obj_msg, icon="⚠️")
                    
                if obj_active:
                     cv2.putText(frame, "WARNING: FORBIDDEN OBJECT!", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                     cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)

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
                        config.thresholds.multi_face_persistence_time
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
                        cv2.putText(frame, "WARNING: MULTIPLE FACES!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
                    
                    # Update Stats
                    stats_markdown = f"""
**Violations**: {st.session_state.violation_tracker.get_violation_count()}

**Face Analysis**  
- Faces Detected: {face_count} (Max: {config.thresholds.max_num_faces})

**Object Analysis**
- People Count: {obj_data.get('person_count', 0)}
- Phone Detected: {obj_data.get('phone_detected', False)}

**Identity Verification**
- Status: {"Verified" if not getattr(st.session_state, 'identity_violation_active', False) else "Mismatch"}
- Distance: {f"{st.session_state.identity_score:.4f}" if st.session_state.identity_score is not None else "N/A"} (Threshold: 0.68)
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
                                     direction, config.thresholds.head_pose_persistence_time
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
                        config.thresholds.no_face_persistence_time
                    )
                    
                    if nf_triggered:
                        st.toast(nf_msg, icon="⚠️")
                        
                    if nf_active:
                         # Overlay Warning
                        cv2.putText(frame, "WARNING: NO FACE DETECTED!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
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
