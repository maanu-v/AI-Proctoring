"""
Single-video processor for batch proctoring analysis.

Processes a video file frame-by-frame using all engine analyzers
(face mesh, head pose, gaze, blink, object detection) and produces
structured violation data with timestamps.
"""

import cv2
import time
import os
import re
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    import imageio_ffmpeg
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)


def parse_ground_truth(gt_path: str) -> List[Dict[str, Any]]:
    """
    Parse a ground truth file (gt.txt).
    
    Format: START_TIME  END_TIME  TYPE
    Times are in MMSS format (e.g., 0135 = 1 min 35 sec).
    
    Cheating types (from the dataset):
    1 = Looking at notes/book
    2 = Looking at phone/device
    3 = Talking to someone
    4 = Passing notes
    5 = Using unauthorized materials
    6 = Other suspicious behavior
    """
    cheating_type_labels = {
        1: "Looking at notes/book",
        2: "Looking at phone/device",
        3: "Talking to someone",
        4: "Passing notes",
        5: "Using unauthorized materials",
        6: "Other suspicious behavior",
    }
    
    ground_truth = []
    if not os.path.exists(gt_path):
        return ground_truth
    
    try:
        with open(gt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = re.split(r'\s+', line)
                if len(parts) >= 3:
                    start_str, end_str, cheat_type_str = parts[0], parts[1], parts[2]
                    
                    # Parse MMSS to seconds
                    start_sec = int(start_str[:2]) * 60 + int(start_str[2:])
                    end_sec = int(end_str[:2]) * 60 + int(end_str[2:])
                    cheat_type = int(cheat_type_str)
                    
                    ground_truth.append({
                        "start_time": start_sec,
                        "end_time": end_sec,
                        "start_str": f"{int(start_str[:2])}:{start_str[2:]}",
                        "end_str": f"{int(end_str[:2])}:{end_str[2:]}",
                        "type": cheat_type,
                        "type_label": cheating_type_labels.get(cheat_type, f"Unknown ({cheat_type})")
                    })
    except Exception as e:
        logger.error(f"Error parsing ground truth {gt_path}: {e}")
    
    return ground_truth


def process_single_video(
    video_path: str,
    subject_id: str,
    gt_path: str,
    output_dir: str,
    sample_rate: int = 3,
) -> Dict[str, Any]:
    """
    Process a single video file through all engine analyzers.
    
    Args:
        video_path: Path to the .avi video file
        subject_id: Subject identifier (e.g., 'subject1')
        gt_path: Path to ground truth file
        output_dir: Directory to write results JSON
        sample_rate: Process every Nth frame (default: 3)
    
    Returns:
        Result dictionary with violations and metadata
    """
    # Import engine modules here to avoid initialization issues with multiprocessing
    from src.engine.face.mesh_detector import MeshDetector
    from src.engine.face.head_pose import HeadPoseEstimator
    from src.engine.face.gaze_estimation import GazeEstimator
    from src.engine.face.blink_estimation import BlinkEstimator
    from src.engine.obj_detection.obj_detect import ObjectDetector
    from src.utils.config import config

    logger.info(f"[{subject_id}] Starting processing: {video_path}")
    start_time = time.time()

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"[{subject_id}] Failed to open video: {video_path}")
        return {"error": f"Failed to open video: {video_path}"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    logger.info(f"[{subject_id}] Video: {total_frames} frames, {fps:.1f} fps, {width}x{height}, {duration:.1f}s")

    # Initialize engine components
    try:
        mesh_detector = MeshDetector()
        head_pose_estimator = HeadPoseEstimator(
            yaw_threshold=config.head_pose.yaw_threshold,
            pitch_threshold=config.head_pose.pitch_threshold,
            roll_threshold=config.head_pose.roll_threshold,
        )
        gaze_estimator = GazeEstimator()
        blink_estimator = BlinkEstimator()
        obj_detector = ObjectDetector()
        # Note: ViolationTracker is handled inline via _check_and_track_violation
    except Exception as e:
        logger.error(f"[{subject_id}] Failed to initialize engine: {e}")
        cap.release()
        return {"error": f"Engine init failed: {e}"}

    # Parse ground truth
    ground_truth = parse_ground_truth(gt_path)

    # Frame-level analysis data (sampled for storage)
    frame_analyses = []
    
    # Track last known state for drawing
    last_mesh_result = None
    last_head_poses = None
    last_gaze_dir = "Center"
    last_obj_data = {"person_count": 0, "phone_detected": False, "detections": []}

    # Setup FFMPEG process for overlying video
    os.makedirs(output_dir, exist_ok=True)
    overlaid_mp4_path = os.path.join(output_dir, f"{subject_id}_overlaid.mp4")
    
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()  # type: ignore[name-defined]
        ffmpeg_cmd = [
            ffmpeg_exe, '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f"{width}x{height}", '-pix_fmt', 'bgr24', '-r', str(fps),
            '-i', '-',  # stdin
            '-i', video_path,  # original video for audio
            '-c:v', 'libx264', '-crf', '28', '-preset', 'fast',
            '-c:a', 'aac', '-b:a', '128k',
            '-map', '0:v:0', '-map', '1:a:0?',
            overlaid_mp4_path
        ]
        ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except Exception as e:
        logger.error(f"[{subject_id}] Failed to start FFMPEG process: {e}")
        ffmpeg_proc = None
    
    # Track violation segments (merged from ViolationTracker)
    active_violations: Dict[str, Dict] = {}  # type -> {start, last_seen, message}
    violation_segments: List[Dict[str, Any]] = []

    frame_idx = 0
    processed_count = 0
    last_timestamp_ms = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Check if we should process this frame with heavy models
        is_sampled_frame = (frame_idx % sample_rate == 0)
        
        if is_sampled_frame:
            processed_count += 1
        
        current_time_sec = frame_idx / fps
        timestamp_ms = int(current_time_sec * 1000)
        
        # Ensure monotonically increasing timestamps for MediaPipe
        if timestamp_ms <= last_timestamp_ms:
            timestamp_ms = last_timestamp_ms + 1
        last_timestamp_ms = timestamp_ms

        # --- Run analyzers (only on sampled frames) ---
        if is_sampled_frame:
            frame_data = {
                "frame": frame_idx,
                "time": round(current_time_sec, 2),
            }

            # 1. Face mesh detection
            try:
                mesh_result = mesh_detector.process(frame, timestamp_ms)
                face_count = len(mesh_result.face_landmarks) if (mesh_result and mesh_result.face_landmarks) else 0
                frame_data["face_count"] = face_count
                last_mesh_result = mesh_result
            except Exception as e:
                logger.debug(f"[{subject_id}] Mesh error at frame {frame_idx}: {e}")
                face_count = 0
                frame_data["face_count"] = 0
                mesh_result = None
                last_mesh_result = None

            # 2. Head pose
            head_direction = "Forward"
            if mesh_result and mesh_result.face_landmarks and len(mesh_result.face_landmarks) > 0:
                try:
                    poses = head_pose_estimator.extract_pose(mesh_result)
                    last_head_poses = poses
                    if poses:
                        head_direction = head_pose_estimator.classify_direction(poses[0])
                        frame_data["head_pose"] = {
                            "direction": head_direction,
                            "yaw": round(poses[0]["yaw"], 1),
                            "pitch": round(poses[0]["pitch"], 1),
                            "roll": round(poses[0]["roll"], 1),
                        }
                except Exception as e:
                    logger.debug(f"[{subject_id}] Head pose error at frame {frame_idx}: {e}")
                    last_head_poses = None
            else:
                last_head_poses = None

            # 3. Gaze estimation
            gaze_direction = "Center"
            if mesh_result and mesh_result.face_landmarks and len(mesh_result.face_landmarks) > 0:
                try:
                    landmarks = mesh_result.face_landmarks[0]
                    gaze_dir, h_ratio, v_ratio = gaze_estimator.estimate_gaze(landmarks, width, height)
                    gaze_direction = gaze_dir
                    last_gaze_dir = gaze_dir
                    frame_data["gaze"] = {
                        "direction": gaze_direction,
                        "h_ratio": round(h_ratio, 3),
                        "v_ratio": round(v_ratio, 3),
                    }
                except Exception as e:
                    logger.debug(f"[{subject_id}] Gaze error at frame {frame_idx}: {e}")

            # 4. Blink estimation
            if mesh_result and mesh_result.face_landmarks and len(mesh_result.face_landmarks) > 0:
                try:
                    landmarks = mesh_result.face_landmarks[0]
                    is_blinking, ear_l, ear_r, ear_avg = blink_estimator.estimate_blink(
                        landmarks, width, height
                    )
                    frame_data["blink"] = {
                        "is_blinking": is_blinking,
                        "ear_avg": round(ear_avg, 3),
                    }
                except Exception as e:
                    logger.debug(f"[{subject_id}] Blink error at frame {frame_idx}: {e}")

            # 5. Object detection (run less frequently — every 5th sampled frame)
            if processed_count % 5 == 0:
                try:
                    obj_data = obj_detector.detect(frame)
                    last_obj_data = obj_data
                    frame_data["objects"] = {
                        "person_count": obj_data["person_count"],
                        "phone_detected": obj_data["phone_detected"],
                    }
                except Exception as e:
                    logger.debug(f"[{subject_id}] Object detection error at frame {frame_idx}: {e}")
                    obj_data = {"person_count": 0, "phone_detected": False, "detections": []}
            else:
                obj_data = last_obj_data

            # --- Check violations using ViolationTracker ---
            _check_and_track_violation(
                active_violations, violation_segments, current_time_sec,
                face_count, head_direction, gaze_direction,
                obj_data, config
            )

            # Store sampled frame data (every 10th processed frame to limit JSON size)
            if processed_count % 10 == 0:
                frame_analyses.append(frame_data)

        # --- Overlay Drawing for Video Output ---
        if ffmpeg_proc:
            annotated_frame = frame.copy()
            
            # Draw Face Mesh
            if last_mesh_result:
                annotated_frame = mesh_detector.draw_landmarks(annotated_frame, last_mesh_result)
                
                # Draw Head Pose Axes
                if last_head_poses and last_mesh_result.face_landmarks:
                    for i, pose in enumerate(last_head_poses):
                        if i < len(last_mesh_result.face_landmarks):
                            nose_landmark = last_mesh_result.face_landmarks[i][1]
                            nx, ny = int(nose_landmark.x * width), int(nose_landmark.y * height)
                            annotated_frame = head_pose_estimator.draw_axes(
                                annotated_frame, pose['pitch'], pose['yaw'], pose['roll'], nx, ny
                            )
            
            # Draw Object Detections
            if last_obj_data and 'detections' in last_obj_data:
                for box, class_name, conf in last_obj_data['detections']:
                    x1, y1, x2, y2 = map(int, box)
                    color = (0, 0, 255) if class_name == 'cell phone' else (255, 165, 0)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, f"{class_name} {conf:.2f}", (x1, max(0, y1-10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw Active Violations
            y_pos = 30
            for vtype, vdata in active_violations.items():
                cv2.putText(annotated_frame, f"VIOLATION: {vdata['message']}", (20, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                y_pos += 30
                
            # Draw Gaze Direction
            cv2.putText(annotated_frame, f"Gaze: {last_gaze_dir}", (20, height - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            try:
                ffmpeg_proc.stdin.write(annotated_frame.tobytes())
            except Exception as e:
                logger.error(f"[{subject_id}] Error writing to FFMPEG pipe: {e}")
                ffmpeg_proc = None

        # Progress logging (skip frame 0 to avoid immediate 0% spam on startup)
        if frame_idx > 0 and (frame_idx % 500) == 0:
            pct = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
            logger.info(f"[{subject_id}] Progress: {pct:.1f}% ({processed_count} frames processed)")

        frame_idx += 1  # <-- MUST be at the END of the loop so current_time_sec is correct

    # Close any remaining active violations at end of video
    for vtype, vdata in active_violations.items():
        duration_v = vdata["last_seen"] - vdata["start"]
        if duration_v >= 1.0:  # same filter as mid-video — drop sub-second ghosts
            violation_segments.append({
                "type": vtype,
                "start_time": round(vdata["start"], 2),
                "end_time": round(vdata["last_seen"], 2),
                "duration": round(duration_v, 2),
                "message": vdata["message"],
                "severity": _get_severity(vtype),
            })

    cap.release()
    
    if ffmpeg_proc:
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
        
    try:
        mesh_detector.close()
    except:
        pass

    processing_time = time.time() - start_time
    logger.info(f"[{subject_id}] Completed in {processing_time:.1f}s. Found {len(violation_segments)} violation segments.")

    # Build violation summary
    violation_type_counts: Dict[str, int] = {}
    for v in violation_segments:
        vtype = v["type"]  # keep the full type key (e.g. "no_face", "head_pose")
        violation_type_counts[vtype] = violation_type_counts.get(vtype, 0) + 1

    total_violation_duration = sum(v.get("duration", 0) for v in violation_segments)
    risk_score = min(1.0, total_violation_duration / max(duration, 1) * 5)  # Normalized risk

    result = {
        "subject_id": subject_id,
        "video_path": video_path,
        "overlaid_video_path": os.path.relpath(overlaid_mp4_path, PROJECT_ROOT),
        "video_metadata": {
            "duration_seconds": round(duration, 2),
            "fps": round(fps, 1),
            "total_frames": total_frames,
            "frames_processed": processed_count,
            "sample_rate": sample_rate,
            "width": width,
            "height": height,
        },
        "processing": {
            "processing_time_seconds": round(processing_time, 2),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "ground_truth": ground_truth,
        "violations": sorted(violation_segments, key=lambda x: x["start_time"]),
        "summary": {
            "total_violations": len(violation_segments),
            "total_violation_duration_seconds": round(total_violation_duration, 2),
            "violation_types": violation_type_counts,
            "risk_score": round(risk_score, 3),
        },
        "frame_analyses_sample": frame_analyses[:200],  # Cap for JSON size
    }

    # Write per-subject JSON
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{subject_id}_results.json")
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    logger.info(f"[{subject_id}] Results saved to {output_path}")
    return result


def _check_and_track_violation(
    active_violations: Dict[str, Dict],
    violation_segments: List[Dict],
    current_time: float,
    face_count: int,
    head_direction: str,
    gaze_direction: str,
    obj_data: Dict,
    config,
):
    """
    Track violations using video timestamps (not real-time).
    Builds violation segments with start/end times.
    """
    violation_gap_threshold = 2.0  # seconds gap to consider a violation ended

    # Define active violations this frame
    current_violations = set()

    # No face
    if face_count == 0:
        current_violations.add("no_face")
        if "no_face" not in active_violations:
            active_violations["no_face"] = {
                "start": current_time,
                "last_seen": current_time,
                "message": "No face detected",
            }
        else:
            active_violations["no_face"]["last_seen"] = current_time

    # Multiple faces
    if face_count > config.thresholds.max_num_faces:
        vtype = "multiple_faces"
        current_violations.add(vtype)
        if vtype not in active_violations:
            active_violations[vtype] = {
                "start": current_time,
                "last_seen": current_time,
                "message": f"Multiple faces detected: {face_count}",
            }
        else:
            active_violations[vtype]["last_seen"] = current_time
            active_violations[vtype]["message"] = f"Multiple faces detected: {face_count}"

    # Head pose
    if head_direction != "Forward":
        vtype = f"head_pose"
        current_violations.add(vtype)
        if vtype not in active_violations:
            active_violations[vtype] = {
                "start": current_time,
                "last_seen": current_time,
                "message": f"Head: {head_direction}",
            }
        else:
            active_violations[vtype]["last_seen"] = current_time
            active_violations[vtype]["message"] = f"Head: {head_direction}"

    # Gaze
    if gaze_direction != "Center":
        vtype = "gaze"
        current_violations.add(vtype)
        if vtype not in active_violations:
            active_violations[vtype] = {
                "start": current_time,
                "last_seen": current_time,
                "message": f"Gaze: {gaze_direction}",
            }
        else:
            active_violations[vtype]["last_seen"] = current_time
            active_violations[vtype]["message"] = f"Gaze: {gaze_direction}"

    # Phone detected
    if obj_data.get("phone_detected", False):
        vtype = "phone_detected"
        current_violations.add(vtype)
        if vtype not in active_violations:
            active_violations[vtype] = {
                "start": current_time,
                "last_seen": current_time,
                "message": "Mobile phone detected",
            }
        else:
            active_violations[vtype]["last_seen"] = current_time

    # Multiple people (body detection)
    if obj_data.get("person_count", 0) > 1:
        vtype = "multiple_people"
        current_violations.add(vtype)
        if vtype not in active_violations:
            active_violations[vtype] = {
                "start": current_time,
                "last_seen": current_time,
                "message": f"Multiple people: {obj_data['person_count']}",
            }
        else:
            active_violations[vtype]["last_seen"] = current_time
            active_violations[vtype]["message"] = f"Multiple people: {obj_data['person_count']}"

    # Close violations that are no longer active
    ended = []
    for vtype, vdata in active_violations.items():
        if vtype not in current_violations:
            gap = current_time - vdata["last_seen"]
            if gap > violation_gap_threshold:
                duration = vdata["last_seen"] - vdata["start"]
                # Only record violations lasting > 1 second
                if duration >= 1.0:
                    violation_segments.append({
                        "type": vtype,
                        "start_time": round(vdata["start"], 2),
                        "end_time": round(vdata["last_seen"], 2),
                        "duration": round(duration, 2),
                        "message": vdata["message"],
                        "severity": _get_severity(vtype),
                    })
                ended.append(vtype)

    for vtype in ended:
        del active_violations[vtype]


def _get_severity(violation_type: str) -> str:
    """Map violation type to severity level."""
    severity_map = {
        "no_face": "high",
        "multiple_faces": "high",
        "phone_detected": "critical",
        "multiple_people": "high",
        "head_pose": "medium",
        "gaze": "low",
    }
    return severity_map.get(violation_type, "medium")
