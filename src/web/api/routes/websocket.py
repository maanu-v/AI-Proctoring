"""
WebSocket routes for real-time frame streaming
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from datetime import datetime
import json
import logging
import time
import base64
import numpy as np

from src.web.api.dependencies import get_session_manager, get_analyzer_manager
from src.web.core.session_manager import SessionManager
from src.web.core.analyzers import AnalyzerManager
from src.web.utils.image_utils import decode_base64_image
from src.utils.config import config

logger = logging.getLogger(__name__)

router = APIRouter(tags=["WebSocket"])


def convert_to_native(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(item) for item in obj]
    return obj


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    session_mgr: SessionManager = Depends(get_session_manager),
    analyzer_mgr: AnalyzerManager = Depends(get_analyzer_manager)
):
    """
    WebSocket endpoint for real-time frame streaming and analysis
    
    Client sends: {"type": "frame", "frame_base64": "data:image/jpeg;base64,..."}
    Server responds: {"type": "analysis", "data": {...}} or {"type": "error", "message": "..."}
    """
    await websocket.accept()
    
    # Validate session exists
    session = session_mgr.get_session(session_id)
    if session is None:
        await websocket.send_json({
            "type": "error",
            "message": f"Session not found: {session_id}"
        })
        await websocket.close(code=1008)  # Policy violation
        return
    
    logger.info(f"WebSocket connected for session {session_id}")
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            message_type = data.get("type")
            
            if message_type == "frame":
                # Process frame analysis
                try:
                    frame_base64 = data.get("frame_base64")
                    if not frame_base64:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Missing frame_base64 in request"
                        })
                        continue
                    
                    session.increment_frame_count()
                    
                    # Record frame received for no-frame violation tracking
                    session.violation_tracker.record_frame_received()
                    
                    # Decode frame
                    frame = decode_base64_image(frame_base64)
                    
                    # Get analyzers
                    analyzers = analyzer_mgr.get_analyzers()
                    
                    # Storage for analysis results
                    analysis_result = {
                        "face_detection": {},
                        "head_pose": {},
                        "gaze": {},
                        "blink": {},
                        "object_detection": {},
                        "identity_verification": {}
                    }
                    
                    violations = []
                    warnings = []
                    
                    # ====================================================================
                    # Face Detection & Mesh
                    # ====================================================================
                    face_count = 0
                    landmarks = None
                    mesh_result = None
                    
                    if session.settings["enable_face_detection"]:
                        # Use MediaPipe's process method with timestamp
                        timestamp_ms = int(time.time() * 1000)
                        mesh_result = analyzers["mesh"].process(frame, timestamp_ms)
                        
                        # Extract face count and landmarks from MediaPipe result
                        face_count = len(mesh_result.face_landmarks) if mesh_result.face_landmarks else 0
                        landmarks = mesh_result.face_landmarks if mesh_result.face_landmarks else None
                        
                        analysis_result["face_detection"] = {
                            "face_count": face_count,
                            "faces_detected": face_count > 0
                        }
                        
                        # Check for no face violation
                        if session.settings["enable_no_face_warning"]:
                            no_face_active, no_face_triggered, no_face_msg = session.violation_tracker.check_no_face(
                                face_count,
                                enabled=True,
                                persistence_time=config.thresholds.no_face_persistence_time
                            )
                            
                            if no_face_active:
                                warnings.append(no_face_msg)
                                if no_face_triggered:
                                    violations.append({
                                        "type": "no_face",
                                        "message": no_face_msg,
                                        "timestamp": datetime.now().isoformat()
                                    })
                        
                        # Check for multiple faces violation
                        face_count_active, face_count_triggered = session.violation_tracker.check_face_count(
                            face_count,
                            max_faces=config.thresholds.max_num_faces,
                            persistence_time=config.thresholds.multi_face_persistence_time
                        )
                        
                        if face_count_active:
                            msg = f"Multiple faces detected: {face_count}"
                            warnings.append(msg)
                            if face_count_triggered:
                                violations.append({
                                    "type": "multiple_faces",
                                    "message": msg,
                                    "timestamp": datetime.now().isoformat()
                                })
                    
                    # ====================================================================
                    # Head Pose Analysis
                    # ====================================================================
                    if session.settings["enable_head_pose"] and mesh_result is not None and landmarks is not None and len(landmarks) > 0:
                        # Extract poses from MediaPipe result
                        poses = analyzers["head_pose"].extract_pose(mesh_result)
                        
                        if poses and len(poses) > 0:
                            pose = poses[0]  # Use first face
                            direction = analyzers["head_pose"].classify_direction(pose)
                            
                            # Check if looking forward
                            is_looking_forward = (
                                abs(pose["yaw"]) <= config.head_pose.yaw_threshold and
                                abs(pose["pitch"]) <= config.head_pose.pitch_threshold
                            )
                            
                            analysis_result["head_pose"] = {
                                "direction": direction,
                                "yaw": float(pose["yaw"]),
                                "pitch": float(pose["pitch"]),
                                "roll": float(pose["roll"]),
                                "is_looking_forward": is_looking_forward
                            }
                            
                            # Check violations
                            hp_active, hp_triggered, hp_msg = session.violation_tracker.check_head_pose(
                                direction,
                                persistence_time=config.thresholds.head_pose_persistence_time
                            )
                            
                            if hp_active:
                                warnings.append(hp_msg)
                                if hp_triggered:
                                    violations.append({
                                        "type": "head_pose",
                                        "message": hp_msg,
                                        "timestamp": datetime.now().isoformat()
                                    })
                    
                    # ====================================================================
                    # Gaze Tracking
                    # ====================================================================
                    if session.settings["enable_gaze"] and landmarks is not None and len(landmarks) > 0:
                        # Get frame dimensions
                        h, w = frame.shape[:2]
                        direction, horizontal_ratio, vertical_ratio = analyzers["gaze"].estimate_gaze(landmarks[0], w, h)
                        
                        # Check if looking at screen (direction is "Center" or close to it)
                        is_looking_at_screen = direction == "Center"
                        
                        analysis_result["gaze"] = {
                            "direction": direction,
                            "horizontal_ratio": float(horizontal_ratio),
                            "vertical_ratio": float(vertical_ratio),
                            "is_looking_at_screen": is_looking_at_screen
                        }
                        
                        # Check violations
                        gaze_active, gaze_triggered, gaze_msg = session.violation_tracker.check_gaze_violation(
                            direction,
                            persistence_time=config.thresholds.gaze_persistence_time
                        )
                        
                        if gaze_active:
                            warnings.append(gaze_msg)
                            if gaze_triggered:
                                violations.append({
                                    "type": "gaze",
                                    "message": gaze_msg,
                                    "timestamp": datetime.now().isoformat()
                                })
                    
                    # ====================================================================
                    # Blink Detection
                    # ====================================================================
                    if session.settings["enable_blink"] and landmarks is not None and len(landmarks) > 0:
                        # Get frame dimensions
                        h, w = frame.shape[:2]
                        is_blinking, ear_left, ear_right, ear_avg = analyzers["blink"].estimate_blink(landmarks[0], w, h)
                        
                        # Get total blinks from the estimator
                        blink_features = analyzers["blink"].get_features()
                        
                        analysis_result["blink"] = {
                            "ear": float(ear_avg),
                            "ear_left": float(ear_left),
                            "ear_right": float(ear_right),
                            "eyes_closed": bool(is_blinking),
                            "total_blinks": int(blink_features.get("total_blinks", 0))
                        }
                        
                        # Check for prolonged eye closure
                        if blink_features.get("prolonged_closure_detected", False):
                            warnings.append("Prolonged eye closure detected")
                    
                    # ====================================================================
                    # Object Detection
                    # ====================================================================
                    if session.settings["enable_object_detection"]:
                        obj_result = analyzers["object"].detect(frame)
                        
                        # Format detections for response
                        formatted_detections = []
                        for box, class_name, conf in obj_result.get("detections", []):
                            formatted_detections.append({
                                "class": class_name,
                                "confidence": float(conf),
                                "box": [float(x) for x in box]
                            })
                        
                        analysis_result["object_detection"] = {
                            "phone_detected": obj_result.get("phone_detected", False),
                            "person_count": obj_result.get("person_count", 0),
                            "objects": formatted_detections
                        }
                        
                        # Check violations
                        obj_active, obj_triggered, obj_msg = session.violation_tracker.check_object_violation(
                            obj_result,
                            persistence_time=2.0
                        )
                        
                        if obj_active:
                            warnings.append(obj_msg)
                            if obj_triggered:
                                violations.append({
                                    "type": "object_detection",
                                    "message": obj_msg,
                                    "timestamp": datetime.now().isoformat()
                                })
                    
                    # ====================================================================
                    # Identity Verification
                    # ====================================================================
                    if session.settings["enable_identity_verification"] and session.reference_embedding is not None:
                        # Check periodically (not every frame for performance)
                        if session.frame_count % config.thresholds.identity_check_interval_frames == 0:
                            if landmarks is not None and len(landmarks) > 0:
                                current_embedding = analyzers["embedder"].get_embedding(frame)
                                
                                if current_embedding is not None:
                                    is_match, distance = analyzers["embedder"].compare_embeddings(
                                        session.reference_embedding,
                                        current_embedding,
                                        threshold=0.68
                                    )
                                    
                                    analysis_result["identity_verification"] = {
                                        "match": bool(is_match),
                                        "distance": float(distance),
                                        "checked_at_frame": int(session.frame_count)
                                    }
                                    
                                    # Check violations
                                    id_active, id_triggered, id_msg = session.violation_tracker.check_identity(
                                        is_match,
                                        persistence_time=config.thresholds.identity_persistence_time
                                    )
                                    
                                    if id_active:
                                        warnings.append(id_msg)
                                        if id_triggered:
                                            violations.append({
                                                "type": "identity_mismatch",
                                                "message": id_msg,
                                                "timestamp": datetime.now().isoformat()
                                            })
                    
                    # ====================================================================
                    # No Frame Violation Check
                    # ====================================================================
                    if session.settings.get("enable_no_frame_warning", True):
                        nf_active, nf_triggered, nf_msg = session.violation_tracker.check_no_frames(
                            persistence_time=config.thresholds.no_frame_persistence_time
                        )
                        if nf_active:
                            warnings.append(nf_msg)
                            if nf_triggered:
                                violations.append({
                                    "type": "no_frames",
                                    "message": nf_msg,
                                    "timestamp": datetime.now().isoformat()
                                })
                    
                    # ====================================================================
                    # Send Analysis Result
                    # ====================================================================
                    # Convert all numpy types to native Python types for JSON serialization
                    analysis_result = convert_to_native(analysis_result)
                    violations = convert_to_native(violations)
                    
                    response = {
                        "type": "analysis",
                        "data": {
                            "session_id": session_id,
                            "frame_count": session.frame_count,
                            "timestamp": datetime.now().isoformat(),
                            "face_detected": face_count > 0,
                            "face_count": face_count,
                            "violations": violations,
                            "analysis": analysis_result,
                            "warnings": warnings
                        }
                    }
                    
                    await websocket.send_json(response)
                    
                except Exception as e:
                    logger.error(f"Error analyzing frame: {e}", exc_info=True)
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Analysis error: {str(e)}"
                    })
            
            elif message_type == "audio":
                # Process audio for voice activity detection
                try:
                    audio_base64 = data.get("audio_base64")
                    if not audio_base64:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Missing audio_base64 in request"
                        })
                        continue

                    audio_bytes = base64.b64decode(audio_base64)
                    analyzers = analyzer_mgr.get_analyzers()

                    audio_result = {}
                    violations = []
                    warnings = []

                    if session.settings.get("enable_audio_detection", True):
                        vad_result = analyzers["vad"].process(audio_bytes)

                        audio_result = {
                            "is_speech": vad_result["is_speech"],
                            "speech_duration": vad_result["speech_duration"],
                            "speech_segments": vad_result["speech_segments"],
                            "frames_with_speech": vad_result["frames_with_speech"],
                            "total_frames": vad_result["total_frames"],
                        }

                        # Check speech violation
                        speech_active, speech_triggered, speech_msg = session.violation_tracker.check_speech_violation(
                            vad_result["is_speech"],
                            vad_result["speech_duration"],
                            persistence_time=config.thresholds.speech_persistence_time
                        )

                        if speech_active:
                            warnings.append(speech_msg)
                            if speech_triggered:
                                violations.append({
                                    "type": "speech_detected",
                                    "message": speech_msg,
                                    "timestamp": datetime.now().isoformat()
                                })

                    response = {
                        "type": "audio_analysis",
                        "data": {
                            "session_id": session_id,
                            "timestamp": datetime.now().isoformat(),
                            "audio": audio_result,
                            "violations": violations,
                            "warnings": warnings
                        }
                    }

                    await websocket.send_json(response)

                except Exception as e:
                    logger.error(f"Error analyzing audio: {e}", exc_info=True)
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Audio analysis error: {str(e)}"
                    })

            elif message_type == "ping":
                # Respond to ping with pong
                await websocket.send_json({"type": "pong"})
            
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {message_type}"
                })
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Server error: {str(e)}"
            })
        except:
            pass
    finally:
        logger.info(f"WebSocket closed for session {session_id}")
