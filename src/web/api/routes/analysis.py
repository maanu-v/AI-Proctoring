"""
Frame analysis routes
"""

from fastapi import APIRouter, HTTPException, Body, Depends
from datetime import datetime
import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from src.web.api.models import AnalysisResult
from src.web.api.dependencies import get_session, get_analyzer_manager
from src.web.core.session_manager import QuizSession
from src.web.core.analyzers import AnalyzerManager
from src.web.utils.image_utils import decode_base64_image
from src.utils.config import config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/analyze", tags=["Frame Analysis"])


@router.post("/frame/{session_id}", response_model=AnalysisResult)
async def analyze_frame(
    session_id: str,
    frame_base64: str = Body(..., embed=True),
    session: QuizSession = Depends(get_session),
    analyzer_mgr: AnalyzerManager = Depends(get_analyzer_manager)
):
    """
    Analyze a single frame from the student's webcam
    
    - **session_id**: Active session identifier
    - **frame_base64**: Base64 encoded image frame
    """
    session.increment_frame_count()
    
    try:
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
        
        if session.settings["enable_face_detection"]:
            result = analyzers["mesh"].detect(frame)
            face_count = result["face_count"]
            landmarks = result["landmarks"]
            
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
        if session.settings["enable_head_pose"] and landmarks is not None and len(landmarks) > 0:
            pose_result = analyzers["head_pose"].estimate_pose(frame, landmarks[0])
            direction = pose_result["direction"]
            
            analysis_result["head_pose"] = {
                "direction": direction,
                "yaw": pose_result["yaw"],
                "pitch": pose_result["pitch"],
                "roll": pose_result["roll"]
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
            gaze_result = analyzers["gaze"].estimate_gaze(frame, landmarks[0])
            
            analysis_result["gaze"] = {
                "direction": gaze_result["direction"],
                "left_iris": gaze_result["left_iris"],
                "right_iris": gaze_result["right_iris"]
            }
            
            # Check violations
            gaze_active, gaze_triggered, gaze_msg = session.violation_tracker.check_gaze_violation(
                gaze_result["direction"],
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
            blink_result = analyzers["blink"].detect_blink(landmarks[0])
            
            analysis_result["blink"] = {
                "ear": blink_result["ear"],
                "blink_detected": blink_result["blink_detected"],
                "eyes_closed": blink_result["eyes_closed"],
                "blink_count": blink_result["blink_count"]
            }
            
            # Check for prolonged eye closure
            if blink_result.get("prolonged_closure", False):
                warnings.append("Prolonged eye closure detected")
        
        # ====================================================================
        # Object Detection
        # ====================================================================
        if session.settings["enable_object_detection"]:
            obj_result = analyzers["object"].detect(frame)
            
            analysis_result["object_detection"] = {
                "phone_detected": obj_result.get("phone_detected", False),
                "person_count": obj_result.get("person_count", 0),
                "objects": obj_result.get("detections", [])
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
                        is_verified = analyzers["embedder"].compare_embeddings(
                            session.reference_embedding,
                            current_embedding,
                            threshold=0.68
                        )
                        
                        analysis_result["identity_verification"] = {
                            "verified": is_verified,
                            "checked_at_frame": session.frame_count
                        }
                        
                        # Check violations
                        id_active, id_triggered, id_msg = session.violation_tracker.check_identity(
                            is_verified,
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
        # Return Analysis Result
        # ====================================================================
        return AnalysisResult(
            session_id=session_id,
            frame_count=session.frame_count,
            timestamp=datetime.now().isoformat(),
            face_detected=face_count > 0,
            face_count=face_count,
            violations=violations,
            analysis=analysis_result,
            warnings=warnings
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing frame: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")
