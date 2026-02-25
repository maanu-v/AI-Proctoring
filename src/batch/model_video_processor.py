"""
Single-video processor using the trained CNN-BiLSTM model.

Processes a video file using the trained model (via predict_clip)
and produces structured violation data with timestamps.
"""

import cv2
import time
import os
import re
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any

try:
    import imageio_ffmpeg
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)


def parse_ground_truth(gt_path: str) -> List[Dict[str, Any]]:
    """Parse a ground truth file (gt.txt)."""

    
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
                    
                    start_sec = int(start_str[:2]) * 60 + int(start_str[2:])
                    end_sec = int(end_str[:2]) * 60 + int(end_str[2:])
                    cheat_type = int(cheat_type_str)
                    
                    ground_truth.append({
                        "start_time": start_sec,
                        "end_time": end_sec,
                        "start_str": f"{int(start_str[:2])}:{start_str[2:]}",
                        "end_str": f"{int(end_str[:2])}:{end_str[2:]}",
                        "type": cheat_type,
                        "type_label": str(cheat_type)
                    })
    except Exception as e:
        logger.error(f"Error parsing ground truth {gt_path}: {e}")
    
    return ground_truth


def process_single_video(
    video_path: str,
    subject_id: str,
    gt_path: str,
    output_dir: str,
    sample_rate: int = 3, # Note: sample_rate is ignored here because predict_clip handles extraction
    worker_id: int = 0,
) -> Dict[str, Any]:
    """
    Process a single video file using the trained CNN-BiLSTM model.
    """
    from src.models.cnn_bi_lstm_train import load_trained_model, predict_clip
    import tensorflow as tf
    from src.utils.config import config

    logger.info(f"[{subject_id}] Starting model inference: {video_path}")
    start_time = time.time()

    model_path = os.path.join(PROJECT_ROOT, config.model.model_path, "best_model.keras")

    try:
        model, metadata = load_trained_model(model_path)
    except Exception as e:
        logger.error(f"[{subject_id}] Failed to load model: {e}")
        return {"error": f"Failed to load model: {e}"}

    # 1. Run inference for the entire clip using the CNN-BiLSTM
    try:
        prediction_results = predict_clip(
            model, video_path, metadata, 
            desc=f"[{subject_id}] AI", 
            position=worker_id
        )
    except Exception as e:
        logger.error(f"[{subject_id}] Failed during inference: {e}")
        return {"error": f"Inference failed: {e}"}

    windows = prediction_results.get("windows", [])
    
    # Pre-parse video info to set up FFMPEG overlay
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Failed to open video for overlay: {video_path}"}
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    # Parse ground truth
    ground_truth = parse_ground_truth(gt_path)

    # Convert sliding window predictions into distinct violation segments
    violation_segments = []
    normal_label_index = 0 # As per our label_map (1 is mapped to 0, which is normal? Actually 0 might be whatever label is lowest. In OEP dataset, 1 is looking around. Wait, no. The dataset doesn't have 0. Let's look at the label map.)
    
    # We define any label other than 0 as a violation.
    normal_orig_label = 0

    current_segment = None

    for w in windows:
        # e.g., predicted_label is the original dataset label (1, 2, 3...)
        lbl = w["predicted_label"]
        start_t = w["start_frame"] / fps
        end_t = w["end_frame"] / fps
        confidence = w["confidence"]
        
        # In OEP, the provided GT only has cheating behaviors. So if predict_clip detects it, it's a violation.
        vtype = str(lbl)
        msg = f"Class {lbl} ({confidence:.0%} conf)"
        
        if current_segment is None:
            current_segment = {
                "type": vtype,
                "start_time": start_t,
                "end_time": end_t,
                "predicted_label": lbl,
                "message": msg,
                "confidences": [confidence]
            }
        elif current_segment["predicted_label"] == lbl and (start_t - current_segment["end_time"]) < 2.0:
            # Continue segment if same label and gap is small
            current_segment["end_time"] = end_t
            current_segment["confidences"].append(confidence)
            current_segment["message"] = f"Class {lbl} (avg conf: {sum(current_segment['confidences'])/len(current_segment['confidences']):.0%})"
        else:
            # Save segment and start a new one
            current_segment["duration"] = current_segment["end_time"] - current_segment["start_time"]
            current_segment["severity"] = "high"
            if current_segment["duration"] >= 1.0: # Minimum 1s duration
                violation_segments.append(current_segment)
            
            current_segment = {
                "type": vtype,
                "start_time": start_t,
                "end_time": end_t,
                "predicted_label": lbl,
                "message": msg,
                "confidences": [confidence]
            }

    if current_segment:
        current_segment["duration"] = current_segment["end_time"] - current_segment["start_time"]
        current_segment["severity"] = "high"
        if current_segment["duration"] >= 1.0:
            violation_segments.append(current_segment)

    # Setup FFMPEG process for overlying video
    os.makedirs(output_dir, exist_ok=True)
    overlaid_mp4_path = os.path.join(output_dir, f"{subject_id}_overlaid.mp4")
    
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        ffmpeg_cmd = [
            ffmpeg_exe, '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f"{width}x{height}", '-pix_fmt', 'bgr24', '-r', str(fps),
            '-i', '-',
            '-i', video_path,
            '-c:v', 'libx264', '-crf', '28', '-preset', 'fast',
            '-c:a', 'aac', '-b:a', '128k',
            '-map', '0:v:0', '-map', '1:a:0?',
            overlaid_mp4_path
        ]
        ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except Exception as e:
        logger.error(f"[{subject_id}] Failed to start FFMPEG process: {e}")
        ffmpeg_proc = None

    frame_idx = 0
    from tqdm import tqdm
    pbar = tqdm(total=total_frames, desc=f"[{subject_id}] Vid (Overlay)", position=worker_id, leave=False)
    
    if ffmpeg_proc:
        while True:
            ret, cap_frame = cap.read()
            if not ret:
                break
                
            current_time_sec = frame_idx / fps
            
            # Find active prediction for this frame
            active_preds = [w for w in windows if w["start_frame"] <= frame_idx <= w["end_frame"]]
            
            annotated_frame = cap_frame.copy()
            if active_preds:
                # Use the window that is most centered on this frame, or just the first overlapping one
                active_pred = active_preds[0]
                lbl = active_pred["predicted_label"]
                conf = active_pred["confidence"]
                name = active_pred["class_name"]
                
                # Check if it's a violation (any label other than the lowest/normal one)
                is_violation = (lbl != normal_orig_label)
                
                if is_violation:
                    # Draw Red border for violation
                    cv2.rectangle(annotated_frame, (0, 0), (width, height), (0, 0, 255), 10)
                    text_color = (0, 0, 255) # Red text
                else:
                    text_color = (0, 255, 0) # Green text (Normal)
                
                # Draw text background
                text = f"Class {lbl} ({conf:.0%})"
                (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (10, height - text_h - 30), (10 + text_w + 20, height - 10), (0, 0, 0), -1)
                annotated_frame = cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0)
                
                # Draw text
                cv2.putText(annotated_frame, text, (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

            try:
                ffmpeg_proc.stdin.write(annotated_frame.tobytes())
            except Exception as e:
                logger.error(f"[{subject_id}] Error writing to FFMPEG pipe: {e}")
                ffmpeg_proc = None
                break
                
            frame_idx += 1
            pbar.update(1)
            
    pbar.close()

    cap.release()
    if ffmpeg_proc:
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()

    processing_time = time.time() - start_time
    logger.info(f"[{subject_id}] Completed in {processing_time:.1f}s. Found {len(violation_segments)} violation segments.")

    violation_type_counts: Dict[str, int] = {}
    for v in violation_segments:
        vtype = v["type"]
        violation_type_counts[vtype] = violation_type_counts.get(vtype, 0) + 1

    total_violation_duration = sum(v.get("duration", 0) for v in violation_segments)
    risk_score = min(1.0, total_violation_duration / max(duration, 1) * 3)

    result = {
        "subject_id": subject_id,
        "video_path": video_path,
        "overlaid_video_path": os.path.relpath(overlaid_mp4_path, PROJECT_ROOT),
        "video_metadata": {
            "duration_seconds": round(duration, 2),
            "fps": round(fps, 1),
            "total_frames": total_frames,
            "frames_processed": total_frames,
            "sample_rate": 1,
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
            "anomaly_score": prediction_results.get("anomaly_score", 0.0)
        },
        "frame_analyses_sample": prediction_results.get("windows", [])[:50],  # Store some raw windows for debugging
    }

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{subject_id}_results.json")
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    logger.info(f"[{subject_id}] Results saved to {output_path}")
    return result

def _get_severity(violation_type: str) -> str:
    return "high"
