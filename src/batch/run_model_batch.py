"""
Batch processing runner for AI Proctoring.

Discovers all webcam videos in data/raw/database and processes them
in parallel using ProcessPoolExecutor.

Usage:
    python -m src.batch.run_batch
    python -m src.batch.run_batch --subjects subject1 subject2
    python -m src.batch.run_batch --sample-rate 5 --max-workers 4
"""

import os
import sys
import json
import glob
import time
import signal
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Ensure project root is on path
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import config

DATABASE_DIR = os.path.join(PROJECT_ROOT, config.batch.database_dir)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, config.batch.output_dir)


def discover_videos(database_dir: str, subjects: list = None) -> list:
    """
    Discover all webcam videos (*1.avi) in the database.
    Returns list of (subject_id, video_path, gt_path) tuples.
    """
    videos = []
    
    subject_dirs = sorted(glob.glob(os.path.join(database_dir, "subject*")))
    
    for subject_dir in subject_dirs:
        subject_id = os.path.basename(subject_dir)
        
        # Filter by specific subjects if provided
        if subjects and subject_id not in subjects:
            continue
        
        # Find webcam video (*1.avi)
        avi_files = glob.glob(os.path.join(subject_dir, "*1.avi"))
        if not avi_files:
            logger.warning(f"No webcam video found for {subject_id}")
            continue
        
        video_path = avi_files[0]
        gt_path = os.path.join(subject_dir, "gt.txt")
        
        videos.append((subject_id, video_path, gt_path))
    
    return videos


def _process_wrapper(args):
    """Wrapper for ProcessPoolExecutor that unpacks arguments."""
    subject_id, video_path, gt_path, output_dir, sample_rate = args
    
    # Re-setup logging for subprocess
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    
    # Add project root to path in subprocess
    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Configure TensorFlow memory growth to avoid OOM across multiprocessing workers
    import os
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    
    try:
        import tensorflow as tf
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        logger.warning(f"Failed to configure TF memory growth: {e}")
    
    from src.batch.model_video_processor import process_single_video
    
    import multiprocessing
    worker_id = multiprocessing.current_process()._identity[0] if multiprocessing.current_process()._identity else 0
    
    try:
        result = process_single_video(
            video_path=video_path,
            subject_id=subject_id,
            gt_path=gt_path,
            output_dir=output_dir,
            sample_rate=sample_rate,
            worker_id=worker_id,
        )
        return subject_id, result
    except Exception as e:
        logger.error(f"[{subject_id}] Processing failed: {e}")
        return subject_id, {"error": str(e)}


def auto_select_gpu():
    """Automatically finds the GPU with the most free memory and sets CUDA_VISIBLE_DEVICES."""
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, text=True, check=True
        )
        gpu_memory = []
        for line in result.stdout.strip().split('\n'):
            if not line: continue
            idx, free_mem = line.split(',')
            gpu_memory.append((int(idx.strip()), int(free_mem.strip())))
            
        if gpu_memory:
            gpu_memory.sort(key=lambda x: x[1], reverse=True)
            best_gpu = gpu_memory[0][0]
            best_mem = gpu_memory[0][1]
            os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
            logger.info(f"Auto-selected GPU {best_gpu} with {best_mem} MiB free memory.")
    except Exception as e:
        logger.warning(f"Failed to auto-select GPU: {e}. Falling back to default.")

def run_batch(
    subjects: list = None,
    sample_rate: int = 3,
    max_workers: int = 2,
):
    """
    Run batch processing on all (or selected) videos.
    
    Args:
        subjects: List of subject IDs to process (None = all)
        sample_rate: Process every Nth frame (defaults to config)
        max_workers: Number of parallel workers (defaults to config)
    """
    if max_workers is None:
        max_workers = config.batch.max_workers
    if sample_rate is None:
        sample_rate = config.batch.sample_rate
    logger.info("=" * 60)
    logger.info("AI Proctoring — Batch Video Processor")
    logger.info("=" * 60)
    
    # Auto-select the emptiest GPU before spawning workers so they inherit the env var
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        auto_select_gpu()
    
    # Discover videos
    videos = discover_videos(DATABASE_DIR, subjects)
    logger.info(f"Found {len(videos)} videos to process")
    
    if not videos:
        logger.warning("No videos found. Check data/raw/database/ directory.")
        return
    
    for sid, vpath, gpath in videos:
        logger.info(f"  - {sid}: {os.path.basename(vpath)}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Prepare tasks
    tasks = [
        (subject_id, video_path, gt_path, OUTPUT_DIR, sample_rate)
        for subject_id, video_path, gt_path in videos
    ]
    
    # Process in parallel
    start_time = time.time()
    all_results = {}
    completed = 0
    executor = ProcessPoolExecutor(max_workers=max_workers)

    logger.info(f"\nStarting parallel processing with {max_workers} workers, sample_rate={sample_rate}")
    logger.info("-" * 60)

    def _shutdown_executor(exec_instance, futures):
        """Cancel pending futures, shut down executor, and SIGKILL any survivors."""
        logger.warning("Shutting down workers — cancelling pending futures...")
        for f in futures:
            f.cancel()
        # cancel_futures=True available in Python 3.9+
        try:
            exec_instance.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            # Python < 3.9 fallback
            exec_instance.shutdown(wait=False)

        # Force-kill any child processes that are still alive
        for pid, proc in list(exec_instance._processes.items()):
            if proc.is_alive():
                logger.warning(f"Force-killing worker PID {pid}")
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass  # already gone

    try:
        future_to_subject = {
            executor.submit(_process_wrapper, task): task[0]
            for task in tasks
        }

        progress = tqdm(
            as_completed(future_to_subject),
            total=len(future_to_subject),
            desc="Batch",
            unit="video",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
        )

        for future in progress:
            subject_id = future_to_subject[future]
            # Show which video is currently finishing
            progress.set_postfix_str(f"done: {subject_id}", refresh=True)
            try:
                sid, result = future.result()
                all_results[sid] = result
                completed += 1

                if "error" not in result:
                    v_count = result["summary"]["total_violations"]
                    risk = result["summary"]["risk_score"]
                    proc_time = result["processing"]["processing_time_seconds"]
                    status = f"✅ {sid} — {v_count} violations, risk={risk:.2f}, {proc_time:.1f}s"
                    progress.set_postfix_str(status, refresh=True)
                    tqdm.write(f"✅ [{completed}/{len(videos)}] {sid}: "
                               f"{v_count} violations, risk={risk:.2f}, took {proc_time:.1f}s")
                else:
                    status = f"❌ {sid} — {result['error'][:60]}"
                    progress.set_postfix_str(status, refresh=True)
                    tqdm.write(f"❌ [{completed}/{len(videos)}] {sid}: {result['error']}")

            except Exception as e:
                tqdm.write(f"❌ [{completed}/{len(videos)}] {subject_id}: Unexpected error: {e}")
                all_results[subject_id] = {"error": str(e)}
                completed += 1

        progress.close()

    except KeyboardInterrupt:
        logger.warning("\n⚠️  KeyboardInterrupt received — stopping all workers...")
        _shutdown_executor(executor, list(future_to_subject.keys()) if 'future_to_subject' in dir() else [])
        logger.warning("All workers terminated. Exiting.")
        sys.exit(130)  # 128 + SIGINT
    finally:
        executor.shutdown(wait=False)
    
    total_time = time.time() - start_time
    
    # Write aggregate index
    index = {
        "processing_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_videos": len(videos),
        "total_processing_time_seconds": round(total_time, 2),
        "sample_rate": sample_rate,
        "subjects": [],
    }
    
    for subject_id, _, _ in sorted(videos, key=lambda x: x[0]):
        result = all_results.get(subject_id, {})
        if "error" not in result:
            index["subjects"].append({
                "subject_id": subject_id,
                "video_path": result.get("video_path", ""),
                "duration_seconds": result.get("video_metadata", {}).get("duration_seconds", 0),
                "total_violations": result.get("summary", {}).get("total_violations", 0),
                "risk_score": result.get("summary", {}).get("risk_score", 0),
                "violation_types": result.get("summary", {}).get("violation_types", {}),
                "results_file": f"{subject_id}_results.json",
            })
        else:
            index["subjects"].append({
                "subject_id": subject_id,
                "error": result["error"],
                "results_file": None,
            })
    
    index_path = os.path.join(OUTPUT_DIR, "index.json")
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)
    
    # Summary
    logger.info("=" * 60)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info(f"  Total time: {total_time:.1f}s")
    logger.info(f"  Videos processed: {len(all_results)}")
    logger.info(f"  Results saved to: {OUTPUT_DIR}")
    logger.info(f"  Index file: {index_path}")
    
    successful = sum(1 for r in all_results.values() if "error" not in r)
    failed = len(all_results) - successful
    logger.info(f"  Success: {successful}, Failed: {failed}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Batch Video Processing for AI Proctoring")
    parser.add_argument(
        "--subjects", nargs="+", default=None,
        help="Specific subject IDs to process (e.g., subject1 subject2). Default: all."
    )
    parser.add_argument(
        "--sample-rate", type=int, default=None,
        help="Process every Nth frame. Default: from config.yaml."
    )
    parser.add_argument(
        "--max-workers", type=int, default=None,
        help="Number of parallel workers. Default: from config.yaml."
    )
    
    args = parser.parse_args()
    
    run_batch(
        subjects=args.subjects,
        sample_rate=args.sample_rate,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
