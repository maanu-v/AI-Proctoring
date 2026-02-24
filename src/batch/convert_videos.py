"""
Batch Video Converter

Converts all .avi files in the database to .mp4 using imageio-ffmpeg
for browser compatibility in the review UI.

Usage:
    python -m src.batch.convert_videos
"""

import os
import glob
import logging
from pathlib import Path
import subprocess
import concurrent.futures

try:
    import imageio_ffmpeg
except ImportError:
    logger.error("imageio-ffmpeg is required. Run: pip install imageio-ffmpeg")
    import sys
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "database"

def convert_video(avi_path: str) -> bool:
    """Convert a single .avi file to .mp4 if it doesn't exist."""
    mp4_path = avi_path.rsplit('.', 1)[0] + '.mp4'
    
    if os.path.exists(mp4_path):
        logger.info(f"Skipping {os.path.basename(avi_path)} (MP4 already exists)")
        return True
        
    logger.info(f"Converting {os.path.basename(avi_path)} to MP4...")
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        # -y (overwrite), -i (input), -vcodec libx264 (H.264 video), -crf 28 (quality), 
        # -preset fast (compression speed), -c:a aac (AAC audio), -b:a 128k (audio bitrate)
        cmd = [
            ffmpeg_exe, '-y', '-i', avi_path, 
            '-vcodec', 'libx264', '-crf', '28', '-preset', 'fast', 
            '-c:a', 'aac', '-b:a', '128k', 
            mp4_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"✅ Successfully converted {os.path.basename(avi_path)}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to convert {os.path.basename(avi_path)}:\n{e.stderr.decode('utf-8') if e.stderr else str(e)}")
        return False
    except Exception as e:
        logger.error(f"❌ Error converting {os.path.basename(avi_path)}: {e}")
        return False

def main():
    logger.info("============================================================")
    logger.info("AI Proctoring — Batch MP4 Converter")
    logger.info("============================================================")
    
    # Find all .avi files (both webcam *1.avi and wearcam *2.avi)
    # If we only want webcam, we can filter by *1.avi, but converting all is safer for future use
    avi_files = glob.glob(str(DATA_DIR / "*" / "*.avi"))
    
    if not avi_files:
        logger.warning(f"No .avi files found in {DATA_DIR}")
        return

    logger.info(f"Found {len(avi_files)} .avi files to check/convert.")
    
    # Process in parallel
    max_workers = min(os.cpu_count() or 4, 8) # Limit workers to not overwhelm I/O
    logger.info(f"Starting parallel processing with {max_workers} workers...")
    
    success_count = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(convert_video, avi_files))
        success_count = sum(1 for r in results if r)
        
    logger.info("============================================================")
    logger.info(f"CONVERSION COMPLETE: {success_count}/{len(avi_files)} successful.")
    logger.info("============================================================")

if __name__ == "__main__":
    main()
