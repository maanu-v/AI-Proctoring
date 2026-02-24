"""
kill_batch.py — Emergency cleanup for run_batch.py worker processes.

Finds and kills all Python processes related to run_batch / video_processor
that may have been left running after a Ctrl+C or crash.

Usage:
    python -m src.batch.kill_batch
    python src/batch/kill_batch.py
"""

import os
import sys
import signal
import subprocess
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Keywords that identify batch worker processes
PROCESS_KEYWORDS = [
    "run_batch",
    "video_processor",
    "kill_batch",   # exclude self below
]


def find_batch_pids() -> list[int]:
    """Return PIDs of all matching batch/worker processes (excluding self)."""
    own_pid = os.getpid()
    found = []

    try:
        # Use ps to list all python processes with their command lines
        result = subprocess.run(
            ["ps", "-eo", "pid,cmd"],
            capture_output=True, text=True, check=True,
        )
    except FileNotFoundError:
        logger.error("'ps' command not found. Cannot search for processes.")
        return []

    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) < 2:
            continue
        pid_str, cmd = parts
        try:
            pid = int(pid_str)
        except ValueError:
            continue

        if pid == own_pid:
            continue  # never kill ourselves

        # Must look like a Python process
        if "python" not in cmd.lower():
            continue

        # Must match at least one keyword (but not ONLY kill_batch)
        matched = [kw for kw in PROCESS_KEYWORDS if kw in cmd]
        if matched and not (matched == ["kill_batch"]):
            found.append((pid, cmd.strip()))

    return found


def kill_pids(pid_cmd_pairs: list[tuple[int, str]], sig=signal.SIGKILL):
    """Send signal to each PID and report."""
    if not pid_cmd_pairs:
        logger.info("No batch worker processes found. Nothing to kill.")
        return

    sig_name = "SIGKILL" if sig == signal.SIGKILL else "SIGTERM"
    logger.info(f"Sending {sig_name} to {len(pid_cmd_pairs)} process(es):")

    for pid, cmd in pid_cmd_pairs:
        short_cmd = cmd[:80] + ("..." if len(cmd) > 80 else "")
        try:
            os.kill(pid, sig)
            logger.info(f"  ✅  PID {pid} killed  —  {short_cmd}")
        except ProcessLookupError:
            logger.info(f"  ⚠️   PID {pid} already gone  —  {short_cmd}")
        except PermissionError:
            logger.error(f"  ❌  PID {pid} permission denied  —  {short_cmd}")


def main():
    logger.info("=== kill_batch: scanning for running batch workers ===")
    pairs = find_batch_pids()

    if not pairs:
        logger.info("All clear — no stray batch processes detected.")
        sys.exit(0)

    logger.info(f"Found {len(pairs)} process(es) to terminate:")
    for pid, cmd in pairs:
        short_cmd = cmd[:80] + ("..." if len(cmd) > 80 else "")
        logger.info(f"  PID {pid}  —  {short_cmd}")

    # First try graceful SIGTERM, then SIGKILL
    kill_pids(pairs, signal.SIGTERM)

    import time
    time.sleep(2)

    # Re-check and force-kill survivors
    survivors = find_batch_pids()
    if survivors:
        logger.warning(f"{len(survivors)} process(es) still alive — sending SIGKILL...")
        kill_pids(survivors, signal.SIGKILL)
    else:
        logger.info("All processes terminated cleanly.")

    logger.info("=== kill_batch done ===")


if __name__ == "__main__":
    main()
