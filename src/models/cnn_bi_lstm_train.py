"""
CNN-BiLSTM Training Script for OEP (Online Exam Proctoring) Dataset
====================================================================
Trains a MobileNetV2 + Bidirectional LSTM model to classify cheating
behaviors in exam recordings.

Dataset structure expected at --dataset-path:
  subject1/
    username1.avi   ← webcam video
    username2.avi   ← wearcam video
    username.wav
    gt.txt          ← "start_frame  end_frame  label" (tab-separated)
  subject2/
    ...

Labels (from OEP paper):
  0 → Normal (not cheating)
  1 → Looking around / Gaze away
  2 → Using phone / device
  3 → Talking / whispering
  4 → Multiple people in frame

Usage:
  # Train on real data (SCP this file to server first):
  python src/models/cnn_bi_lstm_train.py --dataset-path /path/to/OEP/database

  # Quick smoke-test with synthetic data locally:
  python src/models/cnn_bi_lstm_train.py --dataset-path data/raw/database --epochs 2

  # Run inference on a video clip:
  python src/models/cnn_bi_lstm_train.py --predict --video path/to/clip.avi \
      --model-path models/best_model.keras
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless — no display needed on server
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Resource Management
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    def auto_select_gpu():
        import subprocess, os
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
                os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
                print(f"Auto-selected GPU {best_gpu} with {gpu_memory[0][1]} MiB free memory for training.")
        except Exception as e:
            print(f"Failed to auto-select GPU: {e}")

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        auto_select_gpu()

    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"

# --------------------------------------------------------------------------- #
# Lazy TF import so the script can be imported without TF installed            #
# --------------------------------------------------------------------------- #
try:
    import tensorflow as tf
    
    if __name__ == "__main__":
        try:
            gpus = tf.config.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=30000)] # Cap at 30GB to prevent total memory exhaustion
                )
        except Exception as e:
            print(f"Failed to configure TF memory limits: {e}")

    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.utils import Sequence
    from tensorflow.keras.layers import (
        Dense,
        Dropout,
        LSTM,
        Bidirectional,
        TimeDistributed,
        GlobalAveragePooling2D,
    )
    from tensorflow.keras.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        ReduceLROnPlateau,
        CSVLogger,
    )
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.optimizers import Adam
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report,
    )
except ImportError as e:
    print(f"[ERROR] Required library not installed: {e}")
    print("Install via: pip install tensorflow scikit-learn pandas matplotlib opencv-python")
    sys.exit(1)


IMG_SIZE = (224, 224)  # MobileNetV2 input
FRAME_SAMPLE_RATE = 5  # extract every Nth frame from a segment

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --------------------------------------------------------------------------- #
# Logging                                                                      #
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("cnn_bilstm")


# =========================================================================== #
#  1. ARGUMENT PARSING                                                         #
# =========================================================================== #

def _load_model_config():
    """Load ModelConfig from app.yaml, falling back to dataclass defaults on error."""
    import sys
    _root = Path(__file__).resolve().parent.parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
        
    try:
        from src.utils.config import Config
        return Config(str(_root / "src" / "configs" / "app.yaml")).model
    except Exception as e:
        log.warning(f"Failed to load app.yaml explicitly: {e}. Falling back to config defaults.")
        try:
            from src.utils.config import Config
            return Config().model
        except Exception:
            return None


def parse_args() -> argparse.Namespace:
    # Pre-load YAML config so defaults come from app.yaml
    _cfg = _load_model_config()

    p = argparse.ArgumentParser(
        description="CNN-BiLSTM trainer/predictor for OEP proctoring dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Paths ──────────────────────────────────────────────────────────────── #
    p.add_argument(
        "--dataset-path",
        default=_cfg.dataset_path if _cfg else "data/raw/database",
        help="Root folder containing subjectN dirs",
    )
    p.add_argument(
        "--processed-path",
        default=_cfg.processed_path if _cfg else "data/processed",
        help="Where to cache extracted frames",
    )
    p.add_argument(
        "--model-path",
        default=_cfg.model_path if _cfg else "data/models",
        help="Directory to save trained .keras model files",
    )
    p.add_argument(
        "--reports-path",
        default=_cfg.reports_path if _cfg else "data/reports",
        help="Directory to save evaluation plots and metrics",
    )
    # ── Data pipeline ──────────────────────────────────────────────────────── #
    p.add_argument(
        "--sequence-length", type=int,
        default=_cfg.sequence_length if _cfg else 10,
        help="Frames per sequence window",
    )
    p.add_argument(
        "--overlap", type=float,
        default=_cfg.overlap if _cfg else 0.5,
        help="Overlap fraction between sliding windows",
    )
    p.add_argument(
        "--test-size", type=float,
        default=_cfg.test_size if _cfg else 0.2,
        help="Fraction of subjects held out for testing",
    )
    # ── Training hyperparameters ───────────────────────────────────────────── #
    p.add_argument(
        "--epochs", type=int,
        default=_cfg.epochs if _cfg else 50,
    )
    p.add_argument(
        "--batch-size", type=int,
        default=_cfg.batch_size if _cfg else 8,
    )
    p.add_argument(
        "--lr", type=float,
        default=_cfg.learning_rate if _cfg else 1e-4,
        help="Initial Adam learning rate",
    )
    p.add_argument(
        "--patience", type=int,
        default=_cfg.patience if _cfg else 10,
        help="EarlyStopping patience (epochs)",
    )
    p.add_argument(
        "--lr-reduce-factor", type=float,
        default=_cfg.lr_reduce_factor if _cfg else 0.5,
        help="ReduceLROnPlateau multiplicative factor",
    )
    p.add_argument(
        "--lr-patience", type=int,
        default=_cfg.lr_patience if _cfg else 5,
        help="ReduceLROnPlateau patience (epochs)",
    )
    p.add_argument(
        "--min-lr", type=float,
        default=_cfg.min_lr if _cfg else 1e-7,
        help="Minimum learning rate for scheduler",
    )
    # ── Inference ──────────────────────────────────────────────────────────── #
    p.add_argument("--predict", action="store_true",
                   help="Run inference instead of training")
    p.add_argument("--video", default=None,
                   help="Path to video clip for --predict mode")
    p.add_argument(
        "--anomaly-threshold", type=float,
        default=_cfg.anomaly_threshold if _cfg else 0.3,
        help="Fraction of non-Normal windows to flag as suspicious",
    )
    # ── Misc ───────────────────────────────────────────────────────────────── #
    p.add_argument("--no-extract", action="store_true",
                   help="Skip frame extraction (frames already on disk)")
    p.add_argument("--force-synthetic", action="store_true",
                   help="Use synthetic dataset (smoke-test pipeline)")
    return p.parse_args()


# =========================================================================== #
#  2. DATA PIPELINE                                                            #
# =========================================================================== #

def detect_webcam_video(subject_path: Path) -> Optional[Path]:
    """
    Returns the webcam .avi file for a subject.
    Per the OEP README, webcam videos end with '1.avi' (e.g. yousef1.avi).
    """
    avi_files = sorted(subject_path.glob("*.avi"))
    if not avi_files:
        return None
    # Prefer the file ending in '1.avi' (webcam)
    for f in avi_files:
        if f.stem.endswith("1"):
            return f
    # Fallback: first .avi found
    return avi_files[0]


def parse_gt_file(gt_path: Path) -> Optional[pd.DataFrame]:
    """
    Parses a gt.txt ground-truth file.
    Format: start_frame <TAB> end_frame <TAB> label
    Returns a DataFrame with columns [Start, End, Label] or None on failure.
    """
    for sep in ["\t", r"\s+"]:
        try:
            df = pd.read_csv(
                gt_path, sep=sep, header=None,
                names=["Start", "End", "Label"],
                engine="python", on_bad_lines="skip",
            )
            df = df.dropna(subset=["Label"])
            df["Start"] = pd.to_numeric(df["Start"], errors="coerce").fillna(0).astype(int)
            df["End"] = pd.to_numeric(df["End"], errors="coerce").fillna(0).astype(int)
            df["Label"] = pd.to_numeric(df["Label"], errors="coerce")
            df = df.dropna(subset=["Label"])
            df["Label"] = df["Label"].astype(int)
            if len(df) > 0:
                return df
        except Exception:
            continue
    log.warning("Could not parse gt file: %s", gt_path)
    return None


def extract_frames_for_subject(
    subject_dir: str,
    dataset_path: Path,
    output_dir: Path,
) -> None:
    """
    Extracts labelled frames from the webcam video of a single subject
    and saves them as JPEG images under output_dir/class_<label>/.
    """
    subject_path = dataset_path / subject_dir
    video_path = detect_webcam_video(subject_path)
    gt_path = subject_path / "gt.txt"

    if video_path is None:
        log.warning("[%s] No .avi file found, skipping.", subject_dir)
        return
    if not gt_path.exists():
        log.warning("[%s] gt.txt not found, skipping.", subject_dir)
        return

    gt_df = parse_gt_file(gt_path)
    if gt_df is None or len(gt_df) == 0:
        log.warning("[%s] Empty/unparseable gt.txt, skipping.", subject_dir)
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.error("[%s] Cannot open video %s", subject_dir, video_path)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    log.info("[%s] Video: %d frames @ %.1f FPS", subject_dir, total_frames, fps)

    # Create class directories including Class 0 (Normal)
    classes_to_extract = set(gt_df["Label"].unique())
    classes_to_extract.add(0)
    for label in classes_to_extract:
        (output_dir / f"class_{label}").mkdir(parents=True, exist_ok=True)

    # Compile non-overlapping gap ranges for Class 0 (Normal)
    # The video starts at 0 and ends at total_frames
    normal_segments = []
    current_time = 0
    
    # Ensure anomalies are sorted by start time
    gt_df = gt_df.sort_values(by="Start")
    
    for _, row in gt_df.iterrows():
        start, end, label = int(row["Start"]), int(row["End"]), int(row["Label"])
        if start >= total_frames:
            continue
        end = min(end, total_frames - 1)
        
        # If there's a gap between the current time and the start of this violation
        if start > current_time:
            normal_segments.append((current_time, start - 1))
            
        current_time = max(current_time, end + 1)
        
        # Extract violation frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for frame_idx in range(start, end + 1, FRAME_SAMPLE_RATE):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, IMG_SIZE)
            out_fname = f"{subject_dir}_lbl{label}_f{frame_idx:06d}.jpg"
            cv2.imwrite(str(output_dir / f"class_{label}" / out_fname), frame)

    # Add any remaining time at the end of the video as normal
    if current_time < total_frames:
        normal_segments.append((current_time, total_frames - 1))
        
    # Extract Normal (Class 0) frames from the gaps
    for start, end in normal_segments:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for frame_idx in range(start, end + 1, FRAME_SAMPLE_RATE):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, IMG_SIZE)
            out_fname = f"{subject_dir}_lbl0_f{frame_idx:06d}.jpg"
            cv2.imwrite(str(output_dir / "class_0" / out_fname), frame)

    cap.release()
    log.info("[%s] Frame extraction complete.", subject_dir)


def process_all_subjects(dataset_path: Path, output_dir: Path) -> list[str]:
    """Extract frames for every subject* directory. Returns list of subject names."""
    subjects = sorted([
        d.name for d in dataset_path.iterdir()
        if d.is_dir() and d.name.startswith("subject")
    ])
    log.info("Found %d subjects: %s", len(subjects), subjects)

    for subject in subjects:
        extract_frames_for_subject(subject, dataset_path, output_dir)

    return subjects


def split_subjects(subjects: list[str], test_size: float = 0.2) -> tuple[list, list]:
    """
    Subject-level train/test split to prevent data leakage.
    All frames from a given subject end up exclusively in train OR test.
    """
    if len(subjects) == 0:
        return [], []
    train_subs, test_subs = train_test_split(
        subjects, test_size=test_size, random_state=42
    )
    return list(train_subs), list(test_subs)


def build_sequences(
    frames_dir: Path,
    subject_filter: Optional[list[str]],
    sequence_length: int,
    overlap: float,
    label_map: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray, int, dict]:
    """
    Builds (X, y) arrays from extracted frames.

    Args:
        frames_dir:      Directory with class_<N>/ subdirectories.
        subject_filter:  If given, only include frames from these subjects.
        sequence_length: Number of frames per sequence.
        overlap:         Fractional overlap between consecutive windows (0 … <1).
        label_map:       {original_label_int -> model_index}.  If None (first
                         call on train data), the map is built from labels present
                         in frames_dir.  Pass the train map when building test
                         sequences so both use the same encoding.

    Returns:
        X_paths:    list of lists of paths (N, sequence_length)
        y:          int32 array    (N,) — model-index encoded (0-based)
        num_classes: number of distinct classes
        label_map:  {original_label: model_index}  (same one that was passed in,
                    or the newly built one)
    """
    stride = max(1, int(sequence_length * (1 - overlap)))
    X_paths, y_list = [], []

    class_dirs = sorted([
        d for d in frames_dir.iterdir()
        if d.is_dir() and d.name.startswith("class_")
    ])

    # Discover original labels from directory names
    discovered_labels = []
    for cd in class_dirs:
        try:
            discovered_labels.append(int(cd.name.split("_")[1]))
        except (ValueError, IndexError):
            pass

    # Build label_map on first call (train); reuse on second call (test)
    if label_map is None:
        label_map = {orig: idx for idx, orig in enumerate(sorted(discovered_labels))}

    num_classes = len(label_map)

    for class_dir in class_dirs:
        try:
            orig_label = int(class_dir.name.split("_")[1])
        except (ValueError, IndexError):
            continue

        if orig_label not in label_map:
            log.warning("Label %d in test data not seen during training — skipping.", orig_label)
            continue

        model_label = label_map[orig_label]
        images = sorted(class_dir.glob("*.jpg"))

        # Filter by subject
        if subject_filter is not None:
            images = [img for img in images if any(img.name.startswith(s) for s in subject_filter)]

        # Group by subject to keep temporal ordering
        from collections import defaultdict
        by_subject: dict[str, list[Path]] = defaultdict(list)
        for img in images:
            subject = img.name.split("_lbl")[0]
            by_subject[subject].append(img)

        for subject, imgs in by_subject.items():
            imgs = sorted(imgs, key=lambda p: int(p.stem.split("_f")[-1]))

            for start in range(0, len(imgs) - sequence_length + 1, stride):
                seq_paths = imgs[start: start + sequence_length]
                if len(seq_paths) < sequence_length:
                    break

                if len(seq_paths) == sequence_length:
                    X_paths.append(seq_paths)
                    y_list.append(model_label)

    if not X_paths:
        return [], np.array([]), num_classes, label_map

    y = np.array(y_list, dtype=np.int32)
    label_dist = {f"Label {k}": int(v) for k, v in
                  zip(label_map.keys(), np.bincount(y, minlength=num_classes))}
    log.info("Sequences: %d | Classes: %d | Dist: %s", len(X_paths), num_classes, label_dist)
    return X_paths, y, num_classes, label_map


class VideoSequenceGenerator(Sequence):
    """
    Reads batches of images from disk on the fly to prevent OOM errors.
    """
    def __init__(self, x_paths: list[list[Path]], y: np.ndarray, batch_size: int, img_size: tuple[int, int] = IMG_SIZE, shuffle: bool = True):
        self.x_paths = x_paths
        self.y = y
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.x_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self) -> int:
        return int(np.ceil(len(self.x_paths) / self.batch_size))

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        batch_x_paths = [self.x_paths[k] for k in batch_indexes]
        batch_y = self.y[batch_indexes]

        X_batch = []
        for seq_paths in batch_x_paths:
            seq_frames = []
            for p in seq_paths:
                img = cv2.imread(str(p))
                if img is None:
                    img = np.zeros((*self.img_size, 3), dtype=np.float32)
                else:
                    img = cv2.resize(img, self.img_size).astype(np.float32) / 255.0
                seq_frames.append(img)
            X_batch.append(seq_frames)

        return np.array(X_batch, dtype=np.float32), batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


# =========================================================================== #
#  3. SYNTHETIC DATASET (fallback / smoke-test)                               #
# =========================================================================== #

def create_synthetic_dataset(frames_dir: Path, sequence_length: int) -> tuple[list, list]:
    """
    Generates small synthetic image sequences so the full pipeline can be
    exercised without actual dataset files.
    """
    log.warning("No subjects found — generating SYNTHETIC dataset for smoke-test.")
    subjects_train = [f"synth_train_{i:02d}" for i in range(8)]
    subjects_test  = [f"synth_test_{i:02d}"  for i in range(2)]
    all_subjects   = subjects_train + subjects_test

    num_labels = len(CLASS_NAMES)
    frames_per_class = sequence_length * 4  # enough for at least 2 windows

    for subject in all_subjects:
        for label in CLASS_NAMES:
            class_dir = frames_dir / f"class_{label}"
            class_dir.mkdir(parents=True, exist_ok=True)
            for f_idx in range(frames_per_class):
                img = np.random.randint(0, 256, (*IMG_SIZE, 3), dtype=np.uint8)
                # Add a simple colour cue per class so the model has something to learn
                colour = [(0,255,0),(255,0,0),(0,0,255),(255,255,0),(0,255,255)][label]
                cv2.rectangle(img, (30, 30), (100, 100), colour, -1)
                fname = f"{subject}_lbl{label}_f{f_idx:06d}.jpg"
                cv2.imwrite(str(class_dir / fname), img)

    return subjects_train, subjects_test


# =========================================================================== #
#  4. MODEL DEFINITION                                                         #
# =========================================================================== #

def build_cnn_bilstm(
    num_classes: int,
    sequence_length: int,
    input_shape: tuple = (*IMG_SIZE, 3),
) -> "tf.keras.Model":
    """
    MobileNetV2 CNN (frozen ImageNet weights) feature extractor wrapped in
    TimeDistributed + two Bidirectional LSTM layers for temporal modelling.

    Architecture:
        Input  → (batch, T, 224, 224, 3)
        TimeDistributed(MobileNetV2)  → (batch, T, 7, 7, 1280)
        TimeDistributed(GAP)          → (batch, T, 1280)
        TimeDistributed(Dense 512)    → (batch, T, 512)
        BiLSTM(256)  return_seqs=True → (batch, T, 512)
        BiLSTM(128)                   → (batch, 256)
        Dense 128 → Dense num_classes (softmax)
    """
    # Frozen MobileNetV2 backbone
    backbone = MobileNetV2(
        weights="imagenet", include_top=False,
        input_shape=input_shape,
    )
    backbone.trainable = False

    model = Sequential(name="CNN_BiLSTM_Proctor")
    model.add(TimeDistributed(backbone, input_shape=(sequence_length, *input_shape)))
    model.add(TimeDistributed(GlobalAveragePooling2D()))
    model.add(TimeDistributed(Dense(512, activation="relu")))
    model.add(TimeDistributed(Dropout(0.4)))

    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(Dropout(0.4))
    model.add(Bidirectional(LSTM(128, return_sequences=False)))
    model.add(Dropout(0.4))

    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation="softmax", name="predictions"))

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# =========================================================================== #
#  5. TRAINING                                                                  #
# =========================================================================== #

def train(args: argparse.Namespace) -> None:
    dataset_path   = Path(args.dataset_path)
    processed_path = Path(args.processed_path)
    model_dir      = Path(args.model_path)
    reports_dir    = Path(args.reports_path)

    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    processed_path.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Frame extraction ─────────────────────────────────────────── #
    if args.force_synthetic:
        subjects_train, subjects_test = create_synthetic_dataset(
            processed_path, args.sequence_length
        )
        subjects = subjects_train + subjects_test
    else:
        if not args.no_extract:
            subjects = process_all_subjects(dataset_path, processed_path)
        else:
            subjects = sorted([
                d.name for d in processed_path.iterdir()
                if d.is_dir() and d.name.startswith("class_")
            ])
            subjects = []  # subject-level split relies on filenames; derive below

        if not subjects:
            # No subjects found — fall back to synthetic
            subjects_train, subjects_test = create_synthetic_dataset(
                processed_path, args.sequence_length
            )
            subjects = subjects_train + subjects_test
        else:
            subjects_train, subjects_test = split_subjects(subjects, args.test_size)

    log.info("Train subjects (%d): %s", len(subjects_train), subjects_train)
    log.info("Test  subjects (%d): %s", len(subjects_test),  subjects_test)

    # ── Step 2: Build sequences ──────────────────────────────────────────── #
    log.info("Building training sequences metadata…")
    X_train_paths, y_train, num_classes, label_map = build_sequences(
        processed_path, subjects_train, args.sequence_length, args.overlap
    )
    log.info("Building test sequences metadata…")
    X_test_paths, y_test, _, _ = build_sequences(
        processed_path, subjects_test, args.sequence_length, args.overlap,
        label_map=label_map,   # reuse training map so indices match
    )

    if len(X_train_paths) == 0 or len(X_test_paths) == 0:
        log.error("Sequence building returned empty arrays — check your data.")
        sys.exit(1)

    log.info("X_train sequences: %d | X_test sequences: %d", len(X_train_paths), len(X_test_paths))
    
    train_gen = VideoSequenceGenerator(X_train_paths, y_train, args.batch_size, shuffle=True)
    test_gen  = VideoSequenceGenerator(X_test_paths, y_test, args.batch_size, shuffle=False)

    # ── Step 3: Model ────────────────────────────────────────────────────── #
    model = build_cnn_bilstm(num_classes, args.sequence_length)
    model.summary(print_fn=log.info)

    # ── Step 4: Callbacks ────────────────────────────────────────────────── #
    best_model_path  = str(model_dir / "best_model.keras")
    csv_log_path     = str(reports_dir / "training_log.csv")

    callbacks = [
        ModelCheckpoint(
            best_model_path, monitor="val_accuracy",
            save_best_only=True, mode="max", verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy", patience=args.patience,
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=args.lr_reduce_factor,
            patience=args.lr_patience,
            min_lr=args.min_lr,
            verbose=1,
        ),
        CSVLogger(csv_log_path, append=False),
    ]

    # ── Step 5: Training loop ─────────────────────────────────────────────── #
    log.info("Starting training: %d epochs, batch=%d", args.epochs, args.batch_size)
    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=test_gen,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Step 6: Save final model ──────────────────────────────────────────── #
    final_model_path = str(model_dir / "final_model.keras")
    model.save(final_model_path)
    log.info("Final model saved → %s", final_model_path)

    # ── Step 7: Evaluate ─────────────────────────────────────────────────── #
    # Build human-readable class names from the discovered label_map
    # e.g., {1:0, 2:1, 3:2, 5:3, 6:4}  →  ["Label 1", "Label 2", ...]
    reverse_map = {idx: orig for orig, idx in label_map.items()}
    class_names = [f"Label {reverse_map[i]}" for i in range(num_classes)]
    metrics = evaluate(model, test_gen, y_test, class_names, reports_dir)

    # ── Step 8: Plot training history ─────────────────────────────────────── #
    plot_training_history(history, reports_dir)

    # ── Step 9: Save metadata ─────────────────────────────────────────────── #
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "num_classes": num_classes,
        "class_names": class_names,
        # label_map: original dataset label → model output index
        # e.g. {"1": 0, "2": 1, "3": 2, "5": 3, "6": 4}
        "label_map": {str(k): v for k, v in label_map.items()},
        "sequence_length": args.sequence_length,
        "input_shape": [*IMG_SIZE, 3],
        "overlap": args.overlap,
        "epochs_run": len(history.history["accuracy"]),
        "best_val_accuracy": float(max(history.history.get("val_accuracy", [0]))),
        "final_metrics": metrics,
        "train_subjects": subjects_train,
        "test_subjects": subjects_test,
    }
    meta_path = model_dir / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info("Metadata saved → %s", meta_path)
    log.info("=" * 60)
    log.info("Training complete!")
    log.info("  Best model  : %s", best_model_path)
    log.info("  Final model : %s", final_model_path)
    log.info("  Metadata    : %s", meta_path)
    log.info("  Reports dir : %s", reports_dir)
    log.info("=" * 60)


# =========================================================================== #
#  6. EVALUATION & ANALYTICS                                                   #
# =========================================================================== #

def evaluate(
    model: "tf.keras.Model",
    test_gen: "tf.keras.utils.Sequence",
    y_test: np.ndarray,
    class_names: list[str],
    out_dir: Path,
) -> dict:
    """Full evaluation suite: metrics, confusion matrix, classification report."""
    log.info("Evaluating model …")
    y_pred_prob = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = y_test  # already integer indices (sparse)

    acc  = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec  = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1   = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    metrics = {
        "accuracy":  round(acc,  4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1_score":  round(f1,   4),
    }

    # ── Print ──────────────────────────────────────────────────────────────── #
    separator = "─" * 60
    log.info(separator)
    log.info("EVALUATION RESULTS")
    log.info(separator)
    for k, v in metrics.items():
        log.info("  %-12s : %.4f", k.capitalize(), v)
    log.info(separator)

    report = classification_report(
        y_true, y_pred,
        target_names=[class_names[i] for i in sorted(np.unique(y_true))],
        zero_division=0,
    )
    log.info("Classification Report:\n%s", report)

    # ── Save reports ──────────────────────────────────────────────────────── #
    out_dir.mkdir(parents=True, exist_ok=True)

    # metrics.json
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # classification_report.txt
    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(report)

    # ── Confusion matrix plot ─────────────────────────────────────────────── #
    cm = confusion_matrix(y_true, y_pred)
    present_labels = sorted(np.unique(y_true))
    present_names  = [class_names[i] for i in present_labels]

    fig, ax = plt.subplots(figsize=(max(6, len(present_labels) * 1.5),
                                    max(5, len(present_labels) * 1.3)))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(present_names)),
        yticks=np.arange(len(present_names)),
        xticklabels=present_names,
        yticklabels=present_names,
        title="Confusion Matrix",
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=11)
    fig.tight_layout()
    cm_path = str(out_dir / "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    log.info("Confusion matrix saved → %s", cm_path)

    return metrics


def plot_training_history(history: "tf.keras.callbacks.History", out_dir: Path) -> None:
    """Saves accuracy & loss training curves to out_dir/training_history.png."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, metric, title, ylabel in [
        (axes[0], "accuracy", "Model Accuracy", "Accuracy"),
        (axes[1], "loss",     "Model Loss",     "Loss"),
    ]:
        ax.plot(history.history[metric],     label="Train",      linewidth=2)
        ax.plot(history.history[f"val_{metric}"], label="Validation", linewidth=2, linestyle="--")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("CNN-BiLSTM Training History", fontsize=15, fontweight="bold")
    fig.tight_layout()
    hist_path = str(out_dir / "training_history.png")
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    log.info("Training history saved → %s", hist_path)


# =========================================================================== #
#  7. INFERENCE UTILITY                                                        #
# =========================================================================== #

def load_trained_model(model_path: str) -> tuple["tf.keras.Model", dict]:
    """
    Loads a saved .keras model and its companion metadata JSON.

    Returns:
        (model, metadata_dict)
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = load_model(str(model_path))
    log.info("Model loaded from %s", model_path)

    # Look for metadata next to the model file
    meta_candidates = [
        model_path.parent / "model_metadata.json",
        model_path.with_suffix(".json"),
    ]
    metadata = {}
    for mc in meta_candidates:
        if mc.exists():
            with open(mc) as f:
                metadata = json.load(f)
            log.info("Metadata loaded from %s", mc)
            break

    if not metadata:
        log.warning("No metadata JSON found. Using defaults.")
        metadata = {
            "num_classes": 5,
            "class_names": list(CLASS_NAMES.values()),
            "sequence_length": 10,
            "input_shape": [224, 224, 3],
            "overlap": 0.5,
        }
    return model, metadata


def predict_clip(
    model: "tf.keras.Model",
    video_path: str,
    metadata: dict,
    desc: str = "Inference",
    position: int = 0,
) -> dict:
    from tqdm import tqdm
    """
    Runs inference on a video clip and returns per-window and aggregated
    predictions suitable for consumption by the proctoring engine.

    Args:
        model:      Loaded Keras model.
        video_path: Path to the video file.
        metadata:   Dict from model_metadata.json.

    Returns:
        {
          "windows":        list of {start_frame, end_frame, predicted_class, confidence, class_name},
          "summary":        {class_name: fraction_of_windows},
          "dominant_class": class_name with highest fraction,
          "anomaly_score":  fraction of windows NOT labelled "Normal",
        }
    """
    seq_len     = metadata.get("sequence_length", 10)
    class_names = metadata.get("class_names", [])   # e.g. ["Label 1", "Label 2", ...]
    # label_map stored as {"1": 0, "2": 1, ...} (JSON keys are strings)
    label_map_raw = metadata.get("label_map", {})
    # reverse: model_index -> original_label_int
    rev_label = {v: int(k) for k, v in label_map_raw.items()}
    
    # Ensure Class 0 (Normal) is explicitly handled if class_names is missing
    normal_orig_label = 0
    
    h, w        = metadata.get("input_shape", [224, 224, 3])[:2]
    overlap     = metadata.get("overlap", 0.5)
    stride      = max(1, int(seq_len * (1 - overlap)))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frames: list[np.ndarray] = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    pbar = tqdm(total=total_frames, desc=f"{desc} (Extract)", position=position, leave=False)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (w, h)).astype(np.float32) / 255.0
        frames.append(frame)
        pbar.update(1)
    pbar.close()
    cap.release()

    if len(frames) < seq_len:
        log.warning("Video too short (%d frames) for sequence_length=%d", len(frames), seq_len)
        return {"windows": [], "summary": {}, "dominant_class": None, "anomaly_score": 0.0}

    # Build windows
    windows_data, window_info = [], []
    for start in range(0, len(frames) - seq_len + 1, stride):
        seq = np.array(frames[start: start + seq_len], dtype=np.float32)
        windows_data.append(seq)
        window_info.append((start, start + seq_len - 1))

    X = np.array(windows_data, dtype=np.float32)      # (W, T, H, W, 3)
    
    # Run inference in batches to show progress
    batch_size = 32
    probs_list = []
    
    pbar = tqdm(total=len(X), desc=f"{desc} (Predict)", position=position, leave=False)
    for i in range(0, len(X), batch_size):
        batch = X[i : i + batch_size]
        batch_probs = model.predict(batch, verbose=0)
        probs_list.append(batch_probs)
        pbar.update(len(batch))
    pbar.close()
    
    probs = np.vstack(probs_list)
    pred_indices = np.argmax(probs, axis=1)

    results = []
    for i, (start_f, end_f) in enumerate(window_info):
        model_idx = int(pred_indices[i])
        orig_lbl  = rev_label.get(model_idx, model_idx)  # original dataset label
        name      = class_names[model_idx] if model_idx < len(class_names) else f"Label {orig_lbl}"
        results.append({
            "start_frame":      start_f,
            "end_frame":        end_f,
            "predicted_label":  orig_lbl,   # original integer from dataset
            "class_name":       name,
            "confidence":       float(probs[i, model_idx]),
            "all_probs":        {
                class_names[j] if j < len(class_names) else f"Label {rev_label.get(j,j)}": float(probs[i, j])
                for j in range(probs.shape[1])
            },
        })

    # Aggregate
    from collections import Counter
    name_counts = Counter(r["class_name"] for r in results)
    total = len(results)
    summary = {k: round(v / total, 4) for k, v in name_counts.items()}
    dominant = max(summary, key=summary.get) if summary else None

    # anomaly = fraction of windows where predicted label != 0 (Class 0 / Normal)
    anomaly_fraction = sum(1 for r in results if r["predicted_label"] != normal_orig_label) / max(1, len(results))
    anomaly_score = round(anomaly_fraction, 4)

    return {
        "windows":        results,
        "summary":        summary,
        "dominant_class": dominant,
        "anomaly_score":  anomaly_score,
    }


# =========================================================================== #
#  8. ENTRY POINT                                                              #
# =========================================================================== #

def main() -> None:
    args = parse_args()

    if args.predict:
        # ── Inference mode ─────────────────────────────────────────────────── #
        if args.video is None:
            log.error("--video is required when using --predict")
            sys.exit(1)
        model_file = str(Path(args.model_path) / "best_model.keras")
        if not Path(model_file).exists():
            model_file = args.model_path  # allow passing the file directly
        model, metadata = load_trained_model(model_file)
        result = predict_clip(model, args.video, metadata)

        print("\n" + "═" * 60)
        print(f"  Clip: {args.video}")
        print(f"  Dominant behaviour : {result['dominant_class']}")
        print(f"  Anomaly score      : {result['anomaly_score']:.2%}")
        print("  Class distribution:")
        for cls, frac in sorted(result["summary"].items(), key=lambda x: -x[1]):
            bar = "█" * int(frac * 30)
            print(f"    {cls:<20} {frac:6.2%}  {bar}")
        print("═" * 60 + "\n")
    else:
        # ── Training mode ──────────────────────────────────────────────────── #
        train(args)


if __name__ == "__main__":
    main()
