# CNN-BiLSTM Cheating Behaviour Classifier

## What It Is

A deep learning model that watches a video of someone taking an online exam and classifies whether — and how — they are cheating, second-by-second. It combines a **Convolutional Neural Network (CNN)** for visual feature extraction with a **Bidirectional Long Short-Term Memory (BiLSTM)** network for temporal reasoning about behaviour over time.

---

## Why This Architecture?

Cheating detection is inherently a **video understanding** problem:

- A single frame tells you *what* someone looks like right now.
- A *sequence* of frames tells you *what they are doing* — glancing repeatedly at a phone, whispering over time, turning their head left for 5 seconds.

This is why a pure image classifier (e.g., ResNet) isn't enough — it has no memory. And a pure sequence model (e.g., plain LSTM on raw pixels) would be far too large to train from scratch on a 24-subject dataset.

The CNN-BiLSTM solves this by splitting the job:
1. **CNN** — efficient visual feature extraction (transfer learning from ImageNet)
2. **BiLSTM** — learns what *sequences* of those features mean behaviourally

---

## Dataset

Trained on the **OEP (Online Exam Proctoring) Database** (Atoum et al., IEEE TMM):
- 24 subjects (15 actors + 9 real exam-takers)
- Each session recorded by webcam + wearable camera
- Ground truth labels in `gt.txt`: `start_frame  end_frame  label`

| Label | Behaviour | Description |
|:-----:|-----------|-------------|
| **0** | Normal | Candidate is focused on the exam |
| **1** | Gaze Away | Eyes/head directed away from screen |
| **2** | Using Device | Phone or other electronic device visible |
| **3** | Talking | Speaking or whispering (mouth movement) |
| **4** | Multiple People | A second person enters the frame |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Input video clip  →  T frames of (224 × 224 × 3)           │
└────────────────────────────┬────────────────────────────────┘
                             │  TimeDistributed (applied per frame)
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  MobileNetV2 (frozen, ImageNet weights)                      │
│  Output:  (T, 7, 7, 1280)                                    │
└────────────────────────────┬────────────────────────────────┘
                             ▼
         TimeDistributed(GlobalAveragePooling2D)
                  →  (T, 1280)  — compact frame descriptor
                             ▼
            TimeDistributed(Dense 512, ReLU + Dropout 0.4)
                  →  (T, 512)
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  Bidirectional LSTM  256 units  (return_sequences=True)      │
│  → Sees context from BOTH past and future frames             │
│  → Output: (T, 512)                                          │
└────────────────────────────┬────────────────────────────────┘
                     Dropout 0.4
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  Bidirectional LSTM  128 units  (return_sequences=False)     │
│  → Collapses the sequence into a single behaviour vector     │
│  → Output: (256,)                                            │
└────────────────────────────┬────────────────────────────────┘
                     Dropout 0.4
                             ▼
              Dense 128, ReLU  →  Dense 5, Softmax
                             ▼
          ┌──────────────────────────────────────┐
          │  Output: probability over 5 classes  │
          └──────────────────────────────────────┘
```

| Component | Parameters | Notes |
|-----------|-----------|-------|
| MobileNetV2 backbone | 2.26 M | **Frozen** — not updated during training |
| TimeDistributed Dense | 0.66 M | Trainable |
| BiLSTM layers | 1.57 M + 0.66 M | Trainable |
| Classification head | 0.03 M | Trainable |
| **Total** | **5.18 M** | 2.92 M trainable / 2.26 M frozen |

---

## How It Works — Step by Step

### 1. Frame extraction (`extract_frames_for_subject`)
Every 5th frame within each labelled segment is saved as a 224 × 224 JPEG. Webcam video (`username1.avi`) is preferred over the wearcam (`username2.avi`).

### 2. Subject-level train/test split (`split_subjects`)
80% of *subjects* go to training, 20% to testing. This prevents data leakage — all frames from subject N are exclusively in one split.

### 3. Sliding-window sequencing (`build_sequences`)
Frames are grouped into overlapping windows of length `T` (default 10) with 50% overlap. Each window becomes one training sample:
```
frames: [f0, f1, f2 … f9]  → Label: "Talking"
frames: [f5, f6, f7 … f14] → Label: "Talking"   ← 50% overlap
```

### 4. Training
- **Loss**: Sparse categorical cross-entropy
- **Optimiser**: Adam (initial LR `1e-4`)
- **Callbacks**:
  - `ModelCheckpoint` — saves `best_model.keras` on best `val_accuracy`
  - `EarlyStopping` — stops after `patience` epochs without improvement
  - `ReduceLROnPlateau` — halves LR after 5 stale epochs (min `1e-7`)
  - `CSVLogger` — epoch-by-epoch log at `data/reports/training_log.csv`

### 5. Evaluation
After training the following are saved to `data/reports/`:

| Output | Description |
|--------|-------------|
| `confusion_matrix.png` | Per-class true vs predicted counts |
| `training_history.png` | Accuracy & loss curves over epochs |
| `metrics.json` | Accuracy, Precision, Recall, F1 (weighted) |
| `classification_report.txt` | Per-class precision/recall/F1 + support |
| `training_log.csv` | Loss & accuracy for every epoch |

### 6. Model artefacts saved to `data/models/`

| File | Contents |
|------|----------|
| `best_model.keras` | Weights at best validation accuracy (use this for inference) |
| `final_model.keras` | Weights at end of all epochs |
| `model_metadata.json` | Class names, sequence length, shape, metrics, train/test subjects |

---

## Configuration

All training parameters are in [`src/configs/app.yaml`](../src/configs/app.yaml) under the `model:` key — no code changes needed:

```yaml
model:
  dataset_path:   "data/raw/database"   # Where OEP subjects live
  processed_path: "data/processed"      # Frame cache directory
  model_path:     "data/models"         # Saved .keras artefacts
  reports_path:   "data/reports"        # Plots and metrics

  sequence_length: 10     # Frames per sliding window
  overlap:         0.5    # 50% window overlap
  test_size:       0.2    # 20% subjects held out for testing

  epochs:          50
  batch_size:      8
  learning_rate:   0.0001
  patience:        10     # EarlyStopping
  lr_reduce_factor: 0.5
  lr_patience:     5
  min_lr:          0.0000001

  anomaly_threshold: 0.3  # Flag clip if >30% windows are non-Normal
```

CLI flags (e.g. `--epochs 30`) override YAML values for one-off runs.

---

## Running

### Train (on server with real OEP data)
```bash
python src/models/cnn_bi_lstm_train.py
# or override a specific param:
python src/models/cnn_bi_lstm_train.py --epochs 30 --batch-size 4
```

### Smoke-test locally (no dataset needed)
```bash
python src/models/cnn_bi_lstm_train.py --force-synthetic --epochs 2
```

### Inference on a video clip
```bash
python src/models/cnn_bi_lstm_train.py --predict --video exam_clip.avi
```
Output:
```
  Dominant behaviour : Normal
  Anomaly score      : 12.50%
  Class distribution:
    Normal               87.50%  ██████████████████████████
    Gaze Away            12.50%  ███
```

### Load model from Python (proctoring engine integration)
```python
from src.models.cnn_bi_lstm_train import load_trained_model, predict_clip

model, metadata = load_trained_model("data/models/best_model.keras")
result = predict_clip(model, "exam_recording.avi", metadata)

print(result["dominant_class"])    # e.g. "Gaze Away"
print(result["anomaly_score"])     # e.g. 0.35
print(result["summary"])           # {"Normal": 0.65, "Gaze Away": 0.35}
```

---

## What It Can Be Used For

| Use Case | How |
|----------|-----|
| **Post-exam review** | Run on a full recorded exam; get a timeline of flagged segments |
| **Real-time suspicion scoring** | Call `predict_clip()` on a rolling 10-frame buffer from the webcam feed |
| **Integration with proctoring engine** | `anomaly_score` can replace or supplement the rule-based `ViolationTracker` |
| **Fine-tuning on new data** | Unfreeze MobileNetV2 layers and re-train with a lower LR on domain-specific data |
| **Multi-camera fusion** | Run separately on webcam and wearcam feeds; ensemble predictions |

---

## Limitations

- **Small dataset** — 24 subjects is limited for a 5-class problem; consider data augmentation or transfer from a larger video dataset if accuracy is low.
- **Label granularity** — all behaviour within a labelled segment gets the same label, even if the subject momentarily reverts to normal mid-segment.
- **No audio** — the model uses only visual frames; talking detection relies solely on mouth-movement cues in the image, not the audio channel.
- **Fixed resolution** — input is always resized to 224 × 224; extreme camera distances may affect accuracy.
