# 🎥 AI-Based Video Proctoring System

## 📌 Overview

This project aims to build an AI-powered video proctoring system that detects suspicious behavior during examinations using computer vision and machine learning techniques.

The system extracts real-time and offline video features such as:

- Head pose
- Gaze direction
- Blink rate
- Mouth activity
- Multiple person detection

Based on these behavioral signals, the system classifies whether a candidate is likely cheating or not.

---

## 🎯 Objectives

1. Extract facial and behavioral features from video streams.
2. Support both:
   - Real-time webcam input
   - Pre-recorded video input
3. Aggregate temporal behavior patterns.
4. Classify suspicious behavior using:
   - Rule-based scoring (MVP)
   - Machine learning classifier (Advanced phase)
5. Generate interpretable logs for explainability.

---

## 🏗️ System Architecture

            +----------------------+
            |  Video Input         |
            |  (Live / Recorded)   |
            +----------+-----------+
                       |
                       v
            +----------------------+
            | Frame Extraction     |
            | (OpenCV)             |
            +----------+-----------+
                       |
                       v
            +----------------------+
            | Feature Extraction   |
            | (MediaPipe)          |
            +----------+-----------+
                       |
                       v
            +----------------------+
            | Temporal Aggregation |
            | (5-sec windows)      |
            +----------+-----------+
                       |
                       v
            +----------------------+
            | Classification       |
            | (Rule / ML Model / LSTM Model)    |
            +----------+-----------+
                       |
                       v
            +----------------------+
            | Output & Logs        |
            +----------------------+



---

## 🧠 Extracted Features

### 1️⃣ Head Pose
- Yaw (left/right)
- Pitch (up/down)
- Roll
- Head turn frequency

### 2️⃣ Gaze Direction
- Eye landmark tracking
- Ratio of off-screen gaze
- Duration of looking away

### 3️⃣ Blink Detection
- Eye Aspect Ratio (EAR)
- Blink rate per minute
- Long eye closure detection

### 4️⃣ Mouth Activity
- Mouth Aspect Ratio (MAR)
- Speaking detection
- Excessive mouth movement flag

### 5️⃣ Multiple Person Detection
- Person count per frame
- Suspicious flag if >1 detected

---

