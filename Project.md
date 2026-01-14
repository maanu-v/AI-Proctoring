# AI-Based Online Proctoring System

## 1. Introduction

Online examinations have become an integral part of modern education and recruitment processes. While they provide flexibility and scalability, they also introduce significant challenges related to academic integrity and identity verification. Manual proctoring through live invigilators is expensive, difficult to scale, and prone to human error.

This project aims to design and implement an **AI-based automated proctoring system** that monitors a candidate through a webcam and detects suspicious or abnormal behavior using computer vision and machine learning techniques. The system focuses on *behavioral analysis, environmental awareness, and identity consistency*, rather than making absolute judgments of cheating.

The project is developed incrementally, starting with face-based behavioral features and later extending to object detection and identity verification.

---

## 2. Project Objectives

The primary objectives of this project are:

* To monitor candidate behavior during an online examination using a webcam
* To detect abnormal patterns such as looking away, talking, or absence from the screen
* To identify the presence of forbidden objects (e.g., phone, earphones)
* To verify that the same candidate remains present throughout the examination
* To generate structured logs and risk indicators instead of binary cheating decisions

---

## 3. Scope of the Project

### In Scope

* Real-time face landmark detection
* Head pose estimation (yaw, pitch, roll)
* Eye gaze direction estimation
* Mouth activity detection
* Face presence and multiple-face detection
* External object detection (phone, earphones, additional person)
* Face identity verification against a reference image
* Event logging and basic risk scoring

### Out of Scope (Initial Version)

* Audio-based cheating detection
* Browser-level activity monitoring
* Fully automated exam termination
* Legal or institutional enforcement mechanisms

---

## 4. System Overview

The system processes a live video stream from the candidate's webcam and analyzes it using modular AI components. Each module independently extracts signals, which are later aggregated into higher-level events and risk indicators.

### High-Level Workflow

1. Capture webcam video frames
2. Detect and track facial landmarks
3. Extract behavioral features (head pose, gaze, mouth activity)
4. Detect external objects in the scene
5. Verify candidate identity periodically
6. Log events and compute risk metrics

---

## 5. Key Features (Planned)

### 5.1 Face-Based Behavioral Analysis

* **Head Pose Estimation**: Detects prolonged head turns or downward gaze
* **Eye Gaze Tracking**: Estimates eye direction relative to the screen
* **Mouth Activity Detection**: Identifies excessive talking or reading aloud
* **Face Presence Monitoring**: Detects face absence or multiple faces

### 5.2 Environment & Object Monitoring

* Detection of mobile phones, earphones, books, or additional persons
* Temporal filtering to reduce false positives
* Correlation with face-based behavior

### 5.3 Identity Verification

* Capture a reference face image at exam start
* Extract facial embeddings
* Periodic similarity comparison with live video
* Detection of candidate replacement or impersonation

---

## 6. Design Philosophy

* **Event-based Detection**: The system reports observable events instead of direct accusations
* **Explainability**: Each alert is backed by measurable signals and timestamps
* **Modularity**: Each component can be developed, tested, and improved independently
* **Real-Time Efficiency**: Optimized to run on standard consumer hardware
* **Ethical Awareness**: Minimizes bias and avoids intrusive assumptions

---

## 7. Expected Outcomes

* A working real-time AI proctoring prototype
* Structured logs of suspicious events
* A scalable foundation for further research or product development
* A strong demonstration project combining AI, computer vision, and systems design

---

## 8. Future Enhancements

* Audio analysis for speech detection
* Advanced behavioral modeling using temporal ML models
* Dashboard for human review of flagged sessions
* Deployment as a web-based proctoring service

---

## 9. Conclusion

This project serves as a practical exploration of applied artificial intelligence in a real-world, high-impact domain. By focusing on modularity, explainability, and ethical considerations, the system aims to balance technical rigor with responsible deployment. The incremental development approach ensures continuous learning, validation, and improvement throughout the project lifecycle.
