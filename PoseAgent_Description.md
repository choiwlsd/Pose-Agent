# Pose Analysis & Supervisor Evaluation System

## Overview

This project is a real-time posture analysis system built on top of Google MediaPipe Pose tracking. It extracts human skeletal keypoints, computes biomechanical features, and evaluates posture quality using a hybrid rule-based and machine learning approach.

The system is designed with a dual-agent architecture:

- Pose Agent: interprets raw pose data into structured posture states
- Supervisor Agent: evaluates posture states using a reward lookup table

---

## System Architecture

```
MediaPipe Pose Tracking
↓
FeatureExtractor
↓
Pose Agent
- Feature → Posture State Mapping
- Issue Detection
↓
Supervisor Agent
- State Evaluation
- Reward Lookup (Q-table)
↓
Final Output (State + Reward)
```

---

## Components

### 1. PoseExtractor

Uses MediaPipe to extract real-time pose landmarks from webcam or video input.

### 2. FeatureExtractor

Computes biomechanical features such as:

- Joint distances
- Joint angles
- Wrist velocity
- Temporal feature sequences (30-frame buffer)

### 3. PostureFeedbackAnalyzer

Applies:

- Logistic Regression classifier
- Z-score deviation analysis
- Rule-based biomechanical interpretation

Outputs:

- posture case (stable / unstable / risky)
- final score
- detailed feature-level explanations

### 4. MockPoseRewardTable (Supervisor Agent)

Acts as a reward lookup system:

- Maps posture states to predefined reward scores
- Does not perform learning
- Provides deterministic evaluation

---

## State Definition

A posture state is defined as:
`(posture_issue, fail_count)`

Example:

- (WRIST_SHAKE, 2)
- (LEFT_WRIST, 0)
- (GOOD, 0)

---

## Reward System

Reward is computed based on:

- posture improvement (score delta)
- severity of biomechanical issues
- repetition of the same issue

Range: -1.0 ~ +1.0

---

## Key Features

- Real-time pose tracking
- Explainable biomechanical feature analysis
- Hybrid ML + rule-based evaluation
- State-based posture classification
- Supervisor-level reward lookup system

---

## Important Note

This system is not a full reinforcement learning implementation.  
Although it uses a Q-table-like structure, it functions as a deterministic reward lookup system rather than a learning-based policy model.

---

## Future Work

- Transition from rule-based Pose Agent to learned policy network
- Replace Q-table with function approximation
- Introduce sequence models (LSTM / TCN)
- Enable adaptive reward learning

---

## Dependencies

- MediaPipe
- NumPy
- scikit-learn
