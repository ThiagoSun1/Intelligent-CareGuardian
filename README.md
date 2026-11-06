# Intelligent CareGuardian

Intelligent CareGuardian is a privacy-focused edge-AI patient-monitoring prototype. It uses a camera, a YOLO pose model, and a motorized pan-and-tilt mount to keep a person in view and identify visible behaviors that may indicate a fall or discomfort.

## Prototype features

- Tracks a person with pan-and-tilt servo motors.
- Detects possible falls, chest-holding, stomach-holding, and eye-rubbing gestures.
- Records event counts and durations.
- Sends optional email alerts with privacy-protected screenshots.
- Sends a scheduled daily event report.
- Provides a Flask dashboard with the live feed, event summary, Qwen response, settings, and hardware health.
- Monitors camera-frame and servo-controller failures.
- Runs video processing and AI inference locally on the edge device.

This prototype identifies possible visible gestures. It does not diagnose pain or replace professional supervision or medical judgment.

## Recovered source notice

This repository package was reconstructed from the latest clean source snapshot available after the original Jetson became inaccessible. It may not contain changes made directly on that Jetson after the snapshot was saved.

## Main file

```text
eye_stomach_web.py
```

The application expects the YOLO pose weights at:

```text
yolov8n-pose.pt
```

Model weights, databases, event logs, screenshots, device secrets, and virtual environments are intentionally excluded from Git.

## Setup

Create and activate a Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On a Jetson, use NVIDIA-compatible PyTorch and OpenCV packages for the installed JetPack version. A normal `pip install` may install a CPU-only or incompatible PyTorch build.

Download or provide `yolov8n-pose.pt`, then enter the sender Gmail address and Gmail app password in the protected email configuration section of `eye_stomach_web.py`. Never commit real credentials.

Run:

```bash
python3 eye_stomach_web.py
```

Open:

```text
http://127.0.0.1:5000/dashboard
```

## Hardware used by the recovered version

- NVIDIA Jetson Orin Nano
- USB camera
- PCA9685-compatible 16-channel servo controller
- Two pan-and-tilt servo motors

The recovered configuration uses PCA9685 channel 1 for pan and channel 0 for tilt.

## Privacy and security

Do not commit real patient data, screenshots, event logs, databases, Gmail credentials, or generated device secrets. The included `.gitignore` excludes these common private files.

