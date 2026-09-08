# Agent Guidelines & Repository Context

## Project Overview
This repository contains a YOLOv11-based video detection application focused on traffic analysis.

## Key Files & Assets
- `main.py`: Main entrypoint running detection via Ultralytics YOLO API.
- `yolo11n.pt`: Default pretrained YOLOv11 Nano model checkpoint.
- `yolo_traffic mp4 .mp4`: Benchmark video input.
- `runs/`: Output directory generated automatically by Ultralytics runs (e.g. `runs/detect/predict/`).

## Environment
- Python runtime: Python 3.14 (or >= 3.10)
- Key libraries: `ultralytics`, `torch`, `torchvision`, `opencv-python`
- Platform: Windows (PowerShell)

## Coding Conventions
- Prefer writing modular, clean Python scripts with argument parsing (`argparse` or `click`) when enhancing functionality.
- Support both live display (`show=True`) and file export (`save=True`), handling headless environments gracefully.
- Handle OpenCV video writer releases and file paths with appropriate exception handling.
