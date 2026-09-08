# YOLOv11 Traffic Detection & Analysis

A computer vision project utilizing [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics) for vehicle and object detection on traffic video footage.

## Project Structure

```text
.
├── main.py                  # Entry script for YOLOv11 inference on traffic video
├── yolo11n.pt               # Pretrained YOLOv11 Nano model weights
├── yolo_traffic mp4 .mp4    # Sample traffic video input
├── requirements.txt         # Python package dependencies
├── .gitignore               # Git ignore patterns for runs and temporary files
└── AGENTS.md                # Agent & development guidelines
```

## Setup & Installation

Ensure you have Python 3.10+ installed. Install the dependencies via pip:

```bash
pip install -r requirements.txt
```

## Usage

Run the primary detection script:

```bash
python main.py
```

Results (annotated video output) are saved to `runs/detect/predict/`.

## Next Steps & Potential Enhancements

- **Object Tracking**: Enable ByteTrack or BoT-SORT (`model.track(source=..., tracker="bytetrack.yaml")`) to follow vehicle trajectories.
- **Vehicle Counting & Line Crossing**: Count vehicles crossing designated ROI lines/polygons per class (cars, trucks, buses, motorcycles).
- **Speed & Density Estimation**: Estimate vehicle velocities and road congestion levels.
- **Interactive UI**: Build a Streamlit or Gradio interface for uploading custom video streams and tuning confidence thresholds.
