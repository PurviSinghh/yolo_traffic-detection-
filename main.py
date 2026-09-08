from ultralytics import YOLO

# Load the pretrained YOLO model
model = YOLO("yolo11n.pt")

# Give YOLO your traffic video
results = model("yolo_traffic mp4 .mp4", save=True)

print("Detection complete!")