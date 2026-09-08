from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open your video
cap = cv2.VideoCapture("yolo_traffic mp4 .mp4")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Run YOLO detection
    results = model(frame, verbose=False)

    # Start count
    vehicle_count = 0

    # Draw detections and count vehicles
    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            if class_name in ["car", "bus", "truck", "motorcycle"]:
                vehicle_count += 1

        # Draw bounding boxes on frame
        frame = result.plot()

    # ===== THIS PUTS THE COUNT ON THE VIDEO =====
    cv2.rectangle(frame, (10, 10), (350, 70), (0, 0, 0), -1)

    cv2.putText(
        frame,
        f"VEHICLE COUNT: {vehicle_count}",
        (20, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Show the modified frame
    cv2.imshow("YOLO Vehicle Detection + Count", frame)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()