from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open your video
cap = cv2.VideoCapture("yolo_traffic mp4 .mp4")

# Vehicle classes in COCO dataset
vehicle_classes = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Get frame dimensions
    height, width = frame.shape[:2]

    # Define road ROI
    # Change these values according to your video
    roi_x1 = int(width * 0.10)
    roi_y1 = int(height * 0.30)
    roi_x2 = int(width * 0.90)
    roi_y2 = int(height * 0.95)

    # Draw ROI
    cv2.rectangle(
        frame,
        (roi_x1, roi_y1),
        (roi_x2, roi_y2),
        (255, 255, 0),
        2
    )

    # Run YOLO detection
    results = model(frame, verbose=False)

    # Vehicle counters
    total_vehicles = 0
    car_count = 0
    motorcycle_count = 0
    bus_count = 0
    truck_count = 0

    # Process detections
    for result in results:

        boxes = result.boxes

        for box in boxes:

            # Get class ID
            class_id = int(box.cls[0])

            # Get confidence
            confidence = float(box.conf[0])

            # Only consider vehicles
            if class_id not in vehicle_classes:
                continue

            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Find center of vehicle
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Check if vehicle is inside ROI
            if (
                roi_x1 <= center_x <= roi_x2
                and roi_y1 <= center_y <= roi_y2
            ):

                total_vehicles += 1

                # Count vehicle type
                if class_id == 2:
                    car_count += 1

                elif class_id == 3:
                    motorcycle_count += 1

                elif class_id == 5:
                    bus_count += 1

                elif class_id == 7:
                    truck_count += 1

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Vehicle label
                label = f"{vehicle_classes[class_id]} {confidence:.2f}"

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                # Draw center point
                cv2.circle(
                    frame,
                    (center_x, center_y),
                    4,
                    (0, 0, 255),
                    -1
                )

    # ------------------------------------------------
    # TRAFFIC DENSITY
    # ------------------------------------------------

    # Maximum number of vehicles considered HIGH density
    MAX_VEHICLES = 20

    # Calculate density percentage
    density = (total_vehicles / MAX_VEHICLES) * 100

    # Limit density to 100%
    density = min(density, 100)

    # Determine traffic level
    if total_vehicles <= 5:
        traffic_level = "LOW"

    elif total_vehicles <= 12:
        traffic_level = "MEDIUM"

    else:
        traffic_level = "HIGH"

    # ------------------------------------------------
    # DISPLAY INFORMATION ON VIDEO
    # ------------------------------------------------

    # Background rectangle for information
    cv2.rectangle(
        frame,
        (10, 10),
        (390, 180),
        (0, 0, 0),
        -1
    )

    # Total vehicles
    cv2.putText(
        frame,
        f"Vehicles: {total_vehicles}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # Density
    cv2.putText(
        frame,
        f"Density: {density:.1f}%",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # Traffic level
    cv2.putText(
        frame,
        f"Traffic: {traffic_level}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # Vehicle type counts
    cv2.putText(
        frame,
        f"Car:{car_count}  Bike:{motorcycle_count}",
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Bus:{bus_count}  Truck:{truck_count}",
        (20, 175),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # Show frame
    cv2.imshow("YOLO Traffic Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()