import cv2
import torch
from ultralytics import YOLO

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool


# ============================================================
# 1. START
# ============================================================

print("Starting Real-Time Traffic GNN...")


# ============================================================
# 2. LOAD YOLO
# ============================================================

model = YOLO("yolo11n.pt")

print("YOLO loaded successfully.")


# ============================================================
# 3. OPEN VIDEO
# ============================================================

video_path = "yolo_traffic mp4 .mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()

print("Video opened successfully.")


# ============================================================
# 4. VEHICLE CLASSES
# ============================================================

vehicle_classes = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}


# ============================================================
# 5. ROI
# ============================================================

ROI_X1 = 0.10
ROI_Y1 = 0.30
ROI_X2 = 0.90
ROI_Y2 = 0.95


# ============================================================
# 6. VEHICLE TYPE ENCODING
# ============================================================

type_encoding = {
    "Car": 0,
    "Motorcycle": 1,
    "Bus": 2,
    "Truck": 3
}


# ============================================================
# 7. TRAFFIC LEVELS
# ============================================================

traffic_levels = {
    0: "LOW",
    1: "MEDIUM",
    2: "HIGH"
}


# ============================================================
# 8. GNN MODEL
# ============================================================

class TrafficGNN(torch.nn.Module):

    def __init__(self):
        super().__init__()

        # These names match traffic_gnn.pth
        self.gcn1 = GCNConv(3, 16)
        self.gcn2 = GCNConv(16, 8)
        self.classifier = torch.nn.Linear(8, 3)

    def forward(self, x, edge_index, batch):

        x = self.gcn1(x, edge_index)
        x = torch.relu(x)

        x = self.gcn2(x, edge_index)
        x = torch.relu(x)

        x = global_mean_pool(x, batch)

        x = self.classifier(x)

        return x


# ============================================================
# 9. LOAD TRAINED GNN
# ============================================================

gnn_model = TrafficGNN()

gnn_model.load_state_dict(
    torch.load(
        "traffic_gnn.pth",
        map_location="cpu"
    )
)

gnn_model.eval()

print("Trained GNN loaded successfully.")


# ============================================================
# 10. GRAPH DISTANCE
# ============================================================

DISTANCE_THRESHOLD = 150


# ============================================================
# 11. PROCESS VIDEO
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Video finished.")
        break


    height, width = frame.shape[:2]


    # --------------------------------------------------------
    # ROI coordinates
    # --------------------------------------------------------

    roi_x1 = int(width * ROI_X1)
    roi_y1 = int(height * ROI_Y1)

    roi_x2 = int(width * ROI_X2)
    roi_y2 = int(height * ROI_Y2)


    # --------------------------------------------------------
    # YOLO + BoT-SORT
    # --------------------------------------------------------

    results = model.track(
        frame,
        persist=True,
        tracker="botsort.yaml",
        verbose=False
    )


    # --------------------------------------------------------
    # Store vehicles
    # --------------------------------------------------------

    frame_data = []


    # --------------------------------------------------------
    # Process detections
    # --------------------------------------------------------

    for result in results:

        if result.boxes is None:
            continue


        for i in range(len(result.boxes)):

            class_id = int(
                result.boxes.cls[i].item()
            )


            # Only vehicles
            if class_id not in vehicle_classes:
                continue


            vehicle_type = vehicle_classes[class_id]


            # Bounding box
            x1, y1, x2, y2 = map(
                int,
                result.boxes.xyxy[i].tolist()
            )


            # Center point
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)


            # ------------------------------------------------
            # ROI filtering
            # ------------------------------------------------

            if not (
                roi_x1 <= center_x <= roi_x2
                and
                roi_y1 <= center_y <= roi_y2
            ):
                continue


            # Store vehicle
            frame_data.append({
                "type": vehicle_type,
                "x": center_x,
                "y": center_y
            })


            # ------------------------------------------------
            # Draw bounding box
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # Vehicle label
            cv2.putText(
                frame,
                vehicle_type,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )


    # ========================================================
    # 12. VEHICLE COUNT
    # ========================================================

    vehicle_count = len(frame_data)


    # Default state
    traffic_state = "NO VEHICLES"


    # ========================================================
    # 13. CREATE GRAPH
    # ========================================================

    if vehicle_count > 0:

        nodes = []
        positions = []


        # ----------------------------------------------------
        # Create graph nodes
        # ----------------------------------------------------

        for vehicle in frame_data:

            vehicle_type = vehicle["type"]
            x_position = vehicle["x"]
            y_position = vehicle["y"]


            nodes.append([
                type_encoding[vehicle_type],
                x_position,
                y_position
            ])


            positions.append(
                (x_position, y_position)
            )


        # ----------------------------------------------------
        # Node tensor
        # ----------------------------------------------------

        x = torch.tensor(
            nodes,
            dtype=torch.float
        )


        # ----------------------------------------------------
        # Create edges
        # ----------------------------------------------------

        edges = []


        for i in range(len(positions)):

            for j in range(i + 1, len(positions)):

                x1, y1 = positions[i]
                x2, y2 = positions[j]


                distance = (
                    (x1 - x2) ** 2 +
                    (y1 - y2) ** 2
                ) ** 0.5


                if distance < DISTANCE_THRESHOLD:

                    edges.append([i, j])
                    edges.append([j, i])


        # ----------------------------------------------------
        # Edge tensor
        # ----------------------------------------------------

        if len(edges) > 0:

            edge_index = torch.tensor(
                edges,
                dtype=torch.long
            ).t().contiguous()

        else:

            edge_index = torch.empty(
                (2, 0),
                dtype=torch.long
            )


        # ----------------------------------------------------
        # Create graph
        # ----------------------------------------------------

        graph = Data(
            x=x,
            edge_index=edge_index
        )


        # ====================================================
        # 14. GNN PREDICTION
        # ====================================================

        batch = torch.zeros(
            graph.x.size(0),
            dtype=torch.long
        )


        with torch.no_grad():

            output = gnn_model(
                graph.x,
                graph.edge_index,
                batch
            )


            prediction = output.argmax(
                dim=1
            ).item()


            traffic_state = traffic_levels[
                prediction
            ]


    # ========================================================
    # 15. DRAW ROI
    # ========================================================

    cv2.rectangle(
        frame,
        (roi_x1, roi_y1),
        (roi_x2, roi_y2),
        (255, 255, 0),
        2
    )


    # ========================================================
    # 16. DISPLAY VEHICLE COUNT
    # ========================================================

    cv2.putText(
        frame,
        f"VEHICLES: {vehicle_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    # ========================================================
    # 17. DISPLAY GNN PREDICTION
    # ========================================================

    cv2.putText(
        frame,
        f"GNN TRAFFIC: {traffic_state}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )


    # ========================================================
    # 18. SHOW VIDEO
    # ========================================================

    cv2.imshow(
        "Real-Time Traffic GNN",
        frame
    )


    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# 19. CLOSE
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("Real-Time GNN processing completed.")