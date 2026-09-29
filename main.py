print("I AM RUNNING THIS MAIN.PY")
import cv2
import torch

from ultralytics import YOLO

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool


# ============================================================
# 1. LOAD YOLOv11 MODEL
# ============================================================

model = YOLO("yolo11n.pt")


# ============================================================
# 2. VIDEO INPUT
# ============================================================

video_path = "yolo_traffic mp4 .mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


# ============================================================
# 3. VEHICLE CLASSES
# ============================================================

vehicle_classes = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}


# ============================================================
# 4. ROI SETTINGS
# ============================================================

frame_width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

frame_height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

roi_x1 = int(frame_width * 0.10)
roi_y1 = int(frame_height * 0.30)

roi_x2 = int(frame_width * 0.90)
roi_y2 = int(frame_height * 0.95)


# ============================================================
# 5. TRAFFIC SETTINGS
# ============================================================

MAX_VEHICLES = 20

frame_number = 0


# ============================================================
# 6. VEHICLE TRACK STORAGE
# ============================================================

vehicle_tracks = {}


# ============================================================
# 7. TRAFFIC DATA STORAGE
# ============================================================

traffic_data = []

# Stores graphs that will be used for GNN training
graph_dataset = []


# ============================================================
# 8. VEHICLE TYPE ENCODING
# ============================================================

type_encoding = {
    "Car": 0,
    "Motorcycle": 1,
    "Bus": 2,
    "Truck": 3
}


# ============================================================
# 9. CREATE TRAFFIC GRAPH
# ============================================================

def create_traffic_graph(frame_data):

    # --------------------------------------------------------
    # No vehicles
    # --------------------------------------------------------

    if len(frame_data) == 0:

        x = torch.empty(
            (0, 3),
            dtype=torch.float
        )

        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long
        )

        return Data(
            x=x,
            edge_index=edge_index
        )


    # --------------------------------------------------------
    # NODE FEATURES
    # --------------------------------------------------------

    node_features = []

    positions = []

    for vehicle in frame_data:

        vehicle_type = type_encoding[
            vehicle["type"]
        ]

        x_position = vehicle["x"]
        y_position = vehicle["y"]

        node_features.append([
            vehicle_type,
            x_position,
            y_position
        ])

        positions.append(
            (x_position, y_position)
        )


    x = torch.tensor(
        node_features,
        dtype=torch.float
    )


    # --------------------------------------------------------
    # CREATE EDGES
    # --------------------------------------------------------

    edges = []

    DISTANCE_THRESHOLD = 150

    number_of_vehicles = len(positions)


    for i in range(number_of_vehicles):

        for j in range(i + 1, number_of_vehicles):

            x1, y1 = positions[i]
            x2, y2 = positions[j]


            distance = (
                (x1 - x2) ** 2 +
                (y1 - y2) ** 2
            ) ** 0.5


            if distance <= DISTANCE_THRESHOLD:

                # i -> j
                edges.append([i, j])

                # j -> i
                edges.append([j, i])


    # --------------------------------------------------------
    # EDGE TENSOR
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # CREATE GRAPH
    # --------------------------------------------------------

    graph = Data(
        x=x,
        edge_index=edge_index
    )

    return graph


# ============================================================
# 10. GNN MODEL
# ============================================================

class TrafficGNN(torch.nn.Module):

    def __init__(self):

        super().__init__()


        # First GNN layer
        self.gcn1 = GCNConv(
            in_channels=3,
            out_channels=16
        )


        # Second GNN layer
        self.gcn2 = GCNConv(
            in_channels=16,
            out_channels=8
        )


        # Final classifier
        #
        # 8 features from GNN
        # 3 possible traffic classes:
        #
        # 0 = LOW
        # 1 = MEDIUM
        # 2 = HIGH

        self.classifier = torch.nn.Linear(
            8,
            3
        )


    def forward(
        self,
        x,
        edge_index,
        batch
    ):

        # ----------------------------------------------------
        # GCN Layer 1
        # ----------------------------------------------------

        x = self.gcn1(
            x,
            edge_index
        )

        x = torch.relu(x)


        # ----------------------------------------------------
        # GCN Layer 2
        # ----------------------------------------------------

        x = self.gcn2(
            x,
            edge_index
        )

        x = torch.relu(x)


        # ----------------------------------------------------
        # GRAPH POOLING
        # ----------------------------------------------------
        #
        # Converts:
        #
        # multiple vehicle nodes
        #
        # into:
        #
        # one representation of the whole traffic scene

        x = global_mean_pool(
            x,
            batch
        )


        # ----------------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------------

        x = self.classifier(x)


        return x


# ============================================================
# 11. CREATE GNN MODEL
# ============================================================

gnn_model = TrafficGNN()


# ============================================================
# 12. PROCESS VIDEO
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break


    frame_number += 1


    # ========================================================
    # YOLO + BoT-SORT
    # ========================================================

    results = model.track(
        frame,
        persist=True,
        tracker="botsort.yaml",
        verbose=False
    )


    # ========================================================
    # VEHICLE COUNTS
    # ========================================================

    total_vehicles = 0

    car_count = 0
    motorcycle_count = 0
    bus_count = 0
    truck_count = 0


    # Data for current frame
    current_frame_data = []


    # ========================================================
    # PROCESS DETECTIONS
    # ========================================================

    for result in results:

        if result.boxes is None:
            continue


        for box in result.boxes:

            # ------------------------------------------------
            # CLASS
            # ------------------------------------------------

            class_id = int(
                box.cls[0]
            )


            if class_id not in vehicle_classes:
                continue


            # ------------------------------------------------
            # CONFIDENCE
            # ------------------------------------------------

            confidence = float(
                box.conf[0]
            )


            # ------------------------------------------------
            # BOUNDING BOX
            # ------------------------------------------------

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            # ------------------------------------------------
            # CENTER
            # ------------------------------------------------

            center_x = int(
                (x1 + x2) / 2
            )

            center_y = int(
                (y1 + y2) / 2
            )


            # ------------------------------------------------
            # ROI
            # ------------------------------------------------

            if not (
                roi_x1 <= center_x <= roi_x2
                and
                roi_y1 <= center_y <= roi_y2
            ):
                continue


            # ------------------------------------------------
            # TRACK ID
            # ------------------------------------------------

            track_id = None

            if box.id is not None:

                track_id = int(
                    box.id[0]
                )


            # ------------------------------------------------
            # VEHICLE TYPE
            # ------------------------------------------------

            vehicle_type = vehicle_classes[
                class_id
            ]


            # ------------------------------------------------
            # COUNT
            # ------------------------------------------------

            total_vehicles += 1


            if vehicle_type == "Car":

                car_count += 1

            elif vehicle_type == "Motorcycle":

                motorcycle_count += 1

            elif vehicle_type == "Bus":

                bus_count += 1

            elif vehicle_type == "Truck":

                truck_count += 1


            # =================================================
            # TRAJECTORY
            # =================================================

            if track_id is not None:

                if track_id not in vehicle_tracks:

                    vehicle_tracks[
                        track_id
                    ] = []


                vehicle_tracks[
                    track_id
                ].append(
                    (center_x, center_y)
                )


                # Draw recent trajectory

                points = vehicle_tracks[
                    track_id
                ]

                recent_points = points[-30:]


                for i in range(
                    1,
                    len(recent_points)
                ):

                    cv2.line(
                        frame,
                        recent_points[i - 1],
                        recent_points[i],
                        (255, 0, 0),
                        2
                    )


            # =================================================
            # TRAFFIC DATA
            # =================================================

            vehicle_data = {

                "id": track_id,

                "type": vehicle_type,

                "x": center_x,

                "y": center_y,

                "frame": frame_number
            }


            current_frame_data.append(
                vehicle_data
            )

            traffic_data.append(
                vehicle_data
            )


            # =================================================
            # DRAW BOUNDING BOX
            # =================================================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # =================================================
            # LABEL
            # =================================================

            if track_id is not None:

                label = (
                    f"{vehicle_type} "
                    f"ID:{track_id} "
                    f"{confidence:.2f}"
                )

            else:

                label = (
                    f"{vehicle_type} "
                    f"{confidence:.2f}"
                )


            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )


    # ========================================================
    # TRAFFIC DENSITY
    # ========================================================

    density = (
        total_vehicles /
        MAX_VEHICLES
    ) * 100


    density = min(
        density,
        100
    )


    # ========================================================
    # TRAFFIC LEVEL
    # ========================================================

    if total_vehicles <= 5:

        traffic_level = "LOW"

        traffic_label = 0


    elif total_vehicles <= 12:

        traffic_level = "MEDIUM"

        traffic_label = 1


    else:

        traffic_level = "HIGH"

        traffic_label = 2


    # ========================================================
    # CREATE GRAPH FOR CURRENT FRAME
    # ========================================================

    graph = create_traffic_graph(
        current_frame_data
    )


    # ========================================================
    # STORE GRAPH FOR TRAINING
    # ========================================================

    if graph.x.shape[0] > 0:

        # Store traffic level as target label

        graph.y = torch.tensor(
            [traffic_label],
            dtype=torch.long
        )


        graph_dataset.append(
            graph
        )


    # ========================================================
    # DRAW ROI
    # ========================================================

    cv2.rectangle(
        frame,
        (roi_x1, roi_y1),
        (roi_x2, roi_y2),
        (255, 255, 0),
        2
    )


    # ========================================================
    # INFORMATION PANEL
    # ========================================================

    cv2.rectangle(
        frame,
        (10, 10),
        (390, 145),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        frame,
        f"VEHICLE COUNT: {total_vehicles}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Cars: {car_count}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Motorcycles: {motorcycle_count}",
        (20, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Buses: {bus_count}",
        (20, 104),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Trucks: {truck_count}",
        (20, 126),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # ========================================================
    # DENSITY / TRAFFIC LEVEL
    # ========================================================

    cv2.putText(
        frame,
        f"Density: {density:.1f}%",
        (420, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Traffic: {traffic_level}",
        (420, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Frame: {frame_number}",
        (420, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SHOW FRAME
    # ========================================================

    cv2.imshow(
        "Smart Traffic Detection + GNN Training",
        frame
    )


    # ========================================================
    # PRESS Q TO STOP
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# 13. RELEASE VIDEO
# ============================================================

cap.release()

cv2.destroyAllWindows()


# ============================================================
# 14. CHECK DATASET
# ============================================================

print("\n========================================")
print("VIDEO PROCESSING COMPLETE")
print("========================================")

print(
    f"Traffic data points: "
    f"{len(traffic_data)}"
)

print(
    f"Training graphs: "
    f"{len(graph_dataset)}"
)


# ============================================================
# 15. TRAIN GNN
# ============================================================

if len(graph_dataset) == 0:

    print("No graphs available for training.")
    exit()


print("\nStarting GNN training...")


# ------------------------------------------------------------
# DATA LOADER
# ------------------------------------------------------------

train_loader = DataLoader(
    graph_dataset,
    batch_size=16,
    shuffle=True
)


# ------------------------------------------------------------
# LOSS FUNCTION
# ------------------------------------------------------------

criterion = torch.nn.CrossEntropyLoss()


# ------------------------------------------------------------
# OPTIMIZER
# ------------------------------------------------------------

optimizer = torch.optim.Adam(
    gnn_model.parameters(),
    lr=0.01
)


# ============================================================
# TRAINING LOOP
# ============================================================

EPOCHS = 20


for epoch in range(
    EPOCHS
):

    gnn_model.train()

    total_loss = 0

    correct = 0

    total = 0


    for batch in train_loader:

        # ----------------------------------------------------
        # Reset gradients
        # ----------------------------------------------------

        optimizer.zero_grad()


        # ----------------------------------------------------
        # GNN prediction
        # ----------------------------------------------------

        output = gnn_model(
            batch.x,
            batch.edge_index,
            batch.batch
        )


        # ----------------------------------------------------
        # Target
        # ----------------------------------------------------

        target = batch.y


        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = criterion(
            output,
            target
        )


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # Update weights
        # ----------------------------------------------------

        optimizer.step()


        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        total_loss += loss.item()


        predictions = output.argmax(
            dim=1
        )


        correct += (
            predictions == target
        ).sum().item()


        total += target.size(0)


    accuracy = (
        correct / total
    ) * 100


    average_loss = (
        total_loss /
        len(train_loader)
    )


    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {average_loss:.4f} "
        f"Accuracy: {accuracy:.2f}%"
    )


# ============================================================
# 16. SAVE TRAINED GNN
# ============================================================

torch.save(
    gnn_model.state_dict(),
    "traffic_gnn.pth"
)


# ============================================================
# 17. FINISHED
# ============================================================

print("\n========================================")
print("GNN TRAINING COMPLETE")
print("========================================")

print(
    "Trained model saved as: traffic_gnn.pth"
)

print("========================================")