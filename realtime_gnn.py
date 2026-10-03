import cv2
import torch
from ultralytics import YOLO
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool


# SETTINGS


VIDEO_PATH = "yolo_traffic mp4 .mp4"
MODEL_PATH = "traffic_gnn.pth"

DISTANCE_THRESHOLD = 150

CLASS_NAMES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}



# GNN MODEL


class TrafficGNN(torch.nn.Module):

    def __init__(self):
        super().__init__()

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


# START


print("Starting Real-Time Traffic GNN...")


# Load YOLO
yolo_model = YOLO("yolo11n.pt")

print("YOLO loaded successfully.")


# Open video
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()

print("Video opened successfully.")


# Load trained GNN
gnn_model = TrafficGNN()

gnn_model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )
)

gnn_model.eval()

print("Trained GNN loaded successfully.")


# PROCESS VIDEO


while True:

    ret, frame = cap.read()

    if not ret:
        print("Video finished.")
        break

    height, width = frame.shape[:2]

    # ROI
    

    x1 = int(width * 0.10)
    y1 = int(height * 0.30)

    x2 = int(width * 0.90)
    y2 = int(height * 0.95)


    
    # YOLO + BOT-SORT
    

    results = yolo_model.track(
        frame,
        persist=True,
        tracker="botsort.yaml",
        verbose=False
    )

    result = results[0]

    # VEHICLE INFORMATION


    node_features = []
    positions = []

    if result.boxes is not None:

        for box in result.boxes:

            cls = int(box.cls[0])

            if cls not in CLASS_NAMES:
                continue

            # Bounding box
            bx1, by1, bx2, by2 = box.xyxy[0].tolist()

            center_x = (bx1 + bx2) / 2
            center_y = (by1 + by2) / 2


            # Check ROI
            if not (
                x1 <= center_x <= x2
                and y1 <= center_y <= y2
            ):
                continue


            # Vehicle type encoding
            if cls == 2:
                vehicle_type = 0

            elif cls == 3:
                vehicle_type = 1

            elif cls == 5:
                vehicle_type = 2

            else:
                vehicle_type = 3


            # Same type of features used during training
            node_features.append([
                vehicle_type,
                center_x,
                center_y
            ])

            positions.append(
                (center_x, center_y)
            )


    
    # CREATE GRAPH


    if len(node_features) > 0:

        x = torch.tensor(
            node_features,
            dtype=torch.float
        )

        edges = []

        for i in range(len(positions)):

            for j in range(i + 1, len(positions)):

                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]

                distance = (dx * dx + dy * dy) ** 0.5

                if distance <= DISTANCE_THRESHOLD:

                    edges.append([i, j])
                    edges.append([j, i])


        # If no edges exist
        if len(edges) == 0:

            edge_index = torch.empty(
                (2, 0),
                dtype=torch.long
            )

        else:

            edge_index = torch.tensor(
                edges,
                dtype=torch.long
            ).t().contiguous()


        # One graph
        batch = torch.zeros(
            x.size(0),
            dtype=torch.long
        )


        
        # GNN PREDICTION
        

        with torch.no_grad():

            output = gnn_model(
                x,
                edge_index,
                batch
            )

            prediction = torch.argmax(
                output,
                dim=1
            ).item()


        # Convert prediction to traffic level

        if prediction == 0:
            traffic_level = "LOW"

        elif prediction == 1:
            traffic_level = "MEDIUM"

        else:
            traffic_level = "HIGH"


    else:

        traffic_level = "LOW"


    # DRAW YOLO RESULTS


    frame = result.plot()


    
    # DRAW ROI
    

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        2
    )


    
    # DISPLAY GNN RESULT

    cv2.rectangle(
        frame,
        (10, 10),
        (400, 80),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"GNN TRAFFIC: {traffic_level}",
        (20, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )


    
    # SHOW VIDEO
    

    cv2.imshow(
        "Real-Time Traffic GNN",
        frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# CLEANUP

cap.release()

cv2.destroyAllWindows()

print("Real-Time GNN processing completed.")