"""
detection_engine.py
====================
Wraps existing logic from:
  - realtime_gnn.py  (YOLO + BoT-SORT + GNN inference)
  - main.py          (create_traffic_graph, TrafficGNN, vehicle counting)

No algorithm is changed. This module provides a clean OOP interface
so the FastAPI server can call it without touching the original files.
"""

import cv2
import torch
import base64
import time
import threading
from pathlib import Path

from ultralytics import YOLO
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

# ─────────────────────────────────────────────────────────────
# PATHS  (resolve relative to project root)
# ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
VIDEO_PATH   = str(ROOT / "yolo_traffic mp4 .mp4")
YOLO_WEIGHTS = str(ROOT / "yolo11n.pt")
GNN_WEIGHTS  = str(ROOT / "traffic_gnn.pth")

# ─────────────────────────────────────────────────────────────
# CONSTANTS  (unchanged from original code)
# ─────────────────────────────────────────────────────────────
VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
TYPE_ENCODING   = {"Car": 0, "Motorcycle": 1, "Bus": 2, "Truck": 3}
DISTANCE_THRESHOLD = 150
MAX_VEHICLES = 20
TRAFFIC_LABELS = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}


# ─────────────────────────────────────────────────────────────
# GNN MODEL  (identical to main.py / realtime_gnn.py)
# ─────────────────────────────────────────────────────────────
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
        return self.classifier(x)


# ─────────────────────────────────────────────────────────────
# GRAPH BUILDER  (identical to create_traffic_graph in main.py)
# ─────────────────────────────────────────────────────────────
def create_traffic_graph(frame_data: list) -> Data:
    """Build a PyTorch Geometric graph from per-frame vehicle data."""
    if not frame_data:
        return Data(
            x=torch.empty((0, 3), dtype=torch.float),
            edge_index=torch.empty((2, 0), dtype=torch.long),
        )

    node_features, positions = [], []
    for v in frame_data:
        node_features.append([TYPE_ENCODING[v["type"]], v["x"], v["y"]])
        positions.append((v["x"], v["y"]))

    x = torch.tensor(node_features, dtype=torch.float)
    edges = []
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            if (dx * dx + dy * dy) ** 0.5 <= DISTANCE_THRESHOLD:
                edges += [[i, j], [j, i]]

    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    return Data(x=x, edge_index=edge_index)


# ─────────────────────────────────────────────────────────────
# DETECTION ENGINE
# ─────────────────────────────────────────────────────────────
class DetectionEngine:
    """
    Runs YOLO+BoT-SORT+GNN on a background thread and exposes
    the latest snapshot through `get_state()`.
    """

    def __init__(self):
        self._running  = False
        self._thread   = None
        self._lock     = threading.Lock()
        self.yolo_model = None
        self.gnn_model  = None

        # Shared state (updated by background thread, read by main thread)
        self._state = {
            "detection_active": False,
            "vehicle_count":    0,
            "car_count":        0,
            "motorcycle_count": 0,
            "bus_count":        0,
            "truck_count":      0,
            "density":          0.0,
            "traffic_level":    "LOW",
            "gnn_prediction":   "LOW",
            "frame_b64":        None,   # JPEG → base64 string for WebSocket
            "frame_number":     0,
            "detections":       [],     # [{id, type, x, y, conf}]
            "timestamp":        time.time(),
        }

        # History buffers for charts (up to 120 data points ~2 min at 1fps)
        self.history = {
            "timestamps":    [],
            "vehicle_counts":[],
            "densities":     [],
            "traffic_levels":[],
        }

    # ── Public API ───────────────────────────────────────────

    def start(self):
        if self._running:
            return {"status": "already_running"}
        self._running = True
        self._thread  = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        return {"status": "started"}

    def stop(self):
        self._running = False
        with self._lock:
            self._state["detection_active"] = False
        return {"status": "stopped"}

    def get_state(self) -> dict:
        with self._lock:
            return dict(self._state)

    def get_history(self) -> dict:
        return dict(self.history)

    def is_running(self) -> bool:
        return self._running

    # ── Internal loop ────────────────────────────────────────

    def _load_models(self):
        """Load YOLO and GNN models (called once at thread start)."""
        self.yolo_model = YOLO(YOLO_WEIGHTS)

        self.gnn_model = TrafficGNN()
        self.gnn_model.load_state_dict(
            torch.load(GNN_WEIGHTS, map_location="cpu", weights_only=True)
        )
        self.gnn_model.eval()

    def _run_loop(self):
        """
        Main detection loop — mirrors the logic of realtime_gnn.py but
        stores results in shared state instead of showing an OpenCV window.
        """
        try:
            self._load_models()
        except Exception as e:
            print(f"[DetectionEngine] Model load error: {e}")
            self._running = False
            return

        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print(f"[DetectionEngine] Cannot open video: {VIDEO_PATH}")
            self._running = False
            return

        frame_number = 0
        last_chart_update = 0

        while self._running:
            ret, frame = cap.read()
            if not ret:
                # Loop video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_number += 1
            height, width = frame.shape[:2]

            # ROI (unchanged from original)
            roi_x1 = int(width * 0.10)
            roi_y1 = int(height * 0.30)
            roi_x2 = int(width * 0.90)
            roi_y2 = int(height * 0.95)

            # YOLO + BoT-SORT tracking (unchanged)
            results = self.yolo_model.track(
                frame, persist=True, tracker="botsort.yaml", verbose=False
            )

            # Count & collect frame data
            vehicle_count = car_count = motorcycle_count = bus_count = truck_count = 0
            frame_data = []
            detections = []

            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    cls = int(boxes.cls[i].item())
                    if cls not in VEHICLE_CLASSES:
                        continue
                    conf = float(boxes.conf[i].item())
                    x1v, y1v, x2v, y2v = map(int, boxes.xyxy[i].tolist())
                    cx = (x1v + x2v) // 2
                    cy = (y1v + y2v) // 2

                    # ROI check
                    if not (roi_x1 <= cx <= roi_x2 and roi_y1 <= cy <= roi_y2):
                        continue

                    vtype = VEHICLE_CLASSES[cls]
                    track_id = int(boxes.id[i].item()) if boxes.id is not None else None

                    vehicle_count += 1
                    if vtype == "Car":        car_count += 1
                    elif vtype == "Motorcycle": motorcycle_count += 1
                    elif vtype == "Bus":       bus_count += 1
                    elif vtype == "Truck":     truck_count += 1

                    frame_data.append({"type": vtype, "x": cx, "y": cy})
                    detections.append({
                        "id":   track_id,
                        "type": vtype,
                        "x":    cx, "y": cy,
                        "conf": round(conf, 2),
                        "bbox": [x1v, y1v, x2v, y2v],
                    })

                    # Draw bounding box on frame
                    cv2.rectangle(frame, (x1v, y1v), (x2v, y2v), (99, 102, 241), 2)
                    lbl = f"{vtype} {'ID:'+str(track_id)+' ' if track_id else ''}{conf:.2f}"
                    cv2.putText(frame, lbl, (x1v, max(y1v - 10, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Density & traffic level (unchanged from original)
            density = min((vehicle_count / MAX_VEHICLES) * 100, 100.0)
            if vehicle_count <= 5:
                traffic_level, traffic_label = "LOW", 0
            elif vehicle_count <= 12:
                traffic_level, traffic_label = "MEDIUM", 1
            else:
                traffic_level, traffic_label = "HIGH", 2

            # GNN inference (unchanged from realtime_gnn.py)
            gnn_prediction = traffic_level
            if frame_data:
                graph = create_traffic_graph(frame_data)
                if graph.x.shape[0] > 0:
                    batch = torch.zeros(graph.x.size(0), dtype=torch.long)
                    with torch.no_grad():
                        out = self.gnn_model(graph.x, graph.edge_index, batch)
                        pred_idx = torch.argmax(out, dim=1).item()
                    gnn_prediction = TRAFFIC_LABELS[pred_idx]

            # Draw ROI rectangle
            cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 200, 0), 2)

            # Overlay info panel on frame
            cv2.rectangle(frame, (10, 10), (330, 120), (15, 23, 42), -1)
            for txt, y_off in [
                (f"Vehicles: {vehicle_count}", 35),
                (f"Density:  {density:.1f}%", 60),
                (f"Traffic:  {traffic_level}", 85),
                (f"GNN:      {gnn_prediction}", 110),
            ]:
                cv2.putText(frame, txt, (20, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Encode frame to JPEG → base64
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_b64 = base64.b64encode(buf).decode("utf-8")

            # Update shared state
            now = time.time()
            with self._lock:
                self._state.update({
                    "detection_active": True,
                    "vehicle_count":    vehicle_count,
                    "car_count":        car_count,
                    "motorcycle_count": motorcycle_count,
                    "bus_count":        bus_count,
                    "truck_count":      truck_count,
                    "density":          round(density, 1),
                    "traffic_level":    traffic_level,
                    "gnn_prediction":   gnn_prediction,
                    "frame_b64":        frame_b64,
                    "frame_number":     frame_number,
                    "detections":       detections,
                    "timestamp":        now,
                })

            # Update history at ~1 Hz for charts
            if now - last_chart_update >= 1.0:
                last_chart_update = now
                ts = time.strftime("%H:%M:%S", time.localtime(now))
                self.history["timestamps"].append(ts)
                self.history["vehicle_counts"].append(vehicle_count)
                self.history["densities"].append(round(density, 1))
                self.history["traffic_levels"].append(traffic_level)
                # Cap at 120 points
                for k in self.history:
                    if len(self.history[k]) > 120:
                        self.history[k] = self.history[k][-120:]

        cap.release()
        with self._lock:
            self._state["detection_active"] = False
        print("[DetectionEngine] Detection loop stopped.")
