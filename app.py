import tkinter as tk
from tkinter import messagebox
import cv2
import threading
from PIL import Image, ImageTk
from ultralytics import YOLO
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

# =========================================================
# TRAFFIC GNN MODEL
# =========================================================
class TrafficGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn1 = GCNConv(
            3,
            16
        )
        self.gcn2 = GCNConv(
            16,
            8
        )
        self.classifier = nn.Linear(
            8,
            3
        )
    def forward(
        self,
        x,
        edge_index,
        batch
    ):
        x = self.gcn1(
            x,
            edge_index
        )
        x = torch.relu(x)
        x = self.gcn2(
            x,
            edge_index
        )
        x = torch.relu(x)
        x = global_mean_pool(
            x,
            batch
        )
        x = self.classifier(x)
        return x

# =========================================================
# COLORS
# =========================================================
BG = "#0B1026"
CARD = "#171D3A"
INPUT = "#242B49"
WHITE = "#FFFFFF"
LIGHT = "#CBD5E1"
GRAY = "#94A3B8"
PURPLE = "#7C3AED"
PINK = "#EC4899"
BLUE = "#3B82F6"
RED = "#EF4444"

# =========================================================
# APPLICATION
# =========================================================
class SmartTrafficApp:
    def __init__(self, root):
        self.root = root

        # =================================================
        # YOLO DETECTION VARIABLES
        # =================================================
        self.video_path = "yolo_traffic mp4 .mp4"
        self.yolo_model = None
        self.video_capture = None
        self.detection_running = False
        self.detection_thread = None
        self.current_frame = None
        self.vehicle_count = 0
        self.vehicle_density = 0
        self.traffic_level = "LOW"
        self.video_label = None
        self.vehicle_value = None
        self.density_value = None
        self.traffic_value = None
        self.detection_value = None
        self.root.title("Smart Traffic AI")
        self.root.geometry("1100x700")
        self.root.minsize(900, 600)
        self.root.configure(bg=BG)
        self.center_window()
        self.show_login()

        # =================================================
        # LOAD GNN MODEL
        # =================================================

        self.gnn_model = TrafficGNN()
        self.gnn_model.load_state_dict(
            torch.load(
                "traffic_gnn.pth",
                map_location=torch.device("cpu")
            )
        )
        self.gnn_model.eval()

    # =====================================================
    # CENTER WINDOW
    # =====================================================
    def center_window(self):
        self.root.update_idletasks()
        width = 1100
        height = 700
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.root.geometry(
            f"{width}x{height}+{x}+{y}"
        )

    # =====================================================
    # CLEAR SCREEN
    # =====================================================
    def clear_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    # =====================================================
    # LOGIN PAGE
    # =====================================================
    def show_login(self):
        self.clear_screen()

        # Main background
        main = tk.Frame(
            self.root,
            bg=BG
        )
        main.pack(
            fill="both",
            expand=True
        )

        # =================================================
        # LEFT SIDE
        # =================================================
        left = tk.Frame(
            main,
            bg="#111735",
            width=500
        )
        left.pack(
            side="left",
            fill="both",
            expand=True
        )
        left.pack_propagate(False)

        # Traffic icon
        tk.Label(
            left,
            text="🚦",
            font=("Segoe UI Emoji", 75),
            bg="#111735",
            fg=WHITE
        ).pack(
            pady=(120, 10)
        )

        # Title
        tk.Label(
            left,
            text="SMART TRAFFIC",
            font=("Segoe UI", 32, "bold"),
            bg="#111735",
            fg=WHITE
        ).pack()
        tk.Label(
            left,
            text="AI",
            font=("Segoe UI", 34, "bold"),
            bg="#111735",
            fg="#A78BFA"
        ).pack()

        # Subtitle
        tk.Label(
            left,
            text="YOLOv11  •  GNN  •  Real-Time Detection",
            font=("Segoe UI", 11),
            bg="#111735",
            fg=LIGHT
        ).pack(
            pady=15
        )

        # Line
        tk.Frame(
            left,
            bg=PURPLE,
            width=180,
            height=4
        ).pack(
            pady=10
        )
        tk.Label(
            left,
            text="Intelligent traffic monitoring\n"
                 "and vehicle detection system",
            font=("Segoe UI", 11),
            bg="#111735",
            fg=GRAY,
            justify="center"
        ).pack(
            pady=15
        )

        # =================================================
        # RIGHT SIDE
        # =================================================
        right = tk.Frame(
            main,
            bg=BG
        )
        right.pack(
            side="right",
            fill="both",
            expand=True
        )

        # =================================================
        # LOGIN CARD
        # =================================================
        card = tk.Frame(
            right,
            bg=CARD,
            padx=45,
            pady=40
        )
        card.place(
            relx=0.5,
            rely=0.5,
            anchor="center",
            relwidth=0.76
        )

        # Heading
        tk.Label(
            card,
            text="Welcome Back",
            font=("Segoe UI", 25, "bold"),
            bg=CARD,
            fg=WHITE
        ).pack(
            pady=(0, 5)
        )
        tk.Label(
            card,
            text="Admin Login",
            font=("Segoe UI", 11),
            bg=CARD,
            fg=GRAY
        ).pack(
            pady=(0, 30)
        )

        # =================================================
        # USERNAME
        # =================================================
        tk.Label(
            card,
            text="👤  Username",
            font=("Segoe UI", 10, "bold"),
            bg=CARD,
            fg=LIGHT,
            anchor="w"
        ).pack(
            fill="x",
            pady=(0, 7)
        )
        self.username = tk.Entry(
            card,
            font=("Segoe UI", 12),
            bg=INPUT,
            fg=WHITE,
            insertbackground=WHITE,
            relief="flat"
        )
        self.username.pack(
            fill="x",
            ipady=10
        )

        # =================================================
        # PASSWORD
        # =================================================
        tk.Label(
            card,
            text="🔒  Password",
            font=("Segoe UI", 10, "bold"),
            bg=CARD,
            fg=LIGHT,
            anchor="w"
        ).pack(
            fill="x",
            pady=(20, 7)
        )

        #========================================================
        # Password container
        password_frame = tk.Frame(
            card,
            bg=INPUT
        )
        password_frame.pack(
            fill="x"
        )
        self.password = tk.Entry(
            password_frame,
            font=("Segoe UI", 12),
            bg=INPUT,
            fg=WHITE,
            insertbackground=WHITE,
            relief="flat",
            show="*",
            bd=0
        )
        self.password.pack(
            side="left",
            fill="x",
            expand=True,
            ipady=10,
            padx=(10, 0)
        )

        # Show / Hide password button
        self.show_password = tk.Button(
            password_frame,
            text="👁",
            font=("Segoe UI Emoji", 11),
            bg=INPUT,
            fg=LIGHT,
            activebackground=INPUT,
            activeforeground=WHITE,
            relief="flat",
            bd=0,
            cursor="hand2",
            command=self.toggle_password
        )
        self.show_password.pack(
            side="right",
            padx=10
        )

        # =================================================
        # LOGIN BUTTON
        # =================================================
        tk.Button(
            card,
            text="🚀  LOGIN",
            font=("Segoe UI", 11, "bold"),
            bg=PURPLE,
            fg=WHITE,
            activebackground=PINK,
            activeforeground=WHITE,
            relief="flat",
            cursor="hand2",
            command=self.login
        ).pack(
            fill="x",
            pady=(30, 15),
            ipady=10
        )

        # Security
        tk.Label(
            card,
            text="🔐 Authorized admin access only",
            font=("Segoe UI", 9),
            bg=CARD,
            fg=GRAY
        ).pack()

        # Enter key
        self.root.bind(
            "<Return>",
            lambda event: self.login()
        )
        self.username.focus()

    # =====================================================
    # LOGIN
    # =====================================================
    def login(self):
        username = self.username.get().strip()
        password = self.password.get()
        if username == "admin" and password == "admin123":
            self.root.unbind("<Return>")
            self.show_dashboard()
        elif username == "" or password == "":
            messagebox.showwarning(
                "Missing Information",
                "Please enter username and password."
            )
        else:
            messagebox.showerror(
                "Login Failed",
                "Invalid username or password."
            )
            self.password.delete(
                0,
                tk.END
            )
            self.password.focus()

    #===================================================
    def toggle_password(self):
        if self.password.cget("show") == "*":
            self.password.config(show="")
            self.show_password.config(text="🙈")
        else:
            self.password.config(show="*")
            self.show_password.config(text="👁")

    # =====================================================
    # DASHBOARD
    # =====================================================
    def show_dashboard(self): 

        self.clear_screen()

        # Background
        dashboard = tk.Frame(
            self.root,
            bg=BG
        )
        dashboard.pack(
            fill="both",
            expand=True
        )

        # =================================================
        # TOP BAR
        # =================================================
        top = tk.Frame(
            dashboard,
            bg=CARD,
            height=75
        )
        top.pack(
            fill="x"
        )
        top.pack_propagate(False)
        tk.Label(
            top,
            text="🚦  Smart Traffic AI",
            font=("Segoe UI", 22, "bold"),
            bg=CARD,
            fg=WHITE
        ).pack(
            side="left",
            padx=30
        )
        tk.Label(
            top,
            text="👤 Admin",
            font=("Segoe UI", 11, "bold"),
            bg=CARD,
            fg=LIGHT
        ).pack(
            side="right",
            padx=30
        )

        # =================================================
        # CONTENT
        # =================================================
        content = tk.Frame(
            dashboard,
            bg=BG
        )
        content.pack(
            fill="both",
            expand=True,
            padx=30,
            pady=30
        )
        tk.Label(
            content,
            text="Traffic Dashboard",
            font=("Segoe UI", 27, "bold"),
            bg=BG,
            fg=WHITE
        ).pack(
            anchor="w"
        )
        tk.Label(
            content,
            text="YOLOv11 + GNN Traffic Monitoring",
            font=("Segoe UI", 11),
            bg=BG,
            fg=GRAY
        ).pack(
            anchor="w",
            pady=(0, 25)
        )

        # =================================================
        # STAT CARDS
        # =================================================
        stats = tk.Frame(
            content,
            bg=BG
        )
        stats.pack(
            fill="x"
        )

        # Vehicle card
        vehicle_card, self.vehicle_value = self.create_dynamic_card(
            stats,
            "🚗",
            "Vehicles",
            "0",
            BLUE
        )
        vehicle_card.pack(
            side="left",
            fill="both",
            expand=True,
            padx=6
        )

        # Density card
        density_card, self.density_value = self.create_dynamic_card(
            stats,
            "📊",
            "Density",
            "0%",
            PURPLE
        )
        density_card.pack(
            side="left",
            fill="both",
            expand=True,
            padx=6
        )

        # Traffic state card
        traffic_card, self.traffic_value = self.create_dynamic_card(
            stats,
            "🧠",
            "Traffic State",
            "LOW",
            PINK
        )
        traffic_card.pack(
            side="left",
            fill="both",
            expand=True,
            padx=6
        )

        # Detection status card
        detection_card, self.detection_value = self.create_dynamic_card(
            stats,
            "🎥",
            "Detection",
            "Ready",
            "#10B981"
        )
        detection_card.pack(
            side="left",
            fill="both",
            expand=True,
            padx=6
        )

        # =================================================
        # VIDEO AREA
        # =================================================
        video = tk.Frame(
            content,
            bg=CARD
        )
        video.pack(
            fill="both",
            expand=True,
            pady=20
        )
        tk.Label(
            video,
            text="🎥  Traffic Detection",
            font=("Segoe UI", 17, "bold"),
            bg=CARD,
            fg=WHITE
        ).pack(
            anchor="w",
            padx=20,
            pady=(12, 5)
        )

        # Actual video display
        self.video_label = tk.Label(
            video,
            text="Press  ▶ Start Detection  to begin",
            font=("Segoe UI", 14),
            bg="#0F172A",
            fg=GRAY
        )
        self.video_label.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=10
        )
        # =================================================
        # DETECTION BUTTONS
        # =================================================
        button_frame = tk.Frame(
            content,
            bg=BG
        )
        button_frame.pack(
            fill="x",
            pady=(0, 15)
        )
        start_button = tk.Button(
            button_frame,
            text="▶  Start Detection",
            font=("Segoe UI", 10, "bold"),
            bg="#10B981",
            fg=WHITE,
            activebackground="#059669",
            activeforeground=WHITE,
            relief="flat",
            cursor="hand2",
            command=self.start_detection
        )
        start_button.pack(
            side="left",
            ipadx=15,
            ipady=8
        )
        stop_button = tk.Button(
            button_frame,
            text="⏹  Stop Detection",
            font=("Segoe UI", 10, "bold"),
            bg=RED,
            fg=WHITE,
            activebackground="#B91C1C",
            activeforeground=WHITE,
            relief="flat",
            cursor="hand2",
            command=self.stop_detection
        )
        stop_button.pack(
            side="left",
            padx=10,
            ipadx=15,
            ipady=8
        )

        # =================================================
        # LOGOUT
        # =================================================
        tk.Button(
            content,
            text="🔐  Logout",
            font=("Segoe UI", 10, "bold"),
            bg="#374151",
            fg=WHITE,
            activebackground=RED,
            activeforeground=WHITE,
            relief="flat",
            cursor="hand2",
            command=self.logout
        ).pack(
            anchor="e"
        )

    #====================================================
    # CREATE DYNAMIC STAT CARD
    # =====================================================
    def create_dynamic_card(
        self,
        parent,
        icon,
        title,
        value,
        icon_color
    ):
        card = tk.Frame(
            parent,
            bg=CARD,
            height=125
        )
        card.pack_propagate(False)
        tk.Label(
            card,
            text=icon,
            font=("Segoe UI Emoji", 26),
            bg=CARD,
            fg=icon_color
        ).pack(
            anchor="w",
            padx=18,
            pady=(12, 0)
        )
        tk.Label(
            card,
            text=title,
            font=("Segoe UI", 9),
            bg=CARD,
            fg=GRAY
        ).pack(
            anchor="w",
            padx=18
        )
        value_label = tk.Label(
            card,
            text=value,
            font=("Segoe UI", 18, "bold"),
            bg=CARD,
            fg=WHITE
        )
        value_label.pack(
            anchor="w",
            padx=18
        )
        return card, value_label

    # =====================================================
    # START YOLO DETECTION
    # =====================================================
    def start_detection(self):
        if self.detection_running:
            return
        self.detection_running = True
        self.detection_value.config(
            text="Loading..."
        )
        self.detection_thread = threading.Thread(
            target=self.run_detection,
            daemon=True
        )
        self.detection_thread.start()

        # Start checking for frames
        self.root.after(
            30,
            self.update_video
        )

    # =====================================================
    # YOLO DETECTION THREAD
    # =====================================================
    def run_detection(self):
        try:

            # Load YOLO
            self.yolo_model = YOLO(
                "yolo11n.pt"
            )

            # Open video
            self.video_capture = cv2.VideoCapture(
                self.video_path
            )
            if not self.video_capture.isOpened():
                self.root.after(
                    0,
                    lambda: messagebox.showerror(
                        "Video Error",
                        f"Could not open video:\n{self.video_path}"
                    )
                )
                self.detection_running = False
                return
            self.root.after(
                0,
                lambda: self.detection_value.config(
                    text="LIVE"
                )
            )
            while self.detection_running:
                success, frame = self.video_capture.read()

                # Video ended
                if not success:
                    self.video_capture.set(
                        cv2.CAP_PROP_POS_FRAMES,
                        0
                    )
                    continue

                # -----------------------------------------
                # FRAME SIZE
                # -----------------------------------------
                height, width = frame.shape[:2]

                # -----------------------------------------
                # ROI
                # -----------------------------------------
                x1 = int(width * 0.10)
                y1 = int(height * 0.30)
                x2 = int(width * 0.90)
                y2 = int(height * 0.95)

                # -----------------------------------------
                # YOLO TRACKING
                # -----------------------------------------
                results = self.yolo_model.track(
                    frame,
                    persist=True,
                    tracker="botsort.yaml",
                    verbose=False
                )
                vehicle_count = 0
                car_count = 0
                motorcycle_count = 0
                bus_count = 0
                truck_count = 0
                frame_data = []

                # -----------------------------------------
                # VEHICLE CLASSES
                # -----------------------------------------
                vehicle_classes = {
                    2: "Car",
                    3: "Motorcycle",
                    5: "Bus",
                    7: "Truck"
                }
                type_encoding = {
                    "Car": 0,
                    "Motorcycle": 1,
                    "Bus": 2,
                    "Truck": 3
                }

                # -----------------------------------------
                # PROCESS DETECTIONS
                # -----------------------------------------
                if results and results[0].boxes:
                    boxes = results[0].boxes
                    for i in range(len(boxes)):
                        cls = int(
                            boxes.cls[i].item()
                        )
                        if cls not in vehicle_classes:
                            continue
                        confidence = float(
                            boxes.conf[i].item()

                        )

                        # Bounding box
                        x_min, y_min, x_max, y_max = map(
                            int,
                            boxes.xyxy[i].tolist()
                        )

                        # Center
                        center_x = int(
                            (x_min + x_max) / 2
                        )
                        center_y = int(
                            (y_min + y_max) / 2
                        )

                        # Check ROI
                        inside_roi = (
                            x1 <= center_x <= x2
                            and
                            y1 <= center_y <= y2
                        )
                        if not inside_roi:
                            continue
                        vehicle_name = vehicle_classes[cls]
                        frame_data.append({
                            "type": vehicle_name,
                            "x": center_x,
                            "y": center_y
                        }) 
                        vehicle_count += 1
                        if vehicle_name == "Car":
                            car_count += 1
                        elif vehicle_name == "Motorcycle":
                            motorcycle_count += 1
                        elif vehicle_name == "Bus":
                            bus_count += 1
                        elif vehicle_name == "Truck":
                            truck_count += 1

                        # ---------------------------------
                        # DRAW BOX
                        # ---------------------------------
                        cv2.rectangle(
                            frame,
                            (x_min, y_min),
                            (x_max, y_max),
                            (99, 102, 241),
                            2
                        )
                        label = (
                            f"{vehicle_name} "
                            f"{confidence:.2f}"
                        )
                        cv2.putText(
                            frame,
                            label,
                            (x_min, max(y_min - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 255, 255),
                            2
                        )

                # -----------------------------------------
                # TRAFFIC DENSITY
                # -----------------------------------------
                density = min(
                    (vehicle_count / 20) * 100,
                    100
                )

                # -----------------------------------------
                # TRAFFIC LEVEL
                # -----------------------------------------
                if vehicle_count <= 5:
                    traffic_level = "LOW"
                elif vehicle_count <= 12:
                    traffic_level = "MEDIUM"
                else:
                    traffic_level = "HIGH"
                # -----------------------------------------
                # DRAW ROI
                # -----------------------------------------
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 200, 0),
                    2
                )
                # -----------------------------------------
                # DRAW INFORMATION
                # ----------------------------------------
                cv2.rectangle(
                    frame,
                    (10, 10),
                    (350, 125),
                    (15, 23, 42),
                    -1
                )
                cv2.putText(
                    frame,
                    f"Vehicles: {vehicle_count}",
                    (25, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                cv2.putText(
                    frame,
                    f"Density: {density:.1f}%",
                    (25, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                cv2.putText(
                    frame,
                    f"Traffic: {traffic_level}",
                    (25, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                # -----------------------------------------
                # SAVE CURRENT DATA
                # -----------------------------------------
                self.vehicle_count = vehicle_count
                self.vehicle_density = density
                self.traffic_level = traffic_level
                self.current_frame = frame
        except Exception as e:
            print(
                "YOLO ERROR:",
                e
            )
            self.detection_running = False
            self.root.after(
                0,
                lambda: messagebox.showerror(
                    "Detection Error",
                    str(e)
                )
            )
        finally:
            if self.video_capture is not None:
                self.video_capture.release()

    # =====================================================
    # UPDATE TKINTER VIDEO
    # =====================================================
    def update_video(self):
        if not self.detection_running:
            return
        if self.current_frame is not None:
            frame = self.current_frame.copy()

            # OpenCV BGR → RGB

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Resize to fit GUI
            frame = cv2.resize(
                frame,
                (760, 430)
            )
            image = Image.fromarray(
                frame
            )
            photo = ImageTk.PhotoImage(
                image=image
            )
            self.video_label.config(
                image=photo,
                text=""
            )

            # Keep image reference
            self.video_label.image = photo

            # Update dashboard cards
            self.vehicle_value.config(
                text=str(
                    self.vehicle_count
                )
            )
            self.density_value.config(
                text=f"{self.vehicle_density:.0f}%"
            )
            self.traffic_value.config(
                text=self.traffic_level
            )
            self.detection_value.config(
                text="LIVE"
            )
        self.root.after(
            30,
            self.update_video
        )
    # =====================================================
    # STOP YOLO DETECTION
    # =====================================================
    def stop_detection(self):
        self.detection_running = False
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        if self.video_label is not None:
            self.video_label.config(
                image="",
                text="Press  ▶ Start Detection  to begin"
            )
        if self.video_label is not None:
            self.video_label.image = None
        if self.vehicle_value is not None:
            self.vehicle_value.config(
                text="0"
            )
        if self.density_value is not None:
            self.density_value.config(
                text="0%"
            )
        if self.traffic_value is not None:
            self.traffic_value.config(
                text="LOW"
            )
        if self.detection_value is not None:
            self.detection_value.config(
                text="Ready"
            )
    # =====================================================
    # STAT CARD
    # =====================================================
    def stat_card(
        self,
        parent,
        icon,
        title,
        value,
        icon_color
    ):
        card = tk.Frame(
            parent,
            bg=CARD,
            height=125
        )
        card.pack_propagate(False)
        tk.Label(
            card,
            text=icon,
            font=("Segoe UI Emoji", 26),
            bg=CARD,
            fg=icon_color
        ).pack(
            anchor="w",
            padx=18,
            pady=(12, 0)
        )
        tk.Label(
            card,
            text=title,
            font=("Segoe UI", 9),
            bg=CARD,
            fg=GRAY
        ).pack(
            anchor="w",
            padx=18
        )
        tk.Label(
            card,
            text=value,
            font=("Segoe UI", 18, "bold"),
            bg=CARD,
            fg=WHITE
        ).pack(
            anchor="w",
            padx=18
        )
        return card
    # =====================================================
    # LOGOUT
    # =====================================================
    def logout(self):
        if self.detection_running:
            self.stop_detection()
        answer = messagebox.askyesno(
            "Logout",
            "Are you sure you want to logout?"
        )
        if answer:
            self.show_login()   

# =========================================================
# START
# =========================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = SmartTrafficApp(root)
    root.mainloop()