"""
server.py
=========
FastAPI backend for the Smart Traffic Management Dashboard.

Architecture:
  - HTTP REST endpoints for control (start/stop detection, SUMO, etc.)
  - WebSocket /ws/live  → pushes real-time state every ~100 ms
  - Static file serving for the frontend (../frontend/)

Existing modules used:
  backend/detection_engine.py  → wraps realtime_gnn.py / main.py logic
  backend/sumo_bridge.py       → SUMO stub (NOT IMPLEMENTED YET)
  backend/marl_agent.py        → MARL + emergency vehicle stubs

Run with:
  cd backend
  uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import json
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from detection_engine import DetectionEngine
from sumo_bridge import SumoBridge
from marl_agent import MARLAgent, EmergencyVehicleDetector

# ─────────────────────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Smart Traffic Management API",
    description="Real-time traffic monitoring using YOLO + GNN + SUMO + MARL",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# MODULE INSTANCES
# ─────────────────────────────────────────────────────────────
detection_engine = DetectionEngine()
sumo_bridge      = SumoBridge()
marl_agent       = MARLAgent()
emergency_det    = EmergencyVehicleDetector()

# ─────────────────────────────────────────────────────────────
# SERVE FRONTEND
# ─────────────────────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")

@app.get("/")
async def serve_dashboard():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "Frontend not found. Place frontend files in ../frontend/"}


# ─────────────────────────────────────────────────────────────
# WEBSOCKET — real-time broadcast
# ─────────────────────────────────────────────────────────────
connected_clients: list[WebSocket] = []

@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    await ws.accept()
    connected_clients.append(ws)
    try:
        while True:
            # Build combined state snapshot
            det   = detection_engine.get_state()
            sumo  = sumo_bridge.get_state()
            marl  = marl_agent.get_state()
            inter = marl_agent.get_intersection_states()
            emerg = emergency_det.get_state()
            hist  = detection_engine.get_history()

            payload = {
                "type":         "state_update",
                "timestamp":    time.time(),
                "detection":    det,
                "sumo":         sumo,
                "marl":         marl,
                "intersections":inter,
                "emergency":    emerg,
                "history":      hist,
                "system": {
                    "detection_running": detection_engine.is_running(),
                    "sumo_running":      sumo_bridge.is_running(),
                    "marl_running":      marl_agent.is_running(),
                    "server_time":       time.strftime("%H:%M:%S"),
                },
            }

            # Strip the heavy base64 frame from history (sent separately below)
            frame_b64 = det.pop("frame_b64", None)
            payload["detection"]["frame_b64"] = frame_b64  # re-attach

            await ws.send_text(json.dumps(payload))
            await asyncio.sleep(0.1)   # ~10 fps update rate

    except WebSocketDisconnect:
        connected_clients.remove(ws)
    except Exception as e:
        print(f"[WS] Error: {e}")
        if ws in connected_clients:
            connected_clients.remove(ws)


# ─────────────────────────────────────────────────────────────
# REST ENDPOINTS
# ─────────────────────────────────────────────────────────────

# ── Detection ────────────────────────────────────────────────

@app.post("/api/detection/start")
async def start_detection():
    result = detection_engine.start()
    return result

@app.post("/api/detection/stop")
async def stop_detection():
    result = detection_engine.stop()
    return result

@app.get("/api/detection/state")
async def get_detection_state():
    state = detection_engine.get_state()
    state.pop("frame_b64", None)   # don't send large binary over REST
    return state

@app.get("/api/detection/history")
async def get_detection_history():
    return detection_engine.get_history()

# ── SUMO ─────────────────────────────────────────────────────

@app.post("/api/sumo/start")
async def start_sumo():
    return sumo_bridge.start()

@app.post("/api/sumo/stop")
async def stop_sumo():
    return sumo_bridge.stop()

@app.post("/api/sumo/reset")
async def reset_sumo():
    return sumo_bridge.reset()

@app.get("/api/sumo/state")
async def get_sumo_state():
    return sumo_bridge.get_state()

# ── MARL ─────────────────────────────────────────────────────

@app.get("/api/marl/state")
async def get_marl_state():
    return {
        "marl":         marl_agent.get_state(),
        "intersections":marl_agent.get_intersection_states(),
    }

# ── Emergency ────────────────────────────────────────────────

@app.get("/api/emergency/state")
async def get_emergency_state():
    return emergency_det.get_state()

# ── System ───────────────────────────────────────────────────

@app.get("/api/system/status")
async def system_status():
    return {
        "server":             "ONLINE",
        "server_time":        time.strftime("%Y-%m-%d %H:%M:%S"),
        "detection_running":  detection_engine.is_running(),
        "sumo_running":       sumo_bridge.is_running(),
        "marl_running":       marl_agent.is_running(),
        "connected_clients":  len(connected_clients),
        "module_status": {
            "yolo_botsort_gnn": "IMPLEMENTED",
            "sumo":             SumoBridge.IMPLEMENTATION_STATUS,
            "marl":             MARLAgent.IMPLEMENTATION_STATUS,
            "emergency":        EmergencyVehicleDetector.IMPLEMENTATION_STATUS,
        },
    }

@app.get("/api/system/logs")
async def get_logs():
    """Returns last 50 log events (in-memory placeholder)."""
    return {
        "logs": [
            {
                "timestamp": time.strftime("%H:%M:%S"),
                "level":     "INFO",
                "message":   "Log persistence NOT IMPLEMENTED YET. "
                             "Connect a logging handler to capture events.",
            }
        ]
    }
