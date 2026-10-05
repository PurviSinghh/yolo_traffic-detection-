"""
sumo_bridge.py
==============
SUMO/TraCI integration placeholder.

The sumo/ directory exists in the project but contains only empty
sub-directories (networks/, routes/, simulations/).

This module provides a clean stub API so the rest of the backend
compiles and the GUI shows the correct "NOT IMPLEMENTED YET" status.
When SUMO network files are added, only this file needs to be updated.
"""

import threading
import time


class SumoBridge:
    """
    Stub SUMO bridge.
    Status: NOT IMPLEMENTED YET — SUMO network files are missing.
    """

    IMPLEMENTATION_STATUS = "NOT_IMPLEMENTED"
    REASON = (
        "sumo/networks/, sumo/routes/ and sumo/simulations/ directories "
        "exist but are empty. Add SUMO .net.xml, .rou.xml, and .sumocfg "
        "files and update this class to use TraCI."
    )

    def __init__(self):
        self._running = False
        self._state = {
            "status":          "NOT_IMPLEMENTED",
            "connected":       False,
            "sim_time":        0.0,
            "vehicle_count":   0,
            "traffic_flow":    0.0,
            "avg_wait_time":   0.0,
            "queue_length":    0,
            "intersections":   [],
        }

    def start(self) -> dict:
        return {
            "status":  "NOT_IMPLEMENTED",
            "message": self.REASON,
        }

    def stop(self) -> dict:
        return {"status": "NOT_IMPLEMENTED"}

    def reset(self) -> dict:
        return {"status": "NOT_IMPLEMENTED"}

    def get_state(self) -> dict:
        return dict(self._state)

    def is_running(self) -> bool:
        return False
