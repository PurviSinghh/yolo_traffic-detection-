"""
marl_agent.py
=============
Federated Multi-Agent Reinforcement Learning placeholder.

Status: NOT IMPLEMENTED YET

This module provides a clean stub API so the GUI can display the
correct "NOT IMPLEMENTED YET" panels without crashing. When the RL
code is developed, update this file only — the server and frontend
already have the necessary hooks.
"""


class MARLAgent:
    """
    Stub for Federated MARL.
    Status: NOT IMPLEMENTED YET
    """

    IMPLEMENTATION_STATUS = "NOT_IMPLEMENTED"
    REASON = (
        "Federated Multi-Agent Reinforcement Learning has not been "
        "implemented yet. The GUI section is ready to display data "
        "once this module is built."
    )

    def __init__(self):
        self._state = {
            "status":                "NOT_IMPLEMENTED",
            "num_agents":            0,
            "num_intersections":     0,
            "agent_status":          [],
            "training_active":       False,
            "aggregation_active":    False,
            "global_model_version":  0,
            "latest_rl_action":      None,
            "training_episode":      0,
            "training_round":        0,
            "current_reward":        None,
            "local_training_status": "IDLE",
            "global_agg_status":     "IDLE",
        }

        self._intersection_states = [
            {
                "id":              i + 1,
                "signal":          "RED",
                "remaining_time":  0,
                "density":         0.0,
                "queue_length":    0,
                "rl_action":       None,
                "next_phase":      None,
            }
            for i in range(3)
        ]

    def get_state(self) -> dict:
        return dict(self._state)

    def get_intersection_states(self) -> list:
        return list(self._intersection_states)

    def step(self) -> dict:
        return {"status": "NOT_IMPLEMENTED"}

    def is_running(self) -> bool:
        return False


class EmergencyVehicleDetector:
    """
    Stub for emergency vehicle detection.
    Status: NOT IMPLEMENTED YET
    """

    IMPLEMENTATION_STATUS = "NOT_IMPLEMENTED"
    REASON = (
        "Emergency vehicle detection has not been implemented yet. "
        "The GUI section and API endpoint are ready to display data "
        "once this module is added."
    )

    def __init__(self):
        self._state = {
            "status":               "NOT_IMPLEMENTED",
            "detected":             False,
            "vehicle_type":         None,
            "location":             None,
            "lane":                 None,
            "detection_time":       None,
            "priority_status":      None,
            "affected_intersection":None,
            "signal_priority":      False,
        }

    def get_state(self) -> dict:
        return dict(self._state)

    def process_frame(self, frame_data: list) -> dict:
        return self._state
