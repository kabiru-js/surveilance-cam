import time
import uuid
from typing import Dict, Any, List, Optional

class ComplexEventProcessor:
    """
    Stateful Complex Event Processing (CEP) Engine for real-time security automation.
    Maintains in-memory sliding session states per global_entity_id, tracks zone dwell times,
    evaluates real-time security rules, and enforces alert deduplication cooldown windows.
    """
    LOITERING_ZONES = {"CAM_LOBBY", "Restricted_Zone_A"}
    PARKING_ZONE = "Dropoff_Zone"
    VEHICLE_LABELS = {"car", "bus", "truck", "motorcycle"}

    COOLDOWN_WINDOW_SECONDS = 60.0
    LOITERING_THRESHOLD_SECONDS = 120.0
    PARKING_THRESHOLD_SECONDS = 300.0

    def __init__(self, cooldown_seconds: float = 60.0):
        self.cooldown_seconds = cooldown_seconds
        # Session state: { global_entity_id: SessionStateDict }
        self.sessions: Dict[str, Dict[str, Any]] = {}
        # Cooldown state: { (global_entity_id, rule_name): last_fired_timestamp }
        self.cooldowns: Dict[tuple, float] = {}

    def _get_or_create_session(self, entity_id: str, timestamp: float) -> Dict[str, Any]:
        if entity_id not in self.sessions:
            self.sessions[entity_id] = {
                "global_entity_id": entity_id,
                "current_zone": None,
                "zone_entry_time": None,
                "last_seen_time": timestamp,
                "camera_id": None,
                "label": "unknown",
                "color": "unknown",
                "plate": None
            }
        return self.sessions[entity_id]

    def _can_fire_alert(self, entity_id: str, rule_name: str, timestamp: float) -> bool:
        """
        Deduplication gatekeeper: returns True if rule has never fired for entity,
        or if cooldown_seconds has elapsed since the last firing.
        """
        key = (entity_id, rule_name)
        last_fired = self.cooldowns.get(key)
        if last_fired is None:
            return True
        return (timestamp - last_fired) >= self.cooldown_seconds

    def _mark_alert_fired(self, entity_id: str, rule_name: str, timestamp: float):
        key = (entity_id, rule_name)
        self.cooldowns[key] = timestamp

    def process_event(self, event: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Evaluates an observation event against stateful security rules.
        Returns a list of triggered alert dictionaries (empty if no rules fire or alerts are suppressed).
        """
        entity_id = event.get("global_entity_id")
        if not entity_id:
            return []

        timestamp = float(event.get("timestamp", time.time()))
        camera_id = event.get("camera_id", "UNKNOWN_CAM")
        track_id = event.get("track_id", -1)
        label = event.get("label", "unknown").lower()
        color = event.get("color", "unknown")
        plate = event.get("plate")
        zone_id = event.get("zone_id")
        event_type = event.get("event_type", "OBSERVATION").upper()

        session = self._get_or_create_session(entity_id, timestamp)
        session["last_seen_time"] = timestamp
        session["camera_id"] = camera_id
        session["label"] = label
        if color:
            session["color"] = color
        if plate:
            session["plate"] = plate

        # Update Zone State
        # If camera_id is CAM_LOBBY and zone_id is None, treat CAM_LOBBY as the monitored zone
        effective_zone = zone_id or (camera_id if camera_id in self.LOITERING_ZONES else None)

        if event_type == "ZONE_EXIT":
            session["current_zone"] = None
            session["zone_entry_time"] = None
        elif effective_zone is not None:
            if session["current_zone"] != effective_zone:
                # Entered a new zone
                session["current_zone"] = effective_zone
                session["zone_entry_time"] = timestamp
            # If already in effective_zone, retain existing zone_entry_time
        else:
            # Sighting outside of any zone
            session["current_zone"] = None
            session["zone_entry_time"] = None

        triggered_alerts: List[Dict[str, Any]] = []

        # =========================================================================
        # Rule 1: RESTRICTED_ZONE_BREACH (Immediate Trigger, HIGH Severity)
        # Person enters Restricted_Zone_A
        # =========================================================================
        if label == "person" and session["current_zone"] == "Restricted_Zone_A":
            rule_name = "RESTRICTED_ZONE_BREACH"
            if self._can_fire_alert(entity_id, rule_name, timestamp):
                self._mark_alert_fired(entity_id, rule_name, timestamp)
                triggered_alerts.append({
                    "alert_id": f"alt_{uuid.uuid4().hex[:12]}",
                    "timestamp": timestamp,
                    "rule_name": rule_name,
                    "severity": "HIGH",
                    "global_entity_id": entity_id,
                    "camera_id": camera_id,
                    "zone_id": session["current_zone"],
                    "message": f"CRITICAL: Unauthorized person {entity_id} breached {session['current_zone']} on {camera_id}",
                    "metadata": {
                        "label": label,
                        "color": color,
                        "track_id": track_id,
                        "dwell_seconds": round(timestamp - (session["zone_entry_time"] or timestamp), 1)
                    }
                })

        # =========================================================================
        # Rule 2: LOITERING_DWELL (Temporal Duration > 120s, MEDIUM Severity)
        # Person remains in monitored zone (CAM_LOBBY or Restricted_Zone_A) > 120s
        # =========================================================================
        if label == "person" and session["current_zone"] in self.LOITERING_ZONES and session["zone_entry_time"] is not None:
            dwell_time = timestamp - session["zone_entry_time"]
            if dwell_time > self.LOITERING_THRESHOLD_SECONDS:
                rule_name = "LOITERING_DWELL"
                if self._can_fire_alert(entity_id, rule_name, timestamp):
                    self._mark_alert_fired(entity_id, rule_name, timestamp)
                    triggered_alerts.append({
                        "alert_id": f"alt_{uuid.uuid4().hex[:12]}",
                        "timestamp": timestamp,
                        "rule_name": rule_name,
                        "severity": "MEDIUM",
                        "global_entity_id": entity_id,
                        "camera_id": camera_id,
                        "zone_id": session["current_zone"],
                        "message": f"WARNING: Loitering detected. Person {entity_id} remained in {session['current_zone']} for {dwell_time:.1f}s (>120s limit)",
                        "metadata": {
                            "label": label,
                            "color": color,
                            "track_id": track_id,
                            "dwell_seconds": round(dwell_time, 1)
                        }
                    })

        # =========================================================================
        # Rule 3: UNAUTHORIZED_PARKING (Vehicle Dwell > 300s, LOW Severity)
        # Vehicle stationary in Dropoff_Zone > 300s
        # =========================================================================
        if label in self.VEHICLE_LABELS and session["current_zone"] == self.PARKING_ZONE and session["zone_entry_time"] is not None:
            dwell_time = timestamp - session["zone_entry_time"]
            if dwell_time > self.PARKING_THRESHOLD_SECONDS:
                rule_name = "UNAUTHORIZED_PARKING"
                if self._can_fire_alert(entity_id, rule_name, timestamp):
                    self._mark_alert_fired(entity_id, rule_name, timestamp)
                    triggered_alerts.append({
                        "alert_id": f"alt_{uuid.uuid4().hex[:12]}",
                        "timestamp": timestamp,
                        "rule_name": rule_name,
                        "severity": "LOW",
                        "global_entity_id": entity_id,
                        "camera_id": camera_id,
                        "zone_id": session["current_zone"],
                        "message": f"NOTICE: Vehicle {entity_id} ({plate or 'No Plate'}) stationary in {session['current_zone']} for {dwell_time:.1f}s (>300s limit)",
                        "metadata": {
                            "label": label,
                            "color": color,
                            "plate": plate,
                            "track_id": track_id,
                            "dwell_seconds": round(dwell_time, 1)
                        }
                    })

        return triggered_alerts

    def get_session(self, entity_id: str) -> Optional[Dict[str, Any]]:
        return self.sessions.get(entity_id)

    def reset_state(self):
        """Clears all in-memory sessions and cooldown records."""
        self.sessions.clear()
        self.cooldowns.clear()
