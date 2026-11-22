import os
import sys
import json
import time

# Ensure module directory is on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from automation_engine import EventAutomationEngine

def run_tests():
    print("=" * 70)
    print("SPECTRA SAFETY V1 — MODULE 3 (EVENT & AUTOMATION) UNIT TESTS")
    print("=" * 70)

    db_path = os.path.join(CURRENT_DIR, "test_events.db")
    for f in [db_path, db_path + "-wal", db_path + "-shm"]:
        if os.path.exists(f):
            os.remove(f)

    # Initialize Engine (runs WebSocket bus on port 8766 to avoid any port conflicts)
    engine = EventAutomationEngine(db_path=db_path, ws_port=8766, start_bus=True, cooldown_seconds=60.0)

    # -------------------------------------------------------------------------
    # Scenario A: Immediate RESTRICTED_ZONE_BREACH
    # -------------------------------------------------------------------------
    print("\n[SCENARIO A] Immediate Restricted Zone Breach:")
    print(" -> Person (ent_per_01) enters Restricted_Zone_A at t=1000.0s")
    evt_a = {
        "event_id": "evt_001",
        "timestamp": 1000.0,
        "global_entity_id": "ent_per_01",
        "camera_id": "CAM_GATE_A",
        "track_id": 1,
        "label": "person",
        "color": "blue",
        "plate": None,
        "zone_id": "Restricted_Zone_A",
        "event_type": "ZONE_ENTER"
    }
    alerts_a = engine.process_event(evt_a)
    print(f" -> Triggered Alerts: {len(alerts_a)}")
    assert len(alerts_a) == 1, "Scenario A: Must immediately trigger exactly 1 alert"
    assert alerts_a[0]["rule_name"] == "RESTRICTED_ZONE_BREACH"
    assert alerts_a[0]["severity"] == "HIGH"
    assert alerts_a[0]["global_entity_id"] == "ent_per_01"
    print(f" -> Alert: [{alerts_a[0]['severity']}] {alerts_a[0]['rule_name']} - {alerts_a[0]['message']}")
    print("[PASS] Scenario A: Immediate high-severity breach alert triggered successfully.")

    # -------------------------------------------------------------------------
    # Scenario B: Alert Deduplication & Cooldown Recovery
    # -------------------------------------------------------------------------
    print("\n[SCENARIO B] Alert Deduplication & Cooldown Recovery:")
    print(" -> Sighting 10s later at t=1010.0s (within 60s cooldown window)")
    evt_b1 = {
        "event_id": "evt_002",
        "timestamp": 1010.0,
        "global_entity_id": "ent_per_01",
        "camera_id": "CAM_GATE_A",
        "track_id": 1,
        "label": "person",
        "color": "blue",
        "plate": None,
        "zone_id": "Restricted_Zone_A",
        "event_type": "OBSERVATION"
    }
    alerts_b1 = engine.process_event(evt_b1)
    print(f" -> Triggered Alerts at t=1010s: {len(alerts_b1)} (Expected: 0)")
    assert len(alerts_b1) == 0, "Scenario B: Duplicate alert must be suppressed during cooldown"
    print(" -> Duplicate alert successfully suppressed by 60s cooldown: PASSED")

    print(" -> Sighting 65s later at t=1065.0s (cooldown window expired)")
    evt_b2 = {
        "event_id": "evt_003",
        "timestamp": 1065.0,
        "global_entity_id": "ent_per_01",
        "camera_id": "CAM_GATE_A",
        "track_id": 1,
        "label": "person",
        "color": "blue",
        "plate": None,
        "zone_id": "Restricted_Zone_A",
        "event_type": "OBSERVATION"
    }
    alerts_b2 = engine.process_event(evt_b2)
    print(f" -> Triggered Alerts at t=1065s: {len(alerts_b2)} (Expected: 1)")
    assert len(alerts_b2) == 1, "Scenario B: Must allow re-triggering after cooldown expires"
    assert alerts_b2[0]["rule_name"] == "RESTRICTED_ZONE_BREACH"
    print("[PASS] Scenario B: Deduplication and cooldown recovery validated.")

    # -------------------------------------------------------------------------
    # Scenario C: LOITERING_DWELL Temporal Threshold (120s)
    # -------------------------------------------------------------------------
    print("\n[SCENARIO C] Loitering Dwell Time Verification (120s limit):")
    print(" -> Person (ent_per_02) enters CAM_LOBBY at t=2000.0s")
    evt_c1 = {
        "event_id": "evt_004",
        "timestamp": 2000.0,
        "global_entity_id": "ent_per_02",
        "camera_id": "CAM_LOBBY",
        "track_id": 5,
        "label": "person",
        "color": "gray",
        "plate": None,
        "zone_id": "CAM_LOBBY",
        "event_type": "ZONE_ENTER"
    }
    alerts_c1 = engine.process_event(evt_c1)
    assert len(alerts_c1) == 0, "Scenario C: Entry must not trigger loitering alert immediately"

    print(" -> Person still in CAM_LOBBY at t=2050.0s (dwell = 50s <= 120s)")
    evt_c2 = {
        "event_id": "evt_005",
        "timestamp": 2050.0,
        "global_entity_id": "ent_per_02",
        "camera_id": "CAM_LOBBY",
        "track_id": 5,
        "label": "person",
        "color": "gray",
        "plate": None,
        "zone_id": "CAM_LOBBY",
        "event_type": "OBSERVATION"
    }
    alerts_c2 = engine.process_event(evt_c2)
    print(f" -> Triggered Alerts at 50s dwell: {len(alerts_c2)} (Expected: 0)")
    assert len(alerts_c2) == 0, "Scenario C: Dwell at 50s must NOT trigger loitering alert"

    print(" -> Person still in CAM_LOBBY at t=2125.0s (dwell = 125s > 120s)")
    evt_c3 = {
        "event_id": "evt_006",
        "timestamp": 2125.0,
        "global_entity_id": "ent_per_02",
        "camera_id": "CAM_LOBBY",
        "track_id": 5,
        "label": "person",
        "color": "gray",
        "plate": None,
        "zone_id": "CAM_LOBBY",
        "event_type": "OBSERVATION"
    }
    alerts_c3 = engine.process_event(evt_c3)
    print(f" -> Triggered Alerts at 125s dwell: {len(alerts_c3)} (Expected: 1)")
    assert len(alerts_c3) == 1, "Scenario C: Must trigger loitering alert once dwell > 120s"
    assert alerts_c3[0]["rule_name"] == "LOITERING_DWELL"
    assert alerts_c3[0]["severity"] == "MEDIUM"
    assert alerts_c3[0]["metadata"]["dwell_seconds"] == 125.0
    print(f" -> Alert: [{alerts_c3[0]['severity']}] {alerts_c3[0]['message']}")
    print("[PASS] Scenario C: Loitering temporal dwell threshold validated.")

    # -------------------------------------------------------------------------
    # Scenario D: UNAUTHORIZED_PARKING Vehicle Dwell (300s limit) & DB Verification
    # -------------------------------------------------------------------------
    print("\n[SCENARIO D] Unauthorized Vehicle Parking (300s limit) & DB Verification:")
    print(" -> Car (ent_car_01, CA8899) enters Dropoff_Zone at t=3000.0s")
    evt_d1 = {
        "event_id": "evt_007",
        "timestamp": 3000.0,
        "global_entity_id": "ent_car_01",
        "camera_id": "CAM_DRIVEWAY",
        "track_id": 12,
        "label": "car",
        "color": "white",
        "plate": "CA8899",
        "zone_id": "Dropoff_Zone",
        "event_type": "ZONE_ENTER"
    }
    alerts_d1 = engine.process_event(evt_d1)
    assert len(alerts_d1) == 0

    print(" -> Car stationary in Dropoff_Zone at t=3305.0s (dwell = 305s > 300s)")
    evt_d2 = {
        "event_id": "evt_008",
        "timestamp": 3305.0,
        "global_entity_id": "ent_car_01",
        "camera_id": "CAM_DRIVEWAY",
        "track_id": 12,
        "label": "car",
        "color": "white",
        "plate": "CA8899",
        "zone_id": "Dropoff_Zone",
        "event_type": "OBSERVATION"
    }
    alerts_d2 = engine.process_event(evt_d2)
    print(f" -> Triggered Alerts at 305s dwell: {len(alerts_d2)} (Expected: 1)")
    assert len(alerts_d2) == 1, "Scenario D: Must trigger unauthorized parking alert once dwell > 300s"
    assert alerts_d2[0]["rule_name"] == "UNAUTHORIZED_PARKING"
    assert alerts_d2[0]["severity"] == "LOW"
    assert alerts_d2[0]["metadata"]["dwell_seconds"] == 305.0
    print(f" -> Alert: [{alerts_d2[0]['severity']}] {alerts_d2[0]['message']}")
    print("[PASS] Scenario D: Unauthorized vehicle parking validated.")

    # --- Database State Verification ---
    print("\n[PERSISTENCE VERIFICATION] Checking SQLite Database:")
    obs_count = engine.store.count_observations()
    alert_count = engine.store.count_alerts()
    print(f" -> Total Observations in SQLite: {obs_count} (Expected: 8)")
    print(f" -> Total Alerts in SQLite: {alert_count} (Expected: 4)")
    assert obs_count == 8, f"Expected 8 observations, got {obs_count}"
    assert alert_count == 4, f"Expected 4 alerts, got {alert_count}"

    # Verify query
    recent = engine.store.get_recent_alerts(limit=10)
    rule_names = [a["rule_name"] for a in recent]
    print(f" -> Stored Alert Rules: {rule_names}")
    assert "UNAUTHORIZED_PARKING" in rule_names
    assert "LOITERING_DWELL" in rule_names
    assert "RESTRICTED_ZONE_BREACH" in rule_names

    # Clean up
    engine.close()
    for f in [db_path, db_path + "-wal", db_path + "-shm"]:
        if os.path.exists(f):
            os.remove(f)

    print("\n" + "=" * 70)
    print("ALL MODULE 3 UNIT TEST SCENARIOS PASSED PERFECTLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
