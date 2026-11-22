import os
import sys
import json
import cv2
import numpy as np

# Ensure module directory is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from memory_engine import SpatialTemporalMemoryEngine

def create_synthetic_pedestrian(torso_bgr, legs_bgr):
    """
    Creates a synthetic 128x64 tight pedestrian bounding-box crop with colored torso and legs.
    Matches YOLOv8 tight object detection bounding boxes.
    """
    img = np.zeros((128, 64, 3), dtype=np.uint8)
    # Head (top 20px): skin tone
    img[:20, :] = (160, 190, 220)
    # Torso (20 to 75px): torso color
    img[20:75, :] = torso_bgr
    # Legs (75 to 128px): pants color
    img[75:, :] = legs_bgr
    return img

def run_tests():
    print("=" * 70)
    print("SPECTRA SAFETY V1 — MODULE 2 (SPATIAL-TEMPORAL MEMORY) UNIT TESTS")
    print("=" * 70)

    db_path = os.path.join(CURRENT_DIR, "test_memory.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    engine = SpatialTemporalMemoryEngine(db_path=db_path, min_similarity=0.75)

    # 1. Prepare synthetic entities
    # Person A: Blue shirt, dark pants
    person_a_crop = create_synthetic_pedestrian(torso_bgr=(255, 0, 0), legs_bgr=(40, 40, 40))
    # Person B: Bright Red shirt, white pants (visually distinct)
    person_b_crop = create_synthetic_pedestrian(torso_bgr=(0, 0, 255), legs_bgr=(240, 240, 240))

    # --- Initial Baseline Observation ---
    print("\n[STEP 0] Baseline Observation: Person A seen at CAM_GATE_A at t=1000.0s")
    obs0 = engine.process_observation(
        camera_id="CAM_GATE_A",
        timestamp=1000.0,
        track_id=1,
        label="person",
        color="blue",
        crop_bgr=person_a_crop
    )
    print(f" -> Result: Action={obs0['action']}, EntityID={obs0['global_entity_id']}")
    assert obs0["action"] == "NEW", "Initial observation must mint a NEW entity"
    entity_a_id = obs0["global_entity_id"]

    # --- Scenario A: Valid Cross-Camera Match ---
    # Person A seen at CAM_LOBBY (50m away) at t=1040.0s (40s later, speed 1.25 m/s)
    print("\n[SCENARIO A] Valid Cross-Camera Transit:")
    print(" -> Person A observed at CAM_LOBBY at t=1040.0s (50m away, delta_t=40s)")
    obs_a = engine.process_observation(
        camera_id="CAM_LOBBY",
        timestamp=1040.0,
        track_id=5,
        label="person",
        color="blue",
        crop_bgr=person_a_crop
    )
    print(f" -> Action: {obs_a['action']}")
    print(f" -> EntityID: {obs_a['global_entity_id']}")
    print(f" -> Match Details: {json.dumps(obs_a['matched_info'], indent=2)}")

    assert obs_a["action"] == "LINKED", "Scenario A: Must LINK to existing entity"
    assert obs_a["global_entity_id"] == entity_a_id, f"Scenario A: Expected {entity_a_id}, got {obs_a['global_entity_id']}"
    print("[PASS] Scenario A: Successfully linked Person A across CAM_GATE_A and CAM_LOBBY.")

    # --- Scenario B: Impossible Teleportation Rejected ---
    # Visually identical crop seen at CAM_PARKING_LOT (300m away) at t=1005.0s (only 5s after t=1000)
    # Required speed = 300m / 5s = 60 m/s (216 km/h) -> Impossible on foot
    print("\n[SCENARIO B] Impossible Teleportation Defense:")
    print(" -> Visually identical crop at CAM_PARKING_LOT at t=1005.0s (300m away, delta_t=5s)")
    obs_b = engine.process_observation(
        camera_id="CAM_PARKING_LOT",
        timestamp=1005.0,
        track_id=9,
        label="person",
        color="blue",
        crop_bgr=person_a_crop
    )
    print(f" -> Action: {obs_b['action']}")
    print(f" -> EntityID: {obs_b['global_entity_id']}")
    print(f" -> Status: {obs_b['matched_info']['spatial_temporal_status']}")

    assert obs_b["action"] == "NEW", "Scenario B: Teleportation must be REJECTED, creating NEW entity"
    assert obs_b["global_entity_id"] != entity_a_id, "Scenario B: Entity ID must NOT match Person A"
    assert "Teleportation violation" in obs_b["matched_info"]["spatial_temporal_status"], "Scenario B: Must cite teleportation violation"
    print("[PASS] Scenario B: Impossible teleportation detected and rejected as spoofing/twin.")

    # --- Scenario C: Different Person Rejected ---
    # Person B (visually distinct) seen at CAM_LOBBY at t=1050.0s
    print("\n[SCENARIO C] Visual Appearance Mismatch Defense:")
    print(" -> Visually distinct Person B (Red shirt) observed at CAM_LOBBY at t=1050.0s")
    obs_c = engine.process_observation(
        camera_id="CAM_LOBBY",
        timestamp=1050.0,
        track_id=14,
        label="person",
        color="red",
        crop_bgr=person_b_crop
    )
    print(f" -> Action: {obs_c['action']}")
    print(f" -> EntityID: {obs_c['global_entity_id']}")
    print(f" -> Status: {obs_c['matched_info']['spatial_temporal_status']}")

    assert obs_c["action"] == "NEW", "Scenario C: Visual mismatch must create NEW entity"
    assert obs_c["global_entity_id"] != entity_a_id, "Scenario C: Entity ID must NOT match Person A"
    assert obs_c["global_entity_id"] != obs_b["global_entity_id"], "Scenario C: Entity ID must NOT match Scenario B"
    print("[PASS] Scenario C: Distinct appearance correctly minted as new independent entity.")

    # Clean up
    engine.close()
    if os.path.exists(db_path):
        os.remove(db_path)

    print("\n" + "=" * 70)
    print("ALL MODULE 2 UNIT TEST SCENARIOS PASSED PERFECTLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
