import os
import sys
import cv2
import numpy as np

# Ensure module directory is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from detector import ObjectDetector
from alpr import ALPREngine
from tracker_zones import MultiObjectTracker, ZoneMonitor

def run_tests():
    print("=" * 70)
    print("SPECTRA SAFETY V1 — MODULE 1 (PERCEPTION SANDBOX) UNIT TESTS")
    print("=" * 70)

    # -------------------------------------------------------------
    # Test 1: Sub-Module 1.2 Object Detector
    # -------------------------------------------------------------
    print("\n[TEST 1] Sub-Module 1.2: ObjectDetector (ONNX + HSV Analyzer)")
    model_path = os.path.join(SCRIPT_DIR, "yolov8n.onnx")
    detector = ObjectDetector(model_path=model_path, conf_threshold=0.5)

    test_video_path = os.path.join(SCRIPT_DIR, "mock_edge_source.mp4")
    if os.path.exists(test_video_path):
        cap = cv2.VideoCapture(test_video_path)
        ret, frame = cap.read()
        cap.release()
        assert ret, "Failed to read test frame from mock_edge_source.mp4"
        
        # Test 360x640 portrait frame
        resized = cv2.resize(frame, (360, 640))
        detections = detector.detect(resized)
        print(f" -> Detected {len(detections)} objects in test frame:")
        for d in detections[:3]:
            print(f"    * Label: {d['label']}, Confidence: {d['confidence']}, Color: {d['attributes']['color']}, BBox: {d['bbox']}")
        
        assert len(detections) > 0, "Detector should detect objects in mock_edge_source.mp4"
        assert all(d["class_id"] in detector.target_classes for d in detections), "All detections must match COCO target classes"
        print("[PASS] Sub-Module 1.2 ObjectDetector functional.")
    else:
        print("[WARN] mock_edge_source.mp4 not found, skipped video asset test.")

    # -------------------------------------------------------------
    # Test 2: Sub-Module 1.3 ALPR Engine
    # -------------------------------------------------------------
    print("\n[TEST 2] Sub-Module 1.3: ALPREngine (Vehicle Gating + RapidOCR)")
    alpr = ALPREngine(min_confidence=0.5)

    # 2a: Non-vehicle check (Person must be skipped)
    person_det = {"label": "person", "confidence": 0.85, "bbox": [50, 50, 200, 400]}
    dummy_frame = np.zeros((500, 500, 3), dtype=np.uint8)
    res_person = alpr.read_plate(dummy_frame, person_det)
    assert res_person is None, "ALPR must immediately return None for non-vehicle labels"
    print(" -> Non-vehicle detection correctly ignored: PASSED")

    # 2b: Synthetic vehicle with license plate
    car_frame = np.full((600, 600, 3), 120, dtype=np.uint8)
    # Car body
    cv2.rectangle(car_frame, (100, 150), (500, 500), (40, 40, 40), -1)
    # Plate rectangle on bumper (lower half)
    cv2.rectangle(car_frame, (220, 380), (380, 430), (255, 255, 255), -1)
    cv2.putText(car_frame, "CA8899", (230, 420), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

    car_det = {"label": "car", "confidence": 0.90, "bbox": [100, 150, 500, 500]}
    res_car = alpr.read_plate(car_frame, car_det)
    print(f" -> Vehicle plate extraction result: {res_car}")
    assert res_car is not None, "Plate must be detected on vehicle crop"
    assert "CA8899" in res_car["text"], f"Expected CA8899 in plate text, got {res_car['text']}"
    assert res_car["confidence"] >= 0.5, "Plate confidence must be >= 0.5"
    print("[PASS] Sub-Module 1.3 ALPREngine functional.")

    # -------------------------------------------------------------
    # Test 3: Sub-Module 1.4 MultiObjectTracker (Kalman + IoU)
    # -------------------------------------------------------------
    print("\n[TEST 3] Sub-Module 1.4: MultiObjectTracker (Persistent Track IDs)")
    tracker = MultiObjectTracker(max_age=5, min_hits=1, iou_threshold=0.3)

    det_f1 = [{"bbox": [100, 100, 150, 200], "label": "person", "attributes": {"color": "blue"}}]
    tracks_f1 = tracker.update(det_f1)
    assert len(tracks_f1) == 1, "Must initialize 1 track"
    tid = tracks_f1[0].id

    # Simulated motion in frame 2
    det_f2 = [{"bbox": [106, 108, 156, 208], "label": "person", "attributes": {"color": "blue"}}]
    tracks_f2 = tracker.update(det_f2)
    assert len(tracks_f2) == 1, "Must maintain track"
    assert tracks_f2[0].id == tid, f"Track ID must persist: expected {tid}, got {tracks_f2[0].id}"
    print(f" -> Persistent Track ID maintained across frames (ID: {tid}): PASSED")

    # -------------------------------------------------------------
    # Test 4: Sub-Module 1.4 ZoneMonitor (Virtual Zones & Tripwires)
    # -------------------------------------------------------------
    print("\n[TEST 4] Sub-Module 1.4: ZoneMonitor (Shapely Event Engine)")
    zones = {"Restricted_Zone": [(50, 50), (200, 50), (200, 200), (50, 200)]}
    tripwires = {"Boundary_Line": [(0, 150), (300, 150)]}
    monitor = ZoneMonitor(zones=zones, tripwires=tripwires)

    test_trk = tracks_f2[0]
    test_trk.x[2] = 20 # width
    test_trk.x[3] = 20 # height

    # 1. Outside zone: center y=10 -> bottom-center y = 10 + 10 = 20 (zone starts at y=50)
    test_trk.x[0], test_trk.x[1] = 125, 10
    evts0 = monitor.update([test_trk])
    assert len(evts0) == 0, f"Must not emit events outside zone, got: {evts0}"

    # 2. Enter zone: center y=90 -> bottom-center y = 90 + 10 = 100 (inside 50..200)
    test_trk.x[0], test_trk.x[1] = 125, 90
    evts_enter = monitor.update([test_trk])
    assert any(e["event_type"] == "ZONE_ENTER" for e in evts_enter), "Must emit ZONE_ENTER"
    print(" -> ZONE_ENTER event emitted: PASSED")

    # 3. Cross tripwire: previous position was y=100, now center y=170 -> bottom-center y = 170 + 10 = 180 (crosses y=150)
    test_trk.x[0], test_trk.x[1] = 125, 170
    evts_cross = monitor.update([test_trk])
    assert any(e["event_type"] == "TRIPWIRE_CROSS" for e in evts_cross), "Must emit TRIPWIRE_CROSS"
    print(" -> TRIPWIRE_CROSS event emitted: PASSED")

    # 4. Exit zone: center y=240 -> bottom-center y = 240 + 10 = 250 (outside 50..200)
    test_trk.x[0], test_trk.x[1] = 125, 240
    evts_exit = monitor.update([test_trk])
    assert any(e["event_type"] == "ZONE_EXIT" for e in evts_exit), "Must emit ZONE_EXIT"
    print(" -> ZONE_EXIT event emitted: PASSED")

    print("\n" + "=" * 70)
    print("ALL MODULE 1 UNIT TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
