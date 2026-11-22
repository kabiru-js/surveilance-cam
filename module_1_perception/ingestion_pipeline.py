import os
import sys
import socket
import struct
import time
import json
import cv2
import numpy as np

# Ensure module directory is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from detector import ObjectDetector
from alpr import ALPREngine
from tracker_zones import MultiObjectTracker, ZoneMonitor

def receive_exact_bytes(sock, num_bytes):
    """
    Receives an exact number of bytes from a TCP socket.
    Raises ConnectionError if the remote end closes or fails.
    """
    buffer = b""
    while len(buffer) < num_bytes:
        chunk = sock.recv(num_bytes - len(buffer))
        if not chunk:
            raise ConnectionError("Stream disconnected unexpectedly.")
        buffer += chunk
    return buffer


def main():
    print("=" * 60)
    print("Spectra Safety V1 - Module 1 (Perception Sandbox)")
    print("=" * 60)

    # 1. Initialize Sub-Modules
    model_path = os.path.join(SCRIPT_DIR, "yolov8n.onnx")
    print(f"[INIT] Loading YOLOv8n ONNX Detector ({model_path})...")
    detector = ObjectDetector(model_path=model_path, conf_threshold=0.5)
    print("[INIT] Object Detector online.")

    print("[INIT] Initializing ALPR Engine (RapidOCR ONNX)...")
    alpr = ALPREngine(min_confidence=0.5)
    print("[INIT] ALPR Engine online.")

    print("[INIT] Initializing Kalman Multi-Object Tracker...")
    tracker = MultiObjectTracker(max_age=15, min_hits=2, iou_threshold=0.3)
    print("[INIT] Tracker online.")

    # Define Security Zones and Perimeter Tripwires for 360x640 stream
    zones = {
        "Restricted_Zone_A": [(40, 180), (320, 180), (320, 480), (40, 480)]
    }
    tripwires = {
        "Perimeter_Tripwire": [(10, 320), (350, 320)]
    }
    print("[INIT] Initializing Shapely Virtual Zones & Tripwires...")
    zone_monitor = ZoneMonitor(zones=zones, tripwires=tripwires)
    print(f"[INIT] Zone Monitor configured with zones: {list(zones.keys())}, tripwires: {list(tripwires.keys())}")

    max_frames = int(os.environ.get("MAX_FRAMES", 0))

    # Reconnection loop
    while True:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        try:
            print("[NETWORK] Connecting to Edge Camera stream on 127.0.0.1:6000...")
            client_socket.connect(('127.0.0.1', 6000))
            print("[NETWORK] Stream connected successfully! Analyzing live frames...")
            
            frame_count = 0
            
            while True:
                start_time = time.time()
                
                # --- Step 1: Ingestion & Binary Envelope Unpacking ---
                header_bytes = receive_exact_bytes(client_socket, 4)
                payload_size = struct.unpack('!I', header_bytes)[0]
                image_bytes = receive_exact_bytes(client_socket, payload_size)
                
                np_array = np.frombuffer(image_bytes, dtype=np.uint8)
                frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                frame_count += 1
                timestamp = time.time()

                # --- Step 2: Sub-Module 1.2 Object Detection & Color Classification ---
                detections = detector.detect(frame)

                # --- Step 3: Sub-Module 1.3 ALPR (Vehicle Plate Recognition) ---
                for det in detections:
                    if det.get("label") in ALPREngine.VEHICLE_CLASSES:
                        plate_result = alpr.read_plate(frame, det)
                        if plate_result:
                            det["attributes"]["plate"] = plate_result

                # --- Step 4: Sub-Module 1.4 Multi-Object Tracking (Kalman + IoU) ---
                active_tracks = tracker.update(detections)

                # --- Step 5: Sub-Module 1.4 Virtual Zones & Tripwires Event Engine ---
                zone_events = zone_monitor.update(active_tracks, timestamp=timestamp)

                # --- Step 6: Structured Telemetry & Event Logging ---
                total_latency_ms = (time.time() - start_time) * 1000

                frame_telemetry = {
                    "frame_id": frame_count,
                    "timestamp": round(timestamp, 3),
                    "latency_ms": round(total_latency_ms, 1),
                    "active_tracks_count": len(active_tracks),
                    "tracks": [
                        {
                            "track_id": trk.id,
                            "label": trk.label,
                            "color": trk.attributes.get("color", "unknown"),
                            "plate": trk.attributes.get("plate", {}).get("text") if isinstance(trk.attributes.get("plate"), dict) else None,
                            "bbox": trk.get_bbox()
                        }
                        for trk in active_tracks
                    ],
                    "events": zone_events
                }

                # Emit events immediately to console in clean JSON
                if zone_events:
                    for evt in zone_events:
                        print(f"\n[EVENT TRIGGERED] >>> {json.dumps(evt, indent=2)}")

                # Periodic telemetry output every 15 frames (~1 second)
                if frame_count % 15 == 0:
                    summary = [
                        f"ID:{t['track_id']} {t['color']} {t['label']}" + (f" [{t['plate']}]" if t['plate'] else "")
                        for t in frame_telemetry["tracks"]
                    ]
                    print(f"[PERCEPTION] Frame #{frame_count} | Latency: {total_latency_ms:.1f}ms | Active: {len(active_tracks)} | Objects: {summary}")

                if max_frames > 0 and frame_count >= max_frames:
                    print(f"[TEST] Processed {frame_count} frames successfully. Test complete.")
                    client_socket.close()
                    cv2.destroyAllWindows()
                    return

                # --- Step 7: Visual Overlays for Preview Window ---
                # Draw Zones and Tripwires
                frame = zone_monitor.draw_overlays(frame)

                # Draw Trackers (Boxes, Labels, Motion Trails, Bottom Center Anchor)
                for trk in active_tracks:
                    x1, y1, x2, y2 = trk.get_bbox()
                    tid = trk.id
                    color_name = trk.attributes.get("color", "unknown")
                    label_name = trk.label
                    plate_str = ""
                    if isinstance(trk.attributes.get("plate"), dict):
                        plate_str = f" [{trk.attributes['plate'].get('text')}]"

                    display_tag = f"#{tid} {color_name} {label_name}{plate_str}"

                    # Box color: blue for vehicle, green for person, orange for other
                    box_color = (0, 255, 0) if label_name == "person" else (255, 120, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                    # Label badge
                    label_size, _ = cv2.getTextSize(display_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    badge_y1 = max(0, y1 - 18)
                    cv2.rectangle(frame, (x1, badge_y1), (x1 + label_size[0] + 6, badge_y1 + 18), box_color, -1)
                    cv2.putText(frame, display_tag, (x1 + 3, badge_y1 + 13),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

                    # Draw motion history trail
                    if len(trk.history) > 1:
                        pts = np.array(trk.history, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame, [pts], isClosed=False, color=(0, 255, 255), thickness=1)

                    # Bottom center ground anchor
                    bc = trk.get_bottom_center()
                    cv2.circle(frame, (int(bc[0]), int(bc[1])), 3, (0, 0, 255), -1)

                # Status Banner
                fps_est = 1000.0 / max(1.0, total_latency_ms)
                hud_text = f"SPECTRA V1 | Frame: #{frame_count} | Latency: {total_latency_ms:.0f}ms | FPS: {fps_est:.1f}"
                cv2.rectangle(frame, (0, 0), (360, 22), (30, 30, 30), -1)
                cv2.putText(frame, hud_text, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 200), 1)

                try:
                    cv2.imshow('Spectra Safety V1 - Live Perception', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("[SHUTDOWN] User requested exit.")
                        client_socket.close()
                        cv2.destroyAllWindows()
                        return
                except cv2.error:
                    pass

        except (ConnectionRefusedError, ConnectionError, socket.error) as e:
            print(f"[NETWORK] Camera stream offline ({e}). Retrying in 2 seconds...")
            try:
                client_socket.close()
                cv2.destroyAllWindows()
            except Exception:
                pass
            time.sleep(2)


if __name__ == "__main__":
    main()
