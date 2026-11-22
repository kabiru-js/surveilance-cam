import time
import numpy as np
import cv2
from shapely.geometry import Point, Polygon, LineString

def compute_iou(boxA, boxB):
    """
    Computes Intersection-over-Union (IoU) between two bounding boxes [x1, y1, x2, y2].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    boxA_area = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxB_area = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    union_area = float(boxA_area + boxB_area - inter_area)
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


class KalmanBoxTracker:
    """
    Lightweight 2D Kalman Filter state estimation for a bounding box.
    State: [cx, cy, w, h, vx, vy, vw, vh]
    Measurement: [cx, cy, w, h]
    """
    _count = 0

    def __init__(self, bbox, label="object", attributes=None):
        KalmanBoxTracker._count += 1
        self.id = KalmanBoxTracker._count
        
        w = max(1.0, float(bbox[2] - bbox[0]))
        h = max(1.0, float(bbox[3] - bbox[1]))
        cx = float(bbox[0]) + w / 2.0
        cy = float(bbox[1]) + h / 2.0

        # State vector: [cx, cy, w, h, vx, vy, vw, vh]
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32)

        # Transition matrix F
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = 1.0  # dt = 1 frame

        # Measurement matrix H
        self.H = np.eye(4, 8, dtype=np.float32)

        # Covariance matrices
        self.P = np.eye(8, dtype=np.float32) * 10.0
        self.P[4:, 4:] *= 100.0  # high initial velocity variance

        self.Q = np.eye(8, dtype=np.float32)
        self.Q[:4, :4] *= 1.0
        self.Q[4:, 4:] *= 0.01

        self.R = np.eye(4, dtype=np.float32) * 2.0

        self.time_since_update = 0
        self.hits = 1
        self.age = 0
        self.label = label
        self.attributes = attributes or {}
        self.history = [self.get_bottom_center()]

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        return self.get_bbox()

    def update(self, bbox, label=None, attributes=None):
        w = max(1.0, float(bbox[2] - bbox[0]))
        h = max(1.0, float(bbox[3] - bbox[1]))
        cx = float(bbox[0]) + w / 2.0
        cy = float(bbox[1]) + h / 2.0
        z = np.array([cx, cy, w, h], dtype=np.float32)

        # Innovation
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(8, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

        self.time_since_update = 0
        self.hits += 1

        curr_pt = self.get_bottom_center()
        self.history.append(curr_pt)
        if len(self.history) > 30:
            self.history.pop(0)

        if label:
            self.label = label
        if attributes:
            self.attributes.update(attributes)

    def get_bbox(self):
        cx, cy, w, h = float(self.x[0]), float(self.x[1]), float(self.x[2]), float(self.x[3])
        x1 = int(round(cx - w / 2.0))
        y1 = int(round(cy - h / 2.0))
        x2 = int(round(cx + w / 2.0))
        y2 = int(round(cy + h / 2.0))
        return [x1, y1, x2, y2]

    def get_bottom_center(self):
        bbox = self.get_bbox()
        return (float((bbox[0] + bbox[2]) / 2.0), float(bbox[3]))


class MultiObjectTracker:
    """
    Lightweight IoU + Kalman Filter multi-object tracker for dual-core CPU.
    Maintains persistent track IDs and updates states smoothly across frame drops.
    """
    def __init__(self, max_age=15, min_hits=2, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []

    def update(self, detections):
        """
        Updates trackers with new detections for current frame.
        detections: list of dicts with 'bbox', 'label', 'attributes', 'confidence'
        Returns list of active confirmed trackers.
        """
        # 1. Predict new locations for existing trackers
        for trk in self.trackers:
            trk.predict()

        # 2. Associate detections with existing trackers
        matched, unmatched_dets, unmatched_trks = self._associate(detections)

        # 3. Update matched trackers
        for d_idx, t_idx in matched:
            det = detections[d_idx]
            self.trackers[t_idx].update(
                bbox=det["bbox"],
                label=det.get("label"),
                attributes=det.get("attributes")
            )

        # 4. Create new trackers for unmatched detections
        for d_idx in unmatched_dets:
            det = detections[d_idx]
            new_trk = KalmanBoxTracker(
                bbox=det["bbox"],
                label=det.get("label", "object"),
                attributes=det.get("attributes", {})
            )
            self.trackers.append(new_trk)

        # 5. Remove dead trackers
        alive_trackers = []
        for trk in self.trackers:
            if trk.time_since_update <= self.max_age:
                alive_trackers.append(trk)
        self.trackers = alive_trackers

        # Return confirmed tracks (active and updated or hit threshold)
        confirmed = [
            trk for trk in self.trackers 
            if (trk.hits >= self.min_hits or trk.time_since_update == 0) and trk.time_since_update <= 1
        ]
        return confirmed

    def _associate(self, detections):
        if len(self.trackers) == 0:
            return [], list(range(len(detections))), []

        iou_matrix = np.zeros((len(detections), len(self.trackers)), dtype=np.float32)
        for d, det in enumerate(detections):
            for t, trk in enumerate(self.trackers):
                iou_matrix[d, t] = compute_iou(det["bbox"], trk.get_bbox())

        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(self.trackers)))

        while len(unmatched_dets) > 0 and len(unmatched_trks) > 0:
            sub = iou_matrix[np.ix_(unmatched_dets, unmatched_trks)]
            max_iou = np.max(sub)
            if max_iou < self.iou_threshold:
                break
            coords = np.unravel_index(np.argmax(sub), sub.shape)
            d_idx = unmatched_dets[coords[0]]
            t_idx = unmatched_trks[coords[1]]

            matched.append((d_idx, t_idx))
            unmatched_dets.remove(d_idx)
            unmatched_trks.remove(t_idx)

        return matched, unmatched_dets, unmatched_trks


class ZoneMonitor:
    """
    Virtual Zones and Tripwires engine powered by Shapely.
    Emits structured JSON events when a tracked object enters/exits a zone
    or crosses a tripwire.
    """
    def __init__(self, zones=None, tripwires=None):
        self.zones = {}
        if zones:
            for zid, coords in zones.items():
                self.zones[zid] = Polygon(coords)

        self.tripwires = {}
        if tripwires:
            for tid, coords in tripwires.items():
                self.tripwires[tid] = LineString(coords)

        self.track_zone_state = {}       # {track_id: set(zone_ids_inside)}
        self.track_prev_pos = {}         # {track_id: (x, y)}
        self.tripwire_cross_history = {} # {(track_id, wire_id): last_cross_time}

    def update(self, tracks, timestamp=None):
        """
        Evaluates current active tracks against zones and tripwires.
        Returns list of structured JSON event dicts.
        """
        if timestamp is None:
            timestamp = time.time()

        events = []
        current_track_ids = set()

        for trk in tracks:
            tid = trk.id
            current_track_ids.add(tid)
            curr_pos = trk.get_bottom_center()
            prev_pos = self.track_prev_pos.get(tid, curr_pos)

            curr_point = Point(curr_pos)
            motion_line = LineString([prev_pos, curr_pos]) if curr_pos != prev_pos else None

            # --- 1. Zone evaluation ---
            curr_inside_zones = set()
            for zid, poly in self.zones.items():
                if poly.contains(curr_point):
                    curr_inside_zones.add(zid)

            prev_inside_zones = self.track_zone_state.get(tid, set())

            # Entered zones
            for zid in (curr_inside_zones - prev_inside_zones):
                events.append({
                    "event_id": f"evt_enter_{tid}_{zid}_{int(timestamp * 1000)}",
                    "timestamp": round(timestamp, 3),
                    "event_type": "ZONE_ENTER",
                    "track_id": tid,
                    "label": trk.label,
                    "color": trk.attributes.get("color", "unknown"),
                    "plate": trk.attributes.get("plate", {}).get("text") if isinstance(trk.attributes.get("plate"), dict) else None,
                    "zone_id": zid,
                    "tripwire_id": None,
                    "position": [round(curr_pos[0], 1), round(curr_pos[1], 1)],
                    "bbox": trk.get_bbox()
                })

            # Exited zones
            for zid in (prev_inside_zones - curr_inside_zones):
                events.append({
                    "event_id": f"evt_exit_{tid}_{zid}_{int(timestamp * 1000)}",
                    "timestamp": round(timestamp, 3),
                    "event_type": "ZONE_EXIT",
                    "track_id": tid,
                    "label": trk.label,
                    "color": trk.attributes.get("color", "unknown"),
                    "plate": trk.attributes.get("plate", {}).get("text") if isinstance(trk.attributes.get("plate"), dict) else None,
                    "zone_id": zid,
                    "tripwire_id": None,
                    "position": [round(curr_pos[0], 1), round(curr_pos[1], 1)],
                    "bbox": trk.get_bbox()
                })

            self.track_zone_state[tid] = curr_inside_zones

            # --- 2. Tripwire evaluation ---
            if motion_line is not None:
                for wire_id, wire in self.tripwires.items():
                    if motion_line.intersects(wire):
                        last_cross = self.tripwire_cross_history.get((tid, wire_id), 0.0)
                        if timestamp - last_cross > 2.0:  # 2.0s debounce
                            self.tripwire_cross_history[(tid, wire_id)] = timestamp
                            events.append({
                                "event_id": f"evt_cross_{tid}_{wire_id}_{int(timestamp * 1000)}",
                                "timestamp": round(timestamp, 3),
                                "event_type": "TRIPWIRE_CROSS",
                                "track_id": tid,
                                "label": trk.label,
                                "color": trk.attributes.get("color", "unknown"),
                                "plate": trk.attributes.get("plate", {}).get("text") if isinstance(trk.attributes.get("plate"), dict) else None,
                                "zone_id": None,
                                "tripwire_id": wire_id,
                                "position": [round(curr_pos[0], 1), round(curr_pos[1], 1)],
                                "bbox": trk.get_bbox()
                            })

            self.track_prev_pos[tid] = curr_pos

        # Cleanup lost tracks
        lost_tids = set(self.track_zone_state.keys()) - current_track_ids
        for dead_id in lost_tids:
            for zid in self.track_zone_state.get(dead_id, set()):
                events.append({
                    "event_id": f"evt_exit_{dead_id}_{zid}_{int(timestamp * 1000)}",
                    "timestamp": round(timestamp, 3),
                    "event_type": "ZONE_EXIT",
                    "track_id": dead_id,
                    "label": "unknown",
                    "color": "unknown",
                    "plate": None,
                    "zone_id": zid,
                    "tripwire_id": None,
                    "position": [round(p, 1) for p in self.track_prev_pos.get(dead_id, [0, 0])],
                    "bbox": []
                })
            self.track_zone_state.pop(dead_id, None)
            self.track_prev_pos.pop(dead_id, None)

        return events

    def draw_overlays(self, frame):
        """
        Draws zones and tripwires directly onto OpenCV frame matrix.
        """
        overlay = frame.copy()

        # 1. Draw Zones (semi-transparent filled polygons)
        for zid, poly in self.zones.items():
            pts = np.array(poly.exterior.coords, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 165, 255), thickness=2)
            cv2.fillPoly(overlay, [pts], color=(0, 165, 255))
            
            # Label
            first_pt = pts[0][0]
            cv2.putText(frame, f"ZONE: {zid}", (first_pt[0] + 5, first_pt[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

        # Blend polygon fill
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        # 2. Draw Tripwires (thick dashed/colored lines)
        for tid, wire in self.tripwires.items():
            pts = np.array(wire.coords, dtype=np.int32)
            if len(pts) >= 2:
                p1, p2 = tuple(pts[0]), tuple(pts[1])
                cv2.line(frame, p1, p2, color=(0, 255, 255), thickness=2)
                mid_x = (p1[0] + p2[0]) // 2
                mid_y = (p1[1] + p2[1]) // 2
                cv2.putText(frame, f"TRIPWIRE: {tid}", (mid_x - 40, mid_y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        return frame
