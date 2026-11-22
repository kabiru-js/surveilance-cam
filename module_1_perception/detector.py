import os
import shutil
import urllib.request
import cv2
import numpy as np
import onnxruntime as ort

MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.onnx"
EXPECTED_MIN_BYTES = 10 * 1024 * 1024  # ~10 MB minimum for yolov8n.onnx

def ensure_model_file(model_path="yolov8n.onnx"):
    """
    Verifies that the ONNX model file exists and is not corrupted.
    If missing or smaller than EXPECTED_MIN_BYTES, downloads it from GitHub releases.
    """
    # If path doesn't exist in cwd, check if it's in the same directory as this file
    resolved_path = model_path
    if not os.path.exists(resolved_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(script_dir, os.path.basename(model_path))
        if os.path.exists(alt_path):
            resolved_path = alt_path

    need_download = False
    if not os.path.exists(resolved_path):
        need_download = True
    elif os.path.getsize(resolved_path) < EXPECTED_MIN_BYTES:
        print(f"Warning: Existing model at {resolved_path} appears corrupted/incomplete ({os.path.getsize(resolved_path)} bytes). Re-downloading.")
        need_download = True

    if need_download:
        save_target = resolved_path if os.path.isabs(resolved_path) else os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.basename(model_path))
        print(f"Downloading yolov8n.onnx from {MODEL_URL} to {save_target}...")
        req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Mozilla/5.0 (SpectraV1-Client)"})
        with urllib.request.urlopen(req, timeout=30) as response, open(save_target, "wb") as out_file:
            shutil.copyfileobj(response, out_file)
        print(f"Download complete: {os.path.getsize(save_target)} bytes.")
        resolved_path = save_target

    return resolved_path


class ObjectDetector:
    def __init__(self, model_path="yolov8n.onnx", conf_threshold=0.5, nms_threshold=0.45):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        self.resolved_model_path = ensure_model_file(model_path)
        
        # Pure ONNX Runtime CPU Execution
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            self.resolved_model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = (640, 640)  # (width, height)
        
        # Strict COCO class mapping for security & traffic perception
        self.target_classes = {
            0: "person",
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck"
        }

    def _extract_dominant_color(self, crop_bgr, label):
        """
        Lightweight HSV color-space crop analyzer to tag dominant shirt or vehicle color.
        Uses torso region for pedestrians and central body for vehicles.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return "unknown"

        h, w = crop_bgr.shape[:2]
        if h < 4 or w < 4:
            return "unknown"

        if label == "person":
            # Torso region: 20%-55% vertically, 25%-75% horizontally
            roi = crop_bgr[int(h * 0.2):int(h * 0.55), int(w * 0.25):int(w * 0.75)]
        else:
            # Vehicle body region: 30%-70% vertically, 20%-80% horizontally
            roi = crop_bgr[int(h * 0.3):int(h * 0.7), int(w * 0.2):int(w * 0.8)]

        if roi.size == 0:
            roi = crop_bgr

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mean_h, mean_s, mean_v = np.mean(hsv, axis=(0, 1))

        # HSV thresholds
        if mean_v < 55:
            return "black"
        if mean_s < 40:
            return "white" if mean_v > 180 else "gray"

        if mean_h < 10 or mean_h >= 165:
            return "red"
        elif 10 <= mean_h < 25:
            return "orange"
        elif 25 <= mean_h < 35:
            return "yellow"
        elif 35 <= mean_h < 85:
            return "green"
        elif 85 <= mean_h < 130:
            return "blue"
        elif 130 <= mean_h < 165:
            return "purple"
        else:
            return "unknown"

    def detect(self, frame):
        """
        Runs object detection on input frame (BGR numpy array).
        Returns list of detection dicts:
        [{
            "class_id": int,
            "label": str,
            "confidence": float,
            "bbox": [x1, y1, x2, y2],
            "attributes": {"color": str}
        }, ...]
        """
        if frame is None or frame.size == 0:
            return []

        orig_h, orig_w = frame.shape[:2]
        input_w, input_h = self.input_shape

        # Preprocessing: resize -> BGR2RGB -> normalize [0, 1] -> NCHW
        resized = cv2.resize(frame, (input_w, input_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_tensor = rgb.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)

        # ONNX inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        # YOLOv8 output shape is (1, 84, 8400) -> squeeze and transpose to (8400, 84)
        predictions = np.squeeze(outputs[0]).T

        boxes = []
        confidences = []
        class_ids = []

        x_factor = orig_w / float(input_w)
        y_factor = orig_h / float(input_h)

        for row in predictions:
            classes_scores = row[4:]
            class_id = int(np.argmax(classes_scores))
            max_score = float(classes_scores[class_id])

            if max_score >= self.conf_threshold and class_id in self.target_classes:
                cx, cy, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                
                left = int((cx - w / 2.0) * x_factor)
                top = int((cy - h / 2.0) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                boxes.append([left, top, width, height])
                confidences.append(max_score)
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        
        detections = []
        if len(indices) > 0:
            for i in np.array(indices).flatten():
                x, y, w, h = boxes[i]
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(orig_w, x + w)
                y2 = min(orig_h, y + h)

                if x2 <= x1 or y2 <= y1:
                    continue

                label = self.target_classes[class_ids[i]]
                crop = frame[y1:y2, x1:x2]
                color_attr = self._extract_dominant_color(crop, label)

                detections.append({
                    "class_id": class_ids[i],
                    "label": label,
                    "confidence": round(confidences[i], 2),
                    "bbox": [x1, y1, x2, y2],
                    "attributes": {"color": color_attr}
                })

        return detections
