import re
import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

class ALPREngine:
    """
    Automatic License Plate Recognition (ALPR) optimized for dual-core CPU.
    Triggers ONLY when a vehicle (car, bus, truck) is detected.
    Uses OpenCV morphological contour filtering to localize license plate candidates,
    followed by lightweight RapidOCR recognition.
    """
    VEHICLE_CLASSES = {"car", "bus", "truck"}

    def __init__(self, min_confidence=0.5):
        self.min_confidence = min_confidence
        # Initialize RapidOCR with CPU execution
        self.ocr = RapidOCR()

    def _is_vehicle(self, label):
        return label.lower() in self.VEHICLE_CLASSES

    def _clean_plate_text(self, raw_text):
        """
        Cleans and normalizes plate text: uppercase, alphanumeric characters only.
        """
        if not raw_text:
            return ""
        cleaned = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
        return cleaned

    def _find_plate_candidates(self, vehicle_crop):
        """
        Locates candidate license plate bounding boxes in vehicle crop using
        morphological filtering and contour analysis.
        Returns list of (x, y, w, h) in vehicle_crop coordinate space.
        """
        vh, vw = vehicle_crop.shape[:2]
        if vh < 30 or vw < 30:
            return []

        # Plates are usually in the lower 50% of the vehicle
        roi_y_start = int(vh * 0.40)
        roi = vehicle_crop[roi_y_start:vh, 0:vw]
        roi_h, roi_w = roi.shape[:2]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Smooth and emphasize horizontal transitions (vertical edges)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        sobelx = cv2.Sobel(blurred, cv2.CV_8U, 1, 0, ksize=3)
        
        # Otsu thresholding
        _, thresh = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Close gaps between characters to form a plate blob
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        min_area = (vw * vh) * 0.005   # at least 0.5% of vehicle area
        max_area = (vw * vh) * 0.30    # at most 30% of vehicle area

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if min_area < area < max_area:
                aspect_ratio = float(w) / max(1, h)
                # Standard plates have aspect ratio between 1.8 and 5.5
                if 1.8 <= aspect_ratio <= 5.5 and w >= 25 and h >= 10:
                    candidates.append((x, y + roi_y_start, w, h))

        # Sort candidate boxes by area descending
        candidates.sort(key=lambda b: b[2] * b[3], reverse=True)
        return candidates[:3]  # Evaluate top 3 candidate regions

    def read_plate(self, frame, detection):
        """
        Processes a single detection.
        If detection is not a vehicle, immediately returns None.
        Otherwise extracts vehicle crop, locates plate, runs OCR,
        and returns {
            "text": str,
            "confidence": float,
            "bbox": [x1, y1, x2, y2]
        } or None if no valid plate recognized.
        """
        label = detection.get("label", "").lower()
        if not self._is_vehicle(label):
            return None

        bx1, by1, bx2, by2 = detection["bbox"]
        frame_h, frame_w = frame.shape[:2]

        # Boundary checks
        x1 = max(0, bx1)
        y1 = max(0, by1)
        x2 = min(frame_w, bx2)
        y2 = min(frame_h, by2)

        if x2 - x1 < 30 or y2 - y1 < 30:
            return None

        vehicle_crop = frame[y1:y2, x1:x2]
        candidates = self._find_plate_candidates(vehicle_crop)

        best_result = None

        # Strategy 1: Test morphological candidate crops with rec-only (fastest)
        for cx, cy, cw, ch in candidates:
            # Slight padding around candidate plate
            pad_x = int(cw * 0.05)
            pad_y = int(ch * 0.10)
            px1 = max(0, cx - pad_x)
            py1 = max(0, cy - pad_y)
            px2 = min(vehicle_crop.shape[1], cx + cw + pad_x)
            py2 = min(vehicle_crop.shape[0], cy + ch + pad_y)

            plate_crop = vehicle_crop[py1:py2, px1:px2]
            if plate_crop.size == 0 or plate_crop.shape[0] < 8 or plate_crop.shape[1] < 20:
                continue

            try:
                # Rec-only mode
                ocr_out, _ = self.ocr(plate_crop, use_det=False, use_cls=False, use_rec=True)
                if ocr_out and len(ocr_out) > 0:
                    text, score = ocr_out[0]
                    cleaned = self._clean_plate_text(text)
                    if 4 <= len(cleaned) <= 10 and score >= self.min_confidence:
                        best_result = {
                            "text": cleaned,
                            "confidence": round(float(score), 2),
                            "bbox": [x1 + px1, y1 + py1, x1 + px2, y1 + py2]
                        }
                        return best_result
            except Exception:
                pass

        # Strategy 2: Fallback to lower-half vehicle detection + recognition if candidates missed
        vh, vw = vehicle_crop.shape[:2]
        lower_y = int(vh * 0.40)
        lower_roi = vehicle_crop[lower_y:vh, 0:vw]
        if lower_roi.size > 0:
            try:
                ocr_out, _ = self.ocr(lower_roi, use_det=True, use_cls=False, use_rec=True)
                if ocr_out:
                    for item in ocr_out:
                        box, text, score = item[0], item[1], item[2]
                        cleaned = self._clean_plate_text(text)
                        if 4 <= len(cleaned) <= 10 and score >= self.min_confidence:
                            box_np = np.array(box)
                            min_bx, min_by = int(np.min(box_np[:, 0])), int(np.min(box_np[:, 1]))
                            max_bx, max_by = int(np.max(box_np[:, 0])), int(np.max(box_np[:, 1]))
                            best_result = {
                                "text": cleaned,
                                "confidence": round(float(score), 2),
                                "bbox": [
                                    x1 + min_bx,
                                    y1 + lower_y + min_by,
                                    x1 + max_bx,
                                    y1 + lower_y + max_by
                                ]
                            }
                            return best_result
            except Exception:
                pass

        return None
