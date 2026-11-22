import os
import cv2
import numpy as np

class VisualFingerprintExtractor:
    """
    Lightweight CPU-friendly visual feature extractor for persons and vehicles.
    Generates a fixed-length 128-dimensional L2-normalized float32 vector embedding.
    
    Uses multi-scale spatial Hue-Saturation-Value (HSV) histograms combined with
    gradient orientation (HOG-like) descriptors across spatial vertical stripes.
    Optionally supports loading an ONNX Re-ID model if a valid model path is provided.
    """
    EMBEDDING_DIM = 128

    def __init__(self, onnx_model_path=None):
        self.onnx_model_path = onnx_model_path
        self.session = None

        if onnx_model_path and os.path.exists(onnx_model_path):
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2
                opts.inter_op_num_threads = 1
                self.session = ort.InferenceSession(
                    onnx_model_path,
                    sess_options=opts,
                    providers=["CPUExecutionProvider"]
                )
                self.input_name = self.session.get_inputs()[0].name
                print(f"[REID] Loaded ONNX Re-ID model from {onnx_model_path}")
            except Exception as e:
                print(f"[REID] Failed to load ONNX model ({e}), falling back to CPU descriptor.")
                self.session = None

    def extract(self, crop_bgr: np.ndarray) -> np.ndarray:
        """
        Extracts a 128-dimensional L2-normalized float32 embedding from crop_bgr.
        Returns np.ndarray of shape (128,) with norm == 1.0.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            fallback = np.zeros(self.EMBEDDING_DIM, dtype=np.float32)
            fallback[0] = 1.0
            return fallback

        # If an ONNX Re-ID session is active, run model inference
        if self.session is not None:
            try:
                resized = cv2.resize(crop_bgr, (128, 256))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                normalized = (rgb - mean) / std
                nchw = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
                out = self.session.run(None, {self.input_name: nchw})[0].flatten().astype(np.float32)
                if len(out) == self.EMBEDDING_DIM:
                    norm = np.linalg.norm(out)
                    return out / max(norm, 1e-6)
            except Exception:
                pass  # Fall back to CPU descriptor

        # Fast deterministic CPU Multi-Scale Spatial Descriptor
        h, w = crop_bgr.shape[:2]
        if h < 8 or w < 8:
            crop_bgr = cv2.resize(crop_bgr, (64, 128))

        resized = cv2.resize(crop_bgr, (64, 128))
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Gradient magnitude and orientation
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        angle = (np.arctan2(gy, gx) * 180.0 / np.pi) % 180.0

        features = []

        # 4 spatial vertical stripes (head/roof, upper body, lower body, feet/tires)
        # Each stripe: 18 dims (6x3 HS hist) + 6 dims (V hist) + 4 dims (gradient) = 28 dims
        stripe_h = 32
        for s in range(4):
            y0 = s * stripe_h
            y1 = (s + 1) * stripe_h
            strip_hsv = hsv[y0:y1, :]

            # 2D Hue-Saturation histogram: 6 Hue bins x 3 Saturation bins = 18 bins
            hs_hist = cv2.calcHist([strip_hsv], [0, 1], None, [6, 3], [0, 180, 0, 256]).flatten()
            # 1D Value histogram: 6 bins
            v_hist = cv2.calcHist([strip_hsv], [2], None, [6], [0, 256]).flatten()

            color_feat = np.concatenate([hs_hist, v_hist]).astype(np.float32)
            color_feat /= (np.sum(color_feat) + 1e-6)

            # Gradient orientation histogram: 4 bins (0-180 deg) weighted by edge magnitude
            g_hist, _ = np.histogram(angle[y0:y1, :], bins=4, range=(0, 180), weights=mag[y0:y1, :])
            g_feat = g_hist.astype(np.float32)
            g_feat /= (np.sum(g_feat) + 1e-6)

            stripe_vec = np.concatenate([color_feat * 0.85, g_feat * 0.15])  # 28 dims
            features.append(stripe_vec)

        # 4 stripes * 28 dims = 112 dims

        # Global features: 16 dims (4x3 HS hist = 12 dims + 4 V hist = 4 dims)
        ghs_hist = cv2.calcHist([hsv], [0, 1], None, [4, 3], [0, 180, 0, 256]).flatten()
        gv_hist = cv2.calcHist([hsv], [2], None, [4], [0, 256]).flatten()
        global_color = np.concatenate([ghs_hist, gv_hist]).astype(np.float32)
        global_color /= (np.sum(global_color) + 1e-6)
        features.append(global_color)

        # Concatenate: 112 + 16 = 128 dimensions exactly
        vector = np.concatenate(features).astype(np.float32)

        # Strict L2 Normalization
        norm = np.linalg.norm(vector)
        if norm > 1e-6:
            vector = vector / norm
        else:
            vector = np.zeros(self.EMBEDDING_DIM, dtype=np.float32)
            vector[0] = 1.0

        return vector
