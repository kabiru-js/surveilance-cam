# How to Test Module 1 — Perception Sandbox

This guide provides step-by-step instructions for testing and validating **Module 1 (Perception Sandbox)** of Spectra Safety V1.

---

## 1. Architecture & Sub-Modules Under Test

Module 1 handles real-time edge video ingestion, AI object detection, color attribute analysis, vehicle license plate recognition (ALPR), multi-object tracking, and virtual zone breach monitoring.

| File | Sub-Module | Responsibility | Hardware Target |
| :--- | :--- | :--- | :--- |
| `detector.py` | Sub-Module 1.2 | Pure ONNX Runtime YOLOv8n object detection + HSV dominant color extraction | CPU (2 intra-op threads) |
| `alpr.py` | Sub-Module 1.3 | Vehicle-gated plate localization via morphological contours + RapidOCR ONNX | CPU |
| `tracker_zones.py` | Sub-Module 1.4 | 8-State Kalman Filter + IoU Tracker + Shapely Virtual Zones & Tripwires | Pure NumPy + Shapely |
| `mock_server.py` | Test Asset | TCP server streaming downsampled JPEG frames with 4-byte `!I` length headers | `127.0.0.1:6000` |
| `ingestion_pipeline.py` | Integration | End-to-end TCP client, AI pipeline orchestration, HUD overlays, telemetry logging | Live Preview & JSON Events |
| `test_module_1.py` | Unit Test | Automated test suite verifying Sub-Modules 1.2, 1.3, and 1.4 | Headless / CI-ready |

---

## 2. Environment & Prerequisites

All commands must be executed using the lightweight virtual environment located at `module_1_perception/stream_env`:

```powershell
# Navigate to repository root
cd C:\Users\USER\spectra_v1

# Activate virtual environment in PowerShell
.\module_1_perception\stream_env\Scripts\Activate.ps1
```

### Required Dependencies
Verify that all lightweight CPU libraries are installed:
```powershell
python -c "import cv2, onnxruntime, numpy, shapely, rapidocr_onnxruntime; print('All perception libraries loaded successfully!')"
```

> **Note**: Do **NOT** install PyTorch, Torchvision, Ultralytics, or CUDA. The pipeline runs strictly on CPU.

---

## 3. Testing Options

### Option A: Fast Automated Unit Test Suite (Headless / CI)

Run the automated test suite to verify all individual sub-modules without starting a network stream:

```powershell
cd C:\Users\USER\spectra_v1\module_1_perception
python test_module_1.py
```

#### What It Tests:
1. **Sub-Module 1.2 (`detector.py`)**:
   - Auto-checks and downloads `yolov8n.onnx` if missing or corrupted.
   - Runs detection on a frame from `mock_edge_source.mp4`.
   - Confirms detections strictly match COCO classes (`person`, `car`, `motorcycle`, `bus`, `truck`) with confidence $\ge 0.5$.
   - Confirms HSV dominant color analyzer assigns human-readable tags (`white`, `blue`, `gray`, etc.).
2. **Sub-Module 1.3 (`alpr.py`)**:
   - Confirms non-vehicles (`person`) are strictly ignored.
   - Runs morphological filtering and OCR on a synthetic vehicle crop with plate text `CA8899`.
   - Confirms plate text is recognized with confidence $\ge 0.5$.
3. **Sub-Module 1.4 (`tracker_zones.py`)**:
   - Tests `MultiObjectTracker`: Confirms persistent `track_id` across moving frames.
   - Tests `ZoneMonitor`: Confirms `ZONE_ENTER`, `TRIPWIRE_CROSS`, and `ZONE_EXIT` events fire correctly with Shapely geometries.

#### Expected Output:
```text
======================================================================
SPECTRA SAFETY V1 — MODULE 1 (PERCEPTION SANDBOX) UNIT TESTS
======================================================================

[TEST 1] Sub-Module 1.2: ObjectDetector (ONNX + HSV Analyzer)
 -> Detected 10 objects in test frame:
    * Label: person, Confidence: 0.84, Color: white, BBox: [15, 252, 62, 402]
    * Label: person, Confidence: 0.83, Color: gray, BBox: [249, 234, 296, 401]
    * Label: person, Confidence: 0.81, Color: blue, BBox: [293, 226, 340, 399]
[PASS] Sub-Module 1.2 ObjectDetector functional.

[TEST 2] Sub-Module 1.3: ALPREngine (Vehicle Gating + RapidOCR)
 -> Non-vehicle detection correctly ignored: PASSED
 -> Vehicle plate extraction result: {'text': 'CA8899', 'confidence': 1.0, 'bbox': [211, 374, 373, 437]}
[PASS] Sub-Module 1.3 ALPREngine functional.

[TEST 3] Sub-Module 1.4: MultiObjectTracker (Persistent Track IDs)
 -> Persistent Track ID maintained across frames (ID: 1): PASSED

[TEST 4] Sub-Module 1.4: ZoneMonitor (Shapely Event Engine)
 -> ZONE_ENTER event emitted: PASSED
 -> TRIPWIRE_CROSS event emitted: PASSED
 -> ZONE_EXIT event emitted: PASSED

======================================================================
ALL MODULE 1 UNIT TESTS COMPLETED SUCCESSFULLY!
======================================================================
```

---

### Option B: Live End-to-End Ingestion & Preview Testing

This tests real-time network streaming over TCP, binary envelope decoding, AI perception, Kalman tracking, zone breach detection, and GUI overlays.

#### Terminal 1 — Start the Edge Camera Server:
```powershell
cd C:\Users\USER\spectra_v1\module_1_perception
python mock_server.py
```
*Expected Console Output:*
```text
Edge Camera Server online on 127.0.0.1:6000.
Waiting for central ingestion pipeline to connect...
```

#### Terminal 2 — Start the Central Ingestion Pipeline:
```powershell
cd C:\Users\USER\spectra_v1\module_1_perception
# Optional: Set MAX_FRAMES=60 for automated benchmark exit, or omit for continuous streaming
$env:MAX_FRAMES="60"
python ingestion_pipeline.py
```

#### What to Verify in Live Mode:
1. **GUI Window**:
   - A window titled `Spectra Safety V1 - Live Perception` should open.
   - Bounding boxes are rendered (green for pedestrians, orange for vehicles).
   - Labels display `#{track_id} {color} {label}` (e.g., `#3 blue person`).
   - Motion history trails (yellow lines) follow pedestrian movements.
   - Virtual `Restricted_Zone_A` (orange polygon) and `Perimeter_Tripwire` (cyan line) are drawn.
   - Top HUD banner shows current frame number, latency (ms), and estimated FPS.
2. **Terminal Console Output**:
   - Structured JSON alert emitted immediately upon zone entry/exit:
     ```json
     [EVENT TRIGGERED] >>> {
       "event_id": "evt_enter_3_Restricted_Zone_A_1794928665643",
       "timestamp": 1794928665.643,
       "event_type": "ZONE_ENTER",
       "track_id": 3,
       "label": "person",
       "color": "white",
       "plate": null,
       "zone_id": "Restricted_Zone_A",
       "tripwire_id": null,
       "position": [42.5, 401.0],
       "bbox": [17, 246, 68, 401]
     }
     ```
   - 1-Second Telemetry Log:
     ```text
     [PERCEPTION] Frame #45 | Latency: 580.4ms | Active: 8 | Objects: ['ID:1 gray person', 'ID:2 blue person', ...]
     ```
3. **Exit**:
   - Press `q` inside the OpenCV preview window to shut down cleanly.

---

## 4. Suggested Datasets for Perception Benchmarking

Other engineers can evaluate and benchmark Module 1 against standard computer vision datasets:

### 1. Pedestrian & Crowd Tracking Datasets
- **MOT17 / MOT20 (Multiple Object Tracking Benchmark)**:
  - *URL*: [https://motchallenge.net](https://motchallenge.net)
  - *Purpose*: Stress-tests Kalman filter tracking under severe occlusion, camera jitter, and high crowd densities.
  - *Testing Tip*: Convert MOT video sequences into MP4 and point `mock_server.py` to test track persistence across heavy pedestrian crossovers.
- **Oxford Town Centre Dataset**:
  - *URL*: [https://www.robots.ox.ac.uk/~vgg/data/data-were-not-search.html](https://www.robots.ox.ac.uk/~vgg/data/data-were-not-search.html)
  - *Purpose*: High-angle semi-crowded street surveillance with real-world walking patterns, ideal for testing virtual perimeter tripwires and zone dwell times.

### 2. Vehicle Detection & Traffic Monitoring Datasets
- **UA-DETRAC Benchmark**:
  - *URL*: [https://detrac-db.rit.albany.edu](https://detrac-db.rit.albany.edu)
  - *Purpose*: Real-world traffic surveillance captured at varying times of day (sunny, rainy, night), challenging color classification and vehicle detection under headlights and glare.
- **BIT-Vehicle Dataset**:
  - *Purpose*: Multi-class vehicle categorization (`bus`, `car`, `truck`, `suv`) with high-resolution vehicle crops to test dominant vehicle color extraction.

### 3. Automated License Plate Recognition (ALPR) Datasets
- **CCPD (Chinese City Parking Dataset)**:
  - *URL*: [https://github.com/detectRecog/CCPD](https://github.com/detectRecog/CCPD)
  - *Purpose*: 250k+ diverse parking surveillance images with extreme tilts, bad lighting, and varying plate distances.
- **UFPR-AMPR Dataset**:
  - *URL*: [https://web.inf.ufpr.br/vri/databases/ufpr-ampr/](https://web.inf.ufpr.br/vri/databases/ufpr-ampr/)
  - *Purpose*: Brazilian multi-camera vehicular footage featuring motorcycles, cars, and buses with forward and rear plates.
- **OpenALPR Benchmark**:
  - *Purpose*: Standardized test suite for alphanumeric character recognition accuracy and OCR inference speed on CPU.

### 4. Synthetic & Stress-Test Generation
Engineers can generate custom test videos locally using FFmpeg or OpenCV:
```powershell
# Generate a test stream at 15 FPS portrait (360x640) from any input video:
ffmpeg -i my_surveillance_source.mp4 -vf "scale=360:640" -r 15 -c:v libx264 -pix_fmt yuv420p mock_edge_source.mp4
```

---

## 5. Troubleshooting & Common Issues

| Issue | Root Cause | Solution |
| :--- | :--- | :--- |
| `Address already in use (port 6000)` | A previous `mock_server.py` instance is still running in the background. | Run `Get-NetTCPConnection -LocalPort 6000` to find the PID, then `Stop-Process -Id <PID> -Force`. |
| `Protobuf parsing failed` on `yolov8n.onnx` | Model file was partially downloaded or corrupted. | Delete `yolov8n.onnx`. `detector.py` will automatically re-download the clean ~12.8 MB release asset from GitHub. |
| `cv2.error: (-215) size.width>0` in `imshow` | Running in a headless environment without an active desktop session. | The pipeline catches `cv2.error` gracefully; verify outputs directly via console JSON telemetry. |
| Memory consumption exceeds limits | Frame buffer backlog under high load. | The TCP receiver reads exact packet byte envelopes sequentially; ensure `mock_server.py` adheres to the 15 FPS pacing (`time.sleep(1/15)`). |
