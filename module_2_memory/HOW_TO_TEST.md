# How to Test Module 2 — Spatial-Temporal Memory Sandbox

This guide provides step-by-step instructions for testing and validating **Module 2 (Spatial-Temporal Memory Sandbox)** of Spectra Safety V1.

---

## 1. Architecture & Sub-Modules Under Test

Module 2 provides cross-camera entity tracking and long-term memory. It ingests observation events from Module 1 (`label`, `color`, `plate`, `crop_bgr`), extracts visual embeddings, validates physical transit plausibility through a topological camera graph, and either links sightings to an existing entity or mints a new global identity.

| File | Sub-Module | Responsibility | Hardware Target |
| :--- | :--- | :--- | :--- |
| `reid_extractor.py` | Sub-Module 2.1 | 128-dim L2-normalized float32 feature extraction (multi-scale spatial HSV + gradient) | Pure NumPy + OpenCV (<5ms CPU) |
| `vector_store.py` | Sub-Module 2.2 | SQLite persistent vector storage + NumPy matrix dot-product similarity search | Zero-dependency local SQLite |
| `camera_graph.py` | Sub-Module 2.3 | NetworkX directed camera topology, shortest-path calculation, kinematic velocity validation | Pure Python + NetworkX |
| `memory_engine.py` | Orchestration | `SpatialTemporalMemoryEngine` unifying visual search, topological filtering, and entity resolution | CPU |
| `test_module_2.py` | Unit Test | Automated test suite validating cross-camera linking, teleportation defense, and appearance mismatch | Headless / CI-ready |

---

## 2. Environment & Prerequisites

Execute all commands using the lightweight virtual environment located at `module_1_perception/stream_env`:

```powershell
# Navigate to repository root
cd C:\Users\USER\spectra_v1

# Activate virtual environment in PowerShell
.\module_1_perception\stream_env\Scripts\Activate.ps1
```

### Required Dependencies
Verify that all lightweight CPU libraries are installed:
```powershell
python -c "import numpy, cv2, networkx, sqlite3; print('All memory module dependencies loaded successfully!')"
```

> **Note**: Do **NOT** install PyTorch, Torchvision, Qdrant, Milvus, or Redis. Module 2 uses pure NumPy matrix vector search backed by persistent SQLite.

---

## 3. Testing Options

### Option A: Automated Unit Test Suite (Headless / CI)

Run the comprehensive unit test script:

```powershell
cd C:\Users\USER\spectra_v1\module_2_memory
python test_module_2.py
```

#### What It Tests:

1. **Step 0 — Baseline Entity Minting**:
   - Observes Person A (Blue shirt, dark pants) at `CAM_GATE_A` at $t=1000.0\text{ s}$.
   - Verifies that a `NEW` entity ID (e.g. `ent_per_1000_xxxxxx`) is generated and stored in SQLite.
2. **Scenario A — Valid Cross-Camera Linking**:
   - Observes Person A at `CAM_LOBBY` (50 meters away) at $t=1040.0\text{ s}$ ($\Delta t = 40.0\text{ s}$).
   - Implied velocity = $50\text{ m} / 40\text{ s} = 1.25\text{ m/s}$ (normal walking pace $\le 5.0\text{ m/s}$).
   - **Assertion**: Action must be `LINKED` to the **SAME** `global_entity_id`.
3. **Scenario B — Impossible Teleportation Defense**:
   - Observes a visually identical crop at `CAM_PARKING_LOT` (300 meters away) at $t=1005.0\text{ s}$ (only 5 seconds after $t=1000.0\text{ s}$).
   - Implied velocity = $300\text{ m} / 5\text{ s} = 60.0\text{ m/s}$ (216 km/h, physically impossible on foot).
   - **Assertion**: Must **REJECT** the spatial-temporal match and mint a **NEW** `global_entity_id`, flagging a teleportation violation.
4. **Scenario C — Visual Appearance Mismatch Defense**:
   - Observes Person B (Bright Red shirt, white pants) at `CAM_LOBBY` at $t=1050.0\text{ s}$.
   - Visual cosine similarity between Blue and Red crops drops to $\sim 0.64$, below the $0.75$ threshold.
   - **Assertion**: Must mint a **NEW** `global_entity_id`, confirming different people are not conflated.

#### Expected Output:
```text
======================================================================
SPECTRA SAFETY V1 — MODULE 2 (SPATIAL-TEMPORAL MEMORY) UNIT TESTS
======================================================================

[STEP 0] Baseline Observation: Person A seen at CAM_GATE_A at t=1000.0s
 -> Result: Action=NEW, EntityID=ent_per_1000_3b1dd4

[SCENARIO A] Valid Cross-Camera Transit:
 -> Person A observed at CAM_LOBBY at t=1040.0s (50m away, delta_t=40s)
 -> Action: LINKED
 -> EntityID: ent_per_1000_3b1dd4
 -> Match Details: {
  "matched_entity_id": "ent_per_1000_3b1dd4",
  "matched_camera": "CAM_GATE_A",
  "matched_timestamp": 1000.0,
  "visual_similarity": 1.0,
  "spatial_temporal_status": "Plausible transition: 50.0m in 40.0s (1.25 m/s)"
}
[PASS] Scenario A: Successfully linked Person A across CAM_GATE_A and CAM_LOBBY.

[SCENARIO B] Impossible Teleportation Defense:
 -> Visually identical crop at CAM_PARKING_LOT at t=1005.0s (300m away, delta_t=5s)
 -> Action: NEW
 -> EntityID: ent_per_1005_3e7b2d
 -> Status: Rejected visual match with ent_per_1000_3b1dd4: Teleportation violation: speed 60.0 m/s exceeds max threshold 5.0 m/s (300.0m in 5.0s)
[PASS] Scenario B: Impossible teleportation detected and rejected as spoofing/twin.

[SCENARIO C] Visual Appearance Mismatch Defense:
 -> Visually distinct Person B (Red shirt) observed at CAM_LOBBY at t=1050.0s
 -> Action: NEW
 -> EntityID: ent_per_1050_6da227
 -> Status: No visual match above threshold
[PASS] Scenario C: Distinct appearance correctly minted as new independent entity.

======================================================================
ALL MODULE 2 UNIT TEST SCENARIOS PASSED PERFECTLY!
======================================================================
```

---

### Option B: Interactive Vehicle & License Plate Memory Testing

To test vehicular cross-camera tracking and exact plate matching:

```powershell
cd C:\Users\USER\spectra_v1\module_2_memory
python -c "
from memory_engine import SpatialTemporalMemoryEngine
import numpy as np

engine = SpatialTemporalMemoryEngine(db_path=':memory:', min_similarity=0.75)
dummy_crop = np.full((120, 120, 3), 150, dtype=np.uint8)

# 1. Car seen at Entrance Gate at t=1000s
r1 = engine.process_observation('CAM_GATE_A', 1000.0, 1, 'car', color='gray', plate='CA8899', crop_bgr=dummy_crop)
print('Gate observation:', r1['action'], r1['global_entity_id'])

# 2. Same Car seen at Parking Lot (300m) at t=1025s (speed = 12 m/s <= 18 m/s vehicle limit)
r2 = engine.process_observation('CAM_PARKING_LOT', 1025.0, 4, 'car', color='gray', plate='CA8899', crop_bgr=dummy_crop)
print('Parking observation:', r2['action'], r2['global_entity_id'])

assert r2['action'] == 'LINKED'
assert r2['global_entity_id'] == r1['global_entity_id']
print('Vehicle Cross-Camera Tracking: PASSED!')
"
```

---

## 4. Suggested Datasets for Re-ID & Spatial-Temporal Benchmarking

Other engineers can test and benchmark Module 2 with standard computer vision and surveillance datasets:

### 1. Person Re-Identification (Re-ID) Datasets
- **Market-1501**:
  - *URL*: [https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html](https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html)
  - *Purpose*: 32,668 bounding boxes of 1,501 identities across 6 distinct camera views.
  - *Testing Tip*: Feed pedestrian crops from different camera views into `VisualFingerprintExtractor.extract()` to evaluate cross-camera ranking (Rank-1, mAP).
- **DukeMTMC-reID**:
  - *Purpose*: Multi-camera pedestrian tracking footage across 8 outdoor campus cameras, with varying viewpoints, walking speeds, and complex background shadows.
- **MSMT17 (Multi-Scene Multi-Time)**:
  - *Purpose*: 126k bounding boxes across 15 cameras (12 outdoor, 3 indoor) over multiple days and times, testing illumination invariance and color constancy.

### 2. Vehicle Re-Identification & License Plate Datasets
- **VeRi-776 Benchmark**:
  - *URL*: [https://vehiclereid.github.io/VeRi/](https://vehiclereid.github.io/VeRi/)
  - *Purpose*: 50,000 track images of 776 vehicles captured by 20 cameras in an urban road network with real-world spatial-temporal camera timestamps.
  - *Testing Tip*: Ideal for benchmarking the combined `plate` + visual fingerprint + `CameraGraph` kinematic speed filter.
- **CityFlow (AI City Challenge)**:
  - *Purpose*: Multi-camera traffic video across multiple intersections, ideal for evaluating vehicle routing across dense camera graphs.

### 3. Synthetic Multi-Camera Trajectory Logs
Engineers can test the database and graph scaling by generating synthetic JSON trajectory logs:
```python
# Example snippet for generating synthetic camera hops:
import random, time

cameras = ["CAM_GATE_A", "CAM_DRIVEWAY", "CAM_PARKING_LOT", "CAM_LOBBY"]
for i in range(1000):
    cam = random.choice(cameras)
    ts = time.time() + i * 15
    # Process simulated events through SpatialTemporalMemoryEngine
```

---

## 5. Troubleshooting & Configuration

| Scenario | Behavior / Solution |
| :--- | :--- |
| **Swapping to a trained Re-ID ONNX model** | Pass `reid_onnx_path="path/to/model.onnx"` to `VisualFingerprintExtractor` or `SpatialTemporalMemoryEngine`. If the file exists, it uses ONNX Runtime CPU; otherwise, it falls back to the native multi-scale spatial descriptor. |
| **Tuning Similarity Sensitivity** | Default similarity threshold is `0.75`. For noisy or wide-angle cameras, adjust `min_similarity` in `SpatialTemporalMemoryEngine(min_similarity=0.70)`. |
| **Adding New Estate Cameras** | Call `engine.camera_graph.add_camera("CAM_NEW", zone="Perimeter")` and `engine.camera_graph.add_edge("CAM_GATE_A", "CAM_NEW", distance_meters=85.0)`. |
| **Resetting Persistent Database** | Delete `memory_vectors.db` or initialize with `db_path=":memory:"` for ephemeral in-memory unit tests. |
