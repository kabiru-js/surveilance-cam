# How to Test Module 3 — Stateful Event & Automation Engine

This guide provides step-by-step instructions for testing and validating **Module 3 (Stateful Event & Automation Engine)** of Spectra Safety V1.

---

## 1. Architecture & Sub-Modules Under Test

Module 3 ingests enriched spatial-temporal observation events from Module 2, records them in a high-speed time-series store, evaluates real-time stateful security rules using in-memory sliding sessions, enforces alert deduplication cooldowns, and broadcasts alerts instantaneously over a WebSocket bus.

| File | Sub-Module | Responsibility | Hardware Target |
| :--- | :--- | :--- | :--- |
| `event_store.py` | Sub-Module 3.1 | High-speed, zero-dependency time-series DB backed by SQLite in WAL mode with composite indices | Pure SQLite WAL (<1ms write) |
| `cep_engine.py` | Sub-Module 3.2 | In-memory Complex Event Processing (CEP) state machine tracking zone dwell times, cooldown deduplication, and security rules | Pure Python (<0.1ms per event) |
| `notification_bus.py` | Sub-Module 3.3 | Real-time asynchronous WebSocket broadcast server on `127.0.0.1:8765` with dry-run queue fallback | Asyncio + WebSockets |
| `automation_engine.py` | Orchestration | `EventAutomationEngine` unifying time-series persistence, CEP rule evaluation, and alert dispatch | CPU |
| `test_module_3.py` | Unit Test | Automated test suite validating Scenarios A, B, C, and D | Headless / CI-ready |

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
python -c "import sqlite3, asyncio, websockets, json; print('All automation module dependencies loaded successfully!')"
```

> **Note**: Do **NOT** install heavy message brokers (Kafka, RabbitMQ, Mosquitto) or distributed CEP engines (Flink, Bytewax). Module 3 is engineered for zero-dependency edge hardware.

---

## 3. Testing Options

### Option A: Automated Scenario Test Suite (Headless / CI)

Run the comprehensive unit test suite:

```powershell
cd C:\Users\USER\spectra_v1\module_3_automation
python test_module_3.py
```

#### What It Tests:

1. **Scenario A — Immediate Restricted Zone Breach (`RESTRICTED_ZONE_BREACH`)**:
   - Person `ent_per_01` enters `Restricted_Zone_A` at $t=1000.0\text{ s}$.
   - **Assertion**: Immediately triggers a **HIGH** severity alert.
2. **Scenario B — Alert Deduplication & Cooldown Recovery**:
   - Sighting of `ent_per_01` 10 seconds later at $t=1010.0\text{ s}$ ($\Delta t = 10\text{ s} < 60\text{ s}$ cooldown window).
   - **Assertion**: Duplicate alert is **suppressed** (0 alerts fired).
   - Sighting at $t=1065.0\text{ s}$ ($\Delta t = 65\text{ s} \ge 60\text{ s}$ cooldown expired).
   - **Assertion**: Re-triggering is **permitted** (1 alert fired).
3. **Scenario C — Loitering Dwell Time (`LOITERING_DWELL`)**:
   - Person `ent_per_02` enters `CAM_LOBBY` at $t=2000.0\text{ s}$.
   - At $t=2050.0\text{ s}$ (dwell = $50\text{ s} \le 120\text{ s}$): **0 alerts** (dwell threshold not met).
   - At $t=2125.0\text{ s}$ (dwell = $125\text{ s} > 120\text{ s}$ limit): **MEDIUM** severity loitering alert fires.
4. **Scenario D — Unauthorized Vehicle Parking (`UNAUTHORIZED_PARKING`) & Persistence**:
   - Vehicle `ent_car_01` (`CA8899`) enters `Dropoff_Zone` at $t=3000.0\text{ s}$.
   - At $t=3305.0\text{ s}$ (dwell = $305\text{ s} > 300\text{ s}$ limit): **LOW** severity parking alert fires.
   - **Database State Assertion**: Verifies exactly 8 observations and 4 alerts are persisted and queryable in SQLite with JSON metadata.

#### Expected Output:
```text
======================================================================
SPECTRA SAFETY V1 — MODULE 3 (EVENT & AUTOMATION) UNIT TESTS
======================================================================

[SCENARIO A] Immediate Restricted Zone Breach:
 -> Person (ent_per_01) enters Restricted_Zone_A at t=1000.0s
 -> Triggered Alerts: 1
 -> Alert: [HIGH] RESTRICTED_ZONE_BREACH - CRITICAL: Unauthorized person ent_per_01 breached Restricted_Zone_A on CAM_GATE_A
[PASS] Scenario A: Immediate high-severity breach alert triggered successfully.

[SCENARIO B] Alert Deduplication & Cooldown Recovery:
 -> Sighting 10s later at t=1010.0s (within 60s cooldown window)
 -> Triggered Alerts at t=1010s: 0 (Expected: 0)
 -> Duplicate alert successfully suppressed by 60s cooldown: PASSED
 -> Sighting 65s later at t=1065.0s (cooldown window expired)
 -> Triggered Alerts at t=1065s: 1 (Expected: 1)
[PASS] Scenario B: Deduplication and cooldown recovery validated.

[SCENARIO C] Loitering Dwell Time Verification (120s limit):
 -> Person (ent_per_02) enters CAM_LOBBY at t=2000.0s
 -> Person still in CAM_LOBBY at t=2050.0s (dwell = 50s <= 120s)
 -> Triggered Alerts at 50s dwell: 0 (Expected: 0)
 -> Person still in CAM_LOBBY at t=2125.0s (dwell = 125s > 120s)
 -> Triggered Alerts at 125s dwell: 1 (Expected: 1)
 -> Alert: [MEDIUM] WARNING: Loitering detected. Person ent_per_02 remained in CAM_LOBBY for 125.0s (>120s limit)
[PASS] Scenario C: Loitering temporal dwell threshold validated.

[SCENARIO D] Unauthorized Vehicle Parking (300s limit) & DB Verification:
 -> Car (ent_car_01, CA8899) enters Dropoff_Zone at t=3000.0s
 -> Car stationary in Dropoff_Zone at t=3305.0s (dwell = 305s > 300s)
 -> Triggered Alerts at 305s dwell: 1 (Expected: 1)
 -> Alert: [LOW] NOTICE: Vehicle ent_car_01 (CA8899) stationary in Dropoff_Zone for 305.0s (>300s limit)
[PASS] Scenario D: Unauthorized vehicle parking validated.

[PERSISTENCE VERIFICATION] Checking SQLite Database:
 -> Total Observations in SQLite: 8 (Expected: 8)
 -> Total Alerts in SQLite: 4 (Expected: 4)
 -> Stored Alert Rules: ['UNAUTHORIZED_PARKING', 'LOITERING_DWELL', 'RESTRICTED_ZONE_BREACH', 'RESTRICTED_ZONE_BREACH']

======================================================================
ALL MODULE 3 UNIT TEST SCENARIOS PASSED PERFECTLY!
======================================================================
```

---

### Option B: Interactive Live WebSocket Dashboard Simulation

Test real-time alert dispatch to external clients:

1. **Start the Automation Engine in interactive Python**:
   ```python
   from automation_engine import EventAutomationEngine
   engine = EventAutomationEngine(ws_port=8765)
   ```

2. **Connect any WebSocket client or browser console** to `ws://127.0.0.1:8765`:
   ```javascript
   const ws = new WebSocket("ws://127.0.0.1:8765");
   ws.onmessage = (event) => console.log("LIVE ALERT RECEIVED:", JSON.parse(event.data));
   ```

3. **Inject an observation event**:
   ```python
   engine.process_event({
       "global_entity_id": "ent_per_live",
       "camera_id": "CAM_GATE_A",
       "track_id": 99,
       "label": "person",
       "color": "red",
       "zone_id": "Restricted_Zone_A",
       "event_type": "ZONE_ENTER"
   })
   ```
   *The connected WebSocket client will immediately receive the JSON alert payload.*

---

## 4. Suggested Datasets & Benchmark Traces for CEP Stress Testing

1. **High-Throughput Synthetic Event Streams**:
   - Generate synthetic JSON observation streams at 1,000+ events/sec to benchmark SQLite WAL concurrency, write latencies, and in-memory session lookups.
2. **Real-World Access Control & Turnstile Event Logs**:
   - Benchmark ingress/egress dwell times against enterprise access control traces.
3. **Adversarial Ordering Scenarios**:
   - Out-of-order event timestamps and intermittent zone departures/re-entries to test state recovery and session resilience.

---

## 5. Troubleshooting & Configuration

| Scenario | Behavior / Solution |
| :--- | :--- |
| `Address already in use (port 8765)` | A previous WebSocket server is running. Pass `ws_port=8766` to `EventAutomationEngine` or terminate the lingering process via `Stop-Process`. |
| Zero dashboard clients connected | The `NotificationBus` handles dry-run mode gracefully, buffering recent alerts in an in-memory deque without throwing exceptions. |
| Database locked error | SQLite is configured in WAL mode (`PRAGMA journal_mode=WAL;`), which supports concurrent readers and a writer. Avoid external file locks. |
