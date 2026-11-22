import sqlite3
import time
import uuid
import json
from typing import List, Dict, Any, Optional

class TimeSeriesEventStore:
    """
    High-speed, zero-dependency time-series database backed by SQLite in WAL mode.
    Maintains append-only observation streams and persistent security alerts with
    composite indices on (timestamp, camera_id) and (global_entity_id, timestamp).
    """
    def __init__(self, db_path: str = "spectra_events.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        with self.conn:
            # WAL mode for high concurrency read/write without locks
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute("PRAGMA synchronous=NORMAL;")
            
            # 1. Observations Table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS observations (
                    event_id TEXT PRIMARY KEY,
                    timestamp REAL NOT NULL,
                    global_entity_id TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    label TEXT NOT NULL,
                    color TEXT,
                    plate TEXT,
                    zone_id TEXT,
                    event_type TEXT NOT NULL
                );
            """)
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_obs_ts_cam ON observations(timestamp, camera_id);")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_obs_entity_ts ON observations(global_entity_id, timestamp);")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_obs_zone ON observations(zone_id);")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_obs_label ON observations(label);")

            # 2. Alerts Table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    timestamp REAL NOT NULL,
                    rule_name TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    global_entity_id TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    zone_id TEXT,
                    message TEXT NOT NULL,
                    metadata_json TEXT
                );
            """)
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(timestamp DESC);")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_entity_ts ON alerts(global_entity_id, timestamp);")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_rule ON alerts(rule_name);")

    def insert_observation(
        self,
        global_entity_id: str,
        camera_id: str,
        track_id: int,
        label: str,
        timestamp: Optional[float] = None,
        color: Optional[str] = None,
        plate: Optional[str] = None,
        zone_id: Optional[str] = None,
        event_type: str = "OBSERVATION",
        event_id: Optional[str] = None
    ) -> str:
        """
        Inserts a single observation event into the time-series store.
        """
        if event_id is None:
            event_id = f"evt_{uuid.uuid4().hex[:12]}"
        if timestamp is None:
            timestamp = time.time()

        with self.conn:
            self.conn.execute(
                """
                INSERT INTO observations (
                    event_id, timestamp, global_entity_id, camera_id,
                    track_id, label, color, plate, zone_id, event_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    float(timestamp),
                    global_entity_id,
                    camera_id,
                    int(track_id),
                    label.lower(),
                    color,
                    plate,
                    zone_id,
                    event_type.upper()
                )
            )
        return event_id

    def insert_alert(
        self,
        rule_name: str,
        severity: str,
        global_entity_id: str,
        camera_id: str,
        message: str,
        timestamp: Optional[float] = None,
        zone_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        alert_id: Optional[str] = None
    ) -> str:
        """
        Inserts a triggered security alert into the alerts store.
        """
        if alert_id is None:
            alert_id = f"alt_{uuid.uuid4().hex[:12]}"
        if timestamp is None:
            timestamp = time.time()

        meta_str = json.dumps(metadata) if metadata is not None else None

        with self.conn:
            self.conn.execute(
                """
                INSERT INTO alerts (
                    alert_id, timestamp, rule_name, severity,
                    global_entity_id, camera_id, zone_id, message, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alert_id,
                    float(timestamp),
                    rule_name.upper(),
                    severity.upper(),
                    global_entity_id,
                    camera_id,
                    zone_id,
                    message,
                    meta_str
                )
            )
        return alert_id

    def query_events(
        self,
        start_time: float,
        end_time: float,
        label: Optional[str] = None,
        zone_id: Optional[str] = None,
        camera_id: Optional[str] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Queries time-series observation events within a temporal window [start_time, end_time].
        """
        query = """
            SELECT event_id, timestamp, global_entity_id, camera_id,
                   track_id, label, color, plate, zone_id, event_type
            FROM observations
            WHERE timestamp >= ? AND timestamp <= ?
        """
        params: List[Any] = [float(start_time), float(end_time)]

        if label:
            query += " AND label = ?"
            params.append(label.lower())
        if zone_id:
            query += " AND zone_id = ?"
            params.append(zone_id)
        if camera_id:
            query += " AND camera_id = ?"
            params.append(camera_id)

        query += " ORDER BY timestamp ASC LIMIT ?"
        params.append(limit)

        cursor = self.conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_recent_alerts(
        self,
        limit: int = 20,
        severity: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves the most recent security alerts ordered by timestamp descending.
        """
        query = """
            SELECT alert_id, timestamp, rule_name, severity,
                   global_entity_id, camera_id, zone_id, message, metadata_json
            FROM alerts
        """
        params: List[Any] = []

        if severity:
            query += " WHERE severity = ?"
            params.append(severity.upper())

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor = self.conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()

        results = []
        for row in rows:
            d = dict(row)
            if d["metadata_json"]:
                try:
                    d["metadata"] = json.loads(d["metadata_json"])
                except Exception:
                    d["metadata"] = {}
            else:
                d["metadata"] = {}
            del d["metadata_json"]
            results.append(d)
        return results

    def count_observations(self) -> int:
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM observations;")
        return cursor.fetchone()[0]

    def count_alerts(self) -> int:
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM alerts;")
        return cursor.fetchone()[0]

    def close(self):
        if self.conn:
            self.conn.close()
