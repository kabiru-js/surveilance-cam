import time
from typing import Dict, Any, List, Optional

from event_store import TimeSeriesEventStore
from cep_engine import ComplexEventProcessor
from notification_bus import NotificationBus

class EventAutomationEngine:
    """
    Module 3 Orchestrator — Stateful Event & Automation Engine.
    Fuses persistent time-series event ingestion, stateful complex event processing (CEP),
    and real-time WebSocket alert broadcasting.
    """
    def __init__(
        self,
        db_path: str = "spectra_events.db",
        ws_host: str = "127.0.0.1",
        ws_port: int = 8765,
        start_bus: bool = True,
        cooldown_seconds: float = 60.0
    ):
        self.store = TimeSeriesEventStore(db_path=db_path)
        self.cep = ComplexEventProcessor(cooldown_seconds=cooldown_seconds)
        self.bus = NotificationBus(host=ws_host, port=ws_port)
        if start_bus:
            self.bus.start()

    def process_event(self, enriched_event: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Processes an incoming enriched observation event from Module 2:
        1. Persists the observation in the time-series store.
        2. Evaluates stateful rules via the Complex Event Processor.
        3. Persists any triggered alerts and broadcasts them over the notification bus.
        Returns a list of triggered alerts.
        """
        # 1. Persist observation event
        self.store.insert_observation(
            event_id=enriched_event.get("event_id"),
            timestamp=enriched_event.get("timestamp"),
            global_entity_id=enriched_event.get("global_entity_id"),
            camera_id=enriched_event.get("camera_id", "UNKNOWN"),
            track_id=enriched_event.get("track_id", -1),
            label=enriched_event.get("label", "unknown"),
            color=enriched_event.get("color"),
            plate=enriched_event.get("plate"),
            zone_id=enriched_event.get("zone_id"),
            event_type=enriched_event.get("event_type", "OBSERVATION")
        )

        # 2. Evaluate CEP rules
        triggered_alerts = self.cep.process_event(enriched_event)

        # 3. Persist and broadcast alerts
        for alert in triggered_alerts:
            self.store.insert_alert(
                alert_id=alert.get("alert_id"),
                timestamp=alert.get("timestamp"),
                rule_name=alert.get("rule_name"),
                severity=alert.get("severity"),
                global_entity_id=alert.get("global_entity_id"),
                camera_id=alert.get("camera_id"),
                zone_id=alert.get("zone_id"),
                message=alert.get("message"),
                metadata=alert.get("metadata")
            )
            # Broadcast over WebSocket bus to connected dashboards
            self.bus.broadcast_alert(alert)

        return triggered_alerts

    def close(self):
        """Clean shutdown of database connections and background WebSocket server."""
        self.store.close()
        self.bus.stop()
