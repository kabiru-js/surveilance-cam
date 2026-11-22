import asyncio
import threading
import json
import time
from collections import deque
from typing import Dict, Any, List, Set, Optional
import websockets

class NotificationBus:
    """
    Real-time asynchronous WebSocket notification and alert bus.
    Broadcasts security alert payloads instantaneously to connected dashboard clients
    on 127.0.0.1:8765. Features a dry-run queue fallback so the engine runs
    seamlessly even when zero dashboard clients are connected.
    """
    def __init__(self, host: str = "127.0.0.1", port: int = 8765, queue_size: int = 100):
        self.host = host
        self.port = port
        self.clients: Set[Any] = set()
        self.alert_queue: deque = deque(maxlen=queue_size)
        self.server = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

    async def _client_handler(self, websocket):
        """
        Handles incoming WebSocket client connection lifecycle.
        """
        with self._lock:
            self.clients.add(websocket)

        # Welcome message + replay recent alerts
        try:
            welcome_payload = {
                "type": "CONNECTION_ESTABLISHED",
                "timestamp": time.time(),
                "message": "Connected to Spectra Safety V1 Real-Time Alert Bus",
                "buffered_alerts_count": len(self.alert_queue)
            }
            await websocket.send(json.dumps(welcome_payload))

            # Keep connection open until client disconnects
            await websocket.wait_closed()
        except Exception:
            pass
        finally:
            with self._lock:
                self.clients.discard(websocket)

    def start(self):
        """
        Starts the WebSocket server in a background daemon thread.
        """
        if self._running:
            return

        self._running = True
        ready_event = threading.Event()

        def _run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

            async def _start_server():
                try:
                    self.server = await websockets.serve(
                        self._client_handler,
                        self.host,
                        self.port
                    )
                    ready_event.set()
                    await self.server.wait_closed()
                except Exception as e:
                    ready_event.set()

            self.loop.run_until_complete(_start_server())

        self.thread = threading.Thread(target=_run_loop, daemon=True, name="NotificationBusThread")
        self.thread.start()
        ready_event.wait(timeout=3.0)

    def broadcast_alert(self, alert_dict: Dict[str, Any]):
        """
        Broadcasts an alert dictionary to all connected dashboard clients.
        If zero clients are connected, buffers the alert in the dry-run queue without error.
        Thread-safe: can be called from any thread.
        """
        with self._lock:
            self.alert_queue.append(alert_dict)

        payload_str = json.dumps(alert_dict)

        with self._lock:
            active_clients = set(self.clients)

        if not active_clients or not self.loop or not self.loop.is_running():
            return  # Clean dry-run fallback

        # Schedule broadcast on the server's event loop
        def _send():
            try:
                websockets.broadcast(active_clients, payload_str)
            except Exception:
                pass

        self.loop.call_soon_threadsafe(_send)

    def get_connected_clients_count(self) -> int:
        with self._lock:
            return len(self.clients)

    def get_queued_alerts(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            items = list(self.alert_queue)
        return items[-limit:]

    def stop(self):
        """
        Shuts down the WebSocket server and closes active connections.
        """
        if not self._running:
            return

        self._running = False
        if self.server and self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.server.close)

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        with self._lock:
            self.clients.clear()
